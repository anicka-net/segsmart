"""SECRET LAB EXPERIMENT — does a time-series foundation model (Chronos-Bolt)
beat lean classical forecasting on SegSmart's revenue curve?

Same predict-then-verify shape as the 2D-attention experiment: try the heavy,
fashionable thing against a cheap baseline on our ACTUAL data, and keep the
receipts. Not part of the product — torch + chronos are optional, nothing here
is imported by the pipeline.

  models:  naive (last value) · seasonal-naive(7) · ETS/Holt-Winters(7, statsmodels)
           · Chronos-Bolt zero-shot (amazon/chronos-bolt-small)
  task:    forecast daily revenue H days ahead, backtested over several origins
  metric:  MASE (vs in-sample seasonal-naive), sMAPE, MAE — lower is better

NOTE on deps: `pip install chronos-forecasting` pulls transformers<5, which can
clash with other envs (transformer-lens wants >=5.4). Run it in a throwaway venv,
or restore transformers afterwards. The whole point of this experiment is to
decide whether that weight is worth carrying — spoiler in the verdict.

Run:  python3 -m experiments.forecast_bakeoff            # synthetic (clean)
      python3 -m experiments.forecast_bakeoff --dataset uci   # real (messy)

RESULTS (4 origins × H=28, MASE, lower better) — the finding SPLITS by data:
  synthetic (clean weekly seasonality):  Chronos 0.694  <  ETS 0.761  -> Chronos wins
  real UCI  (spiky, holiday-driven):     ETS 1.256      <  Chronos 1.430 -> ETS wins
So the foundation model shines on regular series it has seen the shape of a
million times, but plain Holt-Winters wins on the messy real one — and ETS is
milliseconds with statsmodels we already have, vs Chronos's torch+transformers
(a dep clash with the research env) + a model download. Product call: ship ETS
if/when forecasting lands; Chronos stays a documented option for clean or
never-fit-before series. See docs/SYSTEM_CARD on main.
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H = 28              # forecast horizon (4 weeks)
ORIGINS = 4         # rolling backtest origins, stepped one horizon apart
M = 7               # weekly seasonality


def daily_series(source="synthetic"):
    """Daily revenue, gap-filled. source: 'synthetic' (20-month eshop) or 'uci'
    (real UK online-retail)."""
    import pandas as pd
    from seg.external import daily_sales
    if source == "uci":
        from seg.loader import load_uci
        df = load_uci()
    else:
        from seg.loader import load_eshop
        df = load_eshop("data/synthetic_eshop.csv")
    ds = daily_sales(df)
    s = pd.Series(ds["revenue"].to_numpy(),
                  index=pd.to_datetime(ds["date"])).sort_index()
    s = s.asfreq("D").fillna(0.0)          # explicit calendar, no missing days
    return s


# ---------------------------------------------------------------- forecasters
def f_naive(train, h):
    return np.repeat(train[-1], h)


def f_snaive(train, h, m=M):
    return np.array([train[-m + (i % m)] for i in range(h)])


def f_ets(train, h, m=M):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                                   seasonal_periods=m,
                                   initialization_method="estimated").fit()
    return np.asarray(fit.forecast(h))


def f_chronos(train, h, _cache={}):
    import torch
    if "pipe" not in _cache:
        from chronos import BaseChronosPipeline
        _cache["pipe"] = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    pipe = _cache["pipe"]
    ctx = torch.tensor(np.asarray(train, dtype=np.float32))
    q, mean = pipe.predict_quantiles(ctx, h, quantile_levels=[0.5])
    return mean[0].numpy()                  # zero-shot point forecast


# ---------------------------------------------------------------- metrics
def smape(y, f):
    d = np.abs(y) + np.abs(f)
    r = np.zeros_like(d, dtype=float)
    nz = d != 0
    r[nz] = 2 * np.abs(f - y)[nz] / d[nz]
    return float(np.mean(r) * 100)


def mae(y, f):
    return float(np.mean(np.abs(f - y)))


def mase(y, f, train, m=M):
    scale = np.mean(np.abs(train[m:] - train[:-m])) or 1.0   # in-sample snaive error
    return float(np.mean(np.abs(f - y)) / scale)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "uci"])
    args = ap.parse_args()
    print(f"dataset: {args.dataset}")
    s = daily_series(args.dataset)
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    print(f"daily revenue series: {n} days "
          f"({s.index[0].date()} → {s.index[-1].date()})\n")

    models = {"naive": f_naive, "snaive7": f_snaive, "ETS(7)": f_ets}
    try:
        import chronos  # noqa: F401
        models["Chronos-Bolt"] = f_chronos
    except ImportError:
        print("  [chronos not installed — skipping the foundation model]\n")

    # rolling-origin backtest
    acc = {k: {"MASE": [], "sMAPE": [], "MAE": []} for k in models}
    for o in range(ORIGINS):
        cut = n - H * (ORIGINS - o)         # train end
        if cut < 2 * M + 10:
            continue
        train, test = vals[:cut], vals[cut:cut + H]
        for name, fn in models.items():
            try:
                fc = np.asarray(fn(train, H), dtype=float)
            except Exception as e:
                print(f"  [{name} failed at origin {o}: {e}]")
                continue
            acc[name]["MASE"].append(mase(test, fc, train))
            acc[name]["sMAPE"].append(smape(test, fc))
            acc[name]["MAE"].append(mae(test, fc))

    print(f"backtest: {ORIGINS} origins × H={H} days, averaged\n")
    print(f"  {'model':<14} {'MASE':>7} {'sMAPE%':>8} {'MAE':>12}")
    rows = []
    for name in models:
        if not acc[name]["MASE"]:
            continue
        ms = np.mean(acc[name]["MASE"]); sm = np.mean(acc[name]["sMAPE"])
        ma = np.mean(acc[name]["MAE"])
        rows.append((name, ms, sm, ma))
        print(f"  {name:<14} {ms:>7.3f} {sm:>8.1f} {ma:>12.0f}")

    best = min(rows, key=lambda r: r[1])
    print(f"\nbest by MASE: {best[0]} ({best[1]:.3f})")
    if "Chronos-Bolt" in dict((r[0], r) for r in rows):
        ch = next(r for r in rows if r[0] == "Chronos-Bolt")
        cl = min((r for r in rows if r[0] != "Chronos-Bolt"), key=lambda r: r[1])
        verdict = ("Chronos earns its weight" if ch[1] < cl[1]
                   else f"classical ({cl[0]}) ties/wins — the foundation model's "
                        f"weight (torch + transformers, a dep clash) isn't worth it here")
        print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
