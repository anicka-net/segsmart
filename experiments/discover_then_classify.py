"""SECRET LAB EXPERIMENT — the methodologically 'correct' order (instructor's note).

SegSmart ships RFM RULES first (impose 5 named segments) and uses KMeans as a
cross-check. The cleaner research pipeline is the reverse: DISCOVER with
unsupervised first (no preconceptions — let the data say how many groups and
which), then put a SUPERVISED model ON TOP to operationalise them (assign new
customers + explain what drives membership). This tests whether doing it that
way agrees with — or overturns — the hand-written rules.

  step 1  let the data choose k        (silhouette scan, no imposed 5)
  step 2  discover clusters             (KMeans at the chosen k)
  step 3  supervised on top             (shallow decision tree predicts the
                                         cluster from RAW RFM features -> readable
                                         rules + feature importances)
  step 4  compare to the imposed rules  (ARI + cluster × named-segment cross-tab)

No new deps (numpy + sklearn we already ship). Nothing here is imported by the
product — it's the "we took the feedback and measured it" branch.

Run:  python3 -m experiments.discover_then_classify
      python3 -m experiments.discover_then_classify --dataset uci

RESULTS — the answer SPLITS by data (and vindicates BOTH the rules and the note):
  synthetic (5 designed archetypes): the data INDEPENDENTLY picks k=5 (silhouette
    peak), ARI 0.557 (strong) with the hand-written rules, and a depth-3 tree
    reproduces 99% of the clusters from 6 rules. Drivers: monetary 62%, recency
    27%, AOV 10%, frequency ~0%. Discover-first also SPLITS Champions into normal
    vs very-high-AOV whales/B2B (AOV>~5.7k) — a sub-segment the rules don't name.
  real UCI (messy): the data only robustly supports k=2 (big spenders vs rest),
    ARI 0.18 (weak); the 5 named segments are FINER than the geometry naturally
    supports. monetary alone = 92% of the split.
  both: a tiny tree reproduces the clusters 94–99% -> the discovered segments are
    simple & rule-describable (validates the interpretable-rules choice), and the
    supervised-on-top tree hands you data-derived thresholds + a deployable scorer
    for new customers.
  caveat: silhouette leans toward low k on heavy-tailed RFM data, so "real data
    wants k=2" is geometry+metric, not gospel — but the contrast is real.
  takeaway for the instructor's note: discover-first is the honest order; it
    CONFIRMS the rules on clean data, reveals they're finer than real data
    supports, and surfaces structure (the whale sub-segment) + that frequency
    pulls least weight. Keep readable rules for the owner; use discover-first as
    the validation/insight layer on top.
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "uci"])
    args = ap.parse_args()

    from seg.features import build_features, model_matrix, MODEL_FEATURES
    from seg.segment import rfm_segments
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    if args.dataset == "uci":
        from seg.loader import load_uci
        df = load_uci()
    else:
        from seg.loader import load_eshop
        df = load_eshop("data/synthetic_eshop.csv")

    feat = rfm_segments(build_features(df))
    X, cols = model_matrix(feat)               # scaled features the clustering uses
    n = len(feat)
    print(f"\ndataset: {args.dataset} — {n} customers, features: {cols}\n")

    # ── step 1: let the DATA choose how many segments (no imposed 5) ──────────
    print("step 1 — how many segments does the data want? (silhouette per k)")
    scan = {}
    for k in range(2, 9):
        if k >= n:
            break
        lbl = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
        scan[k] = float(silhouette_score(X, lbl, sample_size=min(3000, n), random_state=SEED))
    for k, s in scan.items():
        bar = "█" * int(round(s * 40))
        print(f"    k={k}  {s:.3f}  {bar}")
    best_k = max(scan, key=scan.get)
    print(f"  -> data prefers k = {best_k}  (RFM rules impose 5)\n")

    # ── step 2: discover at the data's chosen k ──────────────────────────────
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(X)
    clusters = km.labels_

    # ── step 4 (agreement): discovered clusters vs the hand-written rules ─────
    ari = adjusted_rand_score(feat["segment"], clusters)
    print(f"step 4 — agreement of DISCOVERED clusters with the IMPOSED RFM rules")
    print(f"  ARI(KMeans@{best_k}, RFM rules) = {ari:.3f}  "
          f"({'strong' if ari>0.5 else 'moderate' if ari>0.2 else 'weak'})\n")

    ct = pd.crosstab(clusters, feat["segment"])
    ct = ct[[c for c in ["Champions","Loyal","At-risk","New","Dormant"] if c in ct.columns]]
    print("  cluster × named-segment cross-tab (where each discovered cluster lands):")
    print("   " + ct.to_string().replace("\n", "\n   "))
    dominant = {cl: ct.loc[cl].idxmax() for cl in ct.index}
    clean = len(set(dominant.values()))
    print(f"\n  each discovered cluster's dominant rule-segment: {dominant}")
    print(f"  -> {clean} distinct named segments covered by {best_k} clusters "
          f"({'clean 1:1-ish map' if clean==best_k else 'some segments merge/split'})\n")

    # ── step 3: SUPERVISED on top — readable rules that reproduce the clusters ─
    raw = feat[MODEL_FEATURES]                 # RAW units -> human-readable thresholds
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(raw, clusters)
    acc = accuracy_score(clusters, tree.predict(raw))
    print(f"step 3 — supervised ON TOP: a depth-3 tree predicting the discovered cluster")
    print(f"  reproduces {acc*100:.1f}% of the clustering from {len(MODEL_FEATURES)} simple rules")
    imp = sorted(zip(MODEL_FEATURES, tree.feature_importances_), key=lambda t: -t[1])
    print("  what actually drives the discovered segments (feature importance):")
    for f, w in imp:
        if w > 0:
            print(f"    {f:<18} {w*100:4.0f}%  {'▮'*int(round(w*30))}")
    rfm_share = sum(w for f, w in imp if f in ("recency", "frequency", "monetary"))
    print(f"  -> R/F/M alone explain {rfm_share*100:.0f}% of the split "
          f"({'data agrees RFM are the right axes' if rfm_share>0.6 else 'data leans on other features too'})\n")

    print("  the DATA-DERIVED rules (learned, not hand-written):")
    print("   " + export_text(tree, feature_names=list(MODEL_FEATURES)).replace("\n", "\n   "))

    print("verdict: did discovering-first agree with the imposed rules? "
          "read the ARI + cross-tab + how few rules reproduce the clusters.")


if __name__ == "__main__":
    main()
