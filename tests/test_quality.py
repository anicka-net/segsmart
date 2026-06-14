"""Ingest report, honesty warnings, drill-down rows, segment migration."""
import json

import pandas as pd
import pytest

import pipeline
from seg.loader import load_dataframe


def _csv_df(rows, span_days=400, n_cust=8, value=100.0):
    # string cells throughout, like a real CSV read — also keeps the
    # corrupt-a-cell tests valid on pandas versions that refuse str->datetime64
    recs = []
    for i in range(rows):
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=(i * span_days) // rows)
        recs.append({"customer_id": f"c{i % n_cust}", "order_id": f"o{i}",
                     "order_date": day.strftime("%Y-%m-%d"),
                     "quantity": "1", "unit_price": str(value)})
    return pd.DataFrame(recs)


# --- ingest report -----------------------------------------------------------

def test_ingest_report_counts_drops():
    raw = _csv_df(50)
    raw.loc[3, "order_date"] = "not a date"
    raw.loc[7, "unit_price"] = "free!"
    raw.loc[9, "customer_id"] = ""
    df = load_dataframe(raw.astype(str), {})
    ing = df.attrs["ingest"]
    assert ing["rows_in"] == 50
    assert ing["rows_kept"] == 47
    assert ing["dropped"]["unparseable date"] == 1
    assert ing["dropped"]["unparseable price"] == 1
    assert ing["dropped"]["missing customer id"] == 1


def test_ingest_report_in_result():
    res = pipeline.analyze(load_dataframe(_csv_df(60), {}), use_llm=False, out=None)
    assert res["quality"]["ingest"]["rows_kept"] == 60


# --- honesty warnings --------------------------------------------------------

def _codes(res):
    return {w["code"] for w in res["quality"]["warnings"]}


def test_short_window_warns():
    res = pipeline.analyze(load_dataframe(_csv_df(60, span_days=55), {}),
                           use_llm=False, out=None)
    assert "short-window" in _codes(res)


def test_long_window_does_not_warn():
    res = pipeline.analyze(load_dataframe(_csv_df(60, span_days=400), {}),
                           use_llm=False, out=None)
    assert "short-window" not in _codes(res)


def test_tiny_money_warns_on_decimal_misparse():
    # '49,90' parsed with decimal='.' becomes 4990 — but '0,49' becomes 0.049-ish;
    # simulate the typical symptom: order values far below 1 currency unit
    res = pipeline.analyze(load_dataframe(_csv_df(60, value=0.04), {}),
                           use_llm=False, out=None)
    assert "tiny-money" in _codes(res)


def test_few_customers_warns():
    res = pipeline.analyze(load_dataframe(_csv_df(60, n_cust=5), {}),
                           use_llm=False, out=None)
    assert "few-customers" in _codes(res)


def test_high_drop_rate_warns():
    raw = _csv_df(100)
    raw.loc[:30, "order_date"] = "garbage"          # >20% unparseable
    res = pipeline.analyze(load_dataframe(raw.astype(str), {}),
                           use_llm=False, out=None)
    assert "high-drop" in _codes(res)


# --- drill-down rows ---------------------------------------------------------

def test_customers_rows_complete_and_sorted():
    res = pipeline.analyze(load_dataframe(_csv_df(60), {}), use_llm=False, out=None)
    cust = res["customers"]
    assert len(cust) == res["kpis"]["total_customers"]
    assert {"id", "segment", "recency", "frequency", "monetary"} <= set(cust[0])
    spend = [c["monetary"] for c in cust]
    assert spend == sorted(spend, reverse=True)


# --- migration ---------------------------------------------------------------

def test_migration_none_on_first_run_then_tracks_moves(tmp_path):
    out = str(tmp_path / "result.json")
    feat1 = pd.DataFrame({"customer_id": ["a", "b"],
                          "segment": ["Champions", "Loyal"]})
    meta = {"date_to": "2026-01-31"}
    assert pipeline._migration(out, feat1, meta) is None       # first run
    feat2 = pd.DataFrame({"customer_id": ["a", "b", "c"],
                          "segment": ["At-risk", "Loyal", "New"]})
    mig = pipeline._migration(out, feat2, meta)
    assert mig["moves"] == [{"from": "Champions", "to": "At-risk", "customers": 1}]
    assert mig["entered"] == 1 and mig["left"] == 0
    snaps = list((tmp_path / "history").glob("segments-*.json"))
    assert len(snaps) == 2
    assert "segments" in json.load(open(snaps[0], encoding="utf-8"))


def test_adhoc_run_leaves_no_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = pipeline.analyze(load_dataframe(_csv_df(60), {}), use_llm=False, out=None)
    assert "migration" not in res
    assert not (tmp_path / "history").exists()
    assert not (tmp_path / "out").exists()
