"""Config file: round-trip, env expansion, source dispatch, run_config."""
import json
import os
import sqlite3

import pandas as pd
import pytest

from seg import config as cfgmod
from seg.util import NoValidData


CSV = ("customer_id,order_id,order_date,quantity,unit_price\n"
       + "\n".join(f"c{i % 5},o{i},2025-0{1 + i % 9}-15,1,{10 + i}" for i in range(40)))


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "orders.csv"
    p.write_text(CSV)
    return str(p)


def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "cfg" / "segsmart.json")
    cfg = {"source": {"type": "file", "path": "data/x.csv"},
           "output": {"currency": "Kč"}}
    cfgmod.save_config(cfg, p)
    assert cfgmod.load_config(p) == cfg
    # the file is plain, hand-editable JSON
    raw = json.loads(open(p, encoding="utf-8").read())
    assert raw["output"]["currency"] == "Kč"


def test_load_missing_returns_empty(tmp_path):
    assert cfgmod.load_config(str(tmp_path / "nope.json")) == {}


def test_hand_edited_file_is_picked_up(tmp_path, csv_path):
    """The contract: edit the file by hand, next run uses it. No save_config."""
    p = tmp_path / "segsmart.json"
    p.write_text(json.dumps({"source": {"type": "file", "path": csv_path}}))
    cfg = cfgmod.load_config(str(p))
    df = cfgmod.fetch_dataframe(cfg["source"], trusted_paths=True)
    assert df.customer_id.nunique() == 5


def test_env_expansion(monkeypatch, csv_path):
    monkeypatch.setenv("ORDERS_FILE", csv_path)
    df = cfgmod.fetch_dataframe({"type": "file", "path": "${ORDERS_FILE}"},
                            trusted_paths=True)
    assert len(df) == 40
    # unknown vars stay literal (and then fail loudly as a missing file)
    with pytest.raises(NoValidData):
        cfgmod.fetch_raw({"type": "file", "path": "${NO_SUCH_VAR_XYZ}"},
                         trusted_paths=True)


def test_fetch_file_with_mapping(tmp_path):
    p = tmp_path / "shop.csv"
    p.write_text("Kunde;Datum;Summe\n" +
                 "\n".join(f"k{i % 4};0{1 + i % 9}.03.2025;1{i},50" for i in range(20)))
    df = cfgmod.fetch_dataframe({
        "type": "file", "path": str(p), "decimal": ",",
        "mapping": {"customer_id": "Kunde", "order_date": "Datum",
                    "line_value": "Summe"}}, trusted_paths=True)
    assert df.customer_id.nunique() == 4
    assert (df.line_value > 0).all()


def test_fetch_sql_sqlite(tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (email TEXT, dt TEXT, qty REAL, price REAL)")
    con.executemany("INSERT INTO t VALUES (?,?,?,?)",
                    [(f"a{i % 3}@x.cz", "2025-04-01", 1, 9.9) for i in range(12)])
    con.commit(); con.close()
    src = {"type": "sql", "connection_url": f"sqlite:///{db}",
           "query": "SELECT * FROM t",
           "mapping": {"customer_id": "email", "order_date": "dt",
                       "quantity": "qty", "unit_price": "price"}}
    raw = cfgmod.fetch_raw(src)              # the /setup preview path
    assert list(raw.columns) == ["email", "dt", "qty", "price"]
    df = cfgmod.fetch_dataframe(src)
    assert df.customer_id.nunique() == 3


def test_fetch_sql_rejects_writes(tmp_path):
    with pytest.raises(ValueError):
        cfgmod.fetch_raw({"type": "sql", "connection_url": "sqlite://",
                          "query": "DROP TABLE t"})


def test_unknown_type_raises():
    with pytest.raises(NoValidData):
        cfgmod.fetch_raw({"type": "carrier_pigeon"})
    with pytest.raises(NoValidData):
        cfgmod.fetch_raw({})


def test_run_config_end_to_end(tmp_path, csv_path):
    import pipeline
    cfg = {"source": {"type": "file", "path": csv_path},
           "output": {"currency": "€"}, "ai": {"use_llm": False}}
    res = pipeline.run_config(cfg, out=str(tmp_path / "result.json"))
    assert res["meta"]["source"] == "file"
    assert res["meta"]["currency"] == "€"
    assert res["kpis"]["total_customers"] == 5
    assert len(res["campaigns"]) >= 1


def test_run_config_no_source():
    import pipeline
    with pytest.raises(NoValidData):
        pipeline.run_config({})


def test_example_config_is_valid_json():
    here = os.path.join(os.path.dirname(__file__), "..", "config",
                        "segsmart.example.json")
    cfg = json.load(open(here, encoding="utf-8"))
    assert cfg["source"]["type"] in cfgmod.SOURCE_TYPES
