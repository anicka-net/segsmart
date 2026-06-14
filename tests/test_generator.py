import csv
from gen.catalog import build_catalog, TEMPLATES
from gen.synth import generate, HEADER, _feminize, _czk


def test_catalog_price_above_cost_and_czech():
    cat = build_catalog(seed=7)
    assert len(cat) > 0
    for p in cat:
        assert p["unit_price_czk"] > p["unit_cost_czk"] > 0
        assert p["kategorie"] in TEMPLATES
    # deterministic given seed
    assert build_catalog(seed=7)[0]["nazev"] == cat[0]["nazev"]


def test_feminize_czech_surnames():
    assert _feminize("Novák") == "Nováková"
    assert _feminize("Černý") == "Černá"
    assert _feminize("Svoboda") == "Svobodová"
    assert _feminize("Němec") == "Němcová"
    assert _feminize("Hájek") == "Hájková"


def test_czk_uses_comma():
    assert _czk(1234.5) == "1234,50"


def test_synth_matches_schema(tmp_path):
    out = tmp_path / "s.csv"
    generate(n_customers=50, seed=3, out=str(out))
    with open(out, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    assert len(rows) > 100
    # money fields use decimal comma; currency + country constant
    rec = dict(zip(HEADER, rows[1]))
    assert "," in rec["unit_price"] or rec["unit_price"] in ("0,00",)
    assert rec["currency"] == "CZK"
    assert rec["country"] == "Česká republika"


def test_synth_customer_key_is_email(tmp_path):
    out = tmp_path / "s.csv"
    generate(n_customers=50, seed=5, out=str(out))
    with open(out, encoding="utf-8") as f:
        keys = [r["customer_key"] for r in csv.DictReader(f)]
    assert all("@" in k for k in keys)
