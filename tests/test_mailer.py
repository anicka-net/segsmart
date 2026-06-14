"""Campaign discount workflow + launch artifacts + review-round fixes."""
import json

import pytest

from seg.campaigns import strip_voucher_codes, apply_discount, PLAYBOOK, PLAYBOOK_CS
from seg.mailer import build_mailing, save_mailing, deliver
from seg.util import atomic_write_json


# --- the model must not ship invented voucher codes --------------------------

@pytest.mark.parametrize("dirty,clean_must_lack", [
    ("Kód ZPĚTVÍTÁZKY pro slevu 15 %", "ZPĚTVÍTÁZKY"),
    ("Sleva 20 % s kódem JARO15", "JARO15"),
    ("Use code SAVE20 at checkout", "SAVE20"),
    ("Get 10% off, coupon WELCOME10!", "WELCOME10"),
])
def test_strip_voucher_codes(dirty, clean_must_lack):
    assert clean_must_lack not in strip_voucher_codes(dirty)


def test_strip_keeps_owner_code():
    out = strip_voucher_codes("Sleva 15 % s kódem JARO15", keep="JARO15")
    assert "JARO15" in out


def test_strip_leaves_normal_copy_alone():
    for s in ("VIP přístup k novinkám + dárek", "Doprava zdarma na druhou objednávku",
              "Free shipping on order #2", "Loyalty points 2x this month"):
        assert strip_voucher_codes(s) == s


def test_playbooks_are_code_free():
    for book in (PLAYBOOK, PLAYBOOK_CS):
        for _, offer, _ch in book.values():
            assert "kód" not in offer.lower() and "code" not in offer.lower()


# --- owner-specified discount rewrite (deterministic fallback path) ----------

def test_apply_discount_percent_cs():
    card = {"segment": "At-risk", "headline": "Vraťte se k nám",
            "offer": "Návratová nabídka", "channel": "E-mail", "rationale": "x"}
    out = apply_discount(card, {"kind": "percent", "value": 15, "code": "JARO15"},
                         lang="cs", currency="Kč", use_llm=False)
    assert "15 %" in out["offer"] and "JARO15" in out["offer"]
    assert out["discount"] == {"kind": "percent", "value": 15, "code": "JARO15"}


def test_apply_discount_free_shipping_en():
    out = apply_discount({"segment": "New", "headline": "h", "offer": "o"},
                         {"kind": "free_shipping", "code": "SHIP-FREE"},
                         lang="en", currency="£", use_llm=False)
    assert "free shipping" in out["offer"].lower()
    assert "SHIP-FREE" in out["offer"]


def test_apply_discount_amount():
    out = apply_discount({"segment": "Loyal", "headline": "h", "offer": "o"},
                         {"kind": "amount", "value": 100, "code": "STO"},
                         lang="cs", currency="Kč", use_llm=False)
    assert "100 Kč" in out["offer"] and "STO" in out["offer"]


# --- launch artifact ----------------------------------------------------------

CARD = {"segment": "Champions", "channel": "E-mail",
        "headline": "Děkujeme za věrnost", "offer": "VIP přístup",
        "discount": {"kind": "percent", "value": 10, "code": "VIP10"},
        "estimate": {"expected_responders": 5, "est_incremental_revenue": 1000,
                     "assumed_response_rate": 0.25}}


def test_build_mailing_contents():
    m = build_mailing(CARD, [{"id": "a@x.cz"}, {"id": "b@x.cz"}, {"id": ""}],
                      lang="cs", currency="Kč")
    assert m["subject"] == "Děkujeme za věrnost"
    assert "VIP10" in m["body_text"] and "Dobrý den" in m["body_text"]
    assert [r["customer_id"] for r in m["recipients"]] == ["a@x.cz", "b@x.cz"]


def test_save_mailing_writes_file(tmp_path):
    m = build_mailing(CARD, [{"id": "a@x.cz"}], lang="cs", currency="Kč")
    p = save_mailing(m, out_dir=str(tmp_path / "mailings"))
    assert json.load(open(p, encoding="utf-8"))["segment"] == "Champions"


def test_deliver_without_webhook_is_graceful():
    rep = deliver({"x": 1}, None)
    assert rep["delivered"] is False and rep["via"] is None
    rep2 = deliver({"x": 1}, {"webhook_url": ""})
    assert rep2["delivered"] is False


def test_deliver_posts_to_webhook(monkeypatch):
    seen = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_open(req, timeout=0):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return FakeResp()

    import seg.mailer as mm
    monkeypatch.setattr(mm.urllib.request, "urlopen", fake_open)
    rep = deliver({"subject": "s"}, {"webhook_url": "http://localhost:1/hook"})
    assert rep["delivered"] is True and seen["body"]["subject"] == "s"


def test_deliver_webhook_failure_does_not_raise():
    rep = deliver({"x": 1}, {"webhook_url": "http://127.0.0.1:1/nope"}, timeout=1)
    assert rep["delivered"] is False and "error" in rep


# --- review-round fixes --------------------------------------------------------

def test_atomic_write_json_roundtrip(tmp_path):
    p = str(tmp_path / "sub" / "x.json")
    atomic_write_json(p, {"a": "ě"})
    assert json.load(open(p, encoding="utf-8")) == {"a": "ě"}
    leftovers = [f for f in (tmp_path / "sub").iterdir() if f.suffix == ".tmp"]
    assert not leftovers


def test_api_file_source_confined_to_data(tmp_path):
    from seg import config as cfgmod
    from seg.util import NoValidData
    secret = tmp_path / "secret.txt"
    secret.write_text("customer_id,order_date,unit_price\nx,2025-01-01,1\n")
    with pytest.raises(NoValidData):
        cfgmod.fetch_raw({"type": "file", "path": str(secret)}, trusted_paths=False)
    with pytest.raises(NoValidData):
        cfgmod.fetch_raw({"type": "file", "path": "/proc/self/environ"},
                         trusted_paths=False)
    # the same path is fine when the config came from local disk
    raw = cfgmod.fetch_raw({"type": "file", "path": str(secret)}, trusted_paths=True)
    assert len(raw) == 1


def test_decimal_grouped_eu_detected():
    from seg.mapping import _heuristic
    h = _heuristic(["cena"], [["1.234,56"], ["2.000"]])
    assert h["decimal"] == ","
    h2 = _heuristic(["price"], [["1,234.56"]])
    assert h2["decimal"] == "."
    # bare EU thousands-grouped integers: the 1000x corruption case
    h3 = _heuristic(["celkem"], [["1.234"], ["12.500"]])
    assert h3["decimal"] == ","


def test_excel_preamble_header(tmp_path):
    import pandas as pd
    from seg.sniff import read_table
    p = tmp_path / "report.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame([["Export objednávek", None, None],
                      [None, None, None],
                      ["email", "datum", "cena"],
                      ["a@x.cz", "2025-01-01", "10"],
                      ["b@x.cz", "2025-01-02", "20"]]
                     ).to_excel(w, index=False, header=False)
    df, info = read_table(open(p, "rb").read(), filename=str(p))
    assert list(df.columns) == ["email", "datum", "cena"]
    assert info["skipped_rows"] == 1          # title row skipped (blank dropped)
    assert len(df) == 2

# --- the model may not promise concrete discounts either ----------------------

def test_has_invented_discount():
    from seg.campaigns import has_invented_discount
    assert has_invented_discount({"offer": "15% sleva a doprava zdarma"})
    assert has_invented_discount({"offer": "Sleva 100 Kč na nákup"})
    assert has_invented_discount({"headline": "20 % off everything"})
    assert not has_invented_discount({"offer": "Návratová nabídka jen na 14 dní"})
    assert not has_invented_discount({"offer": "Doprava zdarma na druhou objednávku"})
    assert not has_invented_discount({"offer": "Dvojnásobné věrnostní body"})


def test_card_with_invented_discount_falls_back(monkeypatch):
    import seg.campaigns as camp
    monkeypatch.setattr(camp, "_ollama",
                        lambda *a, **k: '{"objective":"x","channel":"Email",'
                        '"offer":"25% off everything","headline":"Save 25%!",'
                        '"rationale":"r","priority":"high"}')
    c = camp.card_for_segment("At-risk", {"customers": 100, "share_pct": 10,
        "rev_share_pct": 10, "avg_recency": 90, "avg_frequency": 2,
        "avg_monetary": 500, "avg_order_value": 250},
        {"peak_month": "Dec", "peak_uplift_pct": 40, "low_month": "Feb"},
        use_llm=True)
    assert c["_source"] == "fallback"          # discount-promising copy rejected
    assert "25" not in c["offer"]


def test_owner_discount_survives_policy():
    # apply_discount output legitimately contains the value — policy is only
    # about UNAPPROVED discounts at generation time
    from seg.campaigns import apply_discount
    out = apply_discount({"segment": "Loyal", "headline": "h", "offer": "o"},
                         {"kind": "percent", "value": 15, "code": "OK15"},
                         lang="cs", currency="Kč", use_llm=False)
    assert "15 %" in out["offer"] and out["discount"]["code"] == "OK15"


# --- contact passthrough: mailings get email + name ---------------------------

def test_mailing_recipients_carry_email_and_name():
    m = build_mailing(CARD, [
        {"id": "c1", "email": "a@x.cz", "name": "Jana Nováková"},
        {"id": "b@x.cz"},                       # id IS the e-mail
        {"id": "hash123"},                      # no e-mail at all
    ], lang="cs", currency="Kč")
    r = m["recipients"]
    assert r[0] == {"customer_id": "c1", "email": "a@x.cz", "name": "Jana Nováková"}
    assert r[1]["email"] == "b@x.cz"
    assert r[2]["email"] == ""
    assert m["deliverable"] == 2


def test_contact_columns_flow_to_result(tmp_path):
    import pandas as pd
    import pipeline
    from seg.loader import load_dataframe
    raw = pd.DataFrame({
        "zakaznik": [f"id{i % 6}" for i in range(60)],
        "mail": [f"z{i % 6}@x.cz" for i in range(60)],
        "jmeno": [f"Jméno {i % 6}" for i in range(60)],
        "datum": [f"2025-{1 + i % 12:02d}-10" for i in range(60)],
        "cena": ["100"] * 60,
    })
    df = load_dataframe(raw, {"customer_id": "zakaznik", "order_date": "datum",
                              "unit_price": "cena", "customer_email": "mail",
                              "customer_name": "jmeno"})
    assert "customer_email" in df.columns
    res = pipeline.analyze(df, use_llm=False, out=None)
    c0 = res["customers"][0]
    assert c0["email"].endswith("@x.cz") and c0["name"].startswith("Jméno")


def test_email_shaped_id_used_as_fallback():
    import pandas as pd
    import pipeline
    from seg.loader import load_dataframe
    raw = pd.DataFrame({
        "customer_id": [f"kupec{i % 5}@seznam.cz" for i in range(50)],
        "order_date": [f"2025-{1 + i % 12:02d}-01" for i in range(50)],
        "unit_price": ["50"] * 50,
    })
    res = pipeline.analyze(load_dataframe(raw, {}), use_llm=False, out=None)
    assert all(c["email"] == c["id"] for c in res["customers"])


def test_invented_discount_currency_first():
    # qwen writes malformed currency-first amounts ('Kč3000') — must be caught
    from seg.campaigns import has_invented_discount
    assert has_invented_discount({"offer": "Doprava zdarma nad Kč3000 a poukaz Kč500."})
    assert has_invented_discount({"offer": "€50 voucher for you"})
    assert not has_invented_discount({"offer": "VIP přístup k novinkám a dárek"})


def test_invented_discount_trailing_symbol():
    # symbol currencies AFTER the amount ('50€', '25 $', '50£') — the \b after a
    # symbol never matches, so an earlier regex let these through (review round 3)
    from seg.campaigns import has_invented_discount
    assert has_invented_discount({"offer": "Get 50€ off your next order"})
    assert has_invented_discount({"headline": "25 $ back for you"})
    assert has_invented_discount({"offer": "Sleva 50£ na vše"})
    # word currencies after the amount still caught
    assert has_invented_discount({"offer": "Sleva 100 Kč na nákup"})
    # and innocuous copy still passes
    assert not has_invented_discount({"offer": "Exkluzivní novinky a dárek navíc"})


# --- someone signs the e-mail --------------------------------------------------

def test_signature_default_per_language():
    m_cs = build_mailing(CARD, [{"id": "a@x.cz"}], lang="cs", currency="Kč")
    m_en = build_mailing(CARD, [{"id": "a@x.cz"}], lang="en", currency="£")
    assert m_cs["body_text"].rstrip().endswith("Váš tým")
    assert m_en["body_text"].rstrip().endswith("Your team")


def test_signature_from_config():
    m = build_mailing(CARD, [{"id": "a@x.cz"}], lang="cs", currency="Kč",
                      signature="Tým Drogerie U Lípy")
    assert m["body_text"].rstrip().endswith("Tým Drogerie U Lípy")
    assert m["signature"] == "Tým Drogerie U Lípy"


def test_signature_blank_falls_back_to_default():
    m = build_mailing(CARD, [{"id": "a@x.cz"}], lang="cs", currency="Kč",
                      signature="   ")
    assert m["body_text"].rstrip().endswith("Váš tým")


def test_build_mailing_strips_crlf_from_recipients():
    # a name/e-mail carrying a newline could inject SMTP/CSV headers downstream
    m = build_mailing(
        {"segment": "Champions", "headline": "Hi"},
        [{"id": "c1", "email": "a@x.com\r\nBcc: evil@x.com", "name": "Eve\nNova"}],
        lang="en")
    r = m["recipients"][0]
    assert "\r" not in r["email"] and "\n" not in r["email"]
    assert "\r" not in r["name"] and "\n" not in r["name"]


def test_deliver_rejects_non_http_scheme():
    res = deliver({"subject": "x"}, {"webhook_url": "file:///etc/passwd"})
    assert res["delivered"] is False
    assert "http" in res.get("error", "").lower()
