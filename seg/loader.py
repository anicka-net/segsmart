"""Pluggable transaction loader.

Every adapter returns the SAME canonical line-item frame, so everything
downstream (features, segmentation, campaigns, dashboard) is dataset-agnostic.
When the schoolmate hands over a real e-shop export, write one adapter here
(or a column mapping for `load_csv`) and nothing else changes.

Canonical columns
-----------------
customer_id : str    order_id : str       order_date : datetime64
quantity    : float  unit_price : float   line_value : float (= qty*price)
product     : str    country : str
"""
from __future__ import annotations
import re

import numpy as np
import pandas as pd

from seg.util import NoValidData

CANON = ["customer_id", "order_id", "order_date",
         "quantity", "unit_price", "line_value", "product", "country"]
# optional contact passthrough — kept when the source maps them, so campaign
# launch can produce directly-mailable recipient lists (email + name)
CONTACT = ["customer_email", "customer_name"]
# optional analytics passthrough — category is used by the product-mix chart;
# when the source provides it directly, _finalize passes it through unchanged
_PASSTHROUGH = CONTACT + [
    "category",
    "product_name", "product_desc", "description",
    "cat_1_ID", "cat_1_name", "cat_2_ID", "cat_2_name",
]


def _guess_dayfirst(txt: pd.Series) -> bool:
    """Is '05/03/2026' the 5th of March (EU) or May 3rd (US)? Look for an
    unambiguous row (a component > 12); fall back on the separator: dotted
    dates ('5.3.2026') are the European convention."""
    sample = txt.dropna().astype(str).head(500)
    first_gt12 = second_gt12 = dotted = False
    for v in sample:
        m = re.match(r"\s*(\d{1,2})([./-])(\d{1,2})[./-]", v)
        if not m:
            continue
        a, sep, b = int(m.group(1)), m.group(2), int(m.group(3))
        first_gt12 |= a > 12
        second_gt12 |= b > 12
        dotted |= sep == "."
    if first_gt12 and not second_gt12:
        return True
    if second_gt12 and not first_gt12:
        return False
    return dotted


def _parse_dates(s: pd.Series) -> pd.Series:
    """Whatever the export calls a date -> datetime64. Handles ISO, EU/US
    orders, Excel serial numbers, unix epochs; bad cells become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:                     # purely numeric column
        med = num.median()
        if 20000 <= med <= 80000:                    # Excel serial (1954..2119)
            return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
        if 1e9 <= med <= 3e9:                        # unix seconds
            return pd.to_datetime(num, unit="s", errors="coerce")
        if 1e12 <= med <= 3e12:                      # unix milliseconds
            return pd.to_datetime(num, unit="ms", errors="coerce")
    txt = s.astype(str)
    return pd.to_datetime(txt, errors="coerce", format="mixed",
                          dayfirst=_guess_dayfirst(txt))


def _finalize(df: pd.DataFrame, drop_cancellations=True, drop_nonpositive=True) -> pd.DataFrame:
    """Shared cleaning every adapter funnels through. Tolerates missing
    columns where a sane default exists (the messy-CSV contract):
      no order_id  -> one order per customer per day
      no quantity  -> 1
      no unit_price but a line total -> derived from the total
    Only customer_id + order_date + some money column are truly required."""
    df = df.copy()
    for col in ("customer_id", "order_date"):
        if col not in df:
            raise NoValidData(f"could not find a {col.replace('_', ' ')} column — "
                              "map it manually in the wizard")
    rows_in = len(df)
    dropped = {}                                    # reason -> rows lost

    def _step(d, reason, prev_len):
        if prev_len - len(d):
            dropped[reason] = dropped.get(reason, 0) + (prev_len - len(d))
        return d

    df["order_date"] = _parse_dates(df["order_date"])
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = _step(df.dropna(subset=["order_date"]), "unparseable date", len(df))
    df = _step(df[~df["customer_id"].str.lower().isin(("", "nan", "none"))],
               "missing customer id", len(df))
    synthesized_ids = "order_id" not in df
    if synthesized_ids:                             # transaction dump: synthesize
        df["order_id"] = (df["customer_id"] + "@"
                          + df["order_date"].dt.strftime("%Y-%m-%d"))
    df["order_id"] = df["order_id"].astype(str)
    if "quantity" not in df:
        df["quantity"] = 1.0
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0)
    has_total = "line_value" in df
    if has_total:
        df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")
    if "unit_price" not in df:
        if not has_total:
            raise NoValidData("could not find a price or order-total column — "
                              "map one manually in the wizard")
        qty = df["quantity"].replace(0, np.nan)
        df["unit_price"] = df["line_value"] / qty
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = _step(df.dropna(subset=["unit_price"]), "unparseable price", len(df))
    if drop_cancellations:
        # a cancellation/return: negative quantity, or a UCI-style C-invoice
        # ('C' + digits ONLY — a plain startswith('C') would silently drop
        # every order in an export numbered 'CZ2025-001', and never applies
        # to ids we synthesized ourselves)
        n = len(df)
        if not synthesized_ids:
            df = df[~df["order_id"].str.fullmatch(r"[Cc]\d+")]
        df = _step(df[df["quantity"] > 0], "cancellations / returns", n)
    if drop_nonpositive:
        df = _step(df[df["unit_price"] > 0], "non-positive price", len(df))
    if has_total:                                   # the export's total is authoritative
        df["line_value"] = df["line_value"].fillna(df["quantity"] * df["unit_price"])
    else:
        df["line_value"] = df["quantity"] * df["unit_price"]
    if "product" not in df:
        df["product"] = ""
    if "country" not in df:
        df["country"] = ""
    df["product"] = df["product"].astype(str)
    df["country"] = df["country"].astype(str)
    extra = [c for c in _PASSTHROUGH if c in df.columns]
    for c in extra:
        df[c] = df[c].astype(str).replace("nan", "")
    out = df[CANON + extra].reset_index(drop=True)
    # ingest report — the pipeline reads it right after loading and shows it
    # on the dashboard, so silently-dropped rows are never silent
    out.attrs["ingest"] = {"rows_in": int(rows_in), "rows_kept": int(len(out)),
                           "dropped": {k: int(v) for k, v in dropped.items()}}
    return out


def load_uci(path: str = "data/online_retail.parquet") -> pd.DataFrame:
    """UCI Online Retail (the standard benchmark dataset)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_excel(path)
    df = df.rename(columns={
        "CustomerID": "customer_id", "InvoiceNo": "order_id",
        "InvoiceDate": "order_date", "Quantity": "quantity",
        "UnitPrice": "unit_price", "Description": "product", "Country": "country",
    })
    return _finalize(df)


# Generic adapter for the schoolmate's real export. Just pass a mapping
# {canonical_name: their_column_name}; line_value is derived if absent.
DEFAULT_MAP = {
    "customer_id": "customer_id", "order_id": "order_id", "order_date": "order_date",
    "quantity": "quantity", "unit_price": "unit_price", "line_value": "line_value",
    "product": "product", "country": "country",
    "customer_email": "customer_email", "customer_name": "customer_name",
}


def _to_num(s: pd.Series, decimal: str = ".") -> pd.Series:
    """Parse a possibly-stringy numeric column. decimal=',' handles European
    formats ('1 234,56' / '1.234,56' → 1234.56)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    # drop everything that isn't a digit, separator or sign (currency, spaces, letters)
    x = s.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    if decimal == ",":
        x = x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        x = x.str.replace(",", "", regex=False)        # strip US thousands separators
    return pd.to_numeric(x, errors="coerce")


def load_dataframe(raw: pd.DataFrame, mapping: dict | None = None,
                   decimal: str = ".") -> pd.DataFrame:
    """Map an arbitrary source frame into the canonical schema. The single seam
    every connector (CSV, SQL, BigQuery, Shoptet, …) funnels through.
    decimal=',' for European-formatted numbers."""
    m = {**DEFAULT_MAP, **(mapping or {})}
    df = pd.DataFrame()
    for canon, col in m.items():
        if col in raw.columns:
            df[canon] = raw[col]
    for c in ("quantity", "unit_price", "line_value"):
        if c in df.columns:
            df[c] = _to_num(df[c], decimal)
    return _finalize(df)


def load_csv(path: str, mapping: dict | None = None, decimal: str = ".",
             **read_kwargs) -> pd.DataFrame:
    """Load a CSV/TSV/Excel file of any flavour (encoding, delimiter and header
    position are sniffed by seg.sniff). read_kwargs kept for API compat."""
    from seg.sniff import read_table                  # local import: no cycle
    with open(path, "rb") as f:
        raw, _info = read_table(f.read(), filename=path)
    out = load_dataframe(raw, mapping, decimal)
    try:
        from seg.catalog import fill_missing as _fill_missing, enrich as _enrich_catalog
        cat = _fill_missing(out, key_col="product")
        if cat is not None:
            out = _enrich_catalog(out, cat, key_col="product")
    except Exception as e:
        print(f"  [catalog enrich skipped: {e}]")
    return out


def _czk(s: pd.Series) -> pd.Series:
    """Parse Czech money strings ('1 234,56' / '-129,46') to float."""
    return pd.to_numeric(
        s.astype(str).str.replace(" ", "").str.replace(" ", "").str.replace(",", "."),
        errors="coerce")


# The partner shop's BigQuery export. product_id encodes line TYPE; status is Czech.
NON_PRODUCT_PREFIXES = ("BILLING", "SHIPPING")          # zero-value structural lines
DISCOUNT_PREFIXES = ("COUPON",)                          # negative adjustment lines
NONREVENUE_STATUS = {"Stornována", "Vrácená objednávka", "Platba selhala"}  # not realized


def load_eshop(path: str = "data/bq_export.csv") -> pd.DataFrame:
    """The partner shop's real export (one row per order line).

    customer_key is the customer id (usually the email; hashed in this export).
    Keeps real SKU + GIFT + COUPON lines so order totals net out discounts;
    drops BILLING/SHIPPING structural lines and cancelled/returned/failed orders.
    line_value comes from revenue_per_line (authoritative — includes discounts),
    NOT qty*unit_price.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.DataFrame({
        "customer_id": raw["customer_key"].astype(str),
        "order_id": raw["order_id"].astype(str),
        "order_date": pd.to_datetime(raw["order_datetime"], errors="coerce"),
        "quantity": pd.to_numeric(raw["product_quantity"], errors="coerce"),
        "unit_price": _czk(raw["unit_price"]),
        "line_value": _czk(raw["revenue_per_line"]),
        "line_cost": _czk(raw.get("cost_per_line", pd.Series(index=raw.index))),
        "product": raw["product_id"].astype(str),
        "country": raw["country"].astype(str),
        "status": raw["order_status"].astype(str),
        "zip_prefix": raw.get("zip_prefix", pd.Series("", index=raw.index)).astype(str),
    })
    rows_in, dropped = len(df), {}
    n = len(df)
    df = df.dropna(subset=["customer_id", "order_date", "line_value"])
    df = df[df["customer_id"].str.lower().ne("nan") & df["customer_id"].ne("")]
    if n - len(df):
        dropped["missing customer / date / value"] = int(n - len(df))
    # drop structural billing/shipping rows and non-realized orders
    n = len(df)
    df = df[~df["product"].str.upper().str.startswith(NON_PRODUCT_PREFIXES)]
    if n - len(df):
        dropped["billing / shipping lines"] = int(n - len(df))
    n = len(df)
    df = df[~df["status"].isin(NONREVENUE_STATUS)]
    if n - len(df):
        dropped["cancelled / returned / failed"] = int(n - len(df))
    # flag real product lines (for diversity counts; excludes coupons/gifts)
    df["is_product"] = ~df["product"].str.upper().str.startswith(
        NON_PRODUCT_PREFIXES + DISCOUNT_PREFIXES)
    df["quantity"] = df["quantity"].fillna(0)
    df["unit_price"] = df["unit_price"].fillna(0)
    out = df.reset_index(drop=True)
    try:
        from seg.catalog import fill_missing as _fill_missing, enrich as _enrich_catalog
        cat = _fill_missing(out, key_col="product")
        if cat is not None:
            out = _enrich_catalog(out, cat, key_col="product")
    except Exception as e:
        print(f"  [catalog enrich skipped: {e}]")
    out.attrs["ingest"] = {"rows_in": int(rows_in), "rows_kept": int(len(out)),
                           "dropped": dropped}
    return out


def summary(df: pd.DataFrame) -> dict:
    """One-glance EDA numbers for the dashboard / validation slide."""
    return {
        "rows": int(len(df)),
        "customers": int(df.customer_id.nunique()),
        "orders": int(df.order_id.nunique()),
        "countries": int(df.country.nunique()),
        "date_from": str(df.order_date.min().date()),
        "date_to": str(df.order_date.max().date()),
        "revenue": round(float(df.line_value.sum()), 2),
        "avg_order_value": round(float(df.groupby("order_id").line_value.sum().mean()), 2),
    }


if __name__ == "__main__":
    d = load_uci()
    print(d.head().to_string())
    import json
    print(json.dumps(summary(d), indent=2))
