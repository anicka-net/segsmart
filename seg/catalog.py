"""Product catalog: lookup table mapping product_id -> name + categories.

The catalog CSV lives at data/product_catalog.csv (relative to the project
root) and has columns: product_id, product_name, cat_1_ID, cat_1_name,
cat_2_ID, cat_2_name.  Missing file -> graceful no-op (no crash, no chart).
"""
from __future__ import annotations
import os
import re
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_HERE, "data", "product_catalog.csv")

CATALOG_COLS = ["product_name", "cat_1_ID", "cat_1_name", "cat_2_ID", "cat_2_name"]

# only numeric-looking IDs are real products (BILLING/SHIPPING/COUPON/GIFT skipped)
_REAL_ID = re.compile(r"^\d+$")

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL  = os.environ.get("SEG_LLM_MODEL", "gemma4:e4b")


def load(path: str = DEFAULT_PATH) -> pd.DataFrame | None:
    """Load the catalog; returns None when the file is absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    if "product_id" not in df.columns:
        return None
    return df.set_index("product_id")


def enrich(df: pd.DataFrame, catalog: pd.DataFrame,
           key_col: str = "product_id") -> pd.DataFrame:
    """Left-join catalog columns onto df[key_col]."""
    cols = [c for c in CATALOG_COLS if c in catalog.columns]
    if not cols or key_col not in df.columns:
        return df
    return df.join(catalog[cols], on=key_col, how="left")


def _existing_categories(catalog: pd.DataFrame) -> list[dict]:
    """Unique cat_1/cat_2 pairs from current catalog — passed as context to LLM."""
    seen, result = set(), []
    for _, row in catalog.reset_index().iterrows():
        key = (row.get("cat_1_ID", ""), row.get("cat_2_ID", ""))
        if key not in seen:
            seen.add(key)
            result.append({k: row.get(k, "") for k in
                           ("cat_1_ID", "cat_1_name", "cat_2_ID", "cat_2_name")})
    return result


def _generate_llm(product_ids: list[str], cat_context: list[dict],
                  lang: str = "cs") -> list[dict]:
    """Ask local Ollama to generate product entries for unknown IDs.
    Context: Alza-style Czech electronics e-shop."""
    import json, urllib.request
    from seg.util import extract_json

    ids_block = "\n".join(product_ids[:50])
    cats_json  = json.dumps(cat_context, ensure_ascii=False)

    prompt = (
        "You are building a product catalog for a Czech electronics e-shop similar to Alza.cz.\n"
        "For each product_id below, invent a realistic product name and assign it to a category.\n"
        f"Reuse categories from this list where possible: {cats_json}\n"
        "You may introduce new subcategories, but keep the style (Czech names, electronics focus).\n\n"
        "Return ONLY a JSON array — no extra text:\n"
        '[{"product_id":"...","product_name":"...","cat_1_ID":"...","cat_1_name":"...",'
        '"cat_2_ID":"...","cat_2_name":"..."}, ...]\n\n'
        f"Product IDs:\n{ids_block}"
    )

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())
    result = extract_json(raw["message"]["content"])
    if not isinstance(result, list):
        raise ValueError("LLM returned non-list")
    return result


def _generate_fallback(product_ids: list[str]) -> list[dict]:
    """Placeholder entries when LLM is unavailable."""
    return [
        {"product_id": pid, "product_name": f"Produkt {pid}",
         "cat_1_ID": "C99", "cat_1_name": "Ostatni",
         "cat_2_ID": "C9901", "cat_2_name": "Nezarazeno"}
        for pid in product_ids
    ]


def fill_missing(df: pd.DataFrame,
                 catalog_path: str = DEFAULT_PATH,
                 key_col: str = "product",
                 use_llm: bool = True,
                 lang: str = "cs") -> pd.DataFrame | None:
    """Check df[key_col] against catalog; generate + persist entries for unknown
    real product IDs (numeric only — BILLING/COUPON/GIFT/SHIPPING skipped).
    Returns the updated catalog DataFrame (or None when path is unavailable)."""
    catalog = load(catalog_path)
    known   = set(catalog.index) if catalog is not None else set()

    all_ids = [str(x) for x in df[key_col].dropna().unique()]
    missing = [pid for pid in all_ids
               if pid not in known and _REAL_ID.match(pid)]

    if not missing:
        return catalog

    print(f"  [catalog: {len(missing)} unknown product ID(s) — generating entries]")

    cat_context = _existing_categories(catalog) if catalog is not None else []
    new_rows: list[dict] = []

    if use_llm:
        try:
            for i in range(0, len(missing), 20):   # batch: max 20 IDs per call
                batch = missing[i : i + 20]
                new_rows.extend(_generate_llm(batch, cat_context, lang=lang))
        except Exception as e:
            print(f"  [catalog LLM failed, using fallback: {e}]")
            new_rows = _generate_fallback(missing)
    else:
        new_rows = _generate_fallback(missing)

    if not new_rows:
        return catalog

    # normalise columns and write
    new_df = pd.DataFrame(new_rows)
    for col in ["product_id"] + CATALOG_COLS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[["product_id"] + CATALOG_COLS].fillna("").astype(str)

    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    write_header = not os.path.exists(catalog_path)
    new_df.to_csv(catalog_path, mode="a", header=write_header,
                  index=False, encoding="utf-8")
    print(f"  [catalog: {len(new_df)} new entry/entries appended to {catalog_path}]")

    return load(catalog_path)
