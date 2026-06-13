"""Product-mix analytics: segment × category cross-tab.

Auto-derives product categories when the source does not supply a 'category'
column: tries a local LLM (Ollama) first, falls back to simple text heuristics
so the pipeline never breaks offline.
"""
from __future__ import annotations
import os, re
import pandas as pd

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("SEG_LLM_MODEL", "gemma4:e4b")

_STOP = {"a", "an", "the", "and", "or", "for", "of", "in", "with", "to",
         "na", "pro", "ke", "ve", "ze", "se", "do", "od", "po"}

# catalog category columns — used directly as the category label (no LLM needed)
_CAT_COLS = ["cat_2_name", "cat_1_name"]
# description columns fed to LLM/heuristic when no ready-made category exists
_DESC_CANDIDATES = ["product_name", "product_desc", "description"]


def _heuristic_category(name: str) -> str:
    """First meaningful word from a product name as its category label."""
    words = re.split(r"[\s\-_/|,]+", name.strip())
    for w in words:
        w = w.strip("().[]{}\"'")
        if len(w) >= 3 and w.lower() not in _STOP and not w.isdigit():
            return w.capitalize()
    return name[:20].capitalize() if name.strip() else "Other"


def _categorize_llm(products: list[str], lang: str = "en") -> dict[str, str]:
    """Ask local Ollama to assign a short category label to each unique product.
    Returns {product_name: category_label}. Raises on any error (caller falls back).

    Reasoning models return empty under Ollama format:"json" (AGENTS.md gotcha),
    so we call /api/chat without format and parse the fenced block.
    """
    import json, urllib.request
    from seg.util import extract_json

    items = "\n".join(f"- {p}" for p in products[:200])  # cap to avoid huge prompts
    if lang == "cs":
        prompt = (
            "Tady je seznam názvů produktů z e-shopu. Přiřaď každému krátký český "
            "název kategorie (1–3 slova). Vrať POUZE JSON objekt: "
            '{"název produktu": "kategorie", ...}. Maximálně 15 různých kategorií.\n\n'
            + items
        )
    else:
        prompt = (
            "Here is a list of product names from an e-shop. Assign each a short "
            "category label (1–3 words). Return ONLY a JSON object: "
            '{"product name": "category", ...}. Use at most 15 distinct categories.\n\n'
            + items
        )
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "user", "content": prompt},
    ], "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read())
    mapping = extract_json(raw["message"]["content"])
    if not isinstance(mapping, dict):
        raise ValueError("LLM returned non-dict")
    return {str(k): str(v) for k, v in mapping.items()}


def assign_categories(df: pd.DataFrame, use_llm: bool = True,
                      lang: str = "en") -> pd.Series:
    """Return a category label Series aligned with df's index.

    Priority:
      1. 'category' column already present in df — use it directly.
      2. LLM categorization of unique product names (if use_llm and Ollama up).
      3. Heuristic: first meaningful word of the product name.
    """
    if "category" in df.columns:
        return df["category"].astype(str).replace({"": "Other", "nan": "Other"})

    # catalog provides a ready-made category — no LLM/heuristic needed
    cat_col = next((c for c in _CAT_COLS if c in df.columns), None)
    if cat_col:
        return df[cat_col].astype(str).replace({"": "Other", "nan": "Other"})

    desc_col = next((c for c in _DESC_CANDIDATES if c in df.columns), "product")
    products = df[desc_col].astype(str)
    unique = [p for p in products.unique() if p and p != "nan"]

    mapping: dict[str, str] = {}
    if use_llm and unique:
        try:
            mapping = _categorize_llm(unique, lang=lang)
        except Exception as e:
            print(f"  [product category LLM failed, using heuristic: {e}]")

    for p in unique:
        if p not in mapping:
            mapping[p] = _heuristic_category(p)

    return products.map(lambda p: mapping.get(p, _heuristic_category(p)))


def product_mix(df: pd.DataFrame, feat: pd.DataFrame,
                use_llm: bool = True, lang: str = "en",
                top_n: int = 12) -> list[dict]:
    """Build a segment × category cross-tab of revenue.

    Args:
        df:      canonical line-item frame (from loader).
        feat:    customer feature frame with customer_id + segment columns.
        use_llm: whether to call Ollama for category names.
        lang:    'en' or 'cs' (steers LLM prompts).
        top_n:   keep only the top N categories by total revenue.

    Returns list of {segment, category, revenue, customers} records,
    sorted by segment then category.
    """
    if df.empty or feat.empty or "product" not in df.columns:
        return []

    cats = assign_categories(df, use_llm=use_llm, lang=lang)
    work = df.copy()
    work["_cat"] = cats

    seg_map = feat.set_index("customer_id")["segment"].to_dict()
    work["_seg"] = work["customer_id"].astype(str).map(seg_map)
    work = work.dropna(subset=["_seg"])
    if work.empty:
        return []

    top_cats = (work.groupby("_cat")["line_value"].sum()
                .nlargest(top_n).index.tolist())
    work = work[work["_cat"].isin(top_cats)]

    rev = work.groupby(["_seg", "_cat"])["line_value"].sum()
    custs = work.groupby(["_seg", "_cat"])["customer_id"].nunique()
    cross = pd.DataFrame({"revenue": rev, "customers": custs}).reset_index()

    return sorted([
        {"segment": str(r["_seg"]), "category": str(r["_cat"]),
         "revenue": round(float(r["revenue"]), 2), "customers": int(r["customers"])}
        for _, r in cross.iterrows()
    ], key=lambda x: (x["segment"], x["category"]))
