"""Lookalikes — the lean answer the 2D-attention experiment pointed to.

attention_segments.py trained a SAINT-lite encoder and found:
  * its clusters lose to plain KMeans (silhouette 0.18 vs 0.25), and
  * its lookalikes are marginally WORSE than cosine-kNN on the raw features
    (Champion-purity 0.74 vs 0.78) while dragging in all of torch.

So the useful feature was never the attention — it's nearest-neighbours on the
six features SegSmart already computes. No torch, no training, no new deps:
just seg.features.model_matrix + sklearn (already required). Milliseconds.

Two things an SME owner can actually use:
  lookalikes(feat, customer_id)  -> the k customers most like this one
  expand_segment(feat, "Champions") -> non-Champions most like the Champions
                                       ("find me more of my best customers")

Run:  python3 -m experiments.lookalikes
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _space(feat: pd.DataFrame):
    """The exact scaled feature space KMeans uses, L2-normalised for cosine."""
    from seg.features import model_matrix
    X, cols = model_matrix(feat)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn, cols


def lookalikes(feat: pd.DataFrame, customer_id, k: int = 8) -> pd.DataFrame:
    """The k customers most similar to `customer_id` by cosine on scaled RFM."""
    Xn, _ = _space(feat)
    pos = feat.index[feat["customer_id"] == customer_id]
    if len(pos) == 0:
        raise KeyError(f"unknown customer_id {customer_id!r}")
    a = int(pos[0])
    sims = Xn @ Xn[a]
    sims[a] = -1.0
    top = np.argsort(-sims)[:k]
    out = feat.iloc[top][["customer_id", "recency", "frequency", "monetary",
                          "avg_order_value", "segment"]].copy()
    out.insert(1, "similarity", sims[top].round(3))
    return out.reset_index(drop=True)


def expand_segment(feat: pd.DataFrame, segment: str = "Champions",
                   k: int = 15) -> pd.DataFrame:
    """Prospects to promote: customers NOT in `segment` ranked by mean cosine
    similarity to the segment's members. The 'find more like my best' list."""
    Xn, _ = _space(feat)
    mask = (feat["segment"] == segment).to_numpy()
    if not mask.any():
        raise ValueError(f"no customers in segment {segment!r}")
    centroid_sims = Xn @ Xn[mask].mean(0)        # similarity to the seed's centre
    centroid_sims[mask] = -1.0                    # exclude existing members
    top = np.argsort(-centroid_sims)[:k]
    out = feat.iloc[top][["customer_id", "recency", "frequency", "monetary",
                          "avg_order_value", "segment"]].copy()
    out.insert(1, "similarity", centroid_sims[top].round(3))
    return out.reset_index(drop=True)


if __name__ == "__main__":
    from seg.loader import load_uci
    from seg.features import build_features
    from seg.segment import rfm_segments

    feat = rfm_segments(build_features(load_uci()))
    print(f"{len(feat)} customers\n")

    champ = feat.loc[feat["segment"] == "Champions", "customer_id"].iloc[0]
    print(f"=== lookalikes of Champion {champ} ===")
    print(lookalikes(feat, champ).to_string(index=False))

    print(f"\n=== expand 'Champions' — prospects most like your best ===")
    print(expand_segment(feat, "Champions").to_string(index=False))
