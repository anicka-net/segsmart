"""SECRET LAB EXPERIMENT — 2D attention on the customer table (SAINT-lite).

Not part of the shipped pipeline, not in CI, not covered by the contracts.
Torch is an *optional* import; everything else in SegSmart stays sklearn-light.

Question: a customer table is genuinely 2D (rows = customers, columns = RFM /
behavioural features). What does running attention along BOTH axes buy us over
plain KMeans-on-RFM?

  axis 1 — across COLUMNS within a row : learns feature interactions
  axis 2 — across ROWS within a batch  : a learned soft-kNN over customers
                                         (this is SAINT's "intersample attention",
                                          and the reason a prediction can depend
                                          on who else is in the batch)

We train it self-supervised (no labels): two noised views of each customer must
land in the same place (InfoNCE) while staying reconstructable (denoising MSE).
Then:
  * cluster the learned embeddings and score the labels in the SAME raw feature
    space as the KMeans baseline, so the comparison is apples-to-apples;
  * use embedding cosine similarity as a lookalike graph ("who is most like a
    Champion?") — the thing KMeans cannot give you.

Run:  python3 -m experiments.attention_segments
      python3 -m experiments.attention_segments --dataset data/synthetic_eshop.csv
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("This experiment needs torch (optional dep):  pip install torch")

SEED = 42


# ----------------------------------------------------------------------------- model
class TwoDBlock(nn.Module):
    """One column-attention pass (within a row) then one intersample pass
    (across the rows of the batch). Same MHSA op both times — just a transpose."""
    def __init__(self, n_tokens: int, dim: int, heads: int = 4):
        super().__init__()
        self.col = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.col_norm = nn.LayerNorm(dim)
        # intersample: each row collapses to one (n_tokens*dim) vector = one token
        self.row = nn.MultiheadAttention(n_tokens * dim, heads, batch_first=True)
        self.row_norm = nn.LayerNorm(n_tokens * dim)
        self.n_tokens, self.dim = n_tokens, dim

    def forward(self, x):                       # x: (B rows, T tokens, D dim)
        a, _ = self.col(x, x, x)                # rows independent; tokens attend
        x = self.col_norm(x + a)
        B = x.shape[0]
        r = x.reshape(1, B, self.n_tokens * self.dim)   # batch becomes the sequence
        a, _ = self.row(r, r, r)                # rows attend to each other
        r = self.row_norm(r + a)
        return r.reshape(B, self.n_tokens, self.dim)


class SaintLite(nn.Module):
    def __init__(self, n_feat: int, dim: int = 32, depth: int = 2, heads: int = 4):
        super().__init__()
        self.n_feat = n_feat
        # continuous tokenizer: value -> dim vector, one (w,b) per feature  (FT-Transformer)
        self.w = nn.Parameter(torch.randn(n_feat, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_feat, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TwoDBlock(n_feat + 1, dim, heads) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.recon = nn.Linear(dim, n_feat)    # denoising head off the CLS token

    def encode(self, x):                        # x: (B, n_feat) -> (B, dim) row embedding
        tok = x.unsqueeze(-1) * self.w + self.b         # (B, n_feat, dim)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return tok[:, 0]                        # CLS row embedding

    def forward(self, x):
        h = self.encode(x)
        return h, F.normalize(self.proj(h), dim=-1), self.recon(h)


# ----------------------------------------------------------------------------- training
def _augment(x, drop=0.2, noise=0.1, g=None):
    """Two corrupted views: random feature dropout + gaussian noise."""
    mask = (torch.rand(x.shape, generator=g) > drop).float()
    return x * mask + torch.randn(x.shape, generator=g) * noise


def info_nce(z1, z2, temp=0.2):
    """Each row's two views attract; all other rows in the batch repel."""
    B = z1.shape[0]
    sim = (z1 @ z2.t()) / temp                  # (B, B)
    target = torch.arange(B)
    return 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))


def train(X, dim=32, depth=2, epochs=60, batch=512, lr=2e-3):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    n = Xt.shape[0]
    model = SaintLite(Xt.shape[1], dim, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 8:                    # intersample needs a few rows
                continue
            xb = Xt[idx]
            v1, v2 = _augment(xb, g=g), _augment(xb, g=g)
            _, z1, r1 = model(v1)
            _, z2, r2 = model(v2)
            loss = info_nce(z1, z2) + 0.5 * (F.mse_loss(r1, xb) + F.mse_loss(r2, xb))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot:.3f}")
    # extract embeddings (eval, deterministic; chunked so intersample sees a stable batch)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, n, batch):
            embs.append(model.encode(Xt[i:i + batch]).numpy())
    return np.concatenate(embs, 0)


# ----------------------------------------------------------------------------- evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="CSV path; default = UCI parquet")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    from seg.features import build_features, model_matrix
    from seg.segment import rfm_segments, kmeans_segments, agreement
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    if args.dataset:
        from seg.loader import load_csv
        df = load_csv(args.dataset)
    else:
        from seg.loader import load_uci
        df = load_uci()

    feat = rfm_segments(build_features(df))
    X, cols = model_matrix(feat)
    n = len(feat)
    print(f"\n{n} customers, {len(cols)} features: {cols}\n")

    # --- baseline: KMeans on raw scaled features (what SegSmart ships) ---
    base_lbl, base_sil, _ = kmeans_segments(feat)
    print(f"[baseline]  KMeans-on-RFM      silhouette(raw)= {base_sil:.3f}")

    # --- 2D attention embeddings -> KMeans, scored in the SAME raw space ---
    print("\ntraining SAINT-lite (2D attention, self-supervised)...")
    Z = train(X, epochs=args.epochs)
    k = len(set(base_lbl))
    att_lbl = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Z)
    att_sil = float(silhouette_score(X, att_lbl, sample_size=min(3000, n), random_state=SEED))
    print(f"[attention] KMeans-on-embeds   silhouette(raw)= {att_sil:.3f}  "
          f"({'better' if att_sil > base_sil else 'worse / tie'} than baseline)")

    print(f"\nstructure agreement (ARI):")
    print(f"  attention-clusters vs RFM rules : {agreement(feat['segment'], att_lbl):+.3f}")
    print(f"  attention-clusters vs raw-KMeans: {adjusted_rand_score(base_lbl, att_lbl):+.3f}")

    # --- the payoff KMeans can't give: a lookalike graph ---
    # ...but does the trained net actually beat plain cosine-kNN on the raw
    # scaled features?  If the neighbourhoods match, the transformer (and all of
    # torch) is dead weight for THIS use-case and the feature is ~5 lines of numpy.
    def topk(space, anchor, k=8):
        sn = space / (np.linalg.norm(space, axis=1, keepdims=True) + 1e-9)
        s = sn @ sn[anchor]; s[anchor] = -1
        return np.argsort(-s)[:k], s

    show = ["recency", "frequency", "monetary", "avg_order_value", "segment"]
    champ_idx = feat.index[feat["segment"] == "Champions"].tolist()
    if champ_idx:
        anchor = champ_idx[0]
        emb_top, emb_s = topk(Z, anchor)
        print(f"\nlookalikes of one Champion (customer #{anchor}) by embedding cosine:")
        print("  cos   " + "  ".join(f"{c:>13}" for c in show))
        for j in emb_top:
            vals = "  ".join(f"{feat.iloc[j][c]:>13.1f}" if c != "segment"
                             else f"{feat.iloc[j][c]:>13}" for c in show)
            print(f"  {emb_s[j]:.3f}  {vals}")
        same = sum(feat.iloc[j]["segment"] == "Champions" for j in emb_top)
        print(f"  -> {same}/8 nearest are themselves Champions")

        # the decisive ablation: attention-embeds vs plain cosine-kNN on raw X.
        # overlap = do they agree?  purity = which one keeps like-with-like better?
        seg = feat["segment"].to_numpy()
        sample = champ_idx[:50] if len(champ_idx) >= 50 else champ_idx
        jac, pur_e, pur_r = [], [], []
        for a in sample:
            e, _ = topk(Z, a); r, _ = topk(X, a)
            jac.append(len(set(e.tolist()) & set(r.tolist())) /
                       len(set(e.tolist()) | set(r.tolist())))
            pur_e.append(np.mean(seg[e] == "Champions"))
            pur_r.append(np.mean(seg[r] == "Champions"))
        print(f"\n[ablation] over {len(sample)} Champions, top-8 lookalikes:")
        print(f"  agreement (Jaccard)            : {np.mean(jac):.2f}  "
              f"(low => the net picks DIFFERENT neighbours than raw cosine)")
        print(f"  Champion-purity, attention-kNN : {np.mean(pur_e):.2f}")
        print(f"  Champion-purity, raw cosine-kNN: {np.mean(pur_r):.2f}")
        better = ("attention earns its keep" if np.mean(pur_e) > np.mean(pur_r) + 0.03
                  else "raw cosine-kNN ties/wins => drop torch, ship ~5 lines of sklearn")
        print(f"  -> {better}")

    print("\nverdict: read silhouette + the ablation (agreement + purity).")


if __name__ == "__main__":
    main()
