"""v7 — Interpretable Statistical Transformer.

Mimics a standard transformer, all weights from corpus statistics:

  1. Embedding: SVD on PPMI → [V, d]
  2. Q Projection: T_fwd @ E → [V, d]  ("what w predicts next")
     K Projection: normalized E → [V, d]  ("what w IS")
  3. Multi-Head Causal Self-Attention (standard scaled dot-product)
  4. Residual + LayerNorm (no learnable params)
  5. Multi-layer stacking
  6. Output: last position @ K.T → softmax

No backpropagation. No gradient descent. Fully interpretable.
"""

import math, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Embedding: PPMI → SVD
# ======================================================================

def build_embeddings(ctx_rows, ctx_cols, ctx_counts, unigram, V, d, min_ppmi, device):
    t0 = time.time()

    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    M = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    M_dense = M.to_dense()

    print(f"    SVD on [{V}x{V}] PPMI, d={d}...")
    try:
        U, S, Vt = torch.svd_lowrank(M_dense, q=d, niter=5)
    except Exception:
        U, S, Vt = torch.linalg.svd(M_dense, full_matrices=False)
        U, S, Vt = U[:, :d], S[:d], Vt[:d, :]

    embeddings = (U * S.sqrt().unsqueeze(0)).to(device)

    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    embeddings [{V}x{d}] in {time.time()-t0:.1f}s")
    return embeddings, idf


# ======================================================================
# 2. Statistical Q/K Projections
# ======================================================================

def build_projections(embeddings, bg_trans, V, d, device):
    """Derive Q and K from bigram transitions.

    Q_fwd[w] = T @ E : expected embedding of what follows w.
    K[w]     = E[w] normalized : what w IS.

    This Q/K separation is the key advance over v6.
    """
    t0 = time.time()

    q_fwd = torch.sparse.mm(bg_trans, embeddings)
    q_fwd = q_fwd / q_fwd.norm(dim=1, keepdim=True).clamp(min=1e-9)

    k_embed = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)

    print(f"    Q/K projections [{V}x{d}] in {time.time()-t0:.1f}s")
    return q_fwd, k_embed


# ======================================================================
# 3. KN scoring (bigram + unigram backoff)
# ======================================================================

def build_kn_simple(encoded, V, max_order, device):
    t0 = time.time()
    bg_r, bg_c = [], []
    for arr in encoded:
        if arr.size < 2:
            continue
        bg_r.append(arr[:-1])
        bg_c.append(arr[1:])

    all_r = np.concatenate(bg_r).astype(np.int64)
    all_c = np.concatenate(bg_c).astype(np.int64)
    pair_keys = all_r * V + all_c
    uniq, counts = np.unique(pair_keys, return_counts=True)
    r_u = (uniq // V).astype(np.int64)
    c_u = (uniq % V).astype(np.int64)

    r_t = torch.from_numpy(r_u).to(device)
    c_t = torch.from_numpy(c_u).to(device)
    cnt_t = torch.from_numpy(counts.astype(np.float32)).to(device)
    row_sums = torch.zeros(V, dtype=torch.float32, device=device)
    row_sums.scatter_add_(0, r_t, cnt_t)
    w = cnt_t / row_sums[r_t].clamp(min=1.0)

    bg_trans = torch.sparse_coo_tensor(
        torch.stack([r_t, c_t]), w, (V, V)
    ).coalesce()

    uni = torch.zeros(V, dtype=torch.float32, device=device)
    for arr in encoded:
        if arr.size == 0:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uni.scatter_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    uni_prob = uni / uni.sum().clamp(min=1.0)

    print(f"    KN: {int(cnt_t.numel()):,} bigram pairs ({time.time()-t0:.1f}s)")
    return dict(bg_trans=bg_trans, uni_prob=uni_prob, V=V, device=device)


def kn_score(kn, context_ids):
    V, device = kn["V"], kn["device"]
    bg = kn["bg_trans"]
    uni = kn["uni_prob"]
    if not context_ids:
        return uni.clone()
    cur = context_ids[-1]
    sel = torch.zeros(V, dtype=torch.float32, device=device)
    sel[cur] = 1.0
    bigram = torch.sparse.mm(bg.t(), sel.unsqueeze(1)).squeeze(1)
    lam, eps = 0.3, 0.02
    result = (1 - lam - eps) * bigram + lam * uni + eps * (1.0 / V)
    return result / result.sum().clamp(min=1e-9)


# ======================================================================
# 4. Statistical Transformer
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Interpretable Statistical Transformer.

    Architecture:
      Input tokens → Embed + PosEncode
      → [Self-Attention(Q,K,V) + Residual + LayerNorm] x n_layers
      → Output: last_pos @ K_all.T → softmax

    Layer 1: Q = Q_fwd[tokens]  (statistical: "what follows this word")
             K = K_embed[tokens] (statistical: "what this word IS")
             V = E[tokens]       (raw embeddings)

    Layer 2+: Q = x (refined representation from previous layer)
              K, V = same as layer 1
    """

    def __init__(self, embeddings, q_fwd, k_embed, idf, unigram_prob,
                 n_heads=4, n_layers=2, context_len=8, pos_decay=0.1):
        self.embeddings = embeddings
        self.q_fwd = q_fwd
        self.k_embed = k_embed
        self.idf = idf
        self.unigram = unigram_prob
        self.V, self.d = embeddings.shape
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay
        self.device = embeddings.device

    def _self_attention(self, Q, K, V, causal_mask, key_mask):
        """Standard multi-head scaled dot-product attention."""
        B, S, D = Q.shape
        H = self.n_heads
        d_h = D // H

        Qh = Q.view(B, S, H, d_h).transpose(1, 2)
        Kh = K.view(B, S, H, d_h).transpose(1, 2)
        Vh = V.view(B, S, H, d_h).transpose(1, 2)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill(~causal_mask, -1e9)
        scores = scores.masked_fill(~key_mask, -1e9)

        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, Vh)
        return out.transpose(1, 2).contiguous().view(B, S, D)

    def score_batch(self, contexts):
        """Batched next-token scoring.

        Args:
            contexts: list of list[int], each a token context window.
        Returns:
            [n, V] probability distributions.
        """
        n = len(contexts)
        cl = self.context_len
        trimmed = [c[-cl:] for c in contexts]
        max_len = max(len(c) for c in trimmed) if trimmed else 1

        padded = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
        pad_mask = torch.zeros(n, max_len, dtype=torch.bool, device=self.device)
        for i, c in enumerate(trimmed):
            L = len(c)
            padded[i, max_len - L:] = torch.tensor(c, dtype=torch.long, device=self.device)
            pad_mask[i, max_len - L:] = True

        S = max_len

        # Embedding + positional decay (recent tokens weighted more)
        x = self.embeddings[padded]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1)
        x = x * pad_mask.unsqueeze(-1).float()

        # Masks for attention
        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        key_m = pad_mask.unsqueeze(1).unsqueeze(2)

        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                Q = self.q_fwd[padded]
            else:
                Q = x
            K = self.k_embed[padded]
            V_attn = self.embeddings[padded]

            attn_out = self._self_attention(Q, K, V_attn, causal, key_m)
            x = _layer_norm(x + attn_out)

        q_final = x[:, -1, :]
        q_final = q_final / q_final.norm(dim=1, keepdim=True).clamp(min=1e-9)
        logits = q_final @ self.k_embed.T / math.sqrt(self.d)

        for i, c in enumerate(trimmed):
            for t in c:
                if 0 <= t < self.V:
                    logits[i, t] = -1e9

        return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        """Score a single context. Returns [V] distribution."""
        return self.score_batch([context_ids])[0]
