"""v7 — Interpretable Statistical Transformer.

Pure transformer architecture. All weights from corpus statistics.
Same math as a standard transformer — only the weight source differs.

  Wq, Wk from SVD of transition operator M = E.T @ T @ E
  spectral_weights: per-dimension glow from singular values
  scores = (Q * spectral_weights) @ K.T / sqrt(d)  ← spectrally-weighted attention
  x = LayerNorm(x + attn_output)                    ← same as transformer
  FFN(x) = normalize(ReLU(x @ E.T)) @ E             ← vocabulary memory lookup
  x = LayerNorm(x + FFN(x))                          ← same as transformer

No multi-head splitting — each SVD dimension is its own "head"
with continuous spectral weighting. Simpler and more principled.
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
# 2. Spectral Weights — each dimension = one knowledge circle
# ======================================================================

def build_spectral_heads(embeddings, bg_trans, V, d, n_heads, device):
    """Spectral decomposition of transition operator in embedding space.

    M = E.T @ T @ E  [d, d]
    SVD(M) = U @ diag(S) @ V.T

    Each singular value S_i = glow of knowledge circle i.
    Each dimension gets its own continuous weight (no discrete heads).

    Returns: Wq [d,d], Wk [d,d], Wv [d,d], spectral_weights [d]
    """
    t0 = time.time()

    TE = torch.sparse.mm(bg_trans, embeddings)
    M = embeddings.T @ TE  # [d, d]

    U, S, Vh = torch.linalg.svd(M)

    S_np = S.cpu().numpy()
    ratios = S_np[:-1] / (S_np[1:] + 1e-12)
    gap_idx = int(ratios.argmax()) + 1
    print(f"    spectral gap at {gap_idx} (S[{gap_idx-1}]={S_np[gap_idx-1]:.1f} -> S[{gap_idx}]={S_np[gap_idx]:.1f})")
    print(f"    top-8 singular values: {', '.join(f'{s:.1f}' for s in S_np[:8])}")

    # Per-dimension glow: sigmoid normalization
    mu = S.mean()
    sigma = S.std().clamp(min=1e-9)
    spectral_weights = torch.sigmoid(2.0 * (S - mu) / sigma)  # [d]
    print(f"    spectral weights: min={spectral_weights.min():.3f} max={spectral_weights.max():.3f} mean={spectral_weights.mean():.3f}")

    Wq = U.contiguous()       # [d, d] spectral rotation for queries
    Wk = Vh.T.contiguous()    # [d, d] spectral rotation for keys
    Wv = torch.eye(d, device=device)

    print(f"    Wq,Wk [{d}x{d}] + weights [{d}] in {time.time()-t0:.1f}s")
    return Wq, Wk, Wv, spectral_weights


# ======================================================================
# 3. KN scoring (kept for comparison only)
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
# 4. Statistical Transformer — spectrally-weighted attention
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Pure statistical transformer with spectrally-weighted attention.

    No multi-head splitting. Each SVD dimension is its own "head"
    with continuous spectral weighting:

      Q = x @ Wq                          (rotate to spectral space)
      K = x @ Wk                          (rotate to spectral space)
      scores = (Q * spectral_weights) @ K.T / sqrt(d)
      output = softmax(scores) @ V
      x = LayerNorm(x + output)
      x = LayerNorm(x + FFN(x))

    Simpler than multi-head, each dimension contributes proportionally
    to its spectral importance (glow).
    """

    def __init__(self, embeddings, Wq, Wk, Wv, spectral_weights, idf,
                 unigram_prob, n_layers=2, context_len=8, pos_decay=0.1,
                 devices=None):
        self.device = embeddings.device
        self.embeddings = embeddings
        self.E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.spectral_weights = spectral_weights
        self.idf = idf
        self.unigram = unigram_prob
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay

        # Multi-GPU: replicate FFN data on secondary devices
        self.devices = devices or [self.device]
        self.E_norm_parts = [self.E_norm]
        self.embed_parts = [self.embeddings]
        if len(self.devices) > 1:
            for dev in self.devices[1:]:
                self.E_norm_parts.append(self.E_norm.to(dev))
                self.embed_parts.append(self.embeddings.to(dev))

    def _attention(self, Q, K, V, causal, pad_mask):
        """Spectrally-weighted attention. No multi-head splitting."""
        # Q * weights: each spectral dimension weighted by its glow
        Q_w = Q * self.spectral_weights  # [B, S, d]
        scores = torch.matmul(Q_w, K.transpose(-2, -1)) / math.sqrt(self.d)  # [B, S, S]
        scores = scores.masked_fill(~causal, -1e9)
        scores = scores.masked_fill(~pad_mask, -1e9)
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V)  # [B, S, d]

    def _ffn_chunk(self, xi, E_n, E_raw):
        sim = xi @ E_n.T / math.sqrt(self.d)
        h = F.relu(sim)
        h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return h @ E_raw

    def _ffn(self, x):
        """Statistical FFN: vocabulary as memory bank. Multi-GPU aware."""
        B, S, D = x.shape
        n_dev = len(self.devices)
        max_elems = 500_000_000
        chunk = max(1, max_elems // (S * self.V))
        chunk = min(chunk, B)
        out = torch.zeros_like(x)

        if n_dev <= 1:
            for i in range(0, B, chunk):
                j = min(i + chunk, B)
                out[i:j] = self._ffn_chunk(x[i:j], self.E_norm, self.embeddings)
        else:
            # Distribute chunks across GPUs
            chunks = [(i, min(i + chunk, B)) for i in range(0, B, chunk)]
            for ci, (start, end) in enumerate(chunks):
                dev_idx = ci % n_dev
                dev = self.devices[dev_idx]
                xi = x[start:end].to(dev)
                res = self._ffn_chunk(xi, self.E_norm_parts[dev_idx], self.embed_parts[dev_idx])
                out[start:end] = res.to(self.device)
        return out

    def score_batch(self, contexts):
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

        x = self.embeddings[padded]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1)
        x = x * pad_mask.unsqueeze(-1).float()

        # Masks: causal [S, S], pad [B, 1, S]
        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)  # [B, 1, S]

        for layer_idx in range(self.n_layers):
            Q = x @ self.Wq
            K = x @ self.Wk
            V_attn = x @ self.Wv

            attn_out = self._attention(Q, K, V_attn, causal, p_mask)
            x = _layer_norm(x + attn_out)

            ffn_out = self._ffn(x)
            x = _layer_norm(x + ffn_out)

        q_final = x[:, -1, :]
        q_final = q_final / q_final.norm(dim=1, keepdim=True).clamp(min=1e-9)
        logits = q_final @ self.E_norm.T / math.sqrt(self.d)

        for i, c in enumerate(trimmed):
            for t in c:
                if 0 <= t < self.V:
                    logits[i, t] = -1e9

        return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]
