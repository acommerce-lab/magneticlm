"""v7 — Interpretable Statistical Transformer.

Pure transformer architecture. All weights from corpus statistics.
Same math as a standard transformer — only the weight source differs.

  Q = x @ Wq          where Wq = solve(E.T@E, E.T@T@E)
  K = x @ Wk          where Wk = I (identity)
  V = x @ Wv          where Wv = I (identity)
  scores = Q @ K.T / sqrt(d_head)
  output = softmax(scores) @ V           ← identical to transformer
  x = LayerNorm(x + output)              ← identical to transformer
  FFN(x) = normalize(ReLU(x @ E.T)) @ E ← vocabulary memory lookup
  x = LayerNorm(x + FFN(x))              ← identical to transformer
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
# 2. Spectral Head Discovery — Wq from SVD of bigram transition matrix
# ======================================================================

def build_spectral_heads(embeddings, bg_trans, V, d, n_heads, device):
    """Derive Wq via spectral decomposition of bigram transitions.

    Instead of arbitrary head count, we analyze the transition matrix T
    in embedding space to discover natural "knowledge circles":

      M = E.T @ T @ E   →  [d, d] transition operator in embedding space
      SVD(M) = U @ S @ V.T

    Each singular triplet (u_i, s_i, v_i) is a knowledge circle:
      - u_i: the "query direction" (what this circle asks for)
      - s_i: the "glow" (how strong this circle is)
      - v_i: the "key direction" (what this circle represents)

    The spectral gap in S tells us how many circles are meaningful.
    We take the top n_heads and weight them by normalized glow.

    Wq = U[:, :n_heads*d_h] @ diag(scale)   — spectral query projection
    Wk = V[:, :n_heads*d_h]                  — spectral key projection
    Wv = identity                             — values are raw embeddings

    Returns: Wq [d, d], Wk [d, d], Wv [d, d], head_glow [n_heads]
    """
    t0 = time.time()

    # Build transition operator in embedding space: M = E.T @ T @ E
    TE = torch.sparse.mm(bg_trans, embeddings)  # [V, d]
    M = embeddings.T @ TE  # [d, d]

    # SVD of M → spectral circles
    U, S, Vh = torch.linalg.svd(M)
    # U: [d, d], S: [d], Vh: [d, d]

    # Spectral gap analysis
    S_np = S.cpu().numpy()
    ratios = S_np[:-1] / (S_np[1:] + 1e-12)
    gap_idx = int(ratios.argmax()) + 1
    print(f"    spectral gap at {gap_idx} (S[{gap_idx-1}]={S_np[gap_idx-1]:.1f} → S[{gap_idx}]={S_np[gap_idx]:.1f})")
    print(f"    top-8 singular values: {', '.join(f'{s:.1f}' for s in S_np[:8])}")
    print(f"    using n_heads={n_heads} (natural gap suggests {gap_idx})")

    d_h = d // n_heads

    # Glow normalization: dampen overly bright circles, boost dim ones
    raw_glow = S[:n_heads * d_h]  # one glow per sub-dimension
    # Group by head: average glow per head
    head_glow = raw_glow.view(n_heads, d_h).mean(dim=1)  # [n_heads]
    mu = head_glow.mean()
    sigma = head_glow.std().clamp(min=1e-9)
    # Sigmoid normalization: compress dynamic range
    head_scale = torch.sigmoid(2.0 * (head_glow - mu) / sigma)  # [n_heads]
    print(f"    head glow: {', '.join(f'{g:.2f}' for g in head_glow.tolist())}")
    print(f"    head scale: {', '.join(f'{s:.3f}' for s in head_scale.tolist())}")

    # Build per-dimension scale: expand head_scale to all d_h dims per head
    dim_scale = head_scale.repeat_interleave(d_h)  # [d]

    # Wq = U[:, :d] @ diag(dim_scale) — spectral query with glow weighting
    Wq = U[:, :d] * dim_scale.unsqueeze(0)  # [d, d]

    # Wk = V.T[:d, :] transposed = V[:, :d] — spectral key directions
    Wk = Vh.T[:, :d].contiguous()  # [d, d]

    # Wv = identity
    Wv = torch.eye(d, device=device)

    print(f"    spectral Wq,Wk,Wv [{d}x{d}] in {time.time()-t0:.1f}s")
    return Wq, Wk, Wv, head_glow


# ======================================================================
# 3. KN scoring (kept for comparison only — not part of the transformer)
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
# 4. Statistical Transformer — same architecture as a real transformer
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Pure statistical transformer. Same math as a real transformer.

    Each layer:
      1. Q = x @ Wq,  K = x @ Wk,  V = x @ Wv   ← same as transformer
      2. attn = softmax(Q @ K.T / sqrt(d_h)) @ V   ← same as transformer
      3. x = LayerNorm(x + attn)                    ← same as transformer
      4. ffn = VocabFFN(x)                           ← statistical FFN
      5. x = LayerNorm(x + ffn)                      ← same as transformer

    The ONLY difference from a real transformer: how W matrices are computed.
    """

    def __init__(self, embeddings, Wq, Wk, Wv, idf, unigram_prob,
                 n_heads=4, n_layers=2, context_len=8, pos_decay=0.1):
        self.embeddings = embeddings  # [V, d]
        self.E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self.Wq = Wq  # [d, d]
        self.Wk = Wk  # [d, d]
        self.Wv = Wv  # [d, d]
        self.idf = idf
        self.unigram = unigram_prob
        self.V, self.d = embeddings.shape
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay
        self.device = embeddings.device

    def _self_attention(self, Q, K, V, causal_mask, key_mask):
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

    def _ffn(self, x):
        """Statistical FFN: vocabulary as memory bank.

        Like a transformer FFN but W1=E_norm.T, W2=E:
          h = ReLU(x @ E_norm.T / sqrt(d))   → word activations [V]
          out = normalize(h) @ E               → weighted sum of embeddings

        Interprets as: "find which words this representation activates,
        then combine their embeddings as the output."
        """
        B, S, D = x.shape
        out = torch.zeros_like(x)
        max_intermediate = 2_000_000_000 // 4  # 2GB float32 budget
        chunk = max(1, max_intermediate // (S * self.V))
        chunk = min(chunk, B)

        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            xi = x[i:j]
            sim = xi @ self.E_norm.T / math.sqrt(self.d)
            h = F.relu(sim)
            h_sum = h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            h = h / h_sum
            out[i:j] = h @ self.embeddings
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

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        key_m = pad_mask.unsqueeze(1).unsqueeze(2)

        for layer_idx in range(self.n_layers):
            Q = x @ self.Wq
            K = x @ self.Wk
            V_attn = x @ self.Wv

            attn_out = self._self_attention(Q, K, V_attn, causal, key_m)
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
