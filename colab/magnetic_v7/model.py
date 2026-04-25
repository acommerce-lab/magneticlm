"""v7 — Interpretable Statistical Transformer (Genetic Architecture).

ONE SVD on the bigram transition matrix in PPMI space determines everything:
  - d (embedding dimension) = number of circles above threshold
  - spectral weights = glow per dimension
  - Wq, Wk = spectral rotation matrices
  - Embeddings = PPMI projected through spectral space

Single control: spectral_threshold (default 0.01 of max singular value).
Everything else is derived. Like DNA: one code, all structure.
"""

import math, time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Unified Spectral Decomposition — ONE SVD produces everything
# ======================================================================

def build_all_from_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                            bg_trans, V, spectral_threshold, min_ppmi,
                            max_d, device):
    """Single SVD → embeddings + Wq + Wk + spectral_weights.

    Steps:
      1. Build PPMI matrix [V, V]
      2. Project through bigram transitions: M = PPMI @ T.T
         (combines distributional similarity with sequential patterns)
      3. SVD(M) → U, S, V.T
      4. Threshold on S determines d (number of circles)
      5. Embeddings = U[:, :d] @ sqrt(diag(S[:d]))
      6. Wq = rotation in spectral space (from SVD of E.T @ T @ E)
      7. spectral_weights = normalized S

    Returns: embeddings [V,d], Wq [d,d], Wk [d,d], Wv [d,d],
             spectral_weights [d], d (auto-detected)
    """
    t0 = time.time()

    # Build PPMI
    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    PPMI = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    PPMI_dense = PPMI.to_dense()

    # SVD on PPMI → initial embeddings (with max_d as upper bound)
    d_svd = min(max_d, V - 1)
    print(f"    SVD on [{V}x{V}] PPMI (max_d={d_svd})...")
    try:
        U_full, S_full, Vt_full = torch.svd_lowrank(PPMI_dense, q=d_svd, niter=5)
    except Exception:
        U_full, S_full, Vt_full = torch.linalg.svd(PPMI_dense, full_matrices=False)
        U_full = U_full[:, :d_svd]
        S_full = S_full[:d_svd]
        Vt_full = Vt_full[:d_svd, :]

    # Threshold: keep dimensions where S_i > threshold * S_max
    S_max = S_full[0].item()
    cutoff = spectral_threshold * S_max
    mask = S_full > cutoff
    d = int(mask.sum().item())
    d = max(4, min(d, d_svd))
    print(f"    S_max={S_max:.1f}, cutoff={cutoff:.2f} -> d={d} dimensions (circles)")
    print(f"    top-8 S: {', '.join(f'{s:.1f}' for s in S_full[:8].tolist())}")
    if d < d_svd:
        print(f"    S at boundary: S[{d-1}]={S_full[d-1]:.2f}, S[{d}]={S_full[d]:.2f}")

    U = U_full[:, :d]
    S = S_full[:d]

    # Embeddings = U @ sqrt(S)
    embeddings = (U * S.sqrt().unsqueeze(0)).to(device)
    print(f"    embeddings [{V}x{d}] built")

    # Now derive Wq, Wk from transition operator in the new embedding space
    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE  # [d, d]

    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)

    Wq = U_m.contiguous()
    Wk = Vh_m.T.contiguous()
    Wv = torch.eye(d, device=device)

    # Spectral weights from S_m (transition spectrum)
    mu = S_m.mean()
    sigma = S_m.std().clamp(min=1e-9)
    spectral_weights = torch.sigmoid(2.0 * (S_m - mu) / sigma)
    print(f"    spectral weights [{d}]: min={spectral_weights.min():.3f} max={spectral_weights.max():.3f}")

    # IDF
    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    total build time: {time.time()-t0:.1f}s")
    return embeddings, Wq, Wk, Wv, spectral_weights, idf, d


# ======================================================================
# 2. Statistical Transformer — spectrally-weighted attention
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Pure statistical transformer. Genetic architecture.

    d = number of spectral circles (auto-detected from threshold)
    Each dimension = one knowledge circle with its own glow weight.

      scores = (Q * spectral_weights) @ K.T / sqrt(d)
    """

    def __init__(self, embeddings, Wq, Wk, Wv, spectral_weights, idf,
                 n_layers=2, context_len=8, pos_decay=0.1, devices=None):
        self.device = embeddings.device
        self.embeddings = embeddings
        self.E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.spectral_weights = spectral_weights
        self.idf = idf
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay

        self.devices = devices or [self.device]
        self.E_norm_parts = [self.E_norm]
        self.embed_parts = [self.embeddings]
        if len(self.devices) > 1:
            for dev in self.devices[1:]:
                self.E_norm_parts.append(self.E_norm.to(dev))
                self.embed_parts.append(self.embeddings.to(dev))

    def _attention(self, Q, K, V, causal, pad_mask):
        Q_w = Q * self.spectral_weights
        scores = torch.matmul(Q_w, K.transpose(-2, -1)) / math.sqrt(self.d)
        scores = scores.masked_fill(~causal, -1e9)
        scores = scores.masked_fill(~pad_mask, -1e9)
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V)

    def _ffn_chunk(self, xi, E_n, E_raw):
        sim = xi @ E_n.T / math.sqrt(self.d)
        h = F.relu(sim)
        h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return h @ E_raw

    def _ffn(self, x):
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

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)

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
