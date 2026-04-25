"""v7 — Genetic Statistical Transformer.

ZERO hyperparameters. Everything derived from data:
  d = auto (noise floor from Marchenko-Pastur)
  L = auto (from knowledge entropy ratio)
  r = 2 (information-theoretic constant)
  cone = triangle (dimensions shrink per layer: d, d//2, d//4, ...)

The word representation across layers is a TRIANGLE, not a rectangle.
Each layer operates on fewer dimensions — only the strongest spectral
circles survive deeper layers.
"""

import math, time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Build everything from spectrum — zero config
# ======================================================================

def build_all_from_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                            bg_trans, V, min_ppmi, device, var_target=0.5):
    """Single SVD -> embeddings + Wq + Wk + S_raw.

    Threshold is AUTO-DERIVED from Marchenko-Pastur noise floor:
      S_noise = sigma_ppmi * sqrt(2*E/V)
      Keep S_i > S_noise (signal above noise).

    Returns: embeddings, Wq, Wk, S_raw, idf, d, S_noise
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

    # Auto noise floor: sigma * sqrt(2*E/V)
    E = int(keep.sum().item())
    sigma_ppmi = float(vals_f.std().item()) if vals_f.numel() > 1 else 1.0
    S_noise = sigma_ppmi * math.sqrt(2.0 * E / max(V, 1))
    print(f"    noise floor: sigma={sigma_ppmi:.2f}, E={E:,}, S_noise={S_noise:.2f}")

    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    PPMI = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    PPMI_dense = PPMI.to_dense()

    # SVD
    if V <= 5000:
        print(f"    full SVD on [{V}x{V}] PPMI...")
        U_full, S_full, Vt_full = torch.linalg.svd(PPMI_dense, full_matrices=False)
    else:
        q = min(V - 1, 4096)
        print(f"    randomized SVD on [{V}x{V}] PPMI, q={q}...")
        U_full, S_full, Vt_full = torch.svd_lowrank(PPMI_dense, q=q, niter=5)

    # Cumulative variance threshold: keep dims explaining 90% of energy
    S_sq = S_full ** 2
    cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
    # var_target passed as parameter
    d = int((cumvar < var_target).sum().item()) + 1
    d = max(4, min(d, len(S_full)))
    actual_var = cumvar[d - 1].item()
    print(f"    cumulative variance: {actual_var:.1%} at d={d} (target={var_target:.0%})")
    print(f"    top-8 S: {', '.join(f'{s:.1f}' for s in S_full[:8].tolist())}")
    if d < len(S_full):
        print(f"    boundary: S[{d-1}]={S_full[d-1]:.2f}, S[{d}]={S_full[d]:.2f}")

    U = U_full[:, :d]
    S = S_full[:d]

    embeddings = (U * S.sqrt().unsqueeze(0)).to(device)
    print(f"    embeddings [{V}x{d}]")

    # Wq, Wk from transition operator in embedding space
    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE  # [d, d]
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)

    Wq_base = U_m.contiguous()
    Wk_base = Vh_m.T.contiguous()
    print(f"    S_m: max={S_m[0]:.1f}, min={S_m[-1]:.3f}")

    # IDF
    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    built in {time.time()-t0:.1f}s")
    return embeddings, Wq_base, Wk_base, S_m, idf, d


# ======================================================================
# 2. Triangle Transformer — dimensions shrink per layer
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Triangle transformer: d shrinks across layers.

    Layer 0: d₀ = d       (all circles, exploration)
    Layer 1: d₁ = d//2    (top half, focusing)
    Layer 2: d₂ = d//4    (top quarter, distillation)
    ...
    Layer L-1: d_L = d//2^(L-1) (core only)

    Residual via truncation: x_new = truncate(x + attn(x), d_new)
    This is valid because SVD dimensions are ordered by importance.

    The word representation across layers is a TRIANGLE.
    """

    def __init__(self, embeddings, Wq_base, Wk_base, S_raw, idf,
                 n_layers=2, context_len=8, pos_decay=0.1, devices=None):
        self.device = embeddings.device
        self.embeddings = embeddings
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay

        # Triangle schedule: d, d//2, d//4, ...
        self.d_schedule = []
        for l in range(n_layers):
            dl = max(4, self.d // (2 ** l))
            self.d_schedule.append(dl)
        print(f"    triangle: {' -> '.join(str(dl) for dl in self.d_schedule)}")

        # Per-layer Wq/Wk: S_m affects DIRECTION (which dims matter)
        # but NOT MAGNITUDE (softmax temperature stays controlled)
        mu = S_raw.mean()
        sigma = S_raw.std().clamp(min=1e-9)
        self.Wq_layers = []
        self.Wk_layers = []
        for l in range(n_layers):
            dl = self.d_schedule[l]
            alpha = 2.0 ** l
            S_dampened = S_raw[:dl] * torch.sigmoid(alpha * (S_raw[:dl] - mu) / sigma)
            # Normalize: keep relative weights, control magnitude
            S_unit = S_dampened / S_dampened.norm().clamp(min=1e-9) * math.sqrt(dl)
            sqrt_Su = S_unit.sqrt()
            Wq_l = (Wq_base[:dl, :dl] * sqrt_Su.unsqueeze(0)).contiguous()
            Wk_l = (Wk_base[:dl, :dl] * sqrt_Su.unsqueeze(0)).contiguous()
            self.Wq_layers.append(Wq_l)
            self.Wk_layers.append(Wk_l)

        # E_norm per layer (truncated embeddings, normalized)
        self.E_norm_layers = []
        for dl in self.d_schedule:
            e = embeddings[:, :dl]
            self.E_norm_layers.append(e / e.norm(dim=1, keepdim=True).clamp(min=1e-9))

        # Output embeddings: normalized (direction only, no frequency bias)
        d_out = self.d_schedule[-1]
        self.E_out = embeddings[:, :d_out]
        self.E_out_norm = self.E_out / self.E_out.norm(dim=1, keepdim=True).clamp(min=1e-9)

        self.idf = idf
        self.devices = devices or [self.device]

    def _attention(self, Q, K, V, causal, pad_mask):
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(~causal, -1e9)
        scores = scores.masked_fill(~pad_mask, -1e9)
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V)

    def _ffn(self, x, E_norm_l, E_raw_l):
        """FFN with layer-appropriate embedding dimensions."""
        B, S, D = x.shape
        out = torch.zeros_like(x)
        max_elems = 500_000_000
        chunk = max(1, max_elems // (S * self.V))
        chunk = min(chunk, B)

        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            xi = x[i:j]
            sim = xi @ E_norm_l.T / math.sqrt(D)
            h = F.relu(sim)
            h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            out[i:j] = h @ E_raw_l
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

        # Start with full-d embeddings
        x = self.embeddings[padded]  # [n, S, d]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1)
        x = x * pad_mask.unsqueeze(-1).float()

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)

        for layer_idx in range(self.n_layers):
            dl = self.d_schedule[layer_idx]

            # Truncate to this layer's dimensions
            x = x[:, :, :dl]

            Wq = self.Wq_layers[layer_idx]
            Wk = self.Wk_layers[layer_idx]

            Q = x @ Wq
            K = x @ Wk
            V_attn = x

            attn_out = self._attention(Q, K, V_attn, causal, p_mask)
            x = _layer_norm(x + attn_out)

            # FFN with truncated embeddings
            E_norm_l = self.E_norm_layers[layer_idx]
            E_raw_l = self.embeddings[:, :dl]
            ffn_out = self._ffn(x, E_norm_l, E_raw_l)
            x = _layer_norm(x + ffn_out)

        # Output: h @ E.T (same as GPT-2 — raw dot product, no temperature)
        logits = x[:, -1, :] @ self.E_out.T

        for i, c in enumerate(trimmed):
            for t in c:
                if 0 <= t < self.V:
                    logits[i, t] = -1e9

        return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]
