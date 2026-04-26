"""v7 — Statistical Transformer (Transformer-Matched Architecture).

Matches standard transformer structure exactly:
  - d constant across all layers (no cone, no triangle)
  - Multi-head attention with n_heads groups
  - Wq, Wk from SVD of transition operator
  - FFN as vocabulary memory
  - Residual + LayerNorm

Only difference from standard transformer: how weights are computed
(statistics vs backprop).

Optional: refine S_m via gradient descent (off by default).
"""

import math, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Build everything from spectrum
# ======================================================================

def build_all_from_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                            bg_trans, V, min_ppmi, device, var_target=0.3):
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
    PPMI = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    PPMI_dense = PPMI.to_dense()

    if V <= 5000:
        print(f"    full SVD on [{V}x{V}] PPMI...")
        U_full, S_full, _ = torch.linalg.svd(PPMI_dense, full_matrices=False)
    else:
        q = min(V - 1, 4096)
        print(f"    randomized SVD on [{V}x{V}] PPMI, q={q}...")
        U_full, S_full, _ = torch.svd_lowrank(PPMI_dense, q=q, niter=5)

    S_sq = S_full ** 2
    cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
    d = int((cumvar < var_target).sum().item()) + 1
    d = max(4, min(d, len(S_full)))
    print(f"    cumvar: {cumvar[d-1]:.1%} at d={d} (target={var_target:.0%})")
    print(f"    top-8 S: {', '.join(f'{s:.1f}' for s in S_full[:8].tolist())}")

    embeddings = (U_full[:, :d] * S_full[:d].sqrt().unsqueeze(0)).to(device)
    print(f"    embeddings [{V}x{d}]")

    # Wq, Wk from transition operator — PURE ROTATIONS (no S_m scaling)
    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)

    Wq = U_m.contiguous()       # [d, d] pure rotation
    Wk = Vh_m.T.contiguous()    # [d, d] pure rotation
    print(f"    S_m: max={S_m[0]:.1f}, min={S_m[-1]:.3f}")

    # Auto n_heads: cluster correlated SVD columns via second SVD
    # Correlation matrix of U_m columns
    col_corr = (U_m.T @ U_m).abs()  # [d, d] — how correlated are spectral circles
    # Find block structure: SVD of correlation matrix
    _, S_corr, _ = torch.linalg.svd(col_corr)
    # n_heads = number of significant correlation clusters
    S_corr_norm = S_corr / S_corr.sum()
    cumcorr = torch.cumsum(S_corr_norm, dim=0)
    n_heads_raw = int((cumcorr < 0.9).sum().item()) + 1
    n_heads_raw = max(1, min(n_heads_raw, d))
    # Round to nearest divisor of d
    n_heads = 1
    for h in range(n_heads_raw, 0, -1):
        if d % h == 0:
            n_heads = h
            break
    d_head = d // n_heads
    print(f"    corr clusters -> n_heads={n_heads}, d_head={d_head}")

    # IDF
    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    built in {time.time()-t0:.1f}s")
    return embeddings, Wq, Wk, S_m, idf, d, n_heads


# ======================================================================
# 2. Standard-shape Statistical Transformer
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Statistical transformer matching standard transformer shape.

    d constant across all layers. Multi-head attention.
    No cone. No triangle. Same shape as GPT/BERT.
    """

    def __init__(self, embeddings, Wq, Wk, idf, n_heads,
                 n_layers=2, context_len=8, pos_decay=0.1):
        self.device = embeddings.device
        self.embeddings = embeddings
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = self.d // n_heads
        self.context_len = context_len
        self.pos_decay = pos_decay

        # Wq, Wk: pure rotations — no S_m scaling (matches transformer init)
        self.Wq = Wq  # [d, d]
        self.Wk = Wk  # [d, d]

        # E_norm for FFN
        self.E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self.idf = idf

        print(f"    d={self.d}, n_heads={n_heads}, d_head={self.d_head}, layers={n_layers}")

    def _attention(self, Q, K, V, causal, pad_mask):
        """Standard multi-head attention."""
        B, S, D = Q.shape
        H = self.n_heads
        d_h = self.d_head

        Qh = Q.view(B, S, H, d_h).transpose(1, 2)  # [B, H, S, d_h]
        Kh = K.view(B, S, H, d_h).transpose(1, 2)
        Vh = V.view(B, S, H, d_h).transpose(1, 2)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e9)
        scores = scores.masked_fill(~pad_mask.unsqueeze(1), -1e9)
        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, Vh)
        return out.transpose(1, 2).contiguous().view(B, S, D)

    def _ffn(self, x):
        """FFN: vocabulary as memory bank."""
        B, S, D = x.shape
        out = torch.zeros_like(x)
        chunk = max(1, 500_000_000 // (S * self.V))
        chunk = min(chunk, B)
        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            sim = x[i:j] @ self.E_norm.T / math.sqrt(D)
            h = F.relu(sim)
            h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
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
        x = self.embeddings[padded]  # [n, S, d]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1)
        x = x * pad_mask.unsqueeze(-1).float()

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)  # [B, 1, S]

        for layer_idx in range(self.n_layers):
            # d constant — no truncation
            Q = x @ self.Wq
            K = x @ self.Wk
            V_attn = x

            attn_out = self._attention(Q, K, V_attn, causal, p_mask)
            x = _layer_norm(x + attn_out)

            ffn_out = self._ffn(x)
            x = _layer_norm(x + ffn_out)

        # Output: h @ E.T / sqrt(d) — temperature scales logits for any d
        logits = x[:, -1, :] @ self.embeddings.T / math.sqrt(self.d)
        return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]

    def refine(self, encoded, n_epochs, lr=0.01, batch_size=64):
        """Optional: refine Wq/Wk via gradient descent on training data."""
        Wq_param = self.Wq.clone().detach().requires_grad_(True)
        Wk_param = self.Wk.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([Wq_param, Wk_param], lr=lr)

        all_ctx, all_tgt = [], []
        for arr in encoded:
            if arr.size < 2:
                continue
            ids = arr.astype(np.int64)
            for i in range(len(ids) - 1):
                ctx = ids[max(0, i - self.context_len):i + 1].tolist()
                all_ctx.append(ctx)
                all_tgt.append(int(ids[i + 1]))

        n_total = len(all_ctx)
        if n_total == 0:
            return
        indices = list(range(n_total))

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            total_loss, n_batches = 0.0, 0

            for start in range(0, min(n_total, 5000), batch_size):
                end = min(start + batch_size, n_total)
                batch_idx = indices[start:end]
                batch_ctx = [all_ctx[i] for i in batch_idx]
                batch_tgt = torch.tensor([all_tgt[i] for i in batch_idx],
                                         dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                # Forward with current params
                n = len(batch_ctx)
                cl = self.context_len
                trimmed = [c[-cl:] for c in batch_ctx]
                max_len = max(len(c) for c in trimmed)
                padded = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
                pad_mask = torch.zeros(n, max_len, dtype=torch.bool, device=self.device)
                for i, c in enumerate(trimmed):
                    L = len(c)
                    padded[i, max_len-L:] = torch.tensor(c, dtype=torch.long, device=self.device)
                    pad_mask[i, max_len-L:] = True
                S = max_len
                x = self.embeddings[padded]
                pos = torch.arange(S, dtype=torch.float32, device=self.device)
                pw = torch.exp(-self.pos_decay * (S - 1 - pos))
                x = x * pw.unsqueeze(0).unsqueeze(-1) * pad_mask.unsqueeze(-1).float()
                causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
                p_mask = pad_mask.unsqueeze(1)
                for _ in range(self.n_layers):
                    Q = x @ Wq_param
                    K = x @ Wk_param
                    attn_out = self._attention(Q, K, x, causal, p_mask)
                    x = _layer_norm(x + attn_out)
                    x = _layer_norm(x + self._ffn(x))
                logits = x[:, -1, :] @ self.embeddings.T / math.sqrt(self.d)
                loss = F.cross_entropy(logits, batch_tgt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            ppl = math.exp(total_loss / max(n_batches, 1))
            print(f"    epoch {epoch}: PPL={ppl:.1f}")

        self.Wq = Wq_param.detach()
        self.Wk = Wk_param.detach()
