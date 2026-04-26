"""v7 — Statistical Transformer (Transformer-Matched, Trainable).

Matches standard transformer exactly:
  - Per-layer Wq_l, Wk_l (not shared)
  - d constant across all layers
  - Multi-head attention
  - FFN (vocab memory)
  - Residual + LayerNorm
  - Optional refinement: train ALL weights with early stopping

Initialization from corpus statistics. Refinement optional.
"""

import math, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Build spectrum
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

    embeddings = (U_full[:, :d] * S_full[:d].sqrt().unsqueeze(0)).to(device)
    print(f"    embeddings [{V}x{d}]")

    # Wq, Wk per layer from transition operator
    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)
    Wq_init = U_m.contiguous()
    Wk_init = Vh_m.T.contiguous()
    print(f"    S_m: max={S_m[0]:.1f}, min={S_m[-1]:.3f}")

    # Auto n_heads
    col_corr = (U_m.T @ U_m).abs()
    _, S_corr, _ = torch.linalg.svd(col_corr)
    S_corr_norm = S_corr / S_corr.sum()
    cumcorr = torch.cumsum(S_corr_norm, dim=0)
    n_heads_raw = int((cumcorr < 0.9).sum().item()) + 1
    n_heads = 1
    for h in range(max(1, n_heads_raw), 0, -1):
        if d % h == 0:
            n_heads = h
            break
    print(f"    n_heads={n_heads}, d_head={d // n_heads}")
    print(f"    built in {time.time()-t0:.1f}s")
    return embeddings, Wq_init, Wk_init, d, n_heads


# ======================================================================
# 2. Statistical Transformer
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


def _decay_cache(history, V, device):
    if len(history) < 2:
        return torch.zeros(V, dtype=torch.float32, device=device)
    h = torch.tensor(history, dtype=torch.int64, device=device)
    ages = torch.arange(h.numel() - 1, -1, -1, dtype=torch.float32, device=device)
    b = torch.zeros(V, dtype=torch.float32, device=device)
    b.scatter_add_(0, h, 1.0 / torch.log(2.0 + ages))
    s = b.sum()
    return b / s if s > 1e-9 else b


class StatTransformer:
    """Standard-shape statistical transformer. All weights trainable."""

    def __init__(self, embeddings, Wq_init, Wk_init, n_heads,
                 n_layers=2, context_len=8, pos_decay=0.1):
        self.device = embeddings.device
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = self.d // n_heads
        self.context_len = context_len
        self.pos_decay = pos_decay

        # All weights (initialized from statistics, optionally refined)
        self.embeddings = embeddings.clone()
        self.Wq_layers = [Wq_init.clone() for _ in range(n_layers)]
        self.Wk_layers = [Wk_init.clone() for _ in range(n_layers)]
        self.E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)

        print(f"    d={self.d}, n_heads={n_heads}, d_head={self.d_head}, layers={n_layers}")

    def _attention(self, Q, K, V, causal, pad_mask):
        B, S, D = Q.shape
        H, d_h = self.n_heads, self.d_head
        Qh = Q.view(B, S, H, d_h).transpose(1, 2)
        Kh = K.view(B, S, H, d_h).transpose(1, 2)
        Vh = V.view(B, S, H, d_h).transpose(1, 2)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e9)
        scores = scores.masked_fill(~pad_mask.unsqueeze(1), -1e9)
        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, Vh)
        return out.transpose(1, 2).contiguous().view(B, S, D)

    def _ffn(self, x, E_norm, E_raw):
        B, S, D = x.shape
        out = torch.zeros_like(x)
        chunk = max(1, 500_000_000 // (S * self.V))
        chunk = min(chunk, B)
        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            sim = x[i:j] @ E_norm.T / math.sqrt(D)
            h = F.relu(sim)
            h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            out[i:j] = h @ E_raw
        return out

    def _forward_logits(self, contexts, embeddings, Wq_layers, Wk_layers, E_norm):
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
        x = embeddings[padded]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1) * pad_mask.unsqueeze(-1).float()

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)

        for l in range(self.n_layers):
            Q = x @ Wq_layers[l]
            K = x @ Wk_layers[l]
            attn_out = self._attention(Q, K, x, causal, p_mask)
            x = _layer_norm(x + attn_out)
            ffn_out = self._ffn(x, E_norm, embeddings)
            x = _layer_norm(x + ffn_out)

        logits = x[:, -1, :] @ embeddings.T / math.sqrt(self.d)
        return logits

    def score_batch(self, contexts):
        with torch.no_grad():
            logits = self._forward_logits(
                contexts, self.embeddings, self.Wq_layers, self.Wk_layers, self.E_norm)
            return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]

    def refine(self, encoded, lr=0.01, batch_size=64, max_epochs=20, patience=3):
        """Train ALL weights with early stopping."""
        # Make all weights trainable
        E_param = self.embeddings.clone().detach().requires_grad_(True)
        Wq_params = [w.clone().detach().requires_grad_(True) for w in self.Wq_layers]
        Wk_params = [w.clone().detach().requires_grad_(True) for w in self.Wk_layers]
        all_params = [E_param] + Wq_params + Wk_params
        optimizer = torch.optim.Adam(all_params, lr=lr)

        # Prepare training data
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

        E_norm_param = E_param / E_param.norm(dim=1, keepdim=True).clamp(min=1e-9)
        best_ppl = float('inf')
        no_improve = 0

        for epoch in range(max_epochs):
            np.random.shuffle(indices)
            total_loss, n_batches = 0.0, 0
            E_norm_param = E_param / E_param.norm(dim=1, keepdim=True).clamp(min=1e-9)

            for start in range(0, min(n_total, 10000), batch_size):
                end = min(start + batch_size, n_total)
                batch_idx = indices[start:end]
                batch_ctx = [all_ctx[i] for i in batch_idx]
                batch_tgt = torch.tensor([all_tgt[i] for i in batch_idx],
                                         dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self._forward_logits(
                    batch_ctx, E_param, Wq_params, Wk_params, E_norm_param)
                loss = F.cross_entropy(logits, batch_tgt)
                loss.backward()
                optimizer.step()
                E_norm_param = E_param / E_param.norm(dim=1, keepdim=True).clamp(min=1e-9)
                total_loss += loss.item()
                n_batches += 1

            ppl = math.exp(total_loss / max(n_batches, 1))
            improved = "↓" if ppl < best_ppl else "↑"
            print(f"    epoch {epoch}: PPL={ppl:.1f} {improved}")

            if ppl < best_ppl:
                best_ppl = ppl
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    early stop at epoch {epoch} (no improvement for {patience})")
                    break

        self.embeddings = E_param.detach()
        self.Wq_layers = [w.detach() for w in Wq_params]
        self.Wk_layers = [w.detach() for w in Wk_params]
        self.E_norm = self.embeddings / self.embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
