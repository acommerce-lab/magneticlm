"""v7 — Genetic Statistical Transformer with Spectral Refinement.

All structure from statistics. Only S_m (spectral weights) refined iteratively.
This is NOT full backprop — it's tuning d dials on a pre-built machine.

Architecture: Triangle Transformer (d shrinks per layer)
Refinement: optimize S_m to minimize cross-entropy on training data
Saturation: stop when S_m changes < threshold
"""

import math, time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Build everything from spectrum
# ======================================================================

def build_all_from_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                            bg_trans, V, min_ppmi, device, var_target=0.5):
    t0 = time.time()

    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    E = int(keep.sum().item())
    sigma_ppmi = float(vals_f.std().item()) if vals_f.numel() > 1 else 1.0
    S_noise = sigma_ppmi * math.sqrt(2.0 * E / max(V, 1))
    print(f"    noise floor: sigma={sigma_ppmi:.2f}, E={E:,}, S_noise={S_noise:.2f}")

    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    PPMI = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    PPMI_dense = PPMI.to_dense()

    if V <= 5000:
        print(f"    full SVD on [{V}x{V}] PPMI...")
        U_full, S_full, Vt_full = torch.linalg.svd(PPMI_dense, full_matrices=False)
    else:
        q = min(V - 1, 4096)
        print(f"    randomized SVD on [{V}x{V}] PPMI, q={q}...")
        U_full, S_full, Vt_full = torch.svd_lowrank(PPMI_dense, q=q, niter=5)

    if var_target > 0:
        S_sq = S_full ** 2
        cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
        d = int((cumvar < var_target).sum().item()) + 1
        d = max(4, min(d, len(S_full)))
        print(f"    cumulative variance: {cumvar[d-1]:.1%} at d={d} (target={var_target:.0%})")
    else:
        log_S = torch.log(S_full.clamp(min=1e-10))
        if len(log_S) > 4:
            d2 = log_S[:-2] - 2 * log_S[1:-1] + log_S[2:]
            d = int(d2.argmax().item()) + 2
            d = max(4, min(d, len(S_full)))
        else:
            d = len(S_full)
        S_sq = S_full ** 2
        cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
        print(f"    elbow at d={d} (explains {cumvar[d-1]:.1%} variance)")
    print(f"    top-8 S: {', '.join(f'{s:.1f}' for s in S_full[:8].tolist())}")
    if d < len(S_full):
        print(f"    boundary: S[{d-1}]={S_full[d-1]:.2f}, S[{d}]={S_full[d]:.2f}")

    U = U_full[:, :d]
    S = S_full[:d]
    embeddings = (U * S.sqrt().unsqueeze(0)).to(device)
    print(f"    embeddings [{V}x{d}]")

    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)

    Wq_base = U_m.contiguous()
    Wk_base = Vh_m.T.contiguous()
    print(f"    S_m: max={S_m[0]:.1f}, min={S_m[-1]:.3f}")

    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    built in {time.time()-t0:.1f}s")
    return embeddings, Wq_base, Wk_base, S_m, idf, d


# ======================================================================
# 2. Triangle Transformer with Spectral Refinement
# ======================================================================

def _layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt()


class StatTransformer:
    """Triangle transformer with refinable spectral weights.

    S_raw is the ONLY trainable parameter (d numbers).
    Everything else (U_m, V_m, E, structure) is fixed from statistics.
    """

    def __init__(self, embeddings, Wq_base, Wk_base, S_raw, idf,
                 n_layers=2, context_len=8, pos_decay=0.1, devices=None):
        self.device = embeddings.device
        self.embeddings = embeddings
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay

        self.Wq_base = Wq_base  # [d, d] fixed rotation
        self.Wk_base = Wk_base  # [d, d] fixed rotation
        self.S_raw = S_raw.clone().detach()  # [d] — will be refined

        self.d_schedule = []
        for l in range(n_layers):
            dl = max(4, self.d // (2 ** l))
            self.d_schedule.append(dl)
        print(f"    triangle: {' -> '.join(str(dl) for dl in self.d_schedule)}")

        self.E_norm_layers = []
        for dl in self.d_schedule:
            e = embeddings[:, :dl]
            self.E_norm_layers.append(e / e.norm(dim=1, keepdim=True).clamp(min=1e-9))

        d_out = self.d_schedule[-1]
        self.E_out = embeddings[:, :d_out]
        self.idf = idf
        self.devices = devices or [self.device]

    def _get_Wq_Wk(self, layer_idx, S_param):
        """Compute Wq/Wk on-the-fly from S_param (differentiable)."""
        dl = self.d_schedule[layer_idx]
        alpha = 2.0 ** layer_idx
        mu = S_param.mean()
        sigma = S_param.std().clamp(min=1e-9)
        S_dampened = S_param[:dl] * torch.sigmoid(alpha * (S_param[:dl] - mu) / sigma)
        S_unit = S_dampened / S_dampened.norm().clamp(min=1e-9) * math.sqrt(dl)
        sqrt_Su = S_unit.sqrt()
        Wq = self.Wq_base[:dl, :dl] * sqrt_Su.unsqueeze(0)
        Wk = self.Wk_base[:dl, :dl] * sqrt_Su.unsqueeze(0)
        return Wq, Wk

    def _forward(self, contexts, S_param):
        """Forward pass using given S_param (supports autograd)."""
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
            dl = self.d_schedule[layer_idx]
            x = x[:, :, :dl]
            Wq, Wk = self._get_Wq_Wk(layer_idx, S_param)

            Q = x @ Wq
            K = x @ Wk
            V_attn = x

            d_l = Q.shape[-1]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_l)
            scores = scores.masked_fill(~causal, -1e9)
            scores = scores.masked_fill(~p_mask, -1e9)
            attn_w = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_w, V_attn)
            x = _layer_norm(x + attn_out)

            E_norm_l = self.E_norm_layers[layer_idx]
            E_raw_l = self.embeddings[:, :dl]
            sim = x @ E_norm_l.T / math.sqrt(dl)
            h = F.relu(sim)
            h = h / h.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            ffn_out = h @ E_raw_l
            x = _layer_norm(x + ffn_out)

        logits = x[:, -1, :] @ self.E_out.T
        return logits

    def score_batch(self, contexts):
        with torch.no_grad():
            logits = self._forward(contexts, self.S_raw)
            return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]

    def refine(self, encoded, n_epochs, lr=0.01, batch_size=64):
        """Refine S_raw (d numbers) via gradient descent on training data.

        Only S_raw changes. Everything else is fixed.
        Each epoch: sample contexts, forward, cross-entropy, backward, update S_raw.
        """
        S_param = self.S_raw.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([S_param], lr=lr)

        # Collect training contexts and targets
        all_ctx = []
        all_tgt = []
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
        S_before = S_param.detach().clone()

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, min(n_total, 5000), batch_size):
                end = min(start + batch_size, n_total)
                batch_idx = indices[start:end]
                batch_ctx = [all_ctx[i] for i in batch_idx]
                batch_tgt = torch.tensor([all_tgt[i] for i in batch_idx],
                                         dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self._forward(batch_ctx, S_param)
                loss = F.cross_entropy(logits, batch_tgt)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            ppl = math.exp(total_loss / max(n_batches, 1))
            S_change = (S_param.detach() - S_before).norm() / S_before.norm().clamp(min=1e-9)
            print(f"    epoch {epoch}: PPL={ppl:.1f}, S_change={S_change:.4f}")

            if S_change < 0.001:
                print(f"    saturated at epoch {epoch}")
                break

            S_before = S_param.detach().clone()

        self.S_raw = S_param.detach()
        print(f"    refined S_raw: max={self.S_raw.max():.1f}, min={self.S_raw.min():.3f}")
