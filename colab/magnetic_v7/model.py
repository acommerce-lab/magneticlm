"""v7 — Statistical Transformer (Full Training Match).

Matches standard transformer:
  - Per-layer Wq_l, Wk_l (separate per layer)
  - Trainable FFN (W1, W2 per layer)
  - d constant, multi-head attention
  - All weights trainable with early stopping
  - No sample cap, max_epochs=100
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

    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)
    Wq_init = U_m.contiguous()
    Wk_init = Vh_m.T.contiguous()

    # Auto n_heads: find best divisor of d in range [2, 16]
    best_heads = 1
    for h in [2, 3, 4, 6, 7, 8, 12, 16]:
        if d % h == 0 and h <= d // 2:
            best_heads = h
    n_heads = best_heads
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
    def __init__(self, embeddings, Wq_init, Wk_init, n_heads,
                 n_layers=2, context_len=8, pos_decay=0.1):
        self.device = embeddings.device
        self.V, self.d = embeddings.shape
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = self.d // n_heads
        self.context_len = context_len
        self.pos_decay = pos_decay
        d = self.d

        self.embeddings = embeddings.clone()
        self.Wq_layers = [Wq_init.clone() for _ in range(n_layers)]
        self.Wk_layers = [Wk_init.clone() for _ in range(n_layers)]

        # FFN: W1 [d, 4d], W2 [4d, d] per layer — initialized from statistics
        d_ff = 4 * d
        self.W1_layers = []
        self.W2_layers = []
        E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        for _ in range(n_layers):
            # W1 init: project to vocab space then truncate to d_ff
            # Use top-d_ff words' normalized embeddings as W1 rows
            if self.V >= d_ff:
                W1 = E_norm[:d_ff, :].T.contiguous()  # [d, d_ff]
                W2 = embeddings[:d_ff, :].contiguous()  # [d_ff, d]
            else:
                W1 = torch.randn(d, d_ff, device=self.device) * 0.02
                W2 = torch.randn(d_ff, d, device=self.device) * 0.02
            self.W1_layers.append(W1)
            self.W2_layers.append(W2)

        self.dropout_p = 0.1
        self.training = False

        print(f"    d={d}, n_heads={n_heads}, d_head={self.d_head}, layers={n_layers}, d_ff={d_ff}")

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

    def _forward_logits(self, contexts, E, Wq_list, Wk_list, W1_list, W2_list):
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
        x = E[padded]
        pos = torch.arange(S, dtype=torch.float32, device=self.device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1) * pad_mask.unsqueeze(-1).float()

        causal = torch.tril(torch.ones(S, S, device=self.device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1)

        for l in range(self.n_layers):
            Q = x @ Wq_list[l]
            K = x @ Wk_list[l]
            attn_out = self._attention(Q, K, x, causal, p_mask)
            if self.training:
                attn_out = F.dropout(attn_out, p=self.dropout_p)
            x = _layer_norm(x + attn_out)

            # FFN: ReLU(x @ W1) @ W2
            ffn_h = F.relu(x @ W1_list[l])
            if self.training:
                ffn_h = F.dropout(ffn_h, p=self.dropout_p)
            ffn_out = ffn_h @ W2_list[l]
            x = _layer_norm(x + ffn_out)

        return x[:, -1, :] @ E.T / math.sqrt(self.d)

    def score_batch(self, contexts):
        with torch.no_grad():
            logits = self._forward_logits(
                contexts, self.embeddings,
                self.Wq_layers, self.Wk_layers,
                self.W1_layers, self.W2_layers)
            return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]

    def refine(self, enc_train, enc_valid, lr=0.001, batch_size=64,
               max_epochs=100, patience=5):
        """Train ALL weights with validation-based early stopping."""
        E_p = self.embeddings.clone().detach().requires_grad_(True)
        Wq_p = [w.clone().detach().requires_grad_(True) for w in self.Wq_layers]
        Wk_p = [w.clone().detach().requires_grad_(True) for w in self.Wk_layers]
        W1_p = [w.clone().detach().requires_grad_(True) for w in self.W1_layers]
        W2_p = [w.clone().detach().requires_grad_(True) for w in self.W2_layers]
        all_params = [E_p] + Wq_p + Wk_p + W1_p + W2_p
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=0.01)

        # Training data
        train_ctx, train_tgt = [], []
        for arr in enc_train:
            if arr.size < 2:
                continue
            ids = arr.astype(np.int64)
            for i in range(len(ids) - 1):
                ctx = ids[max(0, i - self.context_len):i + 1].tolist()
                train_ctx.append(ctx)
                train_tgt.append(int(ids[i + 1]))

        # Validation data (for early stopping)
        val_ctx, val_tgt = [], []
        for arr in enc_valid:
            if arr.size < 2:
                continue
            ids = arr.astype(np.int64)
            for i in range(min(len(ids) - 1, 5000)):
                ctx = ids[max(0, i - self.context_len):i + 1].tolist()
                val_ctx.append(ctx)
                val_tgt.append(int(ids[i + 1]))

        n_train = len(train_ctx)
        n_val = len(val_ctx)
        if n_train == 0:
            return
        self.training = True
        indices = list(range(n_train))
        best_val_ppl = float('inf')
        no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            # Train
            np.random.shuffle(indices)
            total_loss, n_batches = 0.0, 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                batch_idx = indices[start:end]
                batch_ctx = [train_ctx[i] for i in batch_idx]
                batch_tgt = torch.tensor([train_tgt[i] for i in batch_idx],
                                         dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self._forward_logits(batch_ctx, E_p, Wq_p, Wk_p, W1_p, W2_p)
                loss = F.cross_entropy(logits, batch_tgt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            train_ppl = math.exp(total_loss / max(n_batches, 1))

            # Validation PPL (dropout off)
            self.training = False
            with torch.no_grad():
                val_loss = 0.0
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    vctx = val_ctx[vs:ve]
                    vtgt = torch.tensor(val_tgt[vs:ve], dtype=torch.long, device=self.device)
                    vlogits = self._forward_logits(vctx, E_p, Wq_p, Wk_p, W1_p, W2_p)
                    val_loss += F.cross_entropy(vlogits, vtgt).item() * (ve - vs)
                val_ppl = math.exp(val_loss / max(n_val, 1))
            self.training = True

            tag = "↓" if val_ppl < best_val_ppl else "↑"
            print(f"    epoch {epoch}: train={train_ppl:.1f} val={val_ppl:.1f} {tag}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                no_improve = 0
                best_state = {
                    "E": E_p.detach().clone(),
                    "Wq": [w.detach().clone() for w in Wq_p],
                    "Wk": [w.detach().clone() for w in Wk_p],
                    "W1": [w.detach().clone() for w in W1_p],
                    "W2": [w.detach().clone() for w in W2_p],
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    early stop at epoch {epoch} (best val={best_val_ppl:.1f})")
                    break

        self.training = False
        if best_state:
            self.embeddings = best_state["E"]
            self.Wq_layers = best_state["Wq"]
            self.Wk_layers = best_state["Wk"]
            self.W1_layers = best_state["W1"]
            self.W2_layers = best_state["W2"]
