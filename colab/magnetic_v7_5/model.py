"""v7.5 — Distributed Statistical Transformer.

Uses Accelerate for automatic device distribution.
nn.Module throughout for proper device handling.
"""

import math, time, os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Spectrum builder (CPU, runs once)
# ======================================================================

def build_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                   bg_trans, V, min_ppmi, var_target=0.3):
    """Build embeddings + Wq/Wk init from PPMI SVD. Runs on CPU."""
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
    PPMI_dense = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce().to_dense()

    if V <= 5000:
        U_full, S_full, _ = torch.linalg.svd(PPMI_dense, full_matrices=False)
    else:
        U_full, S_full, _ = torch.svd_lowrank(PPMI_dense, q=min(V-1, 4096), niter=5)

    S_sq = S_full ** 2
    cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
    d = max(4, min(int((cumvar < var_target).sum().item()) + 1, len(S_full)))
    print(f"    cumvar: {cumvar[d-1]:.1%} at d={d}")
    embeddings = U_full[:, :d] * S_full[:d].sqrt().unsqueeze(0)

    TE = torch.sparse.mm(bg_trans.cpu(), embeddings)
    U_m, _, Vh_m = torch.linalg.svd(embeddings.T @ TE)
    Wq_init, Wk_init = U_m.contiguous(), Vh_m.T.contiguous()

    n_heads = 1
    for h in [2, 3, 4, 6, 7, 8, 12, 16]:
        if d % h == 0 and h <= d // 2:
            n_heads = h
    print(f"    d={d}, n_heads={n_heads}, built in {time.time()-t0:.1f}s")
    return embeddings, Wq_init, Wk_init, d, n_heads


# ======================================================================
# 2. Transformer components as nn.Module
# ======================================================================

class TransformerLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask):
        B, S, D = x.shape
        H, d_h = self.n_heads, self.d_head
        Q = self.Wq(x).view(B, S, H, d_h).transpose(1, 2)
        K = self.Wk(x).view(B, S, H, d_h).transpose(1, 2)
        V = x.view(B, S, H, d_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        scores = scores.masked_fill(~pad_mask.unsqueeze(1), -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, D)
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.W2(self.dropout(F.relu(self.W1(x))))))
        return x


class StatTransformer(nn.Module):
    def __init__(self, V, d, n_heads, n_layers, context_len=8, pos_decay=0.1, dropout=0.1):
        super().__init__()
        self.V, self.d, self.context_len, self.pos_decay = V, d, context_len, pos_decay
        self.embedding = nn.Embedding(V, d)
        self.layers = nn.ModuleList([TransformerLayer(d, n_heads, 4*d, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        print(f"    params={sum(p.numel() for p in self.parameters()):,}")

    def init_from_spectrum(self, embeddings, Wq_init, Wk_init):
        with torch.no_grad():
            self.embedding.weight.copy_(embeddings)
            E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
            d_ff = 4 * self.d
            for layer in self.layers:
                layer.Wq.weight.copy_(Wq_init)
                layer.Wk.weight.copy_(Wk_init)
                if self.V >= d_ff:
                    layer.W1.weight.copy_(E_norm[:d_ff])
                    layer.W2.weight.copy_(embeddings[:d_ff].T)

    def forward(self, input_ids, pad_mask=None):
        input_ids = input_ids.long()
        B, S = input_ids.shape
        dev = input_ids.device
        x = self.embedding(input_ids)
        pos = torch.arange(S, dtype=torch.float32, device=dev)
        x = x * torch.exp(-self.pos_decay * (S-1-pos)).unsqueeze(0).unsqueeze(-1)
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).float()
        causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
        pm = pad_mask.unsqueeze(1) if pad_mask is not None else torch.ones(B,1,S, device=dev, dtype=torch.bool)
        for layer in self.layers:
            x = layer(x, causal, pm)
        x = self.out_norm(x)
        return x[:, -1, :] @ self.embedding.weight.T / math.sqrt(self.d)

    def score_batch(self, contexts):
        cl = self.context_len
        trimmed = [c[-cl:] for c in contexts]
        ml = max(len(c) for c in trimmed) if trimmed else 1
        dev = self.embedding.weight.device
        padded = torch.zeros(len(trimmed), ml, dtype=torch.long, device=dev)
        pm = torch.zeros(len(trimmed), ml, dtype=torch.bool, device=dev)
        for i, c in enumerate(trimmed):
            L = len(c)
            padded[i, ml-L:] = torch.tensor(c, dtype=torch.long, device=dev)
            pm[i, ml-L:] = True
        with torch.no_grad():
            return F.softmax(self.forward(padded, pm), dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]


# ======================================================================
# 3. Dataset
# ======================================================================

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, encoded, context_len, max_samples=0):
        cl = context_len
        all_ids = np.concatenate([a.astype(np.int64) for a in encoded if a.size >= 2])
        n = len(all_ids) - 1
        if max_samples > 0:
            n = min(n, max_samples)
        self.ctx = torch.zeros(n, cl + 1, dtype=torch.int32)
        self.tgt = torch.zeros(n, dtype=torch.int32)
        for i in range(n):
            start = max(0, i - cl)
            c = all_ids[start:i+1]
            self.ctx[i, cl+1-len(c):] = torch.tensor(c, dtype=torch.int32)
            self.tgt[i] = int(all_ids[i+1])

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        return self.ctx[idx], self.tgt[idx]


# ======================================================================
# 4. Training function
# ======================================================================

def train_model(model, train_data, val_data, context_len,
                lr=0.001, batch_size=512, max_epochs=100, patience=5,
                max_train=0, max_val=10000):
    """Train with DataParallel for multi-GPU. No subsample on train by default."""
    train_ds = LMDataset(train_data, context_len, max_train)
    val_ds = LMDataset(val_data, context_len, max_val)
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    nw = 2 if use_cuda else 0

    # Multi-GPU via DataParallel (splits batch across GPUs)
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)
        batch_size = batch_size * gpu_count  # scale batch with GPUs
        print(f"    DataParallel: {gpu_count} GPUs, batch={batch_size}")
    else:
        print(f"    device={device}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=use_cuda)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=use_cuda)
    raw_model = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr, weight_decay=0.01)

    print(f"    train={len(train_ds):,} val={len(val_ds):,} batch={batch_size}")
    best_val, no_improve, best_state = float('inf'), 0, None

    for epoch in range(max_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for ctx_b, tgt_b in train_loader:
            ctx_b, tgt_b = ctx_b.to(device), tgt_b.to(device)
            optimizer.zero_grad()
            logits = model(ctx_b, ctx_b > 0)
            loss = F.cross_entropy(logits, tgt_b.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item()
            n_batches += 1
        train_ppl = math.exp(total_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for ctx_b, tgt_b in val_loader:
                ctx_b, tgt_b = ctx_b.to(device), tgt_b.to(device)
                logits = model(ctx_b, ctx_b > 0)
                vl += F.cross_entropy(logits, tgt_b.long()).detach().cpu().item() * ctx_b.shape[0]
                vn += ctx_b.shape[0]
            val_ppl = math.exp(vl / max(vn, 1))

        tag = "↓" if val_ppl < best_val else "↑"
        print(f"    epoch {epoch}: train={train_ppl:.1f} val={val_ppl:.1f} {tag}")

        if val_ppl < best_val:
            best_val, no_improve = val_ppl, 0
            raw = model.module if hasattr(model, 'module') else model
            best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop epoch {epoch} (best val={best_val:.1f})")
                break

    if best_state:
        raw = model.module if hasattr(model, 'module') else model
        raw.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# ======================================================================
# 5. Cache helper
# ======================================================================

def decay_cache(history, V, device):
    if len(history) < 2:
        return torch.zeros(V, dtype=torch.float32, device=device)
    h = torch.tensor(history, dtype=torch.int64, device=device)
    ages = torch.arange(h.numel()-1, -1, -1, dtype=torch.float32, device=device)
    b = torch.zeros(V, dtype=torch.float32, device=device)
    b.scatter_add_(0, h, 1.0 / torch.log(2.0 + ages))
    s = b.sum()
    return b / s if s > 1e-9 else b
