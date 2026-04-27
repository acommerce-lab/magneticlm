"""v7 — Statistical Transformer as nn.Module.

Works on any device: CPU, GPU, multi-GPU, TPU.
Uses standard PyTorch nn.Module for automatic device handling.
"""

import math, time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Build spectrum (unchanged)
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

    TE = torch.sparse.mm(bg_trans.cpu(), embeddings.cpu()).to(device)
    M_embed = embeddings.T @ TE
    U_m, S_m, Vh_m = torch.linalg.svd(M_embed)
    Wq_init = U_m.contiguous()
    Wk_init = Vh_m.T.contiguous()

    # Auto n_heads
    best_heads = 1
    for h in [2, 3, 4, 6, 7, 8, 12, 16]:
        if d % h == 0 and h <= d // 2:
            best_heads = h
    n_heads = best_heads
    print(f"    d={d}, n_heads={n_heads}, d_head={d // n_heads}")
    print(f"    built in {time.time()-t0:.1f}s")
    return embeddings, Wq_init, Wk_init, d, n_heads


# ======================================================================
# 2. Transformer Layer as nn.Module
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
        # Multi-head attention
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
        # FFN
        ffn = self.W2(self.dropout(F.relu(self.W1(x))))
        x = self.norm2(x + self.dropout(ffn))
        return x


# ======================================================================
# 3. Full Model as nn.Module
# ======================================================================

class StatTransformer(nn.Module):
    def __init__(self, V, d, n_heads, n_layers, context_len=8,
                 pos_decay=0.1, dropout=0.1):
        super().__init__()
        self.V = V
        self.d = d
        self.n_layers = n_layers
        self.context_len = context_len
        self.pos_decay = pos_decay

        self.embedding = nn.Embedding(V, d)
        self.layers = nn.ModuleList([
            TransformerLayer(d, n_heads, 4 * d, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d)
        print(f"    d={d}, n_heads={n_heads}, d_head={d//n_heads}, "
              f"layers={n_layers}, d_ff={4*d}, params={sum(p.numel() for p in self.parameters()):,}")

    def init_from_spectrum(self, embeddings, Wq_init, Wk_init):
        """Initialize weights from spectral decomposition."""
        with torch.no_grad():
            self.embedding.weight.copy_(embeddings)
            E_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
            d_ff = 4 * self.d
            for layer in self.layers:
                layer.Wq.weight.copy_(Wq_init)
                layer.Wk.weight.copy_(Wk_init)
                if self.V >= d_ff:
                    layer.W1.weight.copy_(E_norm[:d_ff, :])
                    layer.W2.weight.copy_(embeddings[:d_ff, :].T)

    def forward(self, input_ids, pad_mask=None):
        B, S = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)
        pos = torch.arange(S, dtype=torch.float32, device=device)
        pw = torch.exp(-self.pos_decay * (S - 1 - pos))
        x = x * pw.unsqueeze(0).unsqueeze(-1)
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).float()

        causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
        p_mask = pad_mask.unsqueeze(1) if pad_mask is not None else \
                 torch.ones(B, 1, S, device=device, dtype=torch.bool)

        for layer in self.layers:
            x = layer(x, causal, p_mask)

        x = self.out_norm(x)
        logits = x[:, -1, :] @ self.embedding.weight.T / math.sqrt(self.d)
        return logits

    def score_batch(self, contexts):
        cl = self.context_len
        trimmed = [c[-cl:] for c in contexts]
        max_len = max(len(c) for c in trimmed) if trimmed else 1
        device = self.embedding.weight.device

        padded = torch.zeros(len(trimmed), max_len, dtype=torch.long, device=device)
        pad_mask = torch.zeros(len(trimmed), max_len, dtype=torch.bool, device=device)
        for i, c in enumerate(trimmed):
            L = len(c)
            padded[i, max_len - L:] = torch.tensor(c, dtype=torch.long, device=device)
            pad_mask[i, max_len - L:] = True

        with torch.no_grad():
            logits = self.forward(padded, pad_mask)
            return F.softmax(logits, dim=1)

    def score_single(self, context_ids):
        return self.score_batch([context_ids])[0]


# ======================================================================
# 4. Training with DataLoader
# ======================================================================

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, encoded, context_len):
        self.pairs = []
        for arr in encoded:
            if arr.size < 2:
                continue
            ids = arr.astype(np.int64)
            for i in range(len(ids) - 1):
                ctx = ids[max(0, i - context_len):i + 1]
                # Pad to context_len + 1
                padded = np.zeros(context_len + 1, dtype=np.int64)
                padded[context_len + 1 - len(ctx):] = ctx
                self.pairs.append((padded, int(ids[i + 1])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ctx, tgt = self.pairs[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def train_model(model, train_data, val_data, context_len,
                lr=0.001, batch_size=512, max_epochs=100, patience=5,
                max_samples=50000):
    """Train with DataLoader — works on any device."""
    device = next(model.parameters()).device

    train_ds = LMDataset(train_data, context_len)
    val_ds = LMDataset(val_data, context_len)

    # Subsample if too large
    if len(train_ds) > max_samples:
        indices = torch.randperm(len(train_ds))[:max_samples].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    best_val_ppl = float('inf')
    no_improve = 0
    best_state = None

    print(f"    train={len(train_ds):,} val={len(val_ds):,} batch={batch_size}")

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss, n_batches = 0.0, 0
        for ctx_batch, tgt_batch in train_loader:
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            pad_mask = ctx_batch > 0

            optimizer.zero_grad()
            logits = model(ctx_batch, pad_mask)
            loss = F.cross_entropy(logits, tgt_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_ppl = math.exp(total_loss / max(n_batches, 1))

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_n = 0.0, 0
            for ctx_batch, tgt_batch in val_loader:
                ctx_batch = ctx_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                pad_mask = ctx_batch > 0
                logits = model(ctx_batch, pad_mask)
                val_loss += F.cross_entropy(logits, tgt_batch).item() * ctx_batch.shape[0]
                val_n += ctx_batch.shape[0]
            val_ppl = math.exp(val_loss / max(val_n, 1))

        tag = "↓" if val_ppl < best_val_ppl else "↑"
        print(f"    epoch {epoch}: train={train_ppl:.1f} val={val_ppl:.1f} {tag}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop at epoch {epoch} (best val={best_val_ppl:.1f})")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})


# ======================================================================
# 5. Cache helper
# ======================================================================

def decay_cache(history, V, device):
    if len(history) < 2:
        return torch.zeros(V, dtype=torch.float32, device=device)
    h = torch.tensor(history, dtype=torch.int64, device=device)
    ages = torch.arange(h.numel() - 1, -1, -1, dtype=torch.float32, device=device)
    b = torch.zeros(V, dtype=torch.float32, device=device)
    b.scatter_add_(0, h, 1.0 / torch.log(2.0 + ages))
    s = b.sum()
    return b / s if s > 1e-9 else b
