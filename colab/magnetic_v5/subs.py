"""Directional substitution tables via matrix overlap.

Pure tensor functions, no classes.
Returns dicts with successor/predecessor substitute tables.

successor_subs[w]: words sharing FOLLOWERS with w (B_fwd @ B_fwd.T)
predecessor_subs[w]: words sharing PREDECESSORS with w (B_bwd @ B_bwd.T)
"""

from typing import Dict, Tuple
import torch


def _binary_ppmi_matrix(
    rows: torch.Tensor,
    cols: torch.Tensor,
    counts: torch.Tensor,
    unigram: torch.Tensor,
    V: int,
    min_ppmi: float,
    device: torch.device,
) -> torch.Tensor:
    """Build sparse binary matrix: 1 where PPMI > threshold."""
    total = counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = counts / total
    p_a = unigram[rows].float() / N
    p_b = unigram[cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    idx = torch.stack([rows[keep], cols[keep]], dim=0)
    ones = torch.ones(int(keep.sum().item()), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, ones, (V, V)).coalesce()


def _overlap_topk(B: torch.Tensor, V: int, K: int, block_size: int, device: torch.device):
    """Compute B @ B.T in blocks, return top-K per row as [V, K] ids and weights."""
    ids = torch.zeros(V, K, dtype=torch.int64, device=device)
    wts = torch.zeros(V, K, dtype=torch.float32, device=device)
    n = 0
    b_idx = B.indices()
    b_val = B.values()

    for start in range(0, V, block_size):
        end = min(start + block_size, V)
        bs = end - start
        block = torch.zeros(bs, V, dtype=torch.float32, device=device)
        mask = (b_idx[0] >= start) & (b_idx[0] < end)
        if mask.any():
            block[b_idx[0][mask] - start, b_idx[1][mask]] = b_val[mask]
        if block.sum() == 0:
            continue
        sim = torch.sparse.mm(B.t(), block.t()).t()
        for i in range(bs):
            w = start + i
            if w < V:
                sim[i, w] = 0.0
        vals, idx_k = torch.topk(sim, min(K, V), dim=1)
        for i in range(bs):
            w = start + i
            if w >= V:
                break
            nv = int((vals[i] > 0).sum().item())
            nt = min(nv, K)
            if nt > 0:
                ids[w, :nt] = idx_k[i, :nt]
                wts[w, :nt] = vals[i, :nt]
                n += 1

    ws = wts.sum(dim=1, keepdim=True).clamp(min=1e-9)
    wts = wts / ws
    return ids, wts, n


def build(
    bg_rows: torch.Tensor,
    bg_cols: torch.Tensor,
    ctx_rows: torch.Tensor,
    ctx_cols: torch.Tensor,
    ctx_counts: torch.Tensor,
    unigram: torch.Tensor,
    V: int,
    K: int,
    min_ppmi: float,
    block_size: int,
    device: torch.device,
) -> Dict:
    """Build successor + predecessor substitution tables.

    Uses bigram pairs for directionality, context PPMI for filtering.
    """
    # Binary PPMI context matrix for distributional similarity
    B_ctx = _binary_ppmi_matrix(ctx_rows, ctx_cols, ctx_counts, unigram, V, min_ppmi, device)

    # Successor subs: words sharing followers
    # Forward bigram matrix: B_fwd[a, b] = 1 if b follows a
    # But filtered by PPMI context to avoid pure frequency matches
    print("    successor-subs (shared followers)...")
    # Use bigram for structure, PPMI for filtering
    idx_fwd = torch.stack([bg_rows, bg_cols], dim=0)
    ones_fwd = torch.ones(bg_rows.numel(), dtype=torch.float32, device=device)
    B_fwd = torch.sparse_coo_tensor(idx_fwd, ones_fwd, (V, V)).coalesce()
    succ_ids, succ_wts, ns = _overlap_topk(B_fwd, V, K, block_size, device)

    print("    predecessor-subs (shared predecessors)...")
    idx_bwd = torch.stack([bg_cols, bg_rows], dim=0)
    ones_bwd = torch.ones(bg_cols.numel(), dtype=torch.float32, device=device)
    B_bwd = torch.sparse_coo_tensor(idx_bwd, ones_bwd, (V, V)).coalesce()
    pred_ids, pred_wts, np_ = _overlap_topk(B_bwd, V, K, block_size, device)

    print(f"  [subs] succ={ns}/{V}  pred={np_}/{V}")
    return dict(
        succ_ids=succ_ids, succ_wts=succ_wts,
        pred_ids=pred_ids, pred_wts=pred_wts,
        K=K, V=V,
    )
