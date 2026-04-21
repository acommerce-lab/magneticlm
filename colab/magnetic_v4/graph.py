"""Sparse directional graph: forward and backward adjacency matrices.

Single responsibility: store the "network" in GPU-native sparse CSR form
ready for the wave propagator's sparse mat-vec.

Invariants:
  - fwd_adj[a, b] = exp(-cost_fwd(a → b))  (how much signal flows a → b)
  - bwd_adj[a, b] = exp(-cost_bwd(a ← b))  (how much signal flows a ← b)
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .stats import Stats, directional_costs


@dataclass
class Graph:
    vocab_size: int
    # Forward: source row, target col, weight = exp(-cost_fwd)
    fwd_adj: torch.Tensor   # sparse_coo [V, V]
    # Backward: source row, target col, weight = exp(-cost_bwd)
    bwd_adj: torch.Tensor   # sparse_coo [V, V]

    def to(self, device: torch.device) -> "Graph":
        return Graph(
            vocab_size=self.vocab_size,
            fwd_adj=self.fwd_adj.to(device),
            bwd_adj=self.bwd_adj.to(device),
        )


def _top_k_per_row(
    rows: torch.Tensor,
    vals: torch.Tensor,
    V: int,
    top_k: int,
) -> torch.Tensor:
    """Return a mask keeping at most top_k highest-val entries per row.

    Small V with many edges — we use sort-per-row via argsort tricks.
    For large V we keep all; caller should set reasonable top_k.
    """
    if top_k <= 0:
        return torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)

    # Rank each edge within its row by descending val
    # Approach: sort by (row, -val), mark positions 0..top_k within each row
    composite = rows.to(torch.int64) * (1 << 32) + (1_000_000 - (vals * 1e6).to(torch.int64))
    order = torch.argsort(composite)
    rows_sorted = rows[order]

    # position within row via consecutive index
    same_as_prev = torch.zeros_like(rows_sorted, dtype=torch.bool)
    same_as_prev[1:] = rows_sorted[1:] == rows_sorted[:-1]
    # cumulative count resets at each row boundary
    row_pos = torch.zeros_like(rows_sorted, dtype=torch.int64)
    # Compute cumulative position within each group using segment arithmetic
    # Simple approach: iterative won't scale; use scatter trick:
    boundaries = (~same_as_prev).to(torch.int64)
    # cumsum of boundaries gives group id for each element
    group_id = torch.cumsum(boundaries, dim=0) - 1
    # global index minus first index of each group = within-group position
    global_idx = torch.arange(rows_sorted.numel(), device=rows_sorted.device, dtype=torch.int64)
    # first index per group via scatter_min trick
    group_first = torch.full_like(group_id, fill_value=rows_sorted.numel())
    group_first.scatter_reduce_(0, group_id, global_idx, reduce="amin", include_self=False)
    first_for_each = group_first[group_id]
    row_pos = global_idx - first_for_each

    keep_sorted = row_pos < top_k
    # Map back to original order
    mask = torch.zeros(rows.numel(), dtype=torch.bool, device=rows.device)
    mask[order] = keep_sorted
    return mask


def build_graph(stats: Stats, cfg) -> Graph:
    """Construct fwd/bwd sparse adjacency from stats.

    Uses context pairs (symmetric window) so both directions are meaningful
    — asymmetry emerges from the directional cost (a|b vs b|a), not from
    an asymmetric pair collection.
    """
    V = stats.vocab_size
    device = stats.unigram_counts.device

    rows = stats.ctx_rows
    cols = stats.ctx_cols
    counts = stats.ctx_counts
    uni = stats.unigram_counts

    cost_fwd, cost_bwd = directional_costs(
        rows, cols, counts, uni,
        smoothing=cfg.cost_smoothing,
        ceiling=cfg.cost_ceiling,
    )
    w_fwd = torch.exp(-cost_fwd)
    w_bwd = torch.exp(-cost_bwd)

    # Top-K filtering per source row to bound out-degree
    if cfg.max_out_edges > 0:
        keep_f = _top_k_per_row(rows, w_fwd, V, cfg.max_out_edges)
        keep_b = _top_k_per_row(cols, w_bwd, V, cfg.max_out_edges)  # b is source for reverse
    else:
        keep_f = torch.ones(rows.numel(), dtype=torch.bool, device=device)
        keep_b = keep_f

    # Forward: a → b with weight w_fwd
    fwd_i = torch.stack([rows[keep_f], cols[keep_f]], dim=0)
    fwd_v = w_fwd[keep_f]
    fwd_adj = torch.sparse_coo_tensor(fwd_i, fwd_v, (V, V)).coalesce()

    # Backward: b → a (i.e. row=b, col=a) with weight w_bwd
    bwd_i = torch.stack([cols[keep_b], rows[keep_b]], dim=0)
    bwd_v = w_bwd[keep_b]
    bwd_adj = torch.sparse_coo_tensor(bwd_i, bwd_v, (V, V)).coalesce()

    # Row-normalize so each row sums to 1 (proper stochastic matrix).
    # Without this, repeated sparse mat-vec converges to the stationary
    # distribution of the graph (= frequency-based hubs dominate everything).
    fwd_adj = _row_normalize_sparse(fwd_adj)
    bwd_adj = _row_normalize_sparse(bwd_adj)

    return Graph(vocab_size=V, fwd_adj=fwd_adj, bwd_adj=bwd_adj)


def _row_normalize_sparse(mat: torch.Tensor) -> torch.Tensor:
    """Row-normalize a sparse COO matrix so each row sums to 1."""
    mat = mat.coalesce()
    indices = mat.indices()
    values = mat.values()
    if values.numel() == 0:
        return mat
    rows = indices[0]
    V = mat.size(0)
    row_sums = torch.zeros(V, device=values.device, dtype=values.dtype)
    row_sums.scatter_add_(0, rows, values)
    row_sums = row_sums.clamp(min=1e-9)
    values_norm = values / row_sums[rows]
    return torch.sparse_coo_tensor(indices, values_norm, mat.size()).coalesce()


def graph_info(g: Graph) -> str:
    f_nnz = int(g.fwd_adj._nnz())
    b_nnz = int(g.bwd_adj._nnz())
    return f"V={g.vocab_size}  fwd_edges={f_nnz:,}  bwd_edges={b_nnz:,}"
