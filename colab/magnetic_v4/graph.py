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
    # Syntactic: p(b|a) based edges for grammar flow (Re channel)
    syn_fwd: torch.Tensor   # sparse_coo [V, V]
    syn_bwd: torch.Tensor   # sparse_coo [V, V]
    # Semantic: PPMI-based edges for concept flow (Im channel)
    sem_fwd: torch.Tensor   # sparse_coo [V, V]
    sem_bwd: torch.Tensor   # sparse_coo [V, V]

    def to(self, device: torch.device) -> "Graph":
        return Graph(
            vocab_size=self.vocab_size,
            syn_fwd=self.syn_fwd.to(device),
            syn_bwd=self.syn_bwd.to(device),
            sem_fwd=self.sem_fwd.to(device),
            sem_bwd=self.sem_bwd.to(device),
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

    Edge SELECTION uses PPMI (semantic surprise) so content-word links survive
    the top-K filter; edge WEIGHTS use directional transition probabilities
    (p(b|a) forward, p(a|b) backward) so signal strength is properly
    asymmetric.
    """
    V = stats.vocab_size
    device = stats.unigram_counts.device

    rows = stats.ctx_rows
    cols = stats.ctx_cols
    counts = stats.ctx_counts
    uni = stats.unigram_counts

    # ---- Compute PPMI for edge-selection scoring --------------------------
    # PPMI(a,b) = max(0, log[ p(a,b) / (p(a)·p(b)) ])
    # High PPMI = "surprising" co-occurrence = semantic link.
    # Low PPMI = just both common = function-word pairing.
    total_pairs = counts.sum().clamp(min=1.0)
    p_ab = counts / total_pairs
    uni_f = uni.to(torch.float32)
    N = uni_f.sum().clamp(min=1.0)
    p_a = uni_f[rows] / N
    p_b = uni_f[cols] / N
    pmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    ppmi = pmi.clamp(min=0.0)

    cost_fwd, cost_bwd = directional_costs(
        rows, cols, counts, uni,
        smoothing=cfg.cost_smoothing,
        ceiling=cfg.cost_ceiling,
    )
    w_fwd = torch.exp(-cost_fwd)
    w_bwd = torch.exp(-cost_bwd)

    # ---- SYNTACTIC edges: top-K by raw transition probability ----
    # These carry grammar: "the" after "ruled", "is" after "water", etc.
    K = cfg.max_out_edges
    if K > 0:
        keep_syn_f = _top_k_per_row(rows, w_fwd, V, K)
        keep_syn_b = _top_k_per_row(cols, w_bwd, V, K)
    else:
        keep_syn_f = torch.ones(rows.numel(), dtype=torch.bool, device=device)
        keep_syn_b = keep_syn_f

    syn_fwd = _build_sparse(rows, cols, w_fwd, keep_syn_f, V)
    syn_bwd = _build_sparse(cols, rows, w_bwd, keep_syn_b, V)

    # ---- SEMANTIC edges: top-K by PPMI ----
    # These carry meaning: king↔queen, paris↔france, sky↔blue
    min_ppmi = float(getattr(cfg, "min_ppmi", 0.5))
    ppmi_mask = ppmi >= min_ppmi
    if K > 0:
        keep_sem = _top_k_per_row(rows, ppmi, V, K) & ppmi_mask
    else:
        keep_sem = ppmi_mask

    sem_fwd = _build_sparse(rows, cols, w_fwd, keep_sem, V)
    sem_bwd = _build_sparse(cols, rows, w_bwd, keep_sem, V)

    # Row-normalize both so propagation is a proper random walk
    syn_fwd = _row_normalize_sparse(syn_fwd)
    syn_bwd = _row_normalize_sparse(syn_bwd)
    sem_fwd = _row_normalize_sparse(sem_fwd)
    sem_bwd = _row_normalize_sparse(sem_bwd)

    return Graph(
        vocab_size=V,
        syn_fwd=syn_fwd, syn_bwd=syn_bwd,
        sem_fwd=sem_fwd, sem_bwd=sem_bwd,
    )


def _build_sparse(
    rows: torch.Tensor, cols: torch.Tensor,
    vals: torch.Tensor, mask: torch.Tensor, V: int,
) -> torch.Tensor:
    i = torch.stack([rows[mask], cols[mask]], dim=0)
    v = vals[mask]
    return torch.sparse_coo_tensor(i, v, (V, V)).coalesce()


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
    sf = int(g.syn_fwd._nnz())
    sb = int(g.syn_bwd._nnz())
    mf = int(g.sem_fwd._nnz())
    mb = int(g.sem_bwd._nnz())
    return (f"V={g.vocab_size}  syn={sf:,}/{sb:,}  sem={mf:,}/{mb:,}")
