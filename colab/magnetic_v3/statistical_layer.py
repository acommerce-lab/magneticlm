"""Statistical foundation: unigram, bigram, PPMI, entropy, capacity.

Computed once before conceptual training. Runs on GPU when available.
All higher layers read from this frozen foundation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Statistics:
    vocab_size: int

    unigram: torch.Tensor            # [V] float32  (probabilities)
    unigram_counts: torch.Tensor     # [V] int64

    # PPMI as sparse coo tensor  (rows=target, cols=context)
    ppmi_indices: torch.Tensor       # [2, nnz] int64
    ppmi_values: torch.Tensor        # [nnz] float32

    # Per-node entropy H(n) over window neighbors
    entropy: torch.Tensor            # [V] float32
    entropy_max: float

    # Capacity per node (int)
    capacity: torch.Tensor           # [V] int32

    # Bigram probabilities (KN-smoothed) as CSR
    bg_row_ptr: torch.Tensor         # [V+1]
    bg_search_key: torch.Tensor      # [nnz]  row*V + col (globally sorted)
    bg_prob: torch.Tensor            # [nnz]
    bg_backoff: torch.Tensor         # [V] (KN lambda per row)

    # Diagnostics
    k_base: float

    def to(self, device: torch.device) -> "Statistics":
        kw = {}
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                kw[name] = val.to(device)
            else:
                kw[name] = val
        return Statistics(**kw)


# ==========================================================================
# Capacity strategy registry
# ==========================================================================

CAPACITY_METHODS: Dict[str, callable] = {}


def register_capacity(name: str):
    def deco(fn):
        CAPACITY_METHODS[name] = fn
        return fn
    return deco


@register_capacity("entropy")
def _cap_entropy(stats, cfg, k_base: float) -> torch.Tensor:
    """C(n) = floor + (k_base - floor) * (1 - H(n)/H_max).

    Floor is capacity_min: every node gets at least this many slots.
    Ceiling is k_base: high-entropy (diffuse) words stay near floor;
    low-entropy (focused) words grow up to k_base. This gives a real
    dynamic range even when k_base is small, unlike the old formula
    which collapsed to floor for everyone when k_base <= floor.
    """
    H = stats["entropy"]
    H_max = stats["entropy_max"]
    floor = float(cfg.capacity_min)
    ceiling = max(float(k_base), floor + 1.0)
    scale = (1.0 - (H / (H_max + 1e-9))).clamp(min=0.0, max=1.0)
    cap = floor + (ceiling - floor) * scale
    return cap * cfg.capacity_multiplier


@register_capacity("log_inv_freq")
def _cap_log_inv_freq(stats, cfg, k_base: float) -> torch.Tensor:
    f = stats["unigram_counts"].float()
    N = f.sum().item()
    val = torch.log(N / (f + 1.0) + 1.0)
    return k_base * cfg.capacity_multiplier * val / (val.max() + 1e-9)


@register_capacity("uniform")
def _cap_uniform(stats, cfg, k_base: float) -> torch.Tensor:
    V = stats["unigram_counts"].numel()
    return torch.full((V,), k_base * cfg.capacity_multiplier)


# ==========================================================================
# Counters — accumulate in dict[(a,b)] -> count using GPU batched pair ids
# ==========================================================================


def _pair_ids(a: torch.Tensor, b: torch.Tensor, V: int) -> torch.Tensor:
    return a.to(torch.int64) * V + b.to(torch.int64)


def _reduce_pairs(pair_ids: torch.Tensor, V: int):
    if pair_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=pair_ids.device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=pair_ids.device)
    sorted_ids, _ = torch.sort(pair_ids)
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    rows = (uniq // V).to(torch.int64)
    cols = (uniq % V).to(torch.int64)
    vals = counts.to(torch.float32)
    return rows, cols, vals


# ==========================================================================
# Main build
# ==========================================================================


def build_statistics(
    encoded: List[np.ndarray],
    vocab_size: int,
    cfg,
    device: torch.device,
) -> Statistics:
    V = vocab_size

    # Unigram
    uni = torch.zeros(V, dtype=torch.int64, device=device)
    for arr in encoded:
        if arr.size == 0:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uni.scatter_add_(0, t, torch.ones_like(t))
    total = uni.sum().clamp(min=1).float()
    uni_prob = uni.float() / total

    # Bigram (direct A->B, window=1) — now returns (rows, cols, counts) directly
    bg_rows, bg_cols, bg_counts = _collect_pairs(
        encoded, V, window=1, symmetric=False, device=device
    )

    # Kneser-Ney smoothed bigram probabilities per row
    bg_row_ptr, bg_search_key, bg_prob, bg_backoff = _kn_smoothed(
        bg_rows, bg_cols, bg_counts, uni.float(), V, discount=cfg.kn_discount, device=device
    )

    # Context pairs for PPMI/entropy (symmetric window)
    c_rows, c_cols, c_counts = _collect_pairs(
        encoded, V, window=cfg.stat_window, symmetric=True, device=device
    )

    # Filter by min count
    keep = c_counts >= cfg.ppmi_min_count
    c_rows, c_cols, c_counts = c_rows[keep], c_cols[keep], c_counts[keep]

    # PPMI = max(0, log[ p(a,b) / (p(a)*p(b)) ])
    row_tot = torch.zeros(V, device=device).scatter_add_(0, c_rows, c_counts)
    total_pairs = c_counts.sum().clamp(min=1)
    p_ab = c_counts / total_pairs
    p_a = row_tot[c_rows] / total_pairs.clamp(min=1)
    p_b = row_tot[c_cols] / total_pairs.clamp(min=1)
    pmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    ppmi = torch.clamp(pmi, min=0.0)

    keep2 = ppmi > cfg.ppmi_threshold
    ppmi_rows = c_rows[keep2]
    ppmi_cols = c_cols[keep2]
    ppmi_vals = ppmi[keep2].to(torch.float32)

    # Entropy H(n) = - sum p(b|n) log p(b|n) over context neighbors
    entropy = _row_entropy(c_rows, c_cols, c_counts, V, device)
    H_max = float(np.log2(max(V, 2)))

    # k_base — derived from the degree distribution of the PPMI graph.
    # The old "median" default collapses to 2-4 and kills capacity dynamic
    # range. "percentile" (default 0.9) tracks the long tail so a meaningful
    # fraction of nodes can grow edges far above the floor.
    method = cfg.k_base_method
    if method in ("percentile", "percentile_neighbors") or method.startswith("percentile"):
        uniq_rows, neighbor_counts = torch.unique(c_rows, return_counts=True)
        if neighbor_counts.numel() == 0:
            k_base = float(cfg.k_base_fixed)
        else:
            pct = float(getattr(cfg, "k_base_percentile", 0.9))
            k_base = float(
                torch.quantile(neighbor_counts.to(torch.float32), pct).item()
            )
    elif method == "median_neighbors":
        uniq_rows, neighbor_counts = torch.unique(c_rows, return_counts=True)
        if neighbor_counts.numel() == 0:
            k_base = float(cfg.k_base_fixed)
        else:
            k_base = float(torch.median(neighbor_counts).item())
    else:
        k_base = float(cfg.k_base_fixed)

    # Capacity
    cap_fn = CAPACITY_METHODS.get(cfg.capacity_method, CAPACITY_METHODS["entropy"])
    raw_cap = cap_fn(
        {
            "entropy": entropy,
            "entropy_max": H_max,
            "unigram_counts": uni,
        },
        cfg,
        k_base,
    )
    capacity = torch.clamp(raw_cap, min=cfg.capacity_min, max=cfg.capacity_max).to(torch.int32)

    return Statistics(
        vocab_size=V,
        unigram=uni_prob.to(torch.float32),
        unigram_counts=uni.to(torch.int64),
        ppmi_indices=torch.stack([ppmi_rows, ppmi_cols], dim=0),
        ppmi_values=ppmi_vals,
        entropy=entropy.to(torch.float32),
        entropy_max=H_max,
        capacity=capacity,
        bg_row_ptr=bg_row_ptr,
        bg_search_key=bg_search_key,
        bg_prob=bg_prob,
        bg_backoff=bg_backoff,
        k_base=k_base,
    )


def _collect_pairs(
    encoded: List[np.ndarray],
    V: int,
    window: int,
    symmetric: bool,
    device: torch.device,
    chunk_tokens: int = 4_000_000,
):
    """Chunked GPU pair collection with on-the-fly reduction.

    Returns (rows, cols, counts) as GPU tensors — already deduplicated.
    Each chunk's pairs are reduced (sort+unique_consecutive) before the
    next chunk processes, keeping peak GPU memory bounded regardless of
    corpus size. Chunk-level reduced results are merged via scatter_add.
    """
    valid = [a for a in encoded if a.size >= 2]
    if not valid:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    # Group sentences into chunks by cumulative token count
    chunks: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    cur_sz = 0
    for a in valid:
        cur.append(a)
        cur_sz += a.size
        if cur_sz >= chunk_tokens:
            chunks.append(cur)
            cur = []
            cur_sz = 0
    if cur:
        chunks.append(cur)

    all_keys_cpu: List[torch.Tensor] = []
    all_counts_cpu: List[torch.Tensor] = []

    for chunk in chunks:
        flat = np.concatenate(chunk).astype(np.int64)
        sent_ids = np.concatenate([
            np.full(a.size, i, dtype=np.int32) for i, a in enumerate(chunk)
        ])
        t = torch.from_numpy(flat).to(device)
        sid = torch.from_numpy(sent_ids).to(device)
        n = t.numel()

        chunk_buf: List[torch.Tensor] = []
        for d in range(1, window + 1):
            if d >= n:
                break
            same = sid[:-d] == sid[d:]
            a = t[:-d][same]
            b = t[d:][same]
            if a.numel() == 0:
                continue
            chunk_buf.append(_pair_ids(a, b, V))
            if symmetric:
                chunk_buf.append(_pair_ids(b, a, V))
        del t, sid

        if not chunk_buf:
            continue

        chunk_ids = torch.cat(chunk_buf)
        del chunk_buf
        sorted_ids, _ = torch.sort(chunk_ids)
        del chunk_ids
        uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        del sorted_ids
        all_keys_cpu.append(uniq.cpu())
        all_counts_cpu.append(counts.to(torch.int64).cpu())
        torch.cuda.empty_cache()

    if not all_keys_cpu:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    all_keys = torch.cat(all_keys_cpu).to(device)
    all_counts = torch.cat(all_counts_cpu).to(device).to(torch.int64)

    sort_idx = torch.argsort(all_keys)
    all_keys = all_keys[sort_idx]
    all_counts = all_counts[sort_idx]

    uniq, inverse = torch.unique_consecutive(all_keys, return_inverse=True)
    final_counts = torch.zeros(uniq.numel(), dtype=torch.int64, device=device)
    final_counts.scatter_add_(0, inverse, all_counts)

    rows = (uniq // V).to(torch.int64)
    cols = (uniq % V).to(torch.int64)
    return rows, cols, final_counts.to(torch.float32)


def _dedup_sort(x: torch.Tensor) -> torch.Tensor:
    # keep duplicates; we'll reduce later
    return x


def _row_entropy(
    rows: torch.Tensor,
    cols: torch.Tensor,
    counts: torch.Tensor,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.zeros(V, dtype=torch.float32, device=device)
    row_tot = torch.zeros(V, device=device).scatter_add_(0, rows, counts)
    p = counts / row_tot[rows].clamp(min=1.0)
    term = -p * torch.log2(p.clamp(min=1e-12))
    H = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(
        0, rows, term.to(torch.float32)
    )
    return H


def _kn_smoothed(
    rows: torch.Tensor,
    cols: torch.Tensor,
    counts: torch.Tensor,
    unigram: torch.Tensor,
    V: int,
    discount: float,
    device: torch.device,
):
    """Absolute-discount Kneser-Ney-style bigram."""
    if rows.numel() == 0:
        return (
            torch.zeros(V + 1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.ones(V, dtype=torch.float32, device=device),
        )

    sort_idx = torch.argsort(rows * V + cols)
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    counts = counts[sort_idx].to(torch.float32)

    # row totals
    row_tot = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(0, rows, counts)
    # row unique neighbors (for backoff mass)
    _, row_unique = torch.unique(rows, return_counts=True)
    unique_full = torch.zeros(V, dtype=torch.float32, device=device)
    uniq_rows = torch.unique(rows)
    unique_full[uniq_rows] = row_unique.to(torch.float32)

    # probability: max(0, c(a,b)-d)/c(a) + lambda(a) * p_cont(b)
    d = discount
    numer = torch.clamp(counts - d, min=0.0)
    denom = row_tot[rows].clamp(min=1.0)
    p_primary = numer / denom

    # continuation prob = uniform fallback (simpler than true continuation count)
    V_f = max(V, 1)
    p_cont = 1.0 / V_f  # uniform — strong simplification for speed

    lam = (d * unique_full) / torch.clamp(row_tot, min=1.0)
    # add backoff later during lookup; here store main only.

    # Build CSR
    bg_row_ptr = torch.zeros(V + 1, dtype=torch.int64, device=device)
    ones = torch.ones_like(rows, dtype=torch.int64)
    bg_row_ptr.scatter_add_(0, rows + 1, ones)
    bg_row_ptr = torch.cumsum(bg_row_ptr, dim=0)

    search_key = rows.to(torch.int64) * V + cols.to(torch.int64)
    return bg_row_ptr, search_key, p_primary.to(torch.float32), lam.to(torch.float32)


def bigram_prob(
    stats: Statistics,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Query p(b|a) with KN-like backoff to uniform."""
    V = stats.vocab_size
    device = stats.bg_row_ptr.device
    a = a.to(device).to(torch.int64)
    b = b.to(device).to(torch.int64)

    nnz = stats.bg_search_key.numel()
    if nnz == 0:
        lam = stats.bg_backoff[a]
        return lam * (1.0 / max(V, 1))

    query = a * V + b
    pos = torch.searchsorted(stats.bg_search_key, query, right=False)
    safe = torch.clamp(pos, max=nnz - 1)
    hit = (pos < nnz) & (stats.bg_search_key[safe] == query)
    p_main = torch.where(hit, stats.bg_prob[safe], torch.zeros_like(stats.bg_prob[safe]))
    lam = stats.bg_backoff[a]
    return p_main + lam * (1.0 / max(V, 1))


def ppmi_prob(stats: Statistics, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return normalized PPMI-based p(b|a)."""
    V = stats.vocab_size
    device = stats.ppmi_values.device
    a = a.to(device).to(torch.int64)
    b = b.to(device).to(torch.int64)

    if not hasattr(stats, "_ppmi_csr"):
        rows = stats.ppmi_indices[0].to(torch.int64)
        cols = stats.ppmi_indices[1].to(torch.int64)
        vals = stats.ppmi_values
        search_key = rows * V + cols
        order = torch.argsort(search_key)
        search_key = search_key[order]
        vals = vals[order]
        row_sum = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(
            0, (search_key // V), vals
        )
        stats._ppmi_csr = (search_key, vals, row_sum)

    search_key, vals_s, row_sum = stats._ppmi_csr
    nnz = search_key.numel()
    if nnz == 0:
        return torch.zeros_like(a, dtype=torch.float32)
    query = a * V + b
    pos = torch.searchsorted(search_key, query, right=False)
    safe = torch.clamp(pos, max=nnz - 1)
    hit = (pos < nnz) & (search_key[safe] == query)
    v = torch.where(hit, vals_s[safe], torch.zeros_like(vals_s[safe]))
    rs = row_sum[a].clamp(min=1e-9)
    return v / rs
