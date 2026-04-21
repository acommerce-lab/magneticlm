"""Directional co-occurrence statistics.

Single responsibility: count pairs and convert them to forward/backward
transition probabilities.

Core idea (v4):
  forward_prob(a→b)  = P(b | a) = count(a,b) / count(a)
  backward_prob(a←b) = P(a | b) = count(a,b) / count(b)

These are NATURALLY asymmetric:
  - rare→common: forward high, backward tiny
  - common→rare: forward tiny, backward (relatively) high

Cost = -log p. Used as edge attenuation in wave propagation.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Stats:
    vocab_size: int
    unigram_counts: torch.Tensor  # [V] int64

    # Symmetric context pairs (for PPMI / concept edges)
    ctx_rows: torch.Tensor        # [E] int64
    ctx_cols: torch.Tensor        # [E] int64
    ctx_counts: torch.Tensor      # [E] float32

    # Directional bigram pairs (A -> B, window=1)
    bg_rows: torch.Tensor         # [E'] int64
    bg_cols: torch.Tensor         # [E'] int64
    bg_counts: torch.Tensor       # [E'] float32


def _pair_keys(a: torch.Tensor, b: torch.Tensor, V: int) -> torch.Tensor:
    return a.to(torch.int64) * V + b.to(torch.int64)


def _collect(
    encoded: List[np.ndarray],
    V: int,
    window: int,
    symmetric: bool,
    device: torch.device,
    chunk_tokens: int = 4_000_000,
):
    """Chunked GPU pair collection with on-the-fly reduction.

    Returns (rows, cols, counts) — already deduplicated across chunks.
    Peak GPU memory stays ~O(chunk_tokens * window) regardless of corpus.
    """
    valid = [a for a in encoded if a.size >= 2]
    if not valid:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    # Build chunks by cumulative token count
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

    reduced_keys: List[torch.Tensor] = []
    reduced_counts: List[torch.Tensor] = []

    for chunk in chunks:
        flat = np.concatenate(chunk).astype(np.int64)
        sent_ids = np.concatenate([
            np.full(a.size, i, dtype=np.int32) for i, a in enumerate(chunk)
        ])
        t = torch.from_numpy(flat).to(device)
        sid = torch.from_numpy(sent_ids).to(device)
        n = t.numel()

        buf: List[torch.Tensor] = []
        for d in range(1, window + 1):
            if d >= n:
                break
            same = sid[:-d] == sid[d:]
            a = t[:-d][same]
            b = t[d:][same]
            if a.numel() == 0:
                continue
            buf.append(_pair_keys(a, b, V))
            if symmetric:
                buf.append(_pair_keys(b, a, V))
        del t, sid
        if not buf:
            continue

        chunk_ids = torch.cat(buf)
        del buf
        sorted_ids, _ = torch.sort(chunk_ids)
        del chunk_ids
        uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        del sorted_ids
        reduced_keys.append(uniq.cpu())
        reduced_counts.append(counts.to(torch.int64).cpu())
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if not reduced_keys:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    all_keys = torch.cat(reduced_keys).to(device)
    all_counts = torch.cat(reduced_counts).to(device).to(torch.int64)
    sort_idx = torch.argsort(all_keys)
    all_keys = all_keys[sort_idx]
    all_counts = all_counts[sort_idx]

    uniq, inverse = torch.unique_consecutive(all_keys, return_inverse=True)
    final = torch.zeros(uniq.numel(), dtype=torch.int64, device=device)
    final.scatter_add_(0, inverse, all_counts)

    rows = (uniq // V).to(torch.int64)
    cols = (uniq % V).to(torch.int64)
    return rows, cols, final.to(torch.float32)


def build_stats(encoded: List[np.ndarray], V: int, cfg, device: torch.device) -> Stats:
    """Collect unigram, symmetric-context, and directional-bigram counts."""
    # Unigram
    uni = torch.zeros(V, dtype=torch.int64, device=device)
    for arr in encoded:
        if arr.size == 0:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uni.scatter_add_(0, t, torch.ones_like(t))

    # Context pairs (symmetric window — for conceptual edges)
    c_rows, c_cols, c_counts = _collect(
        encoded, V, window=cfg.stat_window, symmetric=True, device=device,
    )
    # Filter rare
    if cfg.min_pair_count > 1:
        keep = c_counts >= cfg.min_pair_count
        c_rows, c_cols, c_counts = c_rows[keep], c_cols[keep], c_counts[keep]

    # Bigram pairs (directional A→B, window=1)
    b_rows, b_cols, b_counts = _collect(
        encoded, V, window=cfg.bigram_window, symmetric=False, device=device,
    )

    return Stats(
        vocab_size=V,
        unigram_counts=uni,
        ctx_rows=c_rows, ctx_cols=c_cols, ctx_counts=c_counts,
        bg_rows=b_rows, bg_cols=b_cols, bg_counts=b_counts,
    )


# ==========================================================================
# Directional cost computation
# ==========================================================================

def directional_costs(
    rows: torch.Tensor,
    cols: torch.Tensor,
    counts: torch.Tensor,
    unigram: torch.Tensor,
    smoothing: float = 1.0,
    ceiling: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (forward_cost, backward_cost) for each (row, col, count) triple.

    forward_cost[a,b]  = -log P(b | a)  = -log[(count(a,b)+s) / (count(a) + s·V)]
    backward_cost[a,b] = -log P(a | b)  = -log[(count(a,b)+s) / (count(b) + s·V)]

    Asymmetric by construction — the engine that uses these treats:
      - forward edge (a→b) as a statistical/syntactic bridge
      - backward edge (a←b) as a conceptual reverse-link
    """
    V = int(unigram.numel())
    s = float(smoothing)

    num = counts + s
    denom_a = unigram[rows].to(torch.float32) + s * V
    denom_b = unigram[cols].to(torch.float32) + s * V

    p_fwd = num / denom_a.clamp(min=1e-9)
    p_bwd = num / denom_b.clamp(min=1e-9)

    cost_fwd = (-torch.log(p_fwd.clamp(min=1e-20))).clamp(max=ceiling)
    cost_bwd = (-torch.log(p_bwd.clamp(min=1e-20))).clamp(max=ceiling)
    return cost_fwd, cost_bwd
