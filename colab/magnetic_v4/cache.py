"""Two cache layers: statistical (decay) + conceptual (PPMI trigger).

Statistical cache (DecayCache):
  Tracks recently observed tokens. Boosts their probability with
  logarithmic decay: w = 1/log(2 + age). This is the mechanism that
  took PPL from 84 to 14 in v1/v2.

Conceptual cache (ConceptCache):
  For each context word, looks up its top-K PPMI neighbors and boosts
  them. This is the "glow center" mechanism as a direct lookup table —
  no propagation, no iteration, just a pre-built dictionary.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ======================================================================
# Statistical cache: decay-weighted recently-seen tokens
# ======================================================================

class DecayCache:
    """Window of recently seen tokens with logarithmic decay.

    Usage:
      cache = DecayCache(window=3000)
      cache.observe(token_id)        # call for each token in context
      boost = cache.get_boost(V)     # returns [V] tensor of boosts
    """

    def __init__(self, window: int = 3000):
        self.window = window
        self.history: List[int] = []

    def observe(self, token_id: int):
        self.history.append(token_id)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def observe_sequence(self, token_ids: List[int]):
        for t in token_ids:
            self.observe(t)

    def get_boost(self, V: int, device: torch.device) -> torch.Tensor:
        """Return [V] cache probability distribution."""
        boost = torch.zeros(V, dtype=torch.float32, device=device)
        if len(self.history) < 2:
            return boost

        total_w = 0.0
        n = len(self.history)
        for i, tok in enumerate(self.history):
            age = n - 1 - i
            w = 1.0 / math.log(2.0 + age)
            boost[tok] += w
            total_w += w

        if total_w > 0:
            boost /= total_w
        return boost

    def get_boost_batch(
        self, contexts: List[List[int]], V: int, device: torch.device,
    ) -> torch.Tensor:
        """Return [B, V] cache boosts for a batch of contexts.

        Each context is treated as its own mini-history (no cross-contamination).
        """
        B = len(contexts)
        boosts = torch.zeros(B, V, dtype=torch.float32, device=device)
        for i, ctx in enumerate(contexts):
            if len(ctx) < 2:
                continue
            n = len(ctx)
            total_w = 0.0
            for j, tok in enumerate(ctx):
                age = n - 1 - j
                w = 1.0 / math.log(2.0 + age)
                boosts[i, tok] += w
                total_w += w
            if total_w > 0:
                boosts[i] /= total_w
        return boosts

    def reset(self):
        self.history = []


# ======================================================================
# Conceptual cache: PPMI trigger lookup (= glow centers as a table)
# ======================================================================

@dataclass
class ConceptCache:
    """Pre-built table: for each word → top-K PPMI semantic neighbors.

    At inference, for each context word, we look up its neighbors and
    aggregate their boosts. This IS the glow center mechanism — but as
    a direct lookup instead of iterative propagation.

    Attributes:
      neighbor_ids:     [N_words, K] int tensor — neighbor word ids
      neighbor_weights: [N_words, K] float tensor — PPMI weights (normalized per word)
      has_neighbors:    [V] bool — which words have concept neighbors
    """
    neighbor_ids: torch.Tensor      # [V, K]
    neighbor_weights: torch.Tensor  # [V, K]
    K: int
    V: int


def build_concept_cache(
    ctx_rows: torch.Tensor,
    ctx_cols: torch.Tensor,
    ctx_counts: torch.Tensor,
    unigram: torch.Tensor,
    V: int,
    K: int,
    device: torch.device,
    min_ppmi: float = 0.5,
) -> ConceptCache:
    """Build the concept cache from co-occurrence statistics.

    Computes PPMI for all pairs, then for each word keeps top-K neighbors
    with highest PPMI as its "concept neighborhood."
    """
    # Compute PPMI
    total = ctx_counts.sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    N = unigram.to(torch.float32).sum().clamp(min=1.0)
    p_a = unigram[ctx_rows].to(torch.float32) / N
    p_b = unigram[ctx_cols].to(torch.float32) / N
    pmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    ppmi = pmi.clamp(min=0.0)

    # Filter low PPMI
    keep = ppmi >= min_ppmi
    rows_f = ctx_rows[keep]
    cols_f = ctx_cols[keep]
    ppmi_f = ppmi[keep]

    # For each word (row), collect top-K neighbors by PPMI
    # Use scatter to build per-word neighbor lists
    nbr_ids = torch.zeros(V, K, dtype=torch.int64, device=device)
    nbr_weights = torch.zeros(V, K, dtype=torch.float32, device=device)

    if rows_f.numel() == 0:
        print(f"  [concept_cache] no PPMI edges above {min_ppmi}, empty cache")
        return ConceptCache(nbr_ids, nbr_weights, K, V)

    # Sort by (row, -ppmi) to pick top-K per row
    sort_key = rows_f.to(torch.int64) * 1000000 + (1000000 - (ppmi_f * 1000).to(torch.int64))
    order = torch.argsort(sort_key)
    rows_s = rows_f[order]
    cols_s = cols_f[order]
    ppmi_s = ppmi_f[order]

    # Position within each row
    boundaries = torch.ones_like(rows_s, dtype=torch.bool)
    boundaries[1:] = rows_s[1:] != rows_s[:-1]
    group_id = torch.cumsum(boundaries.to(torch.int64), dim=0) - 1
    global_idx = torch.arange(rows_s.numel(), device=device, dtype=torch.int64)
    group_first = torch.full_like(group_id, fill_value=rows_s.numel())
    group_first.scatter_reduce_(0, group_id, global_idx, reduce="amin", include_self=False)
    pos_in_group = global_idx - group_first[group_id]

    # Keep only first K per group
    keep_k = pos_in_group < K
    r_k = rows_s[keep_k]
    c_k = cols_s[keep_k]
    p_k = ppmi_s[keep_k]
    pos_k = pos_in_group[keep_k]

    # Fill the tables
    nbr_ids[r_k, pos_k] = c_k
    nbr_weights[r_k, pos_k] = p_k

    # Normalize weights per word
    w_sum = nbr_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    nbr_weights = nbr_weights / w_sum

    n_with = int((nbr_weights.sum(dim=1) > 0).sum().item())
    print(f"  [concept_cache] K={K}  words_with_neighbors={n_with}/{V}  "
          f"edges_kept={int(keep_k.sum().item()):,}")

    return ConceptCache(nbr_ids, nbr_weights, K, V)


def concept_boost(
    cache: ConceptCache,
    context_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Given context word ids, return [V] concept boost distribution.

    For each context word, look up its K neighbors and aggregate.
    """
    V = cache.V
    boost = torch.zeros(V, dtype=torch.float32, device=device)

    for w in context_ids:
        if w < 0 or w >= V:
            continue
        # Get this word's concept neighborhood
        ids = cache.neighbor_ids[w]    # [K]
        wts = cache.neighbor_weights[w]  # [K]
        boost.scatter_add_(0, ids, wts)

    s = boost.sum()
    if s > 1e-9:
        boost = boost / s
    return boost


def concept_boost_batch(
    cache: ConceptCache,
    contexts: List[List[int]],
    device: torch.device,
) -> torch.Tensor:
    """Batched concept boost. Returns [B, V]."""
    B = len(contexts)
    V = cache.V
    boosts = torch.zeros(B, V, dtype=torch.float32, device=device)

    for i, ctx in enumerate(contexts):
        for w in ctx:
            if w < 0 or w >= V:
                continue
            ids = cache.neighbor_ids[w]
            wts = cache.neighbor_weights[w]
            boosts[i].scatter_add_(0, ids, wts)

    s = boosts.sum(dim=1, keepdim=True).clamp(min=1e-9)
    boosts = boosts / s
    return boosts
