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
    """Build concept cache from DISTRIBUTIONAL SIMILARITY, not raw PPMI.

    Two words are substitutes if they appear in the SAME contexts —
    measured by overlap count of their PPMI context sets.

    Algorithm (GPU-native, blocked):
      1. Build binary PPMI matrix B[V, V] (sparse)
      2. Process in row-blocks: B[block] @ B.T → [block_size, V]
      3. For each row, top-K by overlap = distributional substitutes
    """
    # Compute PPMI
    total = ctx_counts.sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    N = unigram.to(torch.float32).sum().clamp(min=1.0)
    p_a = unigram[ctx_rows].to(torch.float32) / N
    p_b = unigram[ctx_cols].to(torch.float32) / N
    pmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    ppmi = pmi.clamp(min=0.0)

    keep = ppmi >= min_ppmi
    rows_f = ctx_rows[keep]
    cols_f = ctx_cols[keep]
    ppmi_f = ppmi[keep]

    if rows_f.numel() == 0:
        print(f"  [concept_cache] no edges, empty cache")
        nbr_ids = torch.zeros(V, K, dtype=torch.int64, device=device)
        nbr_weights = torch.zeros(V, K, dtype=torch.float32, device=device)
        return ConceptCache(nbr_ids, nbr_weights, K, V)

    # Build sparse binary PPMI matrix B[V, V] (1 where PPMI > threshold)
    indices = torch.stack([rows_f, cols_f], dim=0)
    ones = torch.ones(rows_f.numel(), dtype=torch.float32, device=device)
    B = torch.sparse_coo_tensor(indices, ones, (V, V)).coalesce()

    # Compute distributional similarity in blocks: overlap = B @ B.T
    nbr_ids = torch.zeros(V, K, dtype=torch.int64, device=device)
    nbr_weights = torch.zeros(V, K, dtype=torch.float32, device=device)

    block_size = min(1000, V)
    n_found = 0

    for start in range(0, V, block_size):
        end = min(start + block_size, V)
        bs = end - start
        # Extract block rows as dense [bs, V]
        block_dense = torch.zeros(bs, V, dtype=torch.float32, device=device)
        # Fill from sparse
        b_idx = B.indices()
        b_val = B.values()
        mask = (b_idx[0] >= start) & (b_idx[0] < end)
        if mask.any():
            local_rows = b_idx[0][mask] - start
            local_cols = b_idx[1][mask]
            block_dense[local_rows, local_cols] = b_val[mask]

        if block_dense.sum() == 0:
            continue

        # Overlap: block_dense @ B.T → [bs, V]
        # B.T @ block_dense.T → [V, bs], then transpose
        sim = torch.sparse.mm(B.t(), block_dense.t()).t()  # [bs, V]

        # Zero out self-similarity
        for i in range(bs):
            w = start + i
            if w < V:
                sim[i, w] = 0.0

        # Top-K per row
        topk_vals, topk_idx = torch.topk(sim, min(K, V), dim=1)

        for i in range(bs):
            w = start + i
            if w >= V:
                break
            valid = topk_vals[i] > 0
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            n_take = min(n_valid, K)
            nbr_ids[w, :n_take] = topk_idx[i, :n_take]
            nbr_weights[w, :n_take] = topk_vals[i, :n_take]
            n_found += 1

    # Normalize weights per word
    w_sum = nbr_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    nbr_weights = nbr_weights / w_sum

    print(f"  [concept_cache] distributional similarity K={K}  "
          f"words_with_subs={n_found}/{V}")

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
