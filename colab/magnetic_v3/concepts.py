"""Concept layer: community detection, many-to-many membership, glow centers.

Concepts are dense subgraphs discovered in the PPMI network.
Each word can belong to multiple concepts (many-to-many).
Glow centers = concepts whose accumulated activation exceeds threshold
during inference; they inject activation into all member words, creating
distant influence across the graph.

Two discovery methods (switchable via cfg.concept_method):
  * "split"     : connected components, then iteratively prune the weakest
                  intra-cluster edges of any oversize component until all
                  clusters fit [min_size, max_size]. Simple and stable.
  * "labelprop" : weighted label propagation. Each node adopts the label
                  of its strongest-voting neighborhood (sum of PPMI).
                  Naturally produces many small clusters on dense graphs.
  * "components": raw connected components (legacy; tends to collapse into
                  one giant cluster on large vocabularies).

The PPMI cutoff used to filter edges before discovery can be either a
fixed threshold or a percentile (default: keep top 20% of PPMI values,
which adapts to vocabulary scale where a fixed 2.0 would leave no edges).
"""

from dataclasses import dataclass

import torch


@dataclass
class ConceptLayer:
    n_concepts: int
    vocab_size: int
    mem_word: torch.Tensor        # [nnz] int64 — word ids
    mem_concept: torch.Tensor     # [nnz] int64 — concept ids
    mem_strength: torch.Tensor    # [nnz] float32 — membership weight
    concept_sizes: torch.Tensor   # [C] int32
    primary_concept: torch.Tensor # [V] int64 (-1 if no concept)

    def word_to_concept_activation(self, word_act: torch.Tensor) -> torch.Tensor:
        if self.n_concepts == 0:
            return torch.empty(0, dtype=torch.float32, device=word_act.device)
        w_act = word_act[self.mem_word]
        weighted = w_act * self.mem_strength
        out = torch.zeros(self.n_concepts, dtype=torch.float32, device=word_act.device)
        out.scatter_add_(0, self.mem_concept, weighted)
        sizes = self.concept_sizes.to(torch.float32).clamp(min=1.0).to(word_act.device)
        return out / sizes

    def concept_to_word_injection(self, concept_act: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        out = torch.zeros(self.vocab_size, dtype=torch.float32, device=concept_act.device)
        if self.n_concepts == 0:
            return out
        c_act = concept_act[self.mem_concept]
        weighted = c_act * self.mem_strength * strength
        out.scatter_add_(0, self.mem_word, weighted)
        return out

    def to(self, device: torch.device) -> "ConceptLayer":
        return ConceptLayer(
            n_concepts=self.n_concepts,
            vocab_size=self.vocab_size,
            mem_word=self.mem_word.to(device),
            mem_concept=self.mem_concept.to(device),
            mem_strength=self.mem_strength.to(device),
            concept_sizes=self.concept_sizes.to(device),
            primary_concept=self.primary_concept.to(device),
        )


# ---------------------------------------------------------------------------
# Discovery entry point
# ---------------------------------------------------------------------------


def _ppmi_cutoff(vals: torch.Tensor, cfg) -> float:
    """Adaptive or fixed threshold for concept-edge PPMI."""
    mode = getattr(cfg, "concept_ppmi_mode", "percentile")
    if mode == "percentile" and vals.numel() > 0:
        pct = float(getattr(cfg, "concept_ppmi_percentile", 0.8))
        return float(torch.quantile(vals, pct).item())
    return float(cfg.concept_ppmi_threshold)


def discover_concepts(
    ppmi_rows: torch.Tensor,
    ppmi_cols: torch.Tensor,
    ppmi_vals: torch.Tensor,
    vocab_size: int,
    cfg,
    device: torch.device,
) -> ConceptLayer:
    V = vocab_size

    vals_all = ppmi_vals.to(torch.float32).to(device)
    cutoff = _ppmi_cutoff(vals_all, cfg)
    mask = vals_all >= cutoff
    rows = ppmi_rows[mask].to(torch.int64).to(device)
    cols = ppmi_cols[mask].to(torch.int64).to(device)
    vals = vals_all[mask]

    if rows.numel() == 0:
        return _empty(V, device)

    # Symmetric edge list
    all_rows = torch.cat([rows, cols])
    all_cols = torch.cat([cols, rows])
    all_vals = torch.cat([vals, vals])

    method = getattr(cfg, "concept_method", "split")
    if method == "labelprop":
        labels = _weighted_label_propagation(all_rows, all_cols, all_vals, V, device)
        keep_rows, keep_cols, keep_vals = all_rows, all_cols, all_vals
    elif method == "components":
        labels = _connected_components(all_rows, all_cols, V, device)
        keep_rows, keep_cols, keep_vals = all_rows, all_cols, all_vals
    else:  # "split" (default)
        keep_rows, keep_cols, keep_vals, labels = _split_oversized(
            all_rows, all_cols, all_vals, V,
            cfg.concept_min_size, cfg.concept_max_size,
            max_iter=int(getattr(cfg, "concept_split_max_iter", 12)),
            device=device,
        )

    concept_ids, word_concept = _compact_labels(
        labels, V, cfg.concept_min_size, cfg.concept_max_size, device
    )
    n_concepts = concept_ids.numel()
    if n_concepts == 0:
        return _empty(V, device)

    mem_w, mem_c, mem_s = _build_membership(
        keep_rows, keep_cols, keep_vals, word_concept, n_concepts, V, device
    )

    sizes = torch.zeros(n_concepts, dtype=torch.int32, device=device)
    valid = word_concept >= 0
    if valid.any():
        sizes.scatter_add_(
            0,
            word_concept[valid],
            torch.ones(valid.sum(), dtype=torch.int32, device=device),
        )

    return ConceptLayer(
        n_concepts=n_concepts,
        vocab_size=V,
        mem_word=mem_w,
        mem_concept=mem_c,
        mem_strength=mem_s,
        concept_sizes=sizes,
        primary_concept=word_concept,
    )


# ---------------------------------------------------------------------------
# Community detection: connected components (base)
# ---------------------------------------------------------------------------


def _connected_components(rows, cols, V, device, max_iter=100):
    labels = torch.arange(V, dtype=torch.int64, device=device)
    if rows.numel() == 0:
        return labels
    for _ in range(max_iter):
        prev = labels.clone()
        min_label = torch.minimum(labels[rows], labels[cols])
        labels.scatter_reduce_(0, rows, min_label, reduce="amin", include_self=True)
        labels.scatter_reduce_(0, cols, min_label, reduce="amin", include_self=True)
        labels = labels[labels]  # pointer jumping
        if (labels == prev).all():
            break
    return labels


# ---------------------------------------------------------------------------
# Community detection: iterative splitting (default "split")
# ---------------------------------------------------------------------------


def _split_oversized(rows, cols, vals, V, min_size, max_size, max_iter, device):
    """Components, then prune weakest intra-cluster edges of oversize clusters.

    At each iteration:
      1. Run connected components on the current edge list.
      2. Identify labels whose cluster size exceeds max_size.
      3. Within those clusters, drop edges whose PPMI is below the intra-cluster
         median. This snips the weakest links holding the giant cluster together.
      4. Repeat until no oversize cluster remains, or the iteration cap hits.
    Edges outside oversize clusters are preserved across iterations.
    """
    labels = _connected_components(rows, cols, V, device)
    for _ in range(max_iter):
        uniq, counts = torch.unique(labels, return_counts=True)
        if counts.numel() == 0:
            break
        oversize_labels = uniq[counts > max_size]
        if oversize_labels.numel() == 0:
            break

        node_in_over = torch.isin(labels, oversize_labels)
        edge_in_over = (
            node_in_over[rows]
            & node_in_over[cols]
            & (labels[rows] == labels[cols])
        )
        if not edge_in_over.any():
            break

        over_vals = vals[edge_in_over]
        threshold = torch.quantile(over_vals, 0.5)
        # Keep: either not in an oversize cluster, or strictly above median PPMI.
        keep_mask = (~edge_in_over) | (vals > threshold)
        if keep_mask.all():
            # Nothing to trim (all edges tied at threshold) — bail.
            break
        rows = rows[keep_mask]
        cols = cols[keep_mask]
        vals = vals[keep_mask]
        labels = _connected_components(rows, cols, V, device)

    # Any isolated node (no edges after pruning) keeps its singleton label and
    # will be filtered out by _compact_labels via min_size.
    return rows, cols, vals, labels


# ---------------------------------------------------------------------------
# Community detection: weighted label propagation
# ---------------------------------------------------------------------------


def _weighted_label_propagation(rows, cols, vals, V, device, max_iter=20):
    """For each node, adopt the neighbor-label with highest summed PPMI.

    Ties broken by smaller label id (gives stable, deterministic clusters).
    Fully vectorized: sort (row, neighbor_label) composite keys and reduce.
    """
    labels = torch.arange(V, dtype=torch.int64, device=device)
    if rows.numel() == 0:
        return labels
    for _ in range(max_iter):
        prev = labels.clone()
        nbr_lbl = labels[cols]
        keys = rows * V + nbr_lbl
        sort_idx = torch.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_vals = vals[sort_idx]
        uniq_keys, inv = torch.unique_consecutive(sorted_keys, return_inverse=True)
        sum_per_key = torch.zeros(uniq_keys.numel(), dtype=torch.float32, device=device)
        sum_per_key.scatter_add_(0, inv, sorted_vals)

        u_rows = (uniq_keys // V).to(torch.int64)
        u_lbls = (uniq_keys % V).to(torch.int64)

        # Best vote sum per row.
        best_sum = torch.full((V,), float("-inf"), dtype=torch.float32, device=device)
        best_sum.scatter_reduce_(0, u_rows, sum_per_key, reduce="amax", include_self=True)

        # Candidates matching the best vote; tie-break by smallest label.
        is_best = sum_per_key >= (best_sum[u_rows] - 1e-6)
        cand_rows = u_rows[is_best]
        cand_lbls = u_lbls[is_best]

        new_labels = labels.clone()
        if cand_rows.numel() > 0:
            big = torch.full((V,), V + 1, dtype=torch.int64, device=device)
            big.scatter_reduce_(0, cand_rows, cand_lbls, reduce="amin", include_self=True)
            touched = big <= V
            new_labels = torch.where(touched, big, new_labels)
        labels = new_labels
        if (labels == prev).all():
            break
    return labels


# ---------------------------------------------------------------------------
# Label compaction + membership
# ---------------------------------------------------------------------------


def _compact_labels(labels, V, min_size, max_size, device):
    uniq, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    valid = (counts >= min_size) & (counts <= max_size)
    n = valid.sum().item()
    if n == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, torch.full((V,), -1, dtype=torch.int64, device=device)

    remap = torch.full((uniq.numel(),), -1, dtype=torch.int64, device=device)
    valid_idx = torch.nonzero(valid).squeeze(-1)
    remap[valid_idx] = torch.arange(n, dtype=torch.int64, device=device)
    word_concept = remap[inverse]

    concept_ids = torch.arange(n, dtype=torch.int64, device=device)
    return concept_ids, word_concept


def _build_membership(all_rows, all_cols, all_vals, word_concept, n_concepts, V, device):
    has_concept = word_concept >= 0
    primary_words = torch.nonzero(has_concept).squeeze(-1)
    primary_concepts = word_concept[primary_words]
    primary_strengths = torch.ones(primary_words.numel(), dtype=torch.float32, device=device)

    ca = word_concept[all_rows]
    cb = word_concept[all_cols]
    cross = (ca >= 0) & (cb >= 0) & (ca != cb)

    if cross.any():
        sec_words = all_rows[cross]
        sec_concepts = cb[cross]
        sec_strengths = torch.clamp(all_vals[cross] / 10.0, max=0.5)
        mem_w = torch.cat([primary_words, sec_words])
        mem_c = torch.cat([primary_concepts, sec_concepts])
        mem_s = torch.cat([primary_strengths, sec_strengths])
    else:
        mem_w = primary_words
        mem_c = primary_concepts
        mem_s = primary_strengths

    if mem_w.numel() == 0:
        e = torch.empty(0, dtype=torch.int64, device=device)
        return e, e, torch.empty(0, dtype=torch.float32, device=device)

    keys = mem_w * n_concepts + mem_c
    uniq_keys, inv = torch.unique(keys, return_inverse=True)
    result_s = torch.zeros(uniq_keys.numel(), dtype=torch.float32, device=device)
    result_s.scatter_reduce_(0, inv, mem_s, reduce="amax", include_self=False)

    result_w = (uniq_keys // n_concepts).to(torch.int64)
    result_c = (uniq_keys % n_concepts).to(torch.int64)
    return result_w, result_c, result_s


def _empty(V, device):
    e = torch.empty(0, dtype=torch.int64, device=device)
    return ConceptLayer(
        n_concepts=0,
        vocab_size=V,
        mem_word=e,
        mem_concept=e,
        mem_strength=torch.empty(0, dtype=torch.float32, device=device),
        concept_sizes=torch.empty(0, dtype=torch.int32, device=device),
        primary_concept=torch.full((V,), -1, dtype=torch.int64, device=device),
    )
