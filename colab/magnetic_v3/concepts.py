"""Concept layer: community detection, many-to-many membership, glow centers.

Concepts are dense subgraphs discovered in the PPMI network.
Each word can belong to multiple concepts (many-to-many).
Glow centers = concepts whose accumulated activation exceeds threshold
during inference; they inject activation into all member words, creating
distant influence across the graph.

Word -> Concept -> Word is a relay path that lets signals bypass direct
edge distance. A glow center activated 30 tokens ago still influences
the current scoring step through its persistent activation.
"""

from dataclasses import dataclass
from typing import Optional

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
        """[V] word activations -> [C] concept activations (membership-weighted mean)."""
        if self.n_concepts == 0:
            return torch.empty(0, dtype=torch.float32, device=word_act.device)
        w_act = word_act[self.mem_word]
        weighted = w_act * self.mem_strength
        out = torch.zeros(self.n_concepts, dtype=torch.float32, device=word_act.device)
        out.scatter_add_(0, self.mem_concept, weighted)
        sizes = self.concept_sizes.to(torch.float32).clamp(min=1.0).to(word_act.device)
        return out / sizes

    def concept_to_word_injection(self, concept_act: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """[C] concept activations -> [V] word-level injection via membership."""
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


def discover_concepts(
    ppmi_rows: torch.Tensor,
    ppmi_cols: torch.Tensor,
    ppmi_vals: torch.Tensor,
    vocab_size: int,
    cfg,
    device: torch.device,
) -> ConceptLayer:
    V = vocab_size

    mask = ppmi_vals >= cfg.concept_ppmi_threshold
    rows = ppmi_rows[mask].to(torch.int64).to(device)
    cols = ppmi_cols[mask].to(torch.int64).to(device)
    vals = ppmi_vals[mask].to(torch.float32).to(device)

    if rows.numel() == 0:
        return _empty(V, device)

    all_rows = torch.cat([rows, cols])
    all_cols = torch.cat([cols, rows])
    all_vals = torch.cat([vals, vals])

    labels = _connected_components(all_rows, all_cols, V, device)
    concept_ids, word_concept = _compact_labels(
        labels, V, cfg.concept_min_size, cfg.concept_max_size, device
    )
    n_concepts = concept_ids.numel()
    if n_concepts == 0:
        return _empty(V, device)

    mem_w, mem_c, mem_s = _build_membership(
        all_rows, all_cols, all_vals, word_concept, n_concepts, V, device
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


def _connected_components(rows, cols, V, device, max_iter=100):
    labels = torch.arange(V, dtype=torch.int64, device=device)
    for _ in range(max_iter):
        prev = labels.clone()
        min_label = torch.minimum(labels[rows], labels[cols])
        labels.scatter_reduce_(0, rows, min_label, reduce="amin", include_self=True)
        labels.scatter_reduce_(0, cols, min_label, reduce="amin", include_self=True)
        # pointer jumping
        labels = labels[labels]
        if (labels == prev).all():
            break
    return labels


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
