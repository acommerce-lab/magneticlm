# magnetic/edges.py
#
# Semantic edge construction. Two-stage build:
#
#   1. Base edges from a +/- K window in the token stream (matches the
#      original C# StrengthSemanticEdge loop with weights 1.0 for
#      abs(offset)==1 and 0.5 for abs(offset)==2, times base_amount).
#
#   2. Optional PMI reweighting. Each edge weight is multiplied by a
#      normalised PPMI factor computed from unigram frequencies. This
#      addresses the "common words dominate" failure mode the OOD
#      cloze exposed: pairs that co-occur only because both words are
#      high frequency (e.g. "the X", "and X") get their weight
#      collapsed toward zero, while rare+predictive pairs keep theirs.
#      Disable with config.use_pmi = False to reproduce C# exactly.
#
# Edges are represented as three parallel 1D tensors on the primary
# device: edge_from, edge_to, edge_weight. The set is always made
# symmetric (both directions are kept) because the physics spring
# force assumes symmetric springs.
#
# For use in inference (excitation.py), edges are additionally exposed
# as a sparse adjacency matrix accessor - see EdgeSet.
#
# The window offsets supported are determined by config.edge_window
# and config.edge_weight_schedule. By default +/-1 gets weight 1.0 and
# +/-2 gets weight 0.5. Set edge_window=5 and extend the schedule to
# capture a wider distributional neighbourhood.

from dataclasses import dataclass
import math
from typing import Optional

try:
    import torch
except ImportError:
    raise ImportError("PyTorch required")


@dataclass
class EdgeSet:
    """Container for a directed, weighted edge list. Edges are kept
    symmetric but stored as directed tuples so the forward and
    backward passes through the graph can be done with the same
    code path."""
    edge_from: torch.Tensor   # (E,) int64
    edge_to: torch.Tensor     # (E,) int64
    edge_weight: torch.Tensor # (E,) float32
    num_nodes: int            # V

    def numel(self) -> int:
        return self.edge_from.numel()

    def filter_threshold(self, threshold: float) -> "EdgeSet":
        """Return a new EdgeSet keeping only edges with weight >=
        threshold."""
        mask = self.edge_weight >= threshold
        return EdgeSet(
            edge_from=self.edge_from[mask].contiguous(),
            edge_to=self.edge_to[mask].contiguous(),
            edge_weight=self.edge_weight[mask].contiguous(),
            num_nodes=self.num_nodes,
        )


class EdgeBuilder:
    """Builds a semantic EdgeSet from a token stream."""

    def __init__(self, config):
        self.config = config

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------
    def build(
        self,
        tokens_gpu: torch.Tensor,
        freq_gpu: torch.Tensor,
        vocab_size: int,
        device: torch.device,
    ) -> EdgeSet:
        """Build the full edge set on `device`. tokens_gpu is the
        training token stream; freq_gpu[w] is the unigram count of
        word w and is used by the PMI reweighting step."""
        cfg = self.config
        if tokens_gpu.numel() < 2 or vocab_size == 0:
            empty_i = torch.empty(0, dtype=torch.int64, device=device)
            empty_w = torch.empty(0, dtype=torch.float32, device=device)
            return EdgeSet(
                edge_from=empty_i,
                edge_to=empty_i.clone(),
                edge_weight=empty_w,
                num_nodes=vocab_size,
            )

        edges = self._build_base(tokens_gpu, vocab_size, device)
        if cfg.use_pmi and edges.numel() > 0:
            edges = self._apply_pmi(edges, freq_gpu, tokens_gpu.numel(), device)
        if cfg.use_jaccard and edges.numel() > 0:
            edges = self._apply_jaccard(edges, device)
        return edges

    # -------------------------------------------------------------------
    # Stage 1: base edges from the +/- K window
    # -------------------------------------------------------------------
    def _build_base(
        self,
        tokens_gpu: torch.Tensor,
        vocab_size: int,
        device: torch.device,
    ) -> EdgeSet:
        cfg = self.config
        tokens = tokens_gpu.to(device)
        V_long = torch.tensor(vocab_size, dtype=torch.int64, device=device)
        T = tokens.numel()

        running_keys = torch.empty(0, dtype=torch.int64, device=device)
        running_w = torch.empty(0, dtype=torch.float32, device=device)

        # Distance weights come from the configured schedule with a
        # final fallback of 0.0 beyond the schedule length.
        schedule = list(cfg.edge_weight_schedule)
        max_offset = min(cfg.edge_window, len(schedule))

        for offset in range(1, max_offset + 1):
            if T <= offset:
                continue
            dist_weight = schedule[offset - 1]
            amount = float(dist_weight) * float(cfg.edge_base_amount)
            if amount <= 0.0:
                continue

            a = tokens[:-offset]
            b = tokens[offset:]
            # Drop self-loops: a word adjacent to a copy of itself.
            mask = a != b
            a = a[mask]
            b = b[mask]
            if a.numel() == 0:
                continue

            # Symmetric: emit (a->b) and (b->a).
            both_f = torch.cat([a, b])
            both_t = torch.cat([b, a])
            del a, b, mask

            keys = both_f * V_long + both_t
            del both_f, both_t

            uk, inv = torch.unique(keys, return_inverse=True)
            del keys
            ones = torch.ones(
                inv.numel(), dtype=torch.float32, device=device) * amount
            w = torch.zeros(
                uk.numel(), dtype=torch.float32, device=device)
            w.scatter_add_(0, inv, ones)
            del inv, ones

            merged_keys = torch.cat([running_keys, uk])
            merged_w = torch.cat([running_w, w])
            del uk, w
            uk2, inv2 = torch.unique(merged_keys, return_inverse=True)
            w2 = torch.zeros(
                uk2.numel(), dtype=torch.float32, device=device)
            w2.scatter_add_(0, inv2, merged_w)
            del merged_keys, merged_w, inv2
            running_keys = uk2
            running_w = w2

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if running_keys.numel() == 0:
            empty_i = torch.empty(0, dtype=torch.int64, device=device)
            empty_w = torch.empty(0, dtype=torch.float32, device=device)
            return EdgeSet(
                edge_from=empty_i,
                edge_to=empty_i.clone(),
                edge_weight=empty_w,
                num_nodes=vocab_size,
            )

        edge_from = running_keys // V_long
        edge_to = running_keys % V_long

        # Apply the minimum-weight threshold now to shrink the graph
        # before physics touches it.
        strong = running_w >= cfg.semantic_threshold
        return EdgeSet(
            edge_from=edge_from[strong].contiguous(),
            edge_to=edge_to[strong].contiguous(),
            edge_weight=running_w[strong].clamp(-10.0, 10.0).contiguous(),
            num_nodes=vocab_size,
        )

    # -------------------------------------------------------------------
    # Stage 2: PMI reweighting (extension, not in C# original)
    # -------------------------------------------------------------------
    #
    # For each edge (a, b) with current weight w_ab, compute the PPMI:
    #
    #   PMI(a, b) = log( (joint / T) / ((freq_a / T) * (freq_b / T)) )
    #             = log( joint * T / (freq_a * freq_b) )
    #   PPMI(a, b) = max(PMI, 0)
    #
    # Normalise to [0, 1] by dividing by pmi_cap, then multiply the
    # existing weight. An edge between two common words with no
    # special association collapses to ~0 weight; a rare+predictive
    # edge retains its weight; extremely surprising pairs are capped.
    def _apply_pmi(
        self,
        edges: EdgeSet,
        freq_gpu: torch.Tensor,
        total_tokens: int,
        device: torch.device,
    ) -> EdgeSet:
        cfg = self.config
        if edges.numel() == 0:
            return edges

        freq = freq_gpu.to(device).float().clamp_min(1.0)
        # joint count is approximately (edge_weight / base_amount) so
        # we recover it robustly. For symmetrised, multi-offset edges
        # this is a rough estimate but stays monotonic with actual
        # co-occurrence count.
        base = float(cfg.edge_base_amount)
        joint = edges.edge_weight.clamp_min(1e-6) / max(base, 1e-6)

        fa = freq[edges.edge_from]
        fb = freq[edges.edge_to]
        T = float(max(total_tokens, 1))

        # PMI in natural log; we take PPMI and cap.
        # Add a tiny epsilon to avoid log(0).
        pmi = torch.log(
            (joint * T) / (fa * fb).clamp_min(1e-6) + 1e-12
        )
        ppmi = pmi.clamp_min(cfg.pmi_floor)
        factor = (ppmi / max(cfg.pmi_cap, 1e-6)).clamp_max(1.0)

        new_w = edges.edge_weight * factor
        mask = new_w >= cfg.semantic_threshold
        return EdgeSet(
            edge_from=edges.edge_from[mask].contiguous(),
            edge_to=edges.edge_to[mask].contiguous(),
            edge_weight=new_w[mask].contiguous(),
            num_nodes=edges.num_nodes,
        )

    # -------------------------------------------------------------------
    # Stage 3: Jaccard degree-ratio reweighting
    # -------------------------------------------------------------------
    # The user's "relational differentiation" idea: the strength of the
    # link between two words should depend on how SIMILAR their
    # connectivity patterns are. Two words that connect to the SAME set
    # of neighbours (synonyms) keep their full weight. A hub word
    # (degree 50k) linked to a specialist (degree 5) gets suppressed
    # because their edge sets barely overlap.
    #
    # Full Jaccard = |N(a) ∩ N(b)| / |N(a) ∪ N(b)| is O(E * degree)
    # and hard to vectorise on GPU for large graphs. This uses the
    # degree-ratio approximation:
    #
    #   ratio = min(deg_a, deg_b) / max(deg_a, deg_b)
    #
    # which is O(E), fully vectorised, and captures the primary signal:
    # degree MISMATCH implies low Jaccard (a node with 5 neighbours
    # can share at most 5 with a node that has 50000, giving Jaccard
    # ≤ 5/50000 = 0.0001, while ratio = 5/50000 = 0.0001).
    #
    # For equal-degree nodes the ratio is 1.0 regardless of whether
    # their actual neighbours overlap, which is an over-estimate of
    # Jaccard. A future improvement could use sampled intersection
    # counts to refine this.
    def _apply_jaccard(
        self,
        edges: EdgeSet,
        device: torch.device,
    ) -> EdgeSet:
        if edges.numel() == 0:
            return edges

        V = edges.num_nodes
        E = edges.numel()

        # Out-degree per node (= number of unique neighbours for
        # symmetric edges).
        degree = torch.zeros(V, dtype=torch.float32, device=device)
        degree.scatter_add_(
            0, edges.edge_from,
            torch.ones(E, dtype=torch.float32, device=device))
        degree = degree.clamp_min(1.0)

        da = degree[edges.edge_from]
        db = degree[edges.edge_to]
        ratio = torch.sqrt(torch.minimum(da, db) / torch.maximum(da, db))

        new_w = edges.edge_weight * ratio
        mask = new_w >= self.config.semantic_threshold
        return EdgeSet(
            edge_from=edges.edge_from[mask].contiguous(),
            edge_to=edges.edge_to[mask].contiguous(),
            edge_weight=new_w[mask].contiguous(),
            num_nodes=V,
        )
