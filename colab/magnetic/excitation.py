# magnetic/excitation.py
#
# Spreading-activation inference engine. This is the piece that was
# thrown away in the GPU rewrite and that the OOD cloze experiments
# showed we can't do without. Faithful to the C# Generator.cs logic
# (ResetExcitations / prompt excitation loop / decay) with one
# extension: configurable multi-hop diffusion.
#
# Per-node state:
#   excitation[w]  float  0..1, decays multiplicatively per step
#   repulsion[w]   float  0..1, decays faster, makes recently used
#                         words less likely to be picked again
#
# Per-step inference:
#   semantic_force(w) = sum over all excited nodes e of
#                       edge_weight(e, w) * excitation[e]
#
# This is exactly the vectorised equivalent of the inner loop in
# C#:
#
#     foreach (var excitedNode in Nodes where Excitation > 0.05)
#         semantic += GetSemanticWeight(excitedNode.Word, candidate)
#                   * excitedNode.Excitation;
#
# computed in one sparse-matmul with scatter_add, so every candidate
# in the vocabulary gets its semantic score in a single GPU kernel
# rather than V * |excited| Python iterations.

import torch
from typing import Iterable


class ExcitationEngine:
    """Holds the excitation / repulsion state and knows how to spread
    activation through the semantic graph."""

    def __init__(
        self,
        config,
        edge_from: torch.Tensor,
        edge_to: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        freq_gpu: torch.Tensor = None,
    ):
        self.config = config
        self.device = device
        self.N = int(num_nodes)

        self.edge_from = edge_from.to(device)
        self.edge_to = edge_to.to(device)
        self.edge_weight = edge_weight.to(device)

        self.max_edge_weight = (
            float(self.edge_weight.abs().max().item())
            if self.edge_weight.numel() > 0 else 1.0)
        if self.max_edge_weight < 1e-9:
            self.max_edge_weight = 1.0

        # Statistical edge relevance: for each edge, compute how
        # "exceptional" it is relative to what's normal for its source
        # and target nodes. Uses Z-score of the edge weight against the
        # per-node mean/std of outgoing edges, then sigmoid to [0, 1].
        #
        # This replaces the crude degree-ratio with the user's proposal:
        # use actual statistical distribution (mean, std, relative to
        # global stats) so that:
        #   - An edge that's much stronger than typical for BOTH endpoints
        #     gets a high semantic relevance (~1.0)
        #   - An average edge gets ~0.5
        #   - A weak/noisy edge gets near 0
        #
        # Common words' edges are naturally suppressed because their
        # per-node mean is high (many similar-weight edges), so no
        # single edge scores as "exceptional".
        E = self.edge_from.numel()
        if E > 0:
            ew = self.edge_weight.float()

            # Per-node outgoing: mean and std of edge weights.
            degree = torch.zeros(self.N, dtype=torch.float32, device=device)
            degree.scatter_add_(0, self.edge_from, torch.ones(E, dtype=torch.float32, device=device))
            degree = degree.clamp_min(1.0)

            weight_sum = torch.zeros(self.N, dtype=torch.float32, device=device)
            weight_sum.scatter_add_(0, self.edge_from, ew)
            node_mean = weight_sum / degree

            weight_sq_sum = torch.zeros(self.N, dtype=torch.float32, device=device)
            weight_sq_sum.scatter_add_(0, self.edge_from, ew * ew)
            node_var = (weight_sq_sum / degree) - (node_mean * node_mean)
            node_std = node_var.clamp_min(0.0).sqrt().clamp_min(1e-6)

            # Z-score for each edge relative to its source node.
            z_from = (ew - node_mean[self.edge_from]) / node_std[self.edge_from]

            # Z-score relative to target node (same stats but looked up
            # via edge_to — gives the "how special am I as an incoming
            # edge for the target" perspective).
            weight_sum_in = torch.zeros(self.N, dtype=torch.float32, device=device)
            weight_sum_in.scatter_add_(0, self.edge_to, ew)
            degree_in = torch.zeros(self.N, dtype=torch.float32, device=device)
            degree_in.scatter_add_(0, self.edge_to, torch.ones(E, dtype=torch.float32, device=device))
            degree_in = degree_in.clamp_min(1.0)
            node_mean_in = weight_sum_in / degree_in

            weight_sq_in = torch.zeros(self.N, dtype=torch.float32, device=device)
            weight_sq_in.scatter_add_(0, self.edge_to, ew * ew)
            node_var_in = (weight_sq_in / degree_in) - (node_mean_in * node_mean_in)
            node_std_in = node_var_in.clamp_min(0.0).sqrt().clamp_min(1e-6)

            z_to = (ew - node_mean_in[self.edge_to]) / node_std_in[self.edge_to]

            # Combined: edge is "special" if above average for BOTH sides.
            z_combined = (z_from + z_to) * 0.5

            # Sigmoid → [0, 1]. Average edges → 0.5, exceptional → ~1.
            self.edge_relevance = torch.sigmoid(z_combined)
        else:
            self.edge_relevance = torch.ones(0, dtype=torch.float32, device=device)

        # IDF weights: log(V / (freq + 1)), normalised to [0, 1].
        # Common words ("the", "and") get low IDF → low excitation.
        # Rare content words ("england", "king") get high IDF → high
        # excitation. This prevents specialist words that only connect
        # to common words from dominating via degree-normalised average.
        if freq_gpu is not None:
            freq = freq_gpu.to(device).float().clamp_min(1.0)
            raw_idf = torch.log(float(num_nodes) / freq)
            self.idf = (raw_idf / raw_idf.max().clamp_min(1e-9)).clamp(0.01, 1.0)
        else:
            self.idf = torch.ones(self.N, dtype=torch.float32, device=device)

        self.excitation = torch.zeros(self.N, dtype=torch.float32, device=device)
        self.repulsion = torch.zeros(self.N, dtype=torch.float32, device=device)

    # -------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------
    def reset(self):
        self.excitation.zero_()
        self.repulsion.zero_()

    # -------------------------------------------------------------------
    # Prompt ingestion: initial excitation + multi-hop spreading
    # -------------------------------------------------------------------
    def excite_prompt(self, prompt_ids: Iterable[int]):
        """Apply initial excitation to prompt words, then diffuse
        through the graph for config.spreading_hops hops."""
        cfg = self.config
        self.reset()

        ids = [int(i) for i in prompt_ids if int(i) >= 0]
        if not ids:
            return

        P = len(ids)
        # Linear ramp from ramp_low at the first token to ramp_high at
        # the last. Matches the C# 0.5 + 0.5 * (i / P).
        low = float(cfg.prompt_ramp_low)
        high = float(cfg.prompt_ramp_high)
        for i, tid in enumerate(ids):
            if P == 1:
                v = high
            else:
                v = low + (high - low) * (i / (P - 1))
            # Scale by IDF: "the" (IDF≈0.02) gets v≈0.01, while
            # "england" (IDF≈0.8) gets v≈0.4. This prevents common
            # words in the prompt from exciting specialist phrases
            # that always co-occur with them (e.g. "midst", "accordance").
            v = v * float(self.idf[tid].item())
            cur = float(self.excitation[tid].item())
            if v > cur:
                self.excitation[tid] = float(v)

        # Diffuse: each hop, newly excited nodes send hop_decay * their
        # current excitation to their neighbours. Excitation is never
        # lowered, only raised via torch.maximum.
        self._diffuse_hops(cfg.spreading_hops, cfg.hop_decay)

    # -------------------------------------------------------------------
    # Multi-hop diffusion
    # -------------------------------------------------------------------
    def _diffuse_hops(self, hops: int, hop_decay: float):
        if hops <= 0 or self.edge_from.numel() == 0:
            return
        current = self.excitation.clone()
        for _ in range(int(hops)):
            # new_excitation[to[i]] += current[from[i]] * weight[i]
            contrib = current[self.edge_from] * self.edge_weight
            # Normalise by max edge weight so the numeric scale of the
            # contribution is comparable across hops.
            contrib = contrib / self.max_edge_weight
            new_exc = torch.zeros_like(current)
            new_exc.scatter_add_(0, self.edge_to, contrib)
            new_exc = new_exc * float(hop_decay)
            # Keep the strongest signal at each node.
            current = torch.maximum(current, new_exc)
        self.excitation = torch.maximum(self.excitation, current)

    # -------------------------------------------------------------------
    # Per-candidate semantic force (the core of inference)
    # -------------------------------------------------------------------
    def semantic_force(self) -> torch.Tensor:
        """Return (N,) float32 where entry [w] is the summed semantic
        pull from every excited node toward candidate w."""
        cfg = self.config
        if self.edge_from.numel() == 0:
            return torch.zeros(self.N, dtype=torch.float32, device=self.device)

        exc = self.excitation
        # Ignore nodes whose excitation is below the threshold. This
        # mirrors `where(n.Excitation > 0.05)` in the C# generator.
        eff = torch.where(
            exc >= float(cfg.excitation_threshold),
            exc,
            torch.zeros_like(exc))

        # contrib[i] = eff[from[i]] * edge_weight[i] * edge_relevance[i]
        # edge_relevance is a Z-score-based measure of how statistically
        # exceptional each edge is for BOTH its source and target nodes.
        # Common-word edges (typical for the node) get ~0.5, exceptional
        # edges get ~1.0, noise gets ~0. This naturally suppresses the
        # influence of common words without arbitrary thresholds.
        contrib = eff[self.edge_from] * self.edge_weight * self.edge_relevance
        out = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        out.scatter_add_(0, self.edge_to, contrib)

        # Degree-normalized: average contribution per excited neighbour.
        excited_contrib_count = torch.zeros(
            self.N, dtype=torch.float32, device=self.device)
        excited_mask = (eff[self.edge_from] > 0).float()
        excited_contrib_count.scatter_add_(
            0, self.edge_to, excited_mask)
        excited_contrib_count = excited_contrib_count.clamp_min(1.0)
        out = out / excited_contrib_count

        # DO NOT normalize to [0,1] — that amplifies noise when total
        # excitation is low (IDF suppresses common prompt words). The
        # raw degree-averaged values are used by the generator with
        # adaptive mixing that handles scaling properly.
        return out

    # -------------------------------------------------------------------
    # Step update: called after a candidate is chosen
    # -------------------------------------------------------------------
    def activate_chosen(self, word_id: int):
        """Excite the newly chosen word and apply its repulsion."""
        if word_id < 0 or word_id >= self.N:
            return
        self.excitation[word_id] = 0.8
        self.repulsion[word_id] = float(self.config.repulsion_strength)

    def apply_decay(self):
        """Multiplicative decay of excitation and repulsion."""
        self.excitation.mul_(float(self.config.excitation_decay))
        self.repulsion.mul_(float(self.config.repulsion_decay))
