"""Inference: 5-component scoring with persistent activation field.

Stateless mode (InferenceEngine): for PPL/hit-rate evaluation.
Stateful mode (InferenceSession): for generation with glow centers.

Scoring formula (mixture mode):
  score(b) = a1 * S_direct(b)   — ctx children of current, modulated by field
           + a2 * S_adopt(b)    — adopted children via semantic neighbors, modulated
           + a3 * S_concept(b)  — spreading through concept relay layer
           + a4 * S_field(b)    — persistent activation from distant glow centers
           + a5 * S_stats(b)    — PPMI + KN bigram baseline

Glow center mechanism:
  - Activation field persists across generation steps
  - Each word observation -> field update -> concept accumulation
  - Concepts above glow_threshold inject activation into all member words
  - This is how a concept activated 30 tokens ago still influences scoring
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .contextual_map import ContextualMap
from .statistical_layer import Statistics, bigram_prob, ppmi_prob
from .semantic_map import SemanticMapState, neighbors as sem_neighbors, to_sparse_tensor
from .concepts import ConceptLayer
from .spreading import spread, spread_with_concepts, compute_node_thresholds


@dataclass
class InferenceEngine:
    ctx: ContextualMap
    sem: SemanticMapState
    stats: Statistics
    concepts: Optional[ConceptLayer]
    cfg: "any"
    device: torch.device
    sem_sparse: Optional[torch.Tensor] = None
    node_stats: Optional[Dict[str, torch.Tensor]] = None
    idf: Optional[torch.Tensor] = None

    def prepare(self):
        self.sem_sparse = to_sparse_tensor(self.sem, self.device)
        if self.cfg.transfer_method == "threshold_auto":
            self.node_stats = compute_node_thresholds(self.sem_sparse)
        V = self.stats.vocab_size
        freq = self.stats.unigram_counts.to(torch.float32).to(self.device)
        N = freq.sum().clamp(min=1.0)
        self.idf = torch.log(N / (freq + 1.0) + 1.0)
        self.idf = self.idf / self.idf.max().clamp(min=1e-9)

    def create_session(self) -> "InferenceSession":
        return InferenceSession(self)

    # ------------------------------------------------------------------
    # Stateless scoring (for PPL, hit-rate — no field, no glow)
    # ------------------------------------------------------------------

    def score_next_token(self, current: int) -> torch.Tensor:
        V = self.stats.vocab_size
        device = self.device
        cfg = self.cfg

        s_direct = self._ctx_scores(current)

        s_adopt = self._adoption_scores(current)

        if cfg.alpha_concept > 0 and self.sem_sparse is not None:
            s_concept = self._concept_spreading(current, field_seed=None)
        else:
            s_concept = torch.zeros(V, dtype=torch.float32, device=device)

        s_stats = self._stats_scores(current)

        if cfg.scoring_method == "stats_only":
            return s_stats
        elif cfg.scoring_method == "concept_only":
            mix = s_direct + s_adopt + s_concept
        else:
            mix = (
                cfg.alpha_direct * s_direct
                + cfg.alpha_adopt * s_adopt
                + cfg.alpha_concept * s_concept
                + cfg.alpha_stats * s_stats
            )
        s = mix.sum().clamp(min=1e-9)
        return mix / s

    # ------------------------------------------------------------------
    # Shared scoring components
    # ------------------------------------------------------------------

    def _ctx_scores(self, current: int) -> torch.Tensor:
        V = self.stats.vocab_size
        device = self.device
        scores = torch.zeros(V, dtype=torch.float32, device=device)
        s = int(self.ctx.row_ptr[current].item())
        e = int(self.ctx.row_ptr[current + 1].item())
        if e > s:
            cols = self.ctx.col_idx[s:e].to(torch.int64)
            cnts = self.ctx.counts[s:e].to(torch.float32)
            # Sublinear + IDF: sqrt dampens raw count; IDF lifts rare words
            dampened = torch.sqrt(cnts)
            if self.idf is not None:
                dampened = dampened * self.idf[cols]
            scores.scatter_add_(0, cols, dampened)
            scores /= scores.sum().clamp(min=1e-9)
        return scores

    def _adoption_scores(self, current: int) -> torch.Tensor:
        V = self.stats.vocab_size
        device = self.device
        cfg = self.cfg
        nbrs = sem_neighbors(self.sem, current, top_k=cfg.adoption_neighbors)
        scores = torch.zeros(V, dtype=torch.float32, device=device)
        for n_id, n_weight in nbrs:
            if n_weight < cfg.adoption_min_weight:
                continue
            s = int(self.ctx.row_ptr[n_id].item())
            e = int(self.ctx.row_ptr[n_id + 1].item())
            if e > s:
                cols = self.ctx.col_idx[s:e].to(torch.int64)
                cnts = self.ctx.counts[s:e].to(torch.float32)
                dampened = torch.sqrt(cnts) * n_weight
                if self.idf is not None:
                    dampened = dampened * self.idf[cols]
                scores.scatter_add_(0, cols, dampened)
        total = scores.sum().clamp(min=1e-9)
        return scores / total

    def _concept_spreading(
        self, current: int, field_seed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        V = self.stats.vocab_size
        device = self.device
        seed = torch.zeros(V, dtype=torch.float32, device=device)
        seed[current] = 1.0
        if field_seed is not None:
            seed = seed + field_seed * 0.1
        if self.concepts is not None and self.concepts.n_concepts > 0:
            activation = spread_with_concepts(
                seed, self.sem_sparse, self.concepts, self.cfg, self.node_stats
            )
        else:
            activation = spread(seed, self.sem_sparse, self.cfg, self.node_stats)
        pos = activation.clamp(min=0.0)
        s = pos.sum().clamp(min=1e-9)
        return pos / s

    def _stats_scores(self, current: int) -> torch.Tensor:
        V = self.stats.vocab_size
        device = self.device
        all_b = torch.arange(V, device=device)
        a = torch.full_like(all_b, current)
        p_ppmi = ppmi_prob(self.stats, a, all_b)
        s1 = p_ppmi.sum().clamp(min=1e-9)
        p_ppmi = p_ppmi / s1
        p_kn = bigram_prob(self.stats, a, all_b)
        s2 = p_kn.sum().clamp(min=1e-9)
        p_kn = p_kn / s2
        # PPMI already corrects for frequency; KN is frequency-dominated.
        # Lean heavily on PPMI, use KN only as backoff for unseen pairs.
        combined = 0.7 * p_ppmi + 0.3 * p_kn
        if self.idf is not None:
            combined = combined * self.idf
        return combined / combined.sum().clamp(min=1e-9)


# ======================================================================
# Stateful inference session with persistent activation field
# ======================================================================


class InferenceSession:
    """Stateful session: maintains activation field across generation steps.

    The field accumulates from every observed token. Concept nodes aggregate
    word-level activation. When a concept's activation crosses glow_threshold,
    it becomes a glow center and injects activation into all member words.

    This is how a concept activated 30 tokens ago (e.g. {royalty}) still
    boosts words like "crown", "throne" at the current generation step,
    even if the current word is "the" with no direct semantic link.
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        V = engine.stats.vocab_size
        C = engine.concepts.n_concepts if engine.concepts else 0
        self.device = engine.device
        self.word_field = torch.zeros(V, dtype=torch.float32, device=self.device)
        self.concept_field = torch.zeros(C, dtype=torch.float32, device=self.device)

    def observe(self, token_id: int):
        """Register a token (from seed or generated output)."""
        self.word_field[token_id] += 1.0
        concepts = self.engine.concepts
        if concepts and concepts.n_concepts > 0:
            delta = torch.zeros_like(self.word_field)
            delta[token_id] = 1.0
            self.concept_field += concepts.word_to_concept_activation(delta)

    def _step(self):
        """Decay + glow injection. Called before each scoring step."""
        cfg = self.engine.cfg
        self.word_field *= (1.0 - cfg.field_decay)
        self.concept_field *= (1.0 - cfg.concept_decay)

        concepts = self.engine.concepts
        if concepts and concepts.n_concepts > 0:
            glow_mask = self.concept_field > cfg.glow_threshold
            if glow_mask.any():
                glow_act = self.concept_field * glow_mask.float()
                injection = concepts.concept_to_word_injection(glow_act, cfg.glow_strength)
                self.word_field += injection

    def score_next(self, current: int) -> torch.Tensor:
        """Full 5-component scoring with field influence."""
        self._step()
        eng = self.engine
        cfg = eng.cfg
        V = eng.stats.vocab_size

        # Modulation: field-based boost for all candidates
        field_mod = 1.0 + self.word_field.clamp(min=0.0)

        # S_direct: contextual children × field modulation
        s_direct = eng._ctx_scores(current) * field_mod

        # S_adopt: adopted children × field modulation
        s_adopt = eng._adoption_scores(current) * field_mod

        # S_concept: spreading through semantic + concept relay, seeded with field
        if cfg.alpha_concept > 0 and eng.sem_sparse is not None:
            s_concept = eng._concept_spreading(current, field_seed=self.word_field)
        else:
            s_concept = torch.zeros(V, dtype=torch.float32, device=self.device)

        # S_field: persistent activation (distant glow center contribution)
        s_field = self.word_field.clamp(min=0.0)
        sf_sum = s_field.sum().clamp(min=1e-9)
        s_field = s_field / sf_sum

        # S_stats: statistical baseline
        s_stats = eng._stats_scores(current)

        # Combine
        if cfg.scoring_method == "stats_only":
            return s_stats
        elif cfg.scoring_method == "concept_only":
            mix = s_direct + s_adopt + s_concept + cfg.alpha_field * s_field
        else:
            mix = (
                cfg.alpha_direct * s_direct
                + cfg.alpha_adopt * s_adopt
                + cfg.alpha_concept * s_concept
                + cfg.alpha_field * s_field
                + cfg.alpha_stats * s_stats
            )

        s = mix.sum().clamp(min=1e-9)
        return mix / s

    def generate(
        self,
        seed_ids: List[int],
        length: int,
        top_k: int = 40,
        temperature: float = 1.0,
    ) -> List[int]:
        for tok in seed_ids:
            self.observe(tok)
        out = list(seed_ids)
        for _ in range(length):
            if not out:
                break
            dist = self.score_next(int(out[-1]))
            if temperature != 1.0:
                dist = torch.pow(dist.clamp(min=1e-12), 1.0 / max(temperature, 1e-6))
                dist = dist / dist.sum().clamp(min=1e-9)
            if top_k > 0:
                vals, idx = torch.topk(dist, min(top_k, dist.numel()))
                probs = vals / vals.sum().clamp(min=1e-9)
                pick = int(idx[torch.multinomial(probs, 1).item()].item())
            else:
                pick = int(torch.multinomial(dist, 1).item())
            out.append(pick)
            self.observe(pick)
        return out

    def get_glow_centers(self) -> torch.Tensor:
        if self.concept_field.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return (self.concept_field > self.engine.cfg.glow_threshold).nonzero().squeeze(-1)
