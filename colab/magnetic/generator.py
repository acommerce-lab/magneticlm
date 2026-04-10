# magnetic/generator.py
#
# Multi-force text generator and cloze scorer.
#
# The scoring recipe combines TWO proven components:
#
#   1. The LEGACY recipe from eval_full_wt103 (KN-5 + position
#      cosine similarity with adaptive lambda bands). This is what
#      produced 14.20 PPL on WT103 and 27% OOD top-10 in the baseline.
#
#   2. The EXCITATION recipe from C# Generator.cs (spreading activation
#      through semantic edges). This is the user's original architecture
#      that was dropped in the GPU rewrite.
#
# The two are combined via an ADAPTIVE mixing scheme:
#
#   When KN is strong (> 0.05):
#     mixed = 0.96 * kn + 0.02 * pos_score + 0.02 * sem_score
#   When KN is medium (> 0.005):
#     mixed = 0.88 * kn + 0.06 * pos_score + 0.06 * sem_score
#   When KN is weak:
#     mixed = 0.76 * kn + 0.12 * pos_score + 0.12 * sem_score
#
# This ensures the semantic force only "fills in" when KN has no
# signal, and never overrides a strong KN prediction. The position
# similarity provides the proven baseline, and the excitation adds
# the user's graph-walking contribution on top.
#
# For GENERATION, the candidate pool is filtered to:
#   - words with KN probability > floor (contextual candidates), PLUS
#   - words with excitation > threshold (excited candidates)
# This matches the C# Generator.cs `contextNeighbors.Union(excitedWords)`
# and prevents the heavy-tail problem where 100k rare words each get
# a tiny probability that collectively swamps the good candidates.

from typing import List, Optional, Tuple

import torch

from .model import MagneticModel
from .tokenizer import tokenize


class MagneticGenerator:
    def __init__(self, model: MagneticModel):
        self.model = model
        self.cfg = model.config

    # -------------------------------------------------------------------
    # Public entry points
    # -------------------------------------------------------------------
    def generate_from_text(
        self,
        prompt: str,
        max_tokens: int = 30,
        greedy: bool = False,
    ) -> Tuple[List[int], List[str], List[str]]:
        ids, unknown = self.model.vocab.tokenize_text(prompt)
        if not ids:
            return [], [], []
        prompt_words = self.model.vocab.lookup(ids)
        all_ids = self.generate_from_ids(ids, max_tokens, greedy)
        gen_words = self.model.vocab.lookup(all_ids[len(ids):])
        return all_ids, prompt_words, gen_words

    def generate_from_ids(
        self,
        prompt_ids: List[int],
        max_tokens: int = 30,
        greedy: bool = False,
    ) -> List[int]:
        ids = [int(i) for i in prompt_ids]
        self.model.excitation.excite_prompt(ids)

        for _ in range(int(max_tokens)):
            scores = self._score_candidates_generation(ids)
            if scores is None:
                break
            nxt = self._select(scores, greedy=greedy)
            if nxt is None or nxt < 0:
                break
            ids.append(int(nxt))
            self.model.excitation.activate_chosen(int(nxt))
            self.model.excitation.apply_decay()

        return ids

    # -------------------------------------------------------------------
    # Scoring for GENERATION (filtered candidates + adaptive mixing)
    # -------------------------------------------------------------------
    def _score_candidates_generation(self, history: List[int]) -> Optional[torch.Tensor]:
        """Score candidates for the next token during generation.
        Only words with KN > floor OR excitation > threshold are scored.
        Returns a (V,) tensor with -inf for non-candidates."""
        model = self.model
        cfg = self.cfg
        dev = model.device
        V = len(model.vocab)

        # Full KN scores for all V candidates.
        kn = self._compute_kn_all(history)

        # Candidate filtering: top-K by KN score + any excited words.
        # This eliminates "ghost words" — the 100k+ rare words that
        # each get a tiny KN backoff probability and collectively swamp
        # the good candidates when sampled via softmax.
        topk = int(cfg.generation_topk)
        if topk > 0 and topk < V:
            _, topk_idx = torch.topk(kn, k=min(topk, V))
            kn_mask = torch.zeros(V, dtype=torch.bool, device=dev)
            kn_mask[topk_idx] = True
        else:
            kn_mask = kn > 1e-6
        exc_mask = model.excitation.excitation > float(cfg.excitation_threshold)
        candidates = kn_mask | exc_mask

        if not candidates.any():
            return None

        # Position similarity for candidates.
        pos_score = self._compute_pos_similarity(history)

        # Semantic force from excitation.
        sem_score = model.excitation.semantic_force()

        # Clamp semantic to [0, max_pos_score] so it doesn't dominate.
        # This keeps the two supplementary signals on the same scale.
        pos_max = float(pos_score.max().clamp_min(0.01).item())
        sem_score = sem_score.clamp(0.0, pos_max)

        # Repulsion (for used words).
        repulsion = model.excitation.repulsion

        # Adaptive mixing (same bands as eval_full_wt103).
        mixed = self._adaptive_mix(kn, pos_score, sem_score)
        mixed = mixed - repulsion
        mixed = mixed.clamp_min(1e-10)

        # Set non-candidates to -inf so they can't be selected.
        result = torch.full((V,), float('-inf'), device=dev)
        result[candidates] = mixed[candidates]
        return result

    # -------------------------------------------------------------------
    # Scoring for CLOZE (all candidates, adaptive mixing)
    # -------------------------------------------------------------------
    def score_cloze(
        self,
        left_ids: List[int],
        right_ids: List[int],
    ) -> torch.Tensor:
        """Score all V candidates for cloze evaluation. Uses KN on left
        context + position similarity on left context + excitation from
        BOTH left and right context (bidirectional)."""
        model = self.model
        dev = model.device

        # Excitation from full bidirectional context.
        all_ctx = [int(i) for i in left_ids if int(i) >= 0]
        all_ctx += [int(i) for i in right_ids if int(i) >= 0]
        model.excitation.excite_prompt(all_ctx)

        # KN on left only (KN is unidirectional).
        kn = self._compute_kn_all(left_ids)

        # Position similarity from left context.
        pos_score = self._compute_pos_similarity(left_ids)

        # Semantic force (bidirectional via excitation).
        sem_score = model.excitation.semantic_force()
        pos_max = float(pos_score.max().clamp_min(0.01).item())
        sem_score = sem_score.clamp(0.0, pos_max)

        return self._adaptive_mix(kn, pos_score, sem_score)

    # -------------------------------------------------------------------
    # Adaptive mixing (the proven recipe from eval_full_wt103)
    # -------------------------------------------------------------------
    def _adaptive_mix(
        self,
        kn: torch.Tensor,
        pos_score: torch.Tensor,
        sem_score: torch.Tensor,
    ) -> torch.Tensor:
        """Mix KN + position similarity + semantic force with adaptive
        bands. When KN is confident, trust it (98%). When KN is weak,
        give supplementary signals more room (24% total)."""
        dev = kn.device
        # band = weight for EACH of pos and sem (so total supplementary
        # = 2 * band).
        band = torch.where(
            kn > 0.05,
            torch.tensor(0.02, device=dev),
            torch.where(
                kn > 0.005,
                torch.tensor(0.06, device=dev),
                torch.tensor(0.12, device=dev)))
        kn_w = 1.0 - 2.0 * band
        mixed = kn_w * kn + band * pos_score + band * sem_score
        return mixed.clamp(1e-10, 0.999)

    # -------------------------------------------------------------------
    # KN computation helper
    # -------------------------------------------------------------------
    def _compute_kn_all(self, context_ids: List[int]) -> torch.Tensor:
        """Compute KN-5 probabilities for all V candidates given a
        context (list of token ids, last K are used)."""
        model = self.model
        dev = model.device
        V = len(model.vocab)
        K = self.cfg.max_ngram_order
        chunk = int(self.cfg.candidate_chunk_size)

        ctx_row = torch.full((K,), -1, dtype=torch.int64, device=dev)
        clean = [int(i) for i in context_ids if int(i) >= 0]
        ctx_len = min(len(clean), K)
        if ctx_len > 0:
            tail = torch.tensor(
                clean[-ctx_len:], dtype=torch.int64, device=dev)
            ctx_row[-ctx_len:] = tail

        kn = torch.empty(V, dtype=torch.float32, device=dev)
        for cs in range(0, V, chunk):
            ce = min(cs + chunk, V)
            B = ce - cs
            ctx_b = ctx_row.unsqueeze(0).expand(B, -1).contiguous()
            nxt_b = torch.arange(cs, ce, dtype=torch.int64, device=dev)
            kn[cs:ce] = model.ngram.kn_score_batch(ctx_b, nxt_b)
        return kn

    # -------------------------------------------------------------------
    # Position similarity (same as eval_full_wt103)
    # -------------------------------------------------------------------
    def _compute_pos_similarity(
        self, context_ids: List[int],
    ) -> torch.Tensor:
        """Compute position-based similarity between context words and
        every candidate, using the physics embedding. This is the same
        formula that produced the 14.20 PPL / 27% OOD baseline."""
        model = self.model
        dev = model.device
        V = len(model.vocab)
        K = self.cfg.max_ngram_order
        pos = model.positions
        imp = model.importance

        clean = [int(i) for i in context_ids if int(i) >= 0]
        ctx_len = min(len(clean), K)
        if ctx_len == 0 or pos is None:
            return torch.zeros(V, dtype=torch.float32, device=dev)

        ctx_ids = torch.tensor(
            clean[-ctx_len:], dtype=torch.int64, device=dev)
        ctx_pos = pos[ctx_ids]            # (Kc, dim)
        ctx_imp = imp[ctx_ids]            # (Kc,)

        chunk = int(self.cfg.candidate_chunk_size)
        out = torch.zeros(V, dtype=torch.float32, device=dev)

        for cs in range(0, V, chunk):
            ce = min(cs + chunk, V)
            nxt_ids = torch.arange(cs, ce, dtype=torch.int64, device=dev)
            nxt_pos = pos[nxt_ids]        # (B, dim)

            # Cosine similarity: (B, Kc)
            dot = nxt_pos @ ctx_pos.T
            nxt_norm = nxt_pos.norm(dim=1, keepdim=True).clamp_min(1e-6)
            ctx_norm = ctx_pos.norm(dim=1, keepdim=True).clamp_min(1e-6)
            sim = dot / (nxt_norm @ ctx_norm.T)
            sim = sim.clamp(-1.0, 1.0)

            # Only keep positive, meaningful similarities.
            valid = sim > 0.05
            sim = torch.where(valid, sim, torch.zeros_like(sim))

            # Importance boost.
            boost = (1.0 + ctx_imp * 0.05).unsqueeze(0)  # (1, Kc)
            contrib = sim * boost

            # Average over valid context words, scaled to [0, ~0.3].
            pos_count = valid.sum(dim=1).clamp_min(1).float()
            has_any = valid.any(dim=1)
            pos_score = contrib.sum(dim=1)
            pos_score = (pos_score / (pos_count * 3.0)).clamp_max(0.3)
            pos_score = torch.where(
                has_any, pos_score, torch.zeros_like(pos_score))

            out[cs:ce] = pos_score

        return out

    # -------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------
    def _select(
        self,
        scores: torch.Tensor,
        greedy: bool = False,
    ) -> Optional[int]:
        cfg = self.cfg
        valid = torch.isfinite(scores)
        if not valid.any():
            return None
        safe_scores = torch.where(
            valid, scores, torch.full_like(scores, -1e9))

        if greedy or cfg.temperature <= 0.01:
            return int(safe_scores.argmax().item())

        logits = (safe_scores - safe_scores.max()) / float(cfg.temperature)
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            return int(safe_scores.argmax().item())
        return int(torch.multinomial(probs, 1).item())
