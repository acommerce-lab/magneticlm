# magnetic/generator.py
#
# Multi-force text generator. This is the direct GPU port of the C#
# Generator.cs logic, with no shortcuts:
#
#   total = alpha * contextual_score       (KN-5 probability)
#         + beta  * semantic_score         (spreading activation)
#         + repulsion_score                (negative, discourages reuse)
#
# The candidate pool is the full vocabulary, scored in chunks so
# per-step memory stays bounded. Selection uses a softmax with
# temperature (temperature -> 0 = greedy).
#
# One generation step:
#   1. Build the current left context (last K tokens).
#   2. Score all V candidates via ngram.kn_score_batch.
#   3. Ask the excitation engine for the current semantic force vector.
#   4. Mix (alpha, beta) and subtract repulsion.
#   5. Sample (or argmax) the next token.
#   6. Activate the chosen token, apply decay, loop.

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
        """Generate from a raw-text prompt. Returns
        (all_token_ids, prompt_words, generated_words)."""
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
        """Core generation loop. prompt_ids is a list of vocabulary
        ids for the prompt; the method returns the full sequence
        including the prompt."""
        ids = [int(i) for i in prompt_ids]

        # Initial prompt excitation + spreading.
        self.model.excitation.excite_prompt(ids)

        for _ in range(int(max_tokens)):
            scores = self._score_all_candidates(ids)
            nxt = self._select(scores, greedy=greedy)
            if nxt is None or nxt < 0:
                break
            ids.append(int(nxt))
            # Update excitation/repulsion state.
            self.model.excitation.activate_chosen(int(nxt))
            self.model.excitation.apply_decay()

        return ids

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------
    def _score_all_candidates(self, history: List[int]) -> torch.Tensor:
        """Return a (V,) float32 total-score tensor for the next
        token given the running history (list of token ids)."""
        cfg = self.cfg
        model = self.model
        dev = model.device
        V = len(model.vocab)
        K = cfg.max_ngram_order
        chunk = int(cfg.candidate_chunk_size)

        # 1. KN contextual score - computed once per step by chunking
        # the candidate axis (not the context) because the context is
        # the same for every candidate.
        ctx_row = torch.full((K,), -1, dtype=torch.int64, device=dev)
        ctx_len = min(len(history), K)
        if ctx_len > 0:
            tail = torch.tensor(
                history[-ctx_len:], dtype=torch.int64, device=dev)
            ctx_row[-ctx_len:] = tail

        kn = torch.empty(V, dtype=torch.float32, device=dev)
        for cs in range(0, V, chunk):
            ce = min(cs + chunk, V)
            B = ce - cs
            ctx_b = ctx_row.unsqueeze(0).expand(B, -1).contiguous()
            nxt_b = torch.arange(cs, ce, dtype=torch.int64, device=dev)
            kn[cs:ce] = model.ngram.kn_score_batch(ctx_b, nxt_b)

        # 2. Semantic score from the excitation engine (full vector).
        semantic = model.excitation.semantic_force()

        # 3. Repulsion (subtracted - strong repulsion on recent words).
        repulsion = model.excitation.repulsion

        # Combine.
        total = (float(cfg.alpha_contextual) * kn
                 + float(cfg.beta_semantic) * semantic
                 - repulsion)

        return total

    # -------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------
    def _select(
        self,
        scores: torch.Tensor,
        greedy: bool = False,
    ) -> Optional[int]:
        cfg = self.cfg
        # We want to treat the scores as relative "goodness" and then
        # sample - softmax with temperature works. Filter out any
        # -inf or NaN defensively.
        valid = torch.isfinite(scores)
        if not valid.any():
            return None
        scores = torch.where(valid, scores, torch.full_like(scores, -1e9))

        if greedy or cfg.temperature <= 0.01:
            return int(scores.argmax().item())

        # Softmax with temperature on raw scores. We shift by max for
        # numeric stability.
        logits = (scores - scores.max()) / float(cfg.temperature)
        probs = torch.softmax(logits, dim=-1)
        # multinomial can fail on all-zero, guard.
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            return int(scores.argmax().item())
        return int(torch.multinomial(probs, 1).item())

    # -------------------------------------------------------------------
    # Cloze: score a single candidate slot with bidirectional context
    # -------------------------------------------------------------------
    def score_cloze(
        self,
        left_ids: List[int],
        right_ids: List[int],
    ) -> torch.Tensor:
        """For cloze evaluation: return a (V,) score vector combining
        KN on the left context, semantic force from an excitation
        built on BOTH sides, and no repulsion (nothing has been
        "used" yet).

        Note: KN is inherently left-to-right, so it uses only
        left_ids. The excitation is built from left_ids followed by
        right_ids so the spreading activation sees both sides; the
        resulting semantic force is symmetric in the graph sense.
        """
        model = self.model
        cfg = self.cfg
        dev = model.device
        V = len(model.vocab)
        K = cfg.max_ngram_order
        chunk = int(cfg.candidate_chunk_size)

        # Build excitation from full bidirectional context.
        all_ctx = [int(i) for i in left_ids if int(i) >= 0]
        all_ctx += [int(i) for i in right_ids if int(i) >= 0]
        model.excitation.excite_prompt(all_ctx)

        # KN on left only.
        ctx_row = torch.full((K,), -1, dtype=torch.int64, device=dev)
        left_clean = [int(i) for i in left_ids if int(i) >= 0]
        ctx_len = min(len(left_clean), K)
        if ctx_len > 0:
            tail = torch.tensor(
                left_clean[-ctx_len:], dtype=torch.int64, device=dev)
            ctx_row[-ctx_len:] = tail

        kn = torch.empty(V, dtype=torch.float32, device=dev)
        for cs in range(0, V, chunk):
            ce = min(cs + chunk, V)
            B = ce - cs
            ctx_b = ctx_row.unsqueeze(0).expand(B, -1).contiguous()
            nxt_b = torch.arange(cs, ce, dtype=torch.int64, device=dev)
            kn[cs:ce] = model.ngram.kn_score_batch(ctx_b, nxt_b)

        semantic = model.excitation.semantic_force()

        total = (float(cfg.alpha_contextual) * kn
                 + float(cfg.beta_semantic) * semantic)
        return total
