"""Inference: scoring candidates using multi-source mixture.

P(w | context) = alpha * P_concept(w | context)
               + beta  * P_ppmi(w | context)
               + gamma * P_kn(w | last_token)

Candidates = ctx_children(current)
           UNION adopted_children (via semantic neighbors of current)
           UNION global top-k (optional)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .contextual_map import ContextualMap
from .statistical_layer import Statistics, bigram_prob, ppmi_prob
from .semantic_map import SemanticMapState, to_sparse_tensor
from .spreading import spread, compute_node_thresholds


@dataclass
class InferenceEngine:
    ctx: ContextualMap
    sem: SemanticMapState
    stats: Statistics
    cfg: "any"
    device: torch.device
    sem_sparse: Optional[torch.Tensor] = None
    node_stats: Optional[Dict[str, torch.Tensor]] = None

    def prepare(self):
        self.sem_sparse = to_sparse_tensor(self.sem, self.device)
        if self.cfg.transfer_method == "threshold_auto":
            self.node_stats = compute_node_thresholds(self.sem_sparse)

    # --------------------------------------------------------------
    def score_next_token(
        self,
        current: int,
        prev: Optional[int] = None,
    ) -> torch.Tensor:
        """Return dense [V] probability distribution for next token."""
        V = self.stats.vocab_size
        device = self.device
        cfg = self.cfg

        # --- concept activation via spreading ---
        seed = torch.zeros(V, dtype=torch.float32, device=device)
        seed[current] = 1.0
        if cfg.alpha_concept > 0 and self.sem_sparse is not None:
            activation = spread(seed, self.sem_sparse, cfg, self.node_stats)
            # positive-only mass
            pos = torch.clamp(activation, min=0.0)
            denom = pos.sum().clamp(min=1e-9)
            p_concept = pos / denom
        else:
            p_concept = torch.zeros(V, dtype=torch.float32, device=device)

        # --- PPMI-based ---
        if cfg.beta_ppmi > 0:
            all_b = torch.arange(V, device=device)
            a = torch.full_like(all_b, current)
            p_ppmi = ppmi_prob(self.stats, a, all_b)
            s = p_ppmi.sum().clamp(min=1e-9)
            p_ppmi = p_ppmi / s
        else:
            p_ppmi = torch.zeros(V, dtype=torch.float32, device=device)

        # --- KN bigram ---
        if cfg.gamma_kn > 0:
            all_b = torch.arange(V, device=device)
            a = torch.full_like(all_b, current)
            p_kn = bigram_prob(self.stats, a, all_b)
            s = p_kn.sum().clamp(min=1e-9)
            p_kn = p_kn / s
        else:
            p_kn = torch.zeros(V, dtype=torch.float32, device=device)

        # --- ctx direct succession: boosts p_concept with direct counts ---
        ctx_row = torch.zeros(V, dtype=torch.float32, device=device)
        starts = int(self.ctx.row_ptr[current].item())
        ends = int(self.ctx.row_ptr[current + 1].item())
        if ends > starts:
            cols = self.ctx.col_idx[starts:ends].to(torch.int64)
            cnts = self.ctx.counts[starts:ends].to(torch.float32)
            ctx_row.scatter_add_(0, cols, cnts)
            s_ctx = ctx_row.sum().clamp(min=1e-9)
            ctx_row = ctx_row / s_ctx
        if cfg.alpha_concept > 0:
            p_concept = 0.5 * p_concept + 0.5 * ctx_row
        else:
            p_concept = ctx_row

        # combine
        if cfg.scoring_method == "concept_only":
            return p_concept
        elif cfg.scoring_method == "stats_only":
            s = cfg.beta_ppmi + cfg.gamma_kn
            if s <= 0:
                return torch.full_like(p_concept, 1.0 / V)
            return (cfg.beta_ppmi * p_ppmi + cfg.gamma_kn * p_kn) / s
        else:
            mix = (
                cfg.alpha_concept * p_concept
                + cfg.beta_ppmi * p_ppmi
                + cfg.gamma_kn * p_kn
            )
            s = mix.sum().clamp(min=1e-9)
            return mix / s

    # --------------------------------------------------------------
    def score_batch(self, contexts: torch.Tensor) -> torch.Tensor:
        """Score a batch of current tokens -> [B, V]. Vectorized version.

        For now this loops; a fully-vectorized spreading pass is a future step.
        """
        B = contexts.numel()
        V = self.stats.vocab_size
        out = torch.zeros(B, V, dtype=torch.float32, device=self.device)
        for i, c in enumerate(contexts.tolist()):
            out[i] = self.score_next_token(int(c))
        return out

    def top_k(self, current: int, k: int = 10):
        dist = self.score_next_token(current)
        vals, idx = torch.topk(dist, k)
        return idx.tolist(), vals.tolist()

    def generate(self, seed: List[int], length: int, top_k: int = 40, temperature: float = 1.0) -> List[int]:
        out = list(seed)
        for _ in range(length):
            if not out:
                break
            dist = self.score_next_token(int(out[-1]))
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
        return out
