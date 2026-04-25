"""SPIM — Spectral Path Integral Model.

Core computation:
  1. Context = path of transitions ΔE(w_k → w_{k+1})
  2. Accumulate path at each cone level with decay
  3. Predicted point = last word embedding + accumulated path
  4. Score all words at each level, combine with spectral weights
"""

import math
from typing import List
import torch
import torch.nn.functional as F


class SPIMModel:
    """Spectral Path Integral Model.

    No attention. No learned weights. No training loop.
    Prediction = integration of context transitions through spectral cone.
    """

    def __init__(self, embeddings, d_schedule, S_levels, pos_decay=0.7):
        self.device = embeddings.device
        self.embeddings = embeddings  # [V, d]
        self.V, self.d = embeddings.shape
        self.d_schedule = d_schedule
        self.S_levels = S_levels  # [L] weights per level
        self.n_levels = len(d_schedule)
        self.pos_decay = pos_decay

        # Pre-compute normalized embeddings per level (for FFN)
        self.E_levels = []
        self.E_norm_levels = []
        for dl in d_schedule:
            e = embeddings[:, :dl]
            self.E_levels.append(e)
            self.E_norm_levels.append(e / e.norm(dim=1, keepdim=True).clamp(min=1e-9))

    def _path_integral(self, context_ids, d_l):
        """Compute accumulated transition path for one context at dimension d_l.

        path = Σ_{k=0}^{n-2} decay^(n-2-k) × (E[w_{k+1}][:d_l] - E[w_k][:d_l])

        Returns: [d_l] accumulated transition vector
        """
        if len(context_ids) < 2:
            return torch.zeros(d_l, device=self.device)

        E = self.embeddings[:, :d_l]
        ids = context_ids
        n = len(ids)

        # Compute transitions: E[w_{k+1}] - E[w_k]
        emb_seq = E[torch.tensor(ids, dtype=torch.long, device=self.device)]  # [n, d_l]
        deltas = emb_seq[1:] - emb_seq[:-1]  # [n-1, d_l]

        # Decay weights: recent transitions matter more
        n_deltas = deltas.shape[0]
        weights = self.pos_decay ** torch.arange(n_deltas - 1, -1, -1,
                                                  dtype=torch.float32, device=self.device)
        weights = weights / weights.sum().clamp(min=1e-9)

        # Weighted sum of transitions
        path = (deltas * weights.unsqueeze(1)).sum(dim=0)  # [d_l]
        return path

    def _score_at_level(self, context_ids, level):
        """Score all words at one cone level.

        predicted = E[last_word][:d_l] + path_integral[:d_l]
        scores = predicted @ E[:, :d_l].T

        Returns: [V] scores
        """
        dl = self.d_schedule[level]
        E_l = self.E_levels[level]

        # Path integral
        path = self._path_integral(context_ids, dl)

        # Predicted point: last word + accumulated transitions
        last_id = context_ids[-1]
        last_emb = self.embeddings[last_id, :dl]
        predicted = last_emb + path  # [d_l]

        # FFN: refine through vocabulary lookup
        E_norm_l = self.E_norm_levels[level]
        sim = predicted @ E_norm_l.T / math.sqrt(dl)
        h = F.relu(sim)
        h_sum = h.sum().clamp(min=1e-9)
        refined = (h / h_sum) @ E_l  # [d_l]

        # Score against all words
        scores = refined @ E_l.T  # [V]
        return scores

    def score_single(self, context_ids):
        """Score one context. Returns [V] probability distribution."""
        # Score at each level, combine with spectral weights
        combined = torch.zeros(self.V, device=self.device)
        for l in range(self.n_levels):
            scores_l = self._score_at_level(context_ids, l)
            combined += self.S_levels[l] * scores_l

        return F.softmax(combined, dim=0)

    def score_batch(self, contexts):
        """Batch scoring. Returns [n, V] distributions."""
        n = len(contexts)
        out = torch.zeros(n, self.V, device=self.device)
        for i, ctx in enumerate(contexts):
            out[i] = self.score_single(ctx)
        return out
