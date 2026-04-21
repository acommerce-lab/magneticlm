"""Scoring: convert the complex wave field into P(next token).

Single responsibility: z ∈ ℂ^V → probability distribution over V.

Registry pattern (open/closed):
  - "magnitude":  score = |z|² = Re² + Im²
  - "projection": score = α·Re + β·Im   (linear combo)
  - "hybrid":     score = Re · (1 + β·Im)  (Permission×Drive style, complex form)
"""

from typing import Callable, Dict, Optional

import torch


SCORERS: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    s = x.sum(dim=-1, keepdim=True).clamp(min=eps)
    return x / s


@register("magnitude")
def score_magnitude(z: torch.Tensor, cfg) -> torch.Tensor:
    """|z|² — the Born-rule style. Combines context + concept isotropically."""
    re, im = z.real, z.imag
    w_re = float(cfg.context_weight)
    w_im = float(cfg.concept_weight)
    scores = w_re * re * re + w_im * im * im
    return _normalize(scores.clamp(min=0.0))


@register("projection")
def score_projection(z: torch.Tensor, cfg) -> torch.Tensor:
    """Linear projection onto context+concept axes. Allows sign."""
    re, im = z.real, z.imag
    w_re = float(cfg.context_weight)
    w_im = float(cfg.concept_weight)
    scores = w_re * re + w_im * im
    # Shift to non-negative
    scores = scores - scores.min(dim=-1, keepdim=True).values
    return _normalize(scores.clamp(min=0.0))


@register("hybrid")
def score_hybrid(z: torch.Tensor, cfg) -> torch.Tensor:
    """Permission × (1 + Drive) in complex form.

    Re = statistical permission, Im = conceptual drive.
    Gate by Re but allow Im to lift when there's contextual slack.
    """
    re = z.real.clamp(min=0.0)
    im = z.imag
    # Normalize drive independently
    im_pos = im.clamp(min=0.0)
    im_s = im_pos.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    drive = im_pos / im_s
    w = float(cfg.concept_weight)
    scores = re * (1.0 + w * drive)
    return _normalize(scores)


def score(z: torch.Tensor, cfg, unk_id: int = -1) -> torch.Tensor:
    """Public API. Optionally masks <unk>."""
    method = getattr(cfg, "scoring_method", "magnitude")
    fn = SCORERS.get(method, SCORERS["magnitude"])
    dist = fn(z, cfg)
    if getattr(cfg, "mask_unk_in_eval", False) and unk_id >= 0:
        if dist.dim() == 1:
            dist = dist.clone()
            dist[unk_id] = 0.0
        else:
            dist = dist.clone()
            dist[..., unk_id] = 0.0
        dist = _normalize(dist)
    return dist
