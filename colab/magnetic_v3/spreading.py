"""Spreading activation with binary transfer equation.

Two propagation modes:
  1. Word-to-word only (original): edges carry signal through transfer equation.
  2. Word-to-word + concept relay: word signals aggregate into concept nodes,
     concept nodes broadcast back to member words. This creates long-range
     paths that bypass direct edge distance.

The concept relay is the mechanism for distant glow center influence:
  word_a -> concept_c -> word_b  (even if no direct edge a->b exists)
"""

from typing import Dict, Optional

import torch

from .transfer import get_transfer


def compute_node_thresholds(sp_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
    sp = sp_mat.coalesce()
    idx = sp.indices()
    val = sp.values()
    rows = idx[0]
    V = sp.size(0)
    device = val.device

    row_sum = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(0, rows, val)
    row_sq = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(0, rows, val * val)
    row_cnt = torch.zeros(V, dtype=torch.float32, device=device).scatter_add_(
        0, rows, torch.ones_like(val)
    )
    mu = row_sum / row_cnt.clamp(min=1.0)
    var = (row_sq / row_cnt.clamp(min=1.0)) - mu * mu
    sd = torch.sqrt(var.clamp(min=0.0))
    return {"mu": mu, "sd": sd}


def spread(
    seed_activation: torch.Tensor,
    sp_mat: torch.Tensor,
    cfg,
    node_stats: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    transfer = get_transfer(cfg.transfer_method)
    if node_stats is None and cfg.transfer_method == "threshold_auto":
        node_stats = compute_node_thresholds(sp_mat)

    activation = seed_activation.clone()
    sp = sp_mat.coalesce()
    idx = sp.indices()
    val = sp.values()
    a = idx[0]
    b = idx[1]

    for _ in range(cfg.spreading_iters):
        in_sig = activation[a]
        out_sig = transfer(in_sig, val, cfg, node_stats)
        new_act = torch.zeros_like(activation)
        new_act.scatter_add_(0, b, out_sig)
        activation = cfg.spreading_damping * new_act + (1 - cfg.spreading_damping) * seed_activation
    return activation


def spread_with_concepts(
    seed_activation: torch.Tensor,
    sp_mat: torch.Tensor,
    concepts,
    cfg,
    node_stats: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Two-level spreading: word edges + concept relay at each iteration.

    At each hop:
      1. Word-to-word: transfer equation on semantic edges
      2. Word-to-concept: aggregate word activations into concept nodes
      3. Concept-to-word: broadcast concept activations back to member words
    This lets signals travel through shared concepts even without direct edges.
    """
    transfer = get_transfer(cfg.transfer_method)
    if node_stats is None and cfg.transfer_method == "threshold_auto":
        node_stats = compute_node_thresholds(sp_mat)

    activation = seed_activation.clone()
    sp = sp_mat.coalesce()
    idx = sp.indices()
    val = sp.values()
    a = idx[0]
    b = idx[1]

    relay_strength = getattr(cfg, "spreading_concept_relay", 0.5)
    has_concepts = concepts is not None and concepts.n_concepts > 0

    for _ in range(cfg.spreading_iters):
        # Level 1: word -> word
        in_sig = activation[a]
        out_sig = transfer(in_sig, val, cfg, node_stats)
        new_act = torch.zeros_like(activation)
        new_act.scatter_add_(0, b, out_sig)

        # Level 2: word -> concept -> word
        if has_concepts:
            concept_act = concepts.word_to_concept_activation(activation)
            relay = concepts.concept_to_word_injection(concept_act, strength=relay_strength)
            new_act += relay

        activation = cfg.spreading_damping * new_act + (1 - cfg.spreading_damping) * seed_activation
    return activation
