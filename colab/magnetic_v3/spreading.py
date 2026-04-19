"""Spreading activation with binary transfer equation.

At each hop, the transfer equation decides per-edge whether to propagate
+signal, -signal or 0 based on thresholds. Node output is the signed sum
of incoming signals post-damping.
"""

from typing import Dict, Optional

import torch

from .transfer import get_transfer


def compute_node_thresholds(sp_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Mean and std of outgoing edge weights per row."""
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
    """
    seed_activation: [V] initial activation vector (usually sparse).
    sp_mat: [V, V] sparse edge weights.
    Returns: [V] final activation.
    """
    transfer = get_transfer(cfg.transfer_method)
    if node_stats is None and cfg.transfer_method == "threshold_auto":
        node_stats = compute_node_thresholds(sp_mat)

    activation = seed_activation.clone()
    for _ in range(cfg.spreading_iters):
        # propagate: each edge (a,b) carries signal activation[a] through transfer(w)
        sp = sp_mat.coalesce()
        idx = sp.indices()
        val = sp.values()
        a = idx[0]
        b = idx[1]
        in_sig = activation[a]
        out_sig = transfer(in_sig, val, cfg, node_stats)
        # accumulate at b
        new_act = torch.zeros_like(activation)
        new_act.scatter_add_(0, b, out_sig)
        activation = cfg.spreading_damping * new_act + (1 - cfg.spreading_damping) * seed_activation
    return activation
