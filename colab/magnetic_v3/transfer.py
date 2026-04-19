"""Transfer equation for propagating activation through edges.

Given incoming signal s_in, edge weight w, and node state,
return an outgoing signal s_out.

Binary (threshold) transfer is the default: each edge decides
to transmit +|w|, -|w| or 0. This implements the user's request
that edges hold the activation/inhibition decision, not nodes.
"""

from typing import Callable, Dict

import torch


TRANSFERS: Dict[str, Callable] = {}


def register_transfer(name: str):
    def deco(fn):
        TRANSFERS[name] = fn
        return fn
    return deco


@register_transfer("threshold_fixed")
def threshold_fixed(signal: torch.Tensor, weights: torch.Tensor, cfg, node_stats=None):
    pos = (weights > cfg.theta_positive).to(signal.dtype)
    neg = (weights < cfg.theta_negative).to(signal.dtype)
    return signal * (pos * weights.clamp(min=0) - neg * weights.clamp(max=0).abs())


@register_transfer("threshold_auto")
def threshold_auto(signal: torch.Tensor, weights: torch.Tensor, cfg, node_stats=None):
    """Per-node mean +/- std thresholds."""
    if node_stats is None or "mu" not in node_stats:
        return threshold_fixed(signal, weights, cfg)
    mu = node_stats["mu"]
    sd = node_stats["sd"]
    theta_p = mu + sd
    theta_n = mu - sd
    pos = (weights > theta_p).to(signal.dtype)
    neg = (weights < theta_n).to(signal.dtype)
    return signal * (pos * weights.clamp(min=0) - neg * weights.clamp(max=0).abs())


@register_transfer("sigmoid")
def sigmoid(signal: torch.Tensor, weights: torch.Tensor, cfg, node_stats=None):
    gate = torch.tanh(weights)
    return signal * gate


@register_transfer("linear")
def linear(signal: torch.Tensor, weights: torch.Tensor, cfg, node_stats=None):
    return signal * weights


def get_transfer(name: str) -> Callable:
    return TRANSFERS.get(name, TRANSFERS["threshold_auto"])
