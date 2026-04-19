"""Pluggable force registry.

Each force is a callable:
    f(weight, state, cfg) -> delta_velocity_contribution

State is a dict carrying per-edge info (age, last_pulse, degree_a, degree_b, ...).
Forces compute contributions to velocity; the trainer integrates them.

Add new forces by calling register_force(name)(fn). No existing file needs edit.
"""

from typing import Callable, Dict, List

import torch


FORCES: Dict[str, Callable] = {}


def register_force(name: str):
    def deco(fn):
        FORCES[name] = fn
        return fn
    return deco


@register_force("spring")
def spring(weight: torch.Tensor, state: dict, cfg) -> torch.Tensor:
    """Attraction that weakens as weight grows. Only on co-occurrence events."""
    event = state.get("event", None)  # +1 for pulse event, 0 otherwise
    K = cfg.K_spring
    f = K / (1.0 + torch.abs(weight))
    if event is not None:
        f = f * event
    return f


@register_force("decay")
def decay(weight: torch.Tensor, state: dict, cfg) -> torch.Tensor:
    """Time-based decay of all edges (pulls toward zero each step)."""
    K = cfg.K_decay
    return -K * weight


@register_force("damping")
def damping(weight: torch.Tensor, state: dict, cfg) -> torch.Tensor:
    """Not a force — handled inside integrator. Kept for registry completeness."""
    return torch.zeros_like(weight)


def compose(active_names: List[str]) -> List[Callable]:
    return [FORCES[n] for n in active_names if n in FORCES]


def integrate(
    weight: torch.Tensor,
    velocity: torch.Tensor,
    forces: List[Callable],
    state: dict,
    cfg,
) -> (torch.Tensor, torch.Tensor):
    net = torch.zeros_like(weight)
    for f in forces:
        net = net + f(weight, state, cfg)
    velocity = (velocity + net) * (1.0 - cfg.damping)
    weight = weight + cfg.force_lr * velocity
    return weight, velocity
