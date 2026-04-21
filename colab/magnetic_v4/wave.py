"""Complex wave-field propagation engine.

Single responsibility: given a graph and an initial impulse, evolve the
complex field z = a + i·b over T steps using sparse mat-vec ops.

Semantics:
  - Re(z_v)  = contextual (syntactic) activation
  - Im(z_v)  = conceptual (semantic) activation

Per step:
  real_new = fwd_adj · Re(z)                      # context flows forward
  imag_new = bwd_adj · Im(z)                      # concept flows backward
  # Deep reflection converts real→imaginary and back (creative-leap channel)
  reflected = ρ · conj(z)                         # reflection coefficient
  z_new = (real_new + i·imag_new) + reflected
  z_new *= damping                                 # stability
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .graph import Graph


@dataclass
class WaveResult:
    """Final complex field after propagation."""
    z: torch.Tensor           # complex [V]
    real_per_step: Optional[torch.Tensor] = None   # [T, V] diagnostic
    imag_per_step: Optional[torch.Tensor] = None   # [T, V] diagnostic


def _sparse_mv(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse matrix-vector multiply, CSR-friendly."""
    return torch.sparse.mm(adj, x.unsqueeze(1)).squeeze(1)


def propagate(
    graph: Graph,
    impulse: torch.Tensor,
    cfg,
    device: torch.device,
    track_history: bool = False,
) -> WaveResult:
    """Run T steps of wave propagation starting from a real-valued impulse.

    Args:
      graph: fwd/bwd sparse adjacency
      impulse: [V] real-valued initial seed (e.g. one-hot at current token)
      cfg: Config (reads wave_iters, wave_damping, reflection_coef,
           reflection_depth)
      device: target device
      track_history: if True, also return per-step real/imag for diagnostics
    """
    V = graph.vocab_size
    T = int(cfg.wave_iters)
    damp = float(cfg.wave_damping)
    rho = float(cfg.reflection_coef)
    depth = int(cfg.reflection_depth)

    # Start purely real
    re = impulse.to(device).to(torch.float32).clone()
    im = torch.zeros_like(re)

    hist_re = []
    hist_im = []

    for t in range(T):
        # Forward pass of the real (contextual) signal
        re_in = _sparse_mv(graph.fwd_adj, re)
        # Backward pass of the imaginary (conceptual) signal
        im_in = _sparse_mv(graph.bwd_adj, im)

        # Reflection kicks in once the wave has propagated into the "deep"
        # (after `depth` steps). conj(z) = re - i·im, and multiplying by ρ
        # converts some real signal back into imaginary at the current node
        # — this is the "creative leap" channel.
        if t >= depth and rho > 0.0:
            # Rotate 90°: new imaginary gets fed by the arriving real,
            # and vice-versa (symmetric reflection).
            re_in = re_in + rho * im           # im→re feedback
            im_in = im_in + rho * re           # re→im feedback

        re = damp * re_in
        im = damp * im_in

        if track_history:
            hist_re.append(re.detach().cpu())
            hist_im.append(im.detach().cpu())

    z = torch.complex(re, im)
    return WaveResult(
        z=z,
        real_per_step=torch.stack(hist_re) if track_history else None,
        imag_per_step=torch.stack(hist_im) if track_history else None,
    )


def propagate_batch(
    graph: Graph,
    impulses: torch.Tensor,         # [B, V] stacked initial seeds
    cfg,
    device: torch.device,
) -> torch.Tensor:
    """Batched propagation: compute T steps for B impulses in parallel.

    Returns z as complex [B, V]. Sparse @ dense [V, B] handles all seeds at
    once — ideal for batched PPL / hit-rate evaluation.
    """
    V = graph.vocab_size
    T = int(cfg.wave_iters)
    damp = float(cfg.wave_damping)
    rho = float(cfg.reflection_coef)
    depth = int(cfg.reflection_depth)

    # Ensure shape [V, B] for sparse @ dense
    re = impulses.to(device).to(torch.float32).transpose(0, 1).contiguous()   # [V, B]
    im = torch.zeros_like(re)

    for t in range(T):
        re_in = torch.sparse.mm(graph.fwd_adj, re)
        im_in = torch.sparse.mm(graph.bwd_adj, im)
        if t >= depth and rho > 0.0:
            re_in = re_in + rho * im
            im_in = im_in + rho * re
        re = damp * re_in
        im = damp * im_in

    # back to [B, V] complex
    z = torch.complex(re.transpose(0, 1).contiguous(),
                      im.transpose(0, 1).contiguous())
    return z
