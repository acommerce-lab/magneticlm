# magnetic/physics.py
#
# Force-directed physics simulation that turns the semantic edge set
# into a dense D-dimensional position embedding. One call to run()
# takes a ready EdgeSet and returns (positions, importance, velocities).
#
# Faithful port of the C# WordGraph.RunPhysicsSimulation logic, with
# two extensions over both C# and the existing GPU runner:
#
#   1. Configurable dim. The C# init was uniform in [-5, 5] for a
#      3-dim space. Here the init range is scaled so E[||x||] stays
#      ~5 regardless of dim, which means max_radius, optimal_dist,
#      gravity, spring and repulsion constants do NOT need per-dim
#      retuning.
#
#   2. Chunked spring force. Without chunking, D=64 with 10M edges
#      OOMs on a 14 GB T4 (the fvec = unit * fmag broadcast allocates
#      2.37 GB per tensor). The spring loop here processes edges in
#      chunks of config.physics_edge_chunk (default 1M) to keep peak
#      memory bounded.

import math
import time
from typing import Tuple

import torch


class PhysicsSimulator:
    def __init__(self, config):
        self.config = config

    def run(
        self,
        num_nodes: int,
        edge_from: torch.Tensor,
        edge_to: torch.Tensor,
        edge_weight: torch.Tensor,
        device: torch.device,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the physics simulation.

        Returns (positions, velocities_final). Both are (N, dim)
        float32 tensors on `device`. Velocities are returned so the
        caller can snapshot the convergence state for diagnostics, but
        they are not usually needed downstream.
        """
        cfg = self.config
        N = int(num_nodes)
        dim = int(cfg.dim)
        E = int(edge_from.numel())

        if N == 0:
            return (
                torch.empty((0, dim), dtype=torch.float32, device=device),
                torch.empty((0, dim), dtype=torch.float32, device=device),
            )

        torch.manual_seed(cfg.seed)

        # Scale the initial range so E[||x||] stays ~5 regardless of dim.
        # Uniform in [-a, a] has var = a^2/3 per coord, so
        # E[||x||^2] = D*a^2/3. Setting this to 25 keeps the initial
        # norm comparable to the D=3 baseline a=5.
        init_range = 5.0 * math.sqrt(3.0 / float(dim))
        positions = (torch.rand((N, dim), device=device, dtype=torch.float32)
                     * (2.0 * init_range) - init_range)
        velocities = torch.zeros((N, dim), device=device, dtype=torch.float32)

        K_context  = float(cfg.K_context)
        K_frequency = float(cfg.K_frequency)
        K_attraction = float(cfg.K_attraction)
        K_repulsion = float(cfg.K_repulsion)
        damping    = float(cfg.damping)
        lr         = float(cfg.physics_lr)
        optimal_dist = float(cfg.optimal_dist)
        max_radius = float(cfg.max_radius)
        sample_size = min(N, int(cfg.physics_sample_size))
        edge_chunk = int(cfg.physics_edge_chunk)

        if verbose:
            print("  Physics: %d iters, dim=%d, N=%d, E=%d on %s" %
                  (cfg.physics_iters, dim, N, E, device),
                  end="", flush=True)

        t0 = time.time()
        for it in range(cfg.physics_iters):
            forces = torch.zeros_like(positions)

            # ---- 1. Spring forces along semantic edges (chunked) ----
            if E > 0:
                for cs in range(0, E, edge_chunk):
                    ce = min(cs + edge_chunk, E)
                    ef = edge_from[cs:ce]
                    et = edge_to[cs:ce]
                    ew = edge_weight[cs:ce]
                    pf = positions[ef]
                    pt = positions[et]
                    diff = pt - pf
                    dist = diff.norm(dim=1, keepdim=True).clamp_min(0.1)
                    unit = diff / dist
                    k_tensor = torch.where(
                        ew > 1.0,
                        torch.full_like(ew, K_context),
                        torch.full_like(ew, K_frequency))
                    fmag = k_tensor * ew / dist.squeeze(1)
                    fvec = unit * fmag.unsqueeze(1)
                    forces.index_add_(0, ef, fvec)
                    del pf, pt, diff, dist, unit, k_tensor, fmag, fvec

            # ---- 2. Sampled repulsion + far-field attraction ----
            if N > sample_size:
                sample_idx = torch.randperm(N, device=device)[:sample_size]
            else:
                sample_idx = torch.arange(N, device=device)
            sample_pos = positions[sample_idx]

            bsz = 4096
            for bs in range(0, N, bsz):
                be = min(bs + bsz, N)
                tp = positions[bs:be]
                d = sample_pos.unsqueeze(0) - tp.unsqueeze(1)
                di = d.norm(dim=2).clamp_min(0.1)
                u = d / di.unsqueeze(2)
                rep = -K_repulsion / (di * di + 1.0)
                rep_vec = u * rep.unsqueeze(2)
                forces[bs:be] += rep_vec.sum(dim=1)
                beyond = (di > optimal_dist)
                att = torch.where(
                    beyond,
                    K_attraction * (di - optimal_dist) * 0.01,
                    torch.zeros_like(di))
                att_vec = u * att.unsqueeze(2)
                forces[bs:be] += att_vec.sum(dim=1)

            # ---- 3. Gravity toward origin + integration ----
            forces.sub_(0.01 * positions)
            velocities = (velocities + forces * lr) * (1.0 - damping)
            positions.add_(velocities * lr)

            # ---- 4. Boundary clamp ----
            mag = positions.norm(dim=1)
            overflow = mag > max_radius
            if overflow.any():
                scale = (max_radius / mag[overflow]).unsqueeze(1)
                positions[overflow] = positions[overflow] * scale
                velocities[overflow] = velocities[overflow] * 0.5

            if verbose and (it + 1) % 5 == 0:
                print(".", end="", flush=True)

        if verbose:
            print(" done (%.0fs)" % (time.time() - t0))

        return positions, velocities


def compute_importance(
    num_nodes: int,
    edge_from: torch.Tensor,
    freq_gpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Node importance = log(1 + degree) * log(1 + freq). Used by the
    generator as a confidence weight on the semantic force."""
    degs = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    if edge_from.numel() > 0:
        degs.index_add_(
            0, edge_from,
            torch.ones(edge_from.numel(), dtype=torch.float32, device=device))
    return torch.log1p(degs) * torch.log1p(freq_gpu.float())
