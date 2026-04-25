"""Spectral decomposition: PPMI -> SVD -> embeddings + cone structure."""

import math, time
from typing import Dict, List, Tuple
import numpy as np
import torch


def build_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                   V, min_ppmi, device, var_target=0.3):
    """Build embeddings and cone structure from PPMI via SVD.

    Returns:
        embeddings: [V, d] word embeddings
        d_schedule: list of dimensions per cone level
        S_levels: [L] spectral weight per level
        d: auto-detected embedding dimension
    """
    t0 = time.time()

    # PPMI
    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    PPMI = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()
    PPMI_dense = PPMI.to_dense()

    # SVD
    if V <= 5000:
        print(f"    full SVD on [{V}x{V}] PPMI...")
        U_full, S_full, _ = torch.linalg.svd(PPMI_dense, full_matrices=False)
    else:
        q = min(V - 1, 4096)
        print(f"    randomized SVD on [{V}x{V}] PPMI, q={q}...")
        U_full, S_full, _ = torch.svd_lowrank(PPMI_dense, q=q, niter=5)

    # Cumulative variance threshold
    S_sq = S_full ** 2
    cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum().clamp(min=1e-9)
    d = int((cumvar < var_target).sum().item()) + 1
    d = max(4, min(d, len(S_full)))
    print(f"    cumvar: {cumvar[d-1]:.1%} at d={d} (target={var_target:.0%})")
    print(f"    top-8 S: {', '.join(f'{s:.1f}' for s in S_full[:8].tolist())}")

    # Embeddings
    embeddings = (U_full[:, :d] * S_full[:d].sqrt().unsqueeze(0)).to(device)

    # Cone schedule: d, d//2, d//4, ...
    d_schedule = [d]
    dl = d
    while dl > 4:
        dl = dl // 2
        if dl >= 4:
            d_schedule.append(dl)

    # Spectral weight per level (from SVD energy in that range)
    S_levels = []
    for l, dl in enumerate(d_schedule):
        dl_prev = d_schedule[l - 1] if l > 0 else d
        energy = S_full[:dl].sum().item()
        S_levels.append(energy)
    S_levels_t = torch.tensor(S_levels, dtype=torch.float32, device=device)
    S_levels_t = S_levels_t / S_levels_t.sum().clamp(min=1e-9)

    print(f"    cone: {' -> '.join(str(dl) for dl in d_schedule)}")
    print(f"    level weights: {', '.join(f'{w:.3f}' for w in S_levels_t.tolist())}")
    print(f"    built in {time.time()-t0:.1f}s")

    return embeddings, d_schedule, S_levels_t, d
