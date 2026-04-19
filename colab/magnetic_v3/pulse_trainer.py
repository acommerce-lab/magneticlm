"""GPU-accelerated pulse training for the semantic map.

Strategy:
  1. Collect all (a,b) co-occurrence pairs from a sentence batch
     using vectorized NumPy slicing (no per-token Python loop).
  2. Unique-ify pairs and count events on CPU (np.unique).
  3. Simulate force integration on GPU in a tight vectorized loop
     (one iteration per max-event-count, each iteration processes
     ALL pairs in parallel).
  4. Merge deltas into the semantic map (sequential, capacity-aware).

This replaces the old approach of:
  - CPU ProcessPoolExecutor workers that each created ~100K scalar
    torch.tensor objects and called force functions per-pair.
  - IPC serialization of large encoded arrays.
  - Per-pair .item() GPU→CPU sync.

At 10K lines / window=2 / ~50K unique pairs per batch:
  Old: 190s (800K scalar tensor allocations + Python loops)
  New: GPU loop of ~5-20 iterations over 50K pairs (< 1s on T4)
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from .semantic_map import SemanticMapState, merge_delta


def _collect_pair_events(
    encoded_batch: List[np.ndarray],
    V: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pair collection using NumPy array slicing.

    For each sentence, for each window offset d, extracts all (a,b) pairs
    as a[:-d] / a[d:] in one operation — no per-token Python loop.

    Returns (unique_pair_keys, event_counts).
    """
    all_keys: List[np.ndarray] = []
    for arr in encoded_batch:
        n = arr.size
        if n < 2:
            continue
        t = arr.astype(np.int64)
        for d in range(1, window + 1):
            if d >= n:
                break
            src = t[:-d]
            tgt = t[d:]
            mask = src != tgt
            s = src[mask]
            g = tgt[mask]
            all_keys.append(s * V + g)
            all_keys.append(g * V + s)

    if not all_keys:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    concatenated = np.concatenate(all_keys)
    unique_keys, counts = np.unique(concatenated, return_counts=True)
    return unique_keys, counts.astype(np.int64)


def _gpu_force_batch(
    unique_keys: np.ndarray,
    counts: np.ndarray,
    K_spring: float,
    K_decay: float,
    damping: float,
    force_lr: float,
    device: torch.device,
) -> Dict[int, float]:
    """Simulate sequential force integration on GPU, vectorized over all pairs.

    Preserves the original per-event behavior: each co-occurrence event
    applies spring + decay forces to the running delta.  The loop runs
    max(counts) iterations; each iteration processes ALL pairs whose
    remaining event count > 0 in one vectorized GPU operation.

    Typical: max(counts) ≈ 5-20 at 10K lines, with 50K-100K pairs
    processed per iteration → sub-second on T4.
    """
    n_pairs = len(unique_keys)
    if n_pairs == 0:
        return {}

    n_events = torch.from_numpy(counts.astype(np.int32)).to(device)
    delta = torch.zeros(n_pairs, dtype=torch.float32, device=device)
    remaining = n_events.clone()
    max_n = int(remaining.max().item())

    damp_factor = 1.0 - damping
    for _ in range(max_n):
        active = remaining > 0
        if not active.any():
            break
        w = delta[active]
        net = K_spring / (1.0 + torch.abs(w)) - K_decay * w
        delta[active] += force_lr * net * damp_factor
        remaining[active] -= 1

    delta_np = delta.cpu().numpy()
    return {int(k): float(d) for k, d in zip(unique_keys, delta_np) if abs(d) > 1e-12}


def train_parallel(
    state: SemanticMapState,
    encoded: List[np.ndarray],
    cfg,
    resources,
    log_every: int = 10,
):
    """GPU-accelerated pulse training."""
    device = resources.primary_device
    batch_size = max(1, cfg.pulse_batch)
    total_sentences = len(encoded)
    processed = 0
    start = time.time()

    V = state.vocab_size
    window = cfg.semantic_window
    K_spring = cfg.K_spring
    K_decay = cfg.K_decay
    damping_val = cfg.damping
    force_lr = cfg.force_lr

    for batch_start in range(0, total_sentences, batch_size):
        batch = encoded[batch_start : batch_start + batch_size]

        unique_keys, counts = _collect_pair_events(batch, V, window)

        deltas = _gpu_force_batch(
            unique_keys, counts,
            K_spring, K_decay, damping_val, force_lr,
            device,
        )

        merge_delta(state, deltas)

        processed += len(batch)
        if (batch_start // batch_size) % log_every == 0:
            elapsed = time.time() - start
            rate = processed / max(elapsed, 1e-6)
            eta = (total_sentences - processed) / max(rate, 1e-6)
            print(
                f"  [pulse] {processed}/{total_sentences} "
                f"({100*processed/total_sentences:.1f}%) "
                f"rate={rate:.1f} sent/s  eta={eta:.1f}s  edges={len(state.edges)}"
            )

    elapsed = time.time() - start
    print(f"  [pulse] done in {elapsed:.1f}s  edges={len(state.edges)}")
