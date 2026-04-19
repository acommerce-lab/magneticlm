"""Parallel pulse training for the semantic map.

For each sentence:
  1. For each token position t, form a window (t-W, t+W)
  2. Collect (a,b) pulse events for all in-window pairs
  3. Apply force integration to update edge weights
  4. Respect node capacity when creating new edges

Parallelization strategy:
  - CPU workers produce "delta dicts" per sentence chunk
  - Main process merges deltas into the semantic map
  - For larger corpora we batch multiple chunks into a single merge

This keeps contention low (merge is sequential) while distributing
the heavy collection+force computation across cores.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .forces import compose, FORCES
from .semantic_map import SemanticMapState, merge_delta, _pair_key


def _compute_chunk(
    encoded_chunk: List[np.ndarray],
    V: int,
    window: int,
    force_names: List[str],
    force_cfg: dict,
    capacity_cpu: np.ndarray,
) -> Dict[int, float]:
    """Worker: compute pair deltas for one chunk of sentences.

    Returns dict[pair_key] -> delta_weight (no velocity persistence).
    """

    # Rebuild a tiny config shim for forces
    class Cfg:
        pass

    cfg = Cfg()
    for k, v in force_cfg.items():
        setattr(cfg, k, v)

    fns = [FORCES[n] for n in force_names if n in FORCES]

    deltas: Dict[int, float] = {}
    for arr in encoded_chunk:
        if arr.size < 2:
            continue
        n = arr.size
        for i in range(n):
            a = int(arr[i])
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                b = int(arr[j])
                if a == b:
                    continue
                key = _pair_key(a, b, V)
                cur = deltas.get(key, 0.0)

                # apply forces with an event=1 pulse
                w = torch.tensor([cur], dtype=torch.float32)
                v = torch.tensor([0.0], dtype=torch.float32)
                state = {"event": torch.tensor([1.0], dtype=torch.float32)}
                net = torch.zeros_like(w)
                for f in fns:
                    net = net + f(w, state, cfg)
                v = (v + net) * (1.0 - cfg.damping)
                deltas[key] = float(cur + cfg.force_lr * v.item())
    return deltas


def _chunkify(data: List[np.ndarray], n_chunks: int) -> List[List[np.ndarray]]:
    if n_chunks <= 1:
        return [data]
    sizes = [a.size for a in data]
    total = sum(sizes)
    target = total / n_chunks
    chunks: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    cur_sum = 0
    for arr in data:
        cur.append(arr)
        cur_sum += arr.size
        if cur_sum >= target:
            chunks.append(cur)
            cur = []
            cur_sum = 0
    if cur:
        chunks.append(cur)
    return chunks


def _force_cfg_dict(cfg) -> dict:
    return {
        "K_spring": cfg.K_spring,
        "K_decay": cfg.K_decay,
        "damping": cfg.damping,
        "force_lr": cfg.force_lr,
    }


def train_parallel(
    state: SemanticMapState,
    encoded: List[np.ndarray],
    cfg,
    resources,
    log_every: int = 10,
):
    """Train in parallel; merge into `state` sequentially in main process."""
    force_names = cfg.force_list()
    force_cfg = _force_cfg_dict(cfg)

    batch_size = max(1, cfg.pulse_batch)
    workers = resources.num_cpus if cfg.pulse_workers < 0 else min(cfg.pulse_workers, resources.num_cpus)
    workers = max(1, workers)

    total_sentences = len(encoded)
    processed = 0
    start = time.time()

    for batch_start in range(0, total_sentences, batch_size):
        batch = encoded[batch_start : batch_start + batch_size]
        chunks = _chunkify(batch, workers)

        if workers == 1 or len(chunks) == 1:
            deltas_list = [
                _compute_chunk(c, state.vocab_size, cfg.semantic_window, force_names, force_cfg, state.capacity)
                for c in chunks
            ]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        _compute_chunk,
                        c,
                        state.vocab_size,
                        cfg.semantic_window,
                        force_names,
                        force_cfg,
                        state.capacity,
                    )
                    for c in chunks
                ]
                deltas_list = [f.result() for f in futures]

        # Merge
        merged: Dict[int, float] = {}
        for d in deltas_list:
            for k, v in d.items():
                merged[k] = merged.get(k, 0.0) + v
        merge_delta(state, merged)

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
