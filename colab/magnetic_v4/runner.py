"""Pipeline orchestrator: wires data → stats → graph → eval.

Single responsibility: call modules in order; report progress & memory.
No domain logic — each step is owned by its own module.
"""

import gc
import json
import os
import time
from typing import Dict

import numpy as np
import torch

from .config import Config
from .data import load_dataset
from .evaluator import run_full_eval
from .graph import build_graph, graph_info
from .resources import Monitor, Resources, detect, setup_cuda_tuning
from .stats import Stats, build_stats
from .tokenizer import build_vocab, encode_stream


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v4 — wave physics")
    print("=" * 72)

    setup_cuda_tuning()
    resources = detect(cfg)
    print(f"Resources: {resources}")

    # Echo config
    print("Config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k} = {v}")
    print("-" * 72)

    monitor = Monitor(cfg)
    monitor.snapshot("startup")

    # ---------- Load data ----------
    print("Loading dataset...")
    t0 = time.time()
    train_lines, valid_lines = load_dataset(cfg)
    print(f"  train={len(train_lines)} lines  valid={len(valid_lines)} lines  ({time.time()-t0:.1f}s)")

    # ---------- Vocab ----------
    print("Building vocabulary...")
    t0 = time.time()
    vocab = build_vocab(train_lines, max_vocab=cfg.max_vocab, min_count=cfg.min_count)
    print(f"  vocab_size={vocab.size}  ({time.time()-t0:.1f}s)")

    # ---------- Encode ----------
    print("Encoding streams...")
    t0 = time.time()
    encoded_train = encode_stream(train_lines, vocab)
    encoded_valid = encode_stream(valid_lines, vocab)
    del train_lines
    del valid_lines
    gc.collect()
    print(f"  encoded in {time.time()-t0:.1f}s")
    monitor.snapshot("after-encode")

    # ---------- Stats (co-occurrences, directional counts) ----------
    print("Building statistics...")
    t0 = time.time()
    stats = build_stats(encoded_train, vocab.size, cfg, resources.primary_device)
    n_ctx = int(stats.ctx_counts.numel())
    n_bg = int(stats.bg_counts.numel())
    print(f"  stats built in {time.time()-t0:.1f}s  "
          f"ctx_pairs={n_ctx:,}  bg_pairs={n_bg:,}")
    monitor.snapshot("after-stats")

    # ---------- Graph ----------
    print("Building wave graph (directional)...")
    t0 = time.time()
    graph = build_graph(stats, cfg)
    print(f"  {graph_info(graph)}  ({time.time()-t0:.1f}s)")
    # Stats no longer needed in memory after graph built
    del stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    monitor.snapshot("after-graph")

    # ---------- Evaluation ----------
    print("Running evaluation...")
    t0 = time.time()
    results = run_full_eval(graph, encoded_valid, vocab, cfg, resources.primary_device)
    print(f"  eval done in {time.time()-t0:.1f}s")
    monitor.snapshot("post-eval")

    # ---------- Save ----------
    peak_gpu = 0.0
    if torch.cuda.is_available():
        peak_gpu = max(
            (torch.cuda.max_memory_allocated(i) / 1e9
             for i in range(torch.cuda.device_count())),
            default=0.0,
        )
    results["peak_gpu_alloc_gb"] = peak_gpu
    results["peak_rss_gb"] = monitor.peak_rss_gb

    os.makedirs(cfg.save_dir, exist_ok=True)
    out_path = os.path.join(cfg.save_dir, "v4_results.json")
    with open(out_path, "w") as f:
        def _coerce(v):
            if isinstance(v, torch.Tensor):
                return v.tolist()
            return v
        json.dump(
            {k: (v if not hasattr(v, "tolist") else v.tolist())
             for k, v in results.items()},
            f, indent=2, default=str,
        )
    print(f"Saved -> {out_path}")
    print(f"Peak GPU alloc: {peak_gpu:.2f} GB  Peak RSS: {monitor.peak_rss_gb:.2f} GB")
    print("Done.")
    return results
