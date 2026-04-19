"""CLI runner for magnetic_v3.

Handles flat Colab/Kaggle layouts: tries sibling-import first, then
falls back to absolute import of the package folder.

Examples:
    python run.py --max_train_lines 10000 --max_valid_lines 500
    python run.py --capacity_method log_inv_freq --active_forces spring,decay
    python run.py --scoring_method stats_only
    python run.py --eval_generation true --gen_samples 5
    python run.py --use_concepts false  # disable concept layer
    python run.py --glow_threshold 0.5 --glow_strength 0.2
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# ---- flat-layout import fallback -------------------------------------------
try:
    from magnetic_v3 import (
        Config, config_from_args, add_cli_args,
        detect_resources,
        build_vocab, encode_stream, load_dataset,
        build_statistics,
        ContextualMap,
        make_empty as make_semantic_empty,
        init_from_ppmi,
        discover_concepts,
        train_parallel,
        InferenceEngine,
        run_full_eval,
    )
except ImportError:
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from magnetic_v3 import (
        Config, config_from_args, add_cli_args,
        detect_resources,
        build_vocab, encode_stream, load_dataset,
        build_statistics,
        ContextualMap,
        make_empty as make_semantic_empty,
        init_from_ppmi,
        discover_concepts,
        train_parallel,
        InferenceEngine,
        run_full_eval,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="magnetic_v3")
    parser = add_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)

    print("=" * 70)
    print("magnetic_v3")
    print("=" * 70)

    resources = detect_resources(cfg)
    print(f"Resources: {resources}")

    print("Config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k} = {v}")
    print("-" * 70)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---- data ----
    print("Loading dataset...")
    train_lines, valid_lines = load_dataset(cfg)
    print(f"  train={len(train_lines)} lines  valid={len(valid_lines)} lines")

    print("Building vocabulary...")
    t0 = time.time()
    vocab = build_vocab(
        train_lines,
        max_vocab=cfg.max_vocab,
        min_count=cfg.min_count,
        unk=cfg.unk_token,
    )
    print(f"  vocab_size={vocab.size}  built in {time.time()-t0:.1f}s")

    print("Encoding streams...")
    t0 = time.time()
    encoded_train = encode_stream(train_lines, vocab)
    encoded_valid = encode_stream(valid_lines, vocab)
    print(f"  encoded in {time.time()-t0:.1f}s")

    # ---- statistical layer ----
    print("Building statistical layer...")
    t0 = time.time()
    stats = build_statistics(encoded_train, vocab.size, cfg, resources.primary_device)
    print(
        f"  stats built in {time.time()-t0:.1f}s  "
        f"k_base={stats.k_base:.1f}  "
        f"ppmi_edges={stats.ppmi_values.numel()}  "
        f"cap_mean={stats.capacity.float().mean().item():.1f}  "
        f"cap_min={stats.capacity.min().item()}  "
        f"cap_max={stats.capacity.max().item()}"
    )

    # ---- concept layer ----
    concepts = None
    if cfg.use_concepts:
        print("Discovering concepts...")
        t0 = time.time()
        concepts = discover_concepts(
            stats.ppmi_indices[0], stats.ppmi_indices[1], stats.ppmi_values,
            vocab.size, cfg, resources.primary_device,
        )
        n_with = (concepts.primary_concept >= 0).sum().item()
        print(
            f"  concepts={concepts.n_concepts}  "
            f"words_with_concept={n_with}/{vocab.size}  "
            f"membership_entries={concepts.mem_word.numel()}  "
            f"in {time.time()-t0:.1f}s"
        )
    else:
        print("Concept layer disabled (--use_concepts false)")

    # ---- contextual map ----
    print("Building contextual map...")
    t0 = time.time()
    ctx = ContextualMap.build(
        encoded_train, vocab.size, cfg.max_ctx_children, resources.primary_device
    )
    print(f"  ctx edges={ctx.col_idx.numel()}  built in {time.time()-t0:.1f}s")

    # ---- semantic map ----
    print("Initializing semantic map...")
    t0 = time.time()
    sem = make_semantic_empty(vocab.size, stats.capacity)
    if cfg.init_from_ppmi:
        init_from_ppmi(sem, stats.ppmi_indices[0], stats.ppmi_indices[1], stats.ppmi_values)
    print(f"  init edges={len(sem.edges)}  in {time.time()-t0:.1f}s")

    # ---- training ----
    if cfg.run_training:
        print("Pulse training (parallel)...")
        train_parallel(sem, encoded_train, cfg, resources, log_every=5)

    # ---- evaluation ----
    if cfg.run_eval:
        print("Preparing inference engine...")
        engine = InferenceEngine(
            ctx=ctx, sem=sem, stats=stats, concepts=concepts,
            cfg=cfg, device=resources.primary_device,
        )
        engine.prepare()

        print("Running evaluation...")
        results = run_full_eval(engine, encoded_valid, vocab, cfg)

        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    kk: (vv if not hasattr(vv, "tolist") else vv.tolist())
                    for kk, vv in v.items()
                    if kk != "ood_details"
                }
            else:
                serializable[k] = v
        out_path = os.path.join(cfg.save_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved results -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
