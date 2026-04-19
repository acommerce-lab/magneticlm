"""CLI runner for magnetic_v3.

Handles flat layouts with any folder name (Colab/Kaggle):
  - If the files live inside a folder named `magnetic_v3/`, regular import works.
  - Otherwise (e.g. `/kaggle/input/.../codev3/`), we bootstrap `magnetic_v3`
    as an in-memory package pointing to the current directory so that
    relative imports like `from .concepts import X` keep working.

Examples:
    python run.py --max_train_lines 10000 --max_valid_lines 500
    python run.py --capacity_method log_inv_freq --active_forces spring,decay
    python run.py --scoring_method stats_only
    python run.py --eval_generation true --gen_samples 5
    python run.py --use_concepts false
    python run.py --glow_threshold 0.5 --glow_strength 0.2
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time

# -- Bootstrap the magnetic_v3 package regardless of folder name --------------

def _bootstrap_package():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.dont_write_bytecode = True  # many Kaggle inputs are read-only

    # Case 1: importable as-is (installed, or parent dir on path, etc.)
    try:
        import magnetic_v3  # noqa: F401
        return
    except ImportError:
        pass

    # Case 2: we're inside a folder called `magnetic_v3/` — add parent to path
    if os.path.basename(here) == "magnetic_v3":
        parent = os.path.dirname(here)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        import magnetic_v3  # noqa: F401
        return

    # Case 3: flat layout with arbitrary folder name. Create an in-memory
    # `magnetic_v3` package whose submodule search path is the current dir.
    init_path = os.path.join(here, "__init__.py")
    if not os.path.exists(init_path):
        raise ImportError(
            f"Cannot bootstrap magnetic_v3: __init__.py missing in {here}"
        )
    spec = importlib.util.spec_from_file_location(
        "magnetic_v3",
        init_path,
        submodule_search_locations=[here],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["magnetic_v3"] = module
    spec.loader.exec_module(module)


_bootstrap_package()

import numpy as np
import torch

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
            unk_id=vocab.unk_id,
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
