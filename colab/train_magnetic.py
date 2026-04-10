#!/usr/bin/env python3
# train_magnetic.py
#
# Train a MagneticLM instance end-to-end and report WT103 full-mode
# perplexity. Uses the modular `magnetic` package; does NOT touch
# MagneticLMFastRunner.py. Use this as the main entry point for
# training-and-evaluation runs.
#
# Example usage:
#
#   # Single T4, dim=16, PMI on (new default), 100 physics iters
#   python train_magnetic.py --train-lines 100000 --physics-iters 100
#
#   # Dual T4, bigger vocab, dim=32
#   python train_magnetic.py --train-lines 1000000 --physics-iters 300 \
#       --dim 32 --multi-gpu
#
#   # C# faithful reproduction: dim=3, no PMI, 1-hop spreading
#   python train_magnetic.py --train-lines 100000 --dim 3 --no-pmi \
#       --spreading-hops 1
#
# When --run-indist-cloze / --run-ood-cloze are passed, the trained
# model is used for cloze evaluation in-process so you get a single
# output that captures training + perplexity + cloze in one run.

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required.", file=sys.stderr)
    sys.exit(1)

from magnetic import (
    MagneticConfig,
    MagneticModel,
    Evaluator,
    ensure_wt103,
    load_wt103_lines,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="MagneticLM modular training + evaluation runner")
    # Data
    ap.add_argument("--data-dir", default="data/wt103")
    ap.add_argument("--train-lines", type=int, default=100000,
                    help="0 = load all")
    ap.add_argument("--test-lines", type=int, default=0,
                    help="0 = load all")

    # Core
    ap.add_argument("--max-order", type=int, default=5,
                    dest="max_order")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--physics-iters", type=int, default=100,
                    dest="physics_iters")
    ap.add_argument("--multi-gpu", action="store_true",
                    dest="multi_gpu")
    ap.add_argument("--low-order-max", type=int, default=3,
                    dest="low_order_max")

    # Edges
    ap.add_argument("--edge-window", type=int, default=2,
                    dest="edge_window")
    ap.add_argument("--no-pmi", action="store_true",
                    help="Disable PMI reweighting to reproduce C# exactly")

    # Inference weights (used only by cloze eval)
    ap.add_argument("--alpha-contextual", type=float, default=0.6,
                    dest="alpha_contextual")
    ap.add_argument("--beta-semantic", type=float, default=0.35,
                    dest="beta_semantic")
    ap.add_argument("--spreading-hops", type=int, default=2,
                    dest="spreading_hops")
    ap.add_argument("--hop-decay", type=float, default=0.3,
                    dest="hop_decay")

    # Evaluation
    ap.add_argument("--run-wt103-ppl", action="store_true",
                    help="Compute WT103 full-mode perplexity (KN layer)")
    ap.add_argument("--run-indist-cloze", action="store_true")
    ap.add_argument("--indist-cloze-n", type=int, default=10)
    ap.add_argument("--eval-batch-size", type=int, default=16384,
                    dest="eval_batch_size")

    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_config(args) -> MagneticConfig:
    cfg = MagneticConfig()
    cfg.max_ngram_order = args.max_order
    cfg.dim = args.dim
    cfg.physics_iters = args.physics_iters
    cfg.multi_gpu = args.multi_gpu
    cfg.low_order_max = args.low_order_max
    cfg.edge_window = args.edge_window
    cfg.use_pmi = not args.no_pmi
    cfg.alpha_contextual = args.alpha_contextual
    cfg.beta_semantic = args.beta_semantic
    cfg.spreading_hops = args.spreading_hops
    cfg.hop_decay = args.hop_decay
    cfg.seed = args.seed
    return cfg


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(2)

    cfg = build_config(args)
    device = torch.device("cuda:0")
    aux_device = None
    n_dev = torch.cuda.device_count()
    if cfg.multi_gpu and n_dev >= 2:
        aux_device = torch.device("cuda:1")
    elif cfg.multi_gpu:
        print("WARNING: --multi-gpu requested but only %d device(s). "
              "Falling back to single-device." % n_dev)
        cfg.multi_gpu = False

    for i in range(n_dev):
        props = torch.cuda.get_device_properties(i)
        print("Device %d: %s (%.1f GB)" %
              (i, torch.cuda.get_device_name(i), props.total_memory / 1024 ** 3))

    print("\n" + "=" * 64)
    print("  MagneticLM Modular Runner")
    print("  max_order=%d  dim=%d  physics_iters=%d" %
          (cfg.max_ngram_order, cfg.dim, cfg.physics_iters))
    print("  edge_window=%d  use_pmi=%s" %
          (cfg.edge_window, cfg.use_pmi))
    print("  alpha=%.2f  beta=%.2f  spreading_hops=%d  hop_decay=%.2f" %
          (cfg.alpha_contextual, cfg.beta_semantic,
           cfg.spreading_hops, cfg.hop_decay))
    print("  multi_gpu=%s" % cfg.multi_gpu)
    print("=" * 64)

    # ---- Data ----
    torch.backends.cuda.matmul.allow_tf32 = True
    train_path, test_path = ensure_wt103(args.data_dir)

    limit = args.train_lines if args.train_lines > 0 else None
    print("\nLoading train lines from %s (limit=%s)" %
          (train_path, "all" if limit is None else str(limit)))
    train = load_wt103_lines(train_path, limit)
    print("Loaded %d train lines" % len(train))

    test_limit = args.test_lines if args.test_lines > 0 else None
    test = load_wt103_lines(test_path, test_limit)
    print("Loaded %d test lines" % len(test))

    # ---- Train ----
    print("\n--- Training ---")
    model = MagneticModel(cfg, device=device, aux_device=aux_device)
    t0 = time.time()
    model.train(train)
    print("  Train+build wall time: %.0fs" % (time.time() - t0))
    print("  Memory: %s" % model.memory_summary())

    # ---- Eval ----
    evaluator = Evaluator(model)

    if args.run_wt103_ppl:
        print("\n--- WT103 Perplexity (KN layer only) ---")
        t0 = time.time()
        ppl = evaluator.wt103_perplexity(test, batch_size=args.eval_batch_size)
        print("  PPL: %.2f (%.0fs)" % (ppl, time.time() - t0))

    if args.run_indist_cloze:
        print("\n--- In-distribution cloze (mid-word mask, full generator) ---")
        t0 = time.time()
        res = evaluator.indist_cloze(
            train, n=args.indist_cloze_n, verbose=True)
        N = res["n"]
        if N > 0:
            print("\n  top-1 = %d/%d (%.0f%%)" %
                  (res["top1"], N, 100.0 * res["top1"] / N))
            print("  top-5 = %d/%d (%.0f%%)" %
                  (res["top5"], N, 100.0 * res["top5"] / N))
            print("  top-10 = %d/%d (%.0f%%)" %
                  (res["top10"], N, 100.0 * res["top10"] / N))
        print("  Time: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
