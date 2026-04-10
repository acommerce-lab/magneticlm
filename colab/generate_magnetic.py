#!/usr/bin/env python3
# generate_magnetic.py
#
# Train a MagneticLM and generate text from one or more prompts.
# Uses the full multi-force generator (alpha*KN + beta*semantic +
# repulsion) so this is the real inference path, not the
# KN+cosine shortcut in the legacy runner.
#
# Example usage:
#
#   python generate_magnetic.py --train-lines 100000 --physics-iters 100 \
#       --prompt "the king of england was" --max-tokens 30
#
#   # Multiple prompts in one run
#   python generate_magnetic.py --train-lines 100000 \
#       --prompt "the king of england was" \
#       --prompt "she opened the door and" \
#       --prompt "the scientist discovered a new" \
#       --max-tokens 40
#
#   # Greedy decoding for reproducibility
#   python generate_magnetic.py --train-lines 100000 \
#       --prompt "the king of england was" --greedy
#
# The same training knobs as train_magnetic.py are supported so you
# can sweep edge window, dim, spreading hops etc. while watching the
# generation quality change.

import argparse
import os
import sys
import time

def _setup_magnetic_import():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.join(script_dir, "magnetic")
    if os.path.isdir(pkg_dir) and os.path.isfile(
            os.path.join(pkg_dir, "__init__.py")):
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        return
    if not os.path.isfile(os.path.join(script_dir, "config.py")):
        print("FATAL: cannot find magnetic package modules.", file=sys.stderr)
        sys.exit(2)
    import shutil, tempfile
    tmp_root = tempfile.mkdtemp(prefix="magnetic_pkg_")
    tmp_pkg = os.path.join(tmp_root, "magnetic")
    os.makedirs(tmp_pkg)
    for fname in sorted(os.listdir(script_dir)):
        if fname.endswith(".py") and "_magnetic" not in fname:
            shutil.copy2(os.path.join(script_dir, fname),
                         os.path.join(tmp_pkg, fname))
    sys.path.insert(0, tmp_root)
    print("  (rebuilt magnetic/ package in %s)" % tmp_pkg)

_setup_magnetic_import()

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required.", file=sys.stderr)
    sys.exit(1)

from magnetic import (
    MagneticConfig,
    MagneticModel,
    MagneticGenerator,
    ensure_wt103,
    load_wt103_lines,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="MagneticLM modular generator")
    ap.add_argument("--data-dir", default="data/wt103")
    ap.add_argument("--train-lines", type=int, default=100000)
    ap.add_argument("--max-order", type=int, default=5, dest="max_order")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--physics-iters", type=int, default=100,
                    dest="physics_iters")
    ap.add_argument("--multi-gpu", action="store_true", dest="multi_gpu")

    ap.add_argument("--edge-window", type=int, default=2,
                    dest="edge_window")
    ap.add_argument("--no-pmi", action="store_true")
    ap.add_argument("--no-jaccard", action="store_true")
    ap.add_argument("--generation-topk", type=int, default=500,
                    dest="generation_topk")

    ap.add_argument("--alpha-contextual", type=float, default=0.6,
                    dest="alpha_contextual")
    ap.add_argument("--beta-semantic", type=float, default=0.35,
                    dest="beta_semantic")
    ap.add_argument("--spreading-hops", type=int, default=2,
                    dest="spreading_hops")
    ap.add_argument("--hop-decay", type=float, default=0.3,
                    dest="hop_decay")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--excitation-decay", type=float, default=0.85,
                    dest="excitation_decay")
    ap.add_argument("--repulsion-strength", type=float, default=0.5,
                    dest="repulsion_strength")

    ap.add_argument("--prompt", action="append", default=[],
                    help="Prompt to continue. Can be passed multiple times.")
    ap.add_argument("--max-tokens", type=int, default=30, dest="max_tokens")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_config(args) -> MagneticConfig:
    cfg = MagneticConfig()
    cfg.max_ngram_order = args.max_order
    cfg.dim = args.dim
    cfg.physics_iters = args.physics_iters
    cfg.multi_gpu = args.multi_gpu
    cfg.edge_window = args.edge_window
    cfg.use_pmi = not args.no_pmi
    cfg.use_jaccard = not args.no_jaccard
    cfg.generation_topk = args.generation_topk
    cfg.alpha_contextual = args.alpha_contextual
    cfg.beta_semantic = args.beta_semantic
    cfg.spreading_hops = args.spreading_hops
    cfg.hop_decay = args.hop_decay
    cfg.temperature = args.temperature
    cfg.excitation_decay = args.excitation_decay
    cfg.repulsion_strength = args.repulsion_strength
    cfg.seed = args.seed
    return cfg


def main():
    args = parse_args()
    if not args.prompt:
        print("ERROR: at least one --prompt is required.", file=sys.stderr)
        sys.exit(2)
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
        print("WARNING: --multi-gpu requested but only %d device(s)." % n_dev)
        cfg.multi_gpu = False

    torch.backends.cuda.matmul.allow_tf32 = True
    train_path, _ = ensure_wt103(args.data_dir)
    limit = args.train_lines if args.train_lines > 0 else None
    train = load_wt103_lines(train_path, limit)
    print("Loaded %d train lines" % len(train))

    print("\n--- Training ---")
    model = MagneticModel(cfg, device=device, aux_device=aux_device)
    t0 = time.time()
    model.train(train)
    print("  Train+build wall time: %.0fs" % (time.time() - t0))
    print("  Memory: %s" % model.memory_summary())

    generator = MagneticGenerator(model)

    print("\n" + "=" * 64)
    print("  GENERATION  mode=%s  alpha=%.2f  beta=%.2f  temperature=%.2f" %
          ("greedy" if args.greedy else "sampling",
           cfg.alpha_contextual, cfg.beta_semantic, cfg.temperature))
    print("  spreading_hops=%d  hop_decay=%.2f  excitation_decay=%.2f" %
          (cfg.spreading_hops, cfg.hop_decay, cfg.excitation_decay))
    print("=" * 64)

    for p in args.prompt:
        print("\nPrompt: %s" % p)
        t0 = time.time()
        all_ids, prompt_words, gen_words = generator.generate_from_text(
            p, max_tokens=args.max_tokens, greedy=args.greedy)
        gen_t = time.time() - t0
        if not prompt_words:
            print("  (no in-vocab tokens in prompt)")
            continue
        print("  Prompt tokens (%d): %s" %
              (len(prompt_words), " ".join(prompt_words)))
        print("  Generated (%d): %s" %
              (len(gen_words), " ".join(gen_words)))
        print("  Time: %.1fs (%.0f ms/tok)" %
              (gen_t, gen_t * 1000.0 / max(1, args.max_tokens)))


if __name__ == "__main__":
    main()
