#!/usr/bin/env python3
# ood_magnetic.py
#
# Out-of-distribution cloze evaluation for the new modular MagneticLM.
# Same test bank as colab/ood_cloze.py so the numbers are directly
# comparable to the legacy KN+cosine baseline we already measured.
#
# The key difference from ood_cloze.py: this one uses the FULL
# multi-force inference path (excitation + spreading activation +
# semantic force + KN), not the cosine-on-positions shortcut. The
# whole point is to see whether bringing back the C# inference logic
# rescues OOD performance from the 27% top-10 floor we measured.
#
# Example usage:
#
#   # Basic: new defaults (dim=16, PMI on, 2-hop spreading)
#   python ood_magnetic.py --train-lines 100000 --physics-iters 100
#
#   # C# faithful reproduction: dim=3, no PMI, 1-hop
#   python ood_magnetic.py --train-lines 100000 --dim 3 --no-pmi \
#       --spreading-hops 1
#
#   # Sweep alpha/beta to see which balance works best
#   python ood_magnetic.py --train-lines 100000 --alpha-contextual 0.6 \
#       --beta-semantic 0.35

import argparse
import os
import sys
import time

# Ensure the magnetic package is importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_PKG_DIR = os.path.join(_SCRIPT_DIR, "magnetic")
if not os.path.isdir(_PKG_DIR):
    print("FATAL: magnetic/ not found at %s" % _PKG_DIR)
    print("Contents: %s" % sorted(os.listdir(_SCRIPT_DIR)))
    sys.exit(2)

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


# --------------------------------------------------------------------------
# Hand-designed OOD test bank. Identical to the one in colab/ood_cloze.py
# so runs are comparable across implementations.
# --------------------------------------------------------------------------
OOD_TESTS = [
    # daily_actions
    ("she drinks her [MASK] every morning",
     ["coffee", "tea", "water", "milk", "juice"], "daily_actions"),
    ("he opened the [MASK] and started reading",
     ["book", "letter", "newspaper", "magazine", "paper"], "daily_actions"),
    ("she put on her [MASK] and went outside",
     ["coat", "jacket", "shoes", "hat", "boots", "dress"], "daily_actions"),
    ("the baby started to [MASK] loudly",
     ["cry", "laugh", "scream", "sing", "shout"], "daily_actions"),
    ("he wrote a [MASK] to his mother",
     ["letter", "note", "poem", "message", "song", "book"], "daily_actions"),
    ("the cook added salt to the [MASK]",
     ["soup", "food", "dish", "meat", "pot"], "daily_actions"),

    # simple_factual
    ("the sun rises in the [MASK]",
     ["east", "morning", "sky"], "simple_factual"),
    ("the earth revolves around the [MASK]",
     ["sun"], "simple_factual"),
    ("a week has seven [MASK]",
     ["days"], "simple_factual"),
    ("water freezes at zero [MASK]",
     ["degrees", "celsius"], "simple_factual"),
    ("there are twelve months in a [MASK]",
     ["year"], "simple_factual"),
    ("birds can [MASK] in the sky",
     ["fly", "soar", "glide"], "simple_factual"),

    # adjectives
    ("she bought a [MASK] apple from the market",
     ["red", "green", "fresh", "ripe", "rotten", "large", "small"],
     "adjectives"),
    ("he drove the [MASK] car to work",
     ["new", "old", "red", "black", "blue", "white", "big", "small"],
     "adjectives"),
    ("the [MASK] sky was full of stars",
     ["night", "dark", "clear", "black", "evening"], "adjectives"),
    ("the building was very [MASK]",
     ["tall", "large", "old", "big", "high", "small", "new"],
     "adjectives"),
    ("the water was too [MASK] to drink",
     ["hot", "cold", "dirty", "salty", "warm"], "adjectives"),

    # animals
    ("the [MASK] barked loudly at the stranger",
     ["dog"], "animals"),
    ("the [MASK] purred quietly on her lap",
     ["cat"], "animals"),
    ("the [MASK] flew across the blue sky",
     ["bird", "eagle", "plane"], "animals"),
    ("the cat chased the [MASK] around the house",
     ["mouse", "dog", "bird", "ball"], "animals"),
    ("the farmer fed his [MASK] every morning",
     ["animals", "horses", "cows", "pigs", "chickens"], "animals"),

    # relationships
    ("he kissed his [MASK] goodbye",
     ["wife", "mother", "daughter", "girlfriend", "son", "family"],
     "relationships"),
    ("the girl ran to hug her [MASK]",
     ["mother", "father", "sister", "friend", "brother"], "relationships"),
    ("he shook her [MASK] firmly",
     ["hand"], "relationships"),
    ("she called her [MASK] on the phone",
     ["mother", "father", "friend", "husband", "sister", "brother"],
     "relationships"),

    # emotions
    ("she felt very [MASK] after winning the prize",
     ["happy", "proud", "excited", "pleased", "glad"], "emotions"),
    ("he was [MASK] when his team lost",
     ["sad", "angry", "disappointed", "upset", "unhappy"], "emotions"),
    ("the children were [MASK] to see their grandmother",
     ["happy", "excited", "glad", "pleased"], "emotions"),

    # common_objects
    ("the key was in his [MASK]",
     ["pocket", "hand", "bag"], "common_objects"),
    ("she climbed the [MASK] to reach the top shelf",
     ["ladder", "chair", "stairs", "stool"], "common_objects"),
    ("he sat on the [MASK] and read the newspaper",
     ["chair", "bench", "sofa", "couch", "seat"], "common_objects"),
    ("the teacher wrote on the [MASK] with chalk",
     ["board", "blackboard", "wall"], "common_objects"),
    ("she put the flowers in a [MASK]",
     ["vase", "pot", "jar", "bowl", "bag"], "common_objects"),

    # body
    ("he covered his [MASK] with both hands",
     ["face", "eyes", "head", "mouth", "ears"], "body"),
    ("she brushed her long [MASK] every night",
     ["hair", "teeth"], "body"),

    # pangram baseline (known to fail)
    ("the quick brown [MASK] jumped over the lazy dog",
     ["fox"], "pangram_baseline"),
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="OOD cloze evaluation for modular MagneticLM")
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

    ap.add_argument("--alpha-contextual", type=float, default=0.6,
                    dest="alpha_contextual")
    ap.add_argument("--beta-semantic", type=float, default=0.35,
                    dest="beta_semantic")
    ap.add_argument("--spreading-hops", type=int, default=2,
                    dest="spreading_hops")
    ap.add_argument("--hop-decay", type=float, default=0.3,
                    dest="hop_decay")

    ap.add_argument("--top-k", type=int, default=20, dest="top_k")
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

    evaluator = Evaluator(model)

    print("\n" + "=" * 72)
    print("  OOD CLOZE (modular multi-force inference)")
    print("  alpha=%.2f  beta=%.2f  spreading_hops=%d  hop_decay=%.2f" %
          (cfg.alpha_contextual, cfg.beta_semantic,
           cfg.spreading_hops, cfg.hop_decay))
    print("  dim=%d  edge_window=%d  use_pmi=%s" %
          (cfg.dim, cfg.edge_window, cfg.use_pmi))
    print("=" * 72)
    t0 = time.time()
    res = evaluator.ood_cloze(OOD_TESTS, top_k=args.top_k, verbose=True)
    eval_t = time.time() - t0

    N = res["n"]
    if N == 0:
        print("\n  No scorable OOD tests. Check vocab coverage.")
        return

    print("\n" + "-" * 72)
    print("  SUMMARY  (%d tests, %.1fs)" % (N, eval_t))
    print("-" * 72)
    print("    top-1  = %d/%d (%.0f%%)" % (res["top1"], N,
                                           100.0 * res["top1"] / N))
    print("    top-5  = %d/%d (%.0f%%)" % (res["top5"], N,
                                           100.0 * res["top5"] / N))
    print("    top-10 = %d/%d (%.0f%%)" % (res["top10"], N,
                                           100.0 * res["top10"] / N))
    print("\n  By category:")
    print("    %-22s  %6s  %5s  %5s  %6s" %
          ("category", "total", "top-1", "top-5", "top-10"))
    for cat in sorted(res["by_category"]):
        s = res["by_category"][cat]
        print("    %-22s  %6d  %5d  %5d  %6d" %
              (cat, s["n"], s["top1"], s["top5"], s["top10"]))

    print("\n  Reference baselines (legacy KN+cosine on dim=3):")
    print("    in-distribution: top-10 ~ 90 %")
    print("    out-of-dist     : top-10 ~ 27 %")
    print("  Anything above 27% here is a real improvement from the")
    print("  restored multi-force inference path.")


if __name__ == "__main__":
    main()
