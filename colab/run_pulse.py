#!/usr/bin/env python3
"""
run_pulse.py — MagneticLM Dual-Map Runner with Pulse Training

No dependency on the magnetic/ package. Self-contained.

Architecture (exactly as described):
  Contextual Map (mandatory): word A → word B (direct bigram succession)
  Semantic Map (advisory): word A ≈ word C (iterative reward/penalty)

Training pipeline:
  1. Tokenize corpus
  2. Build contextual children map (bigram successors) on GPU
  3. Pulse-train semantic map on CPU (per-sentence: window, predict, reward/penalize)
  4. Evaluate

Inference:
  Candidates = ctx_children(current_word) ∪ ctx_children(semantic_neighbors)
  P(next | ctx) = alpha * P_direct + beta * P_adopt + gamma * P_unigram

Usage:
  python run_pulse.py --train-lines 100000 --run-all
  python run_pulse.py --train-lines 860000 --run-ppl --run-ood-cloze
  python run_pulse.py --train-lines 100000 --reward 0.1 --penalty 0.05
"""

import argparse
import array
import os
import re
import sys
import time

# Make pulse/ importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    import numpy as np
    import torch
except ImportError:
    print("ERROR: PyTorch and NumPy required.", file=sys.stderr)
    sys.exit(1)

from pulse.config import PulseConfig
from pulse.semantic_trainer import SemanticMap
from pulse.contextual_map import ContextualMap
from pulse.evaluator import DualMapEvaluator


# =========================================================================
# Minimal tokenizer (same logic as everywhere, no dependency)
# =========================================================================
_SPLIT_RE = re.compile(r'[.,;!?()\[\]{}"]+')


def tokenize(line: str):
    return [w for w in _SPLIT_RE.sub(' ', line.lower()).split() if w]


# =========================================================================
# Minimal vocabulary
# =========================================================================
class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = []

    def add(self, word):
        tid = self.word2id.get(word)
        if tid is None:
            tid = len(self.id2word)
            self.word2id[word] = tid
            self.id2word.append(word)
        return tid

    def __len__(self):
        return len(self.id2word)


# =========================================================================
# WikiText-103 data loading
# =========================================================================
def ensure_wt103(data_dir="data/wt103"):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path
    print("Downloading WikiText-103 (via HF datasets)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    for split, path in (("train", train_path), ("test", test_path)):
        with open(path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                t = item["text"].strip()
                if t and not t.startswith("="):
                    f.write(t + "\n")
    return train_path, test_path


def load_lines(path, limit=None):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if limit and len(lines) >= limit:
                break
    return lines


# =========================================================================
# PulseModel: the trained dual-map model
# =========================================================================
class PulseModel:
    """Holds both maps and all state needed for inference/evaluation."""

    def __init__(self, config: PulseConfig, device: torch.device):
        self.config = config
        self.device = device
        self.vocab = Vocabulary()
        self.ctx_map = None
        self.semantic_map = None
        self.freq_gpu = None

    def train(self, lines):
        cfg = self.config
        dev = self.device
        t_all = time.time()

        # ---- Phase 1: Tokenize ----
        print("\n  Phase 1: Tokenize")
        print("  " + "-" * 50)
        tokens_gpu = self._tokenize_to_gpu(lines)
        T = tokens_gpu.numel()
        V = len(self.vocab)
        print("  Tokens: %d, Vocab: %d" % (T, V))
        if T < 2 or V == 0:
            return

        # Frequency table
        self.freq_gpu = torch.zeros(V, dtype=torch.int64, device=dev)
        self.freq_gpu.scatter_add_(
            0, tokens_gpu, torch.ones_like(tokens_gpu))

        # ---- Phase 2: Contextual map (GPU) ----
        print("\n  Phase 2: Contextual Map (bigram successors)")
        print("  " + "-" * 50)
        self.ctx_map = ContextualMap.build_from_tokens(
            tokens_gpu, V, self.freq_gpu, dev,
            max_children=cfg.max_ctx_children,
            verbose=True)
        stats = self.ctx_map.stats()
        print("    Words with children: %d / %d" %
              (stats["words_with_children"], V))
        print("    Avg children: %.1f, Max: %d" %
              (stats["avg_children"], stats["max_children"]))

        # ---- Phase 3: Semantic pulse training (CPU) ----
        print("\n  Phase 3: Semantic Pulse Training (iterative)")
        print("  " + "-" * 50)
        print("    Window: +/-%d  Reward: +%.3f  Penalty: -%.3f" %
              (cfg.semantic_window, cfg.reward_amount, cfg.penalty_amount))
        print("    Transitive propagation every %d sentences" %
              cfg.transitive_interval)

        sentences = self._split_sentences(lines)
        del tokens_gpu
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        self.semantic_map = SemanticMap(cfg)
        self.semantic_map.train_corpus(sentences, verbose=True)

        sem = self.semantic_map.stats()
        print("    Total edges: %d (%d positive, %d negative)" %
              (sem["total_edges"], sem["positive_edges"],
               sem["negative_edges"]))
        print("    Words with semantic edges: %d" % sem["words_with_edges"])
        print("    Weight range: [%.3f, %.3f]" %
              (sem["min_weight"], sem["max_weight"]))

        print("\n  Training complete. Total: %.0fs" % (time.time() - t_all))

    def _tokenize_to_gpu(self, lines):
        w2i = self.vocab.word2id
        id2w = self.vocab.id2word
        buf = array.array('i')
        n_lines = 0
        t0 = time.time()
        for line in lines:
            words = tokenize(line)
            if len(words) < 2:
                continue
            for w in words:
                tid = w2i.get(w)
                if tid is None:
                    tid = len(id2w)
                    w2i[w] = tid
                    id2w.append(w)
                buf.append(tid)
            n_lines += 1
            if n_lines % 20000 == 0:
                print("\r    %d lines, %d tokens (%.0fs)" %
                      (n_lines, len(buf), time.time() - t0),
                      end="", flush=True)
        print("\r    %d lines, %d tokens (%.0fs) done.%s" %
              (n_lines, len(buf), time.time() - t0, " " * 20))
        np_arr = np.frombuffer(buf, dtype=np.int32)
        return torch.from_numpy(np_arr).to(
            device=self.device, dtype=torch.int64)

    def _split_sentences(self, lines):
        sentences = []
        w2i = self.vocab.word2id
        for line in lines:
            words = tokenize(line)
            if len(words) < 2:
                continue
            ids = [w2i.get(w, -1) for w in words]
            ids = [i for i in ids if i >= 0]
            if len(ids) >= 2:
                sentences.append(ids)
        return sentences

    def memory_summary(self):
        if self.device.type != "cuda":
            return "(CPU)"
        parts = []
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1024 ** 2
            peak = torch.cuda.max_memory_allocated(i) / 1024 ** 2
            parts.append("GPU%d cur=%.0fMiB peak=%.0fMiB" %
                         (i, used, peak))
        return " | ".join(parts)


# =========================================================================
# OOD test set
# =========================================================================
OOD_TESTS = [
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
    ("she bought a [MASK] apple from the market",
     ["red", "green", "fresh", "ripe", "rotten", "large", "small"],
     "adjectives"),
    ("he drove the [MASK] car to work",
     ["new", "old", "red", "black", "blue", "white", "big", "small"],
     "adjectives"),
    ("the [MASK] sky was full of stars",
     ["night", "dark", "clear", "black", "evening"], "adjectives"),
    ("the building was very [MASK]",
     ["tall", "large", "old", "big", "high", "small", "new"], "adjectives"),
    ("the water was too [MASK] to drink",
     ["hot", "cold", "dirty", "salty", "warm"], "adjectives"),
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
    ("she felt very [MASK] after winning the prize",
     ["happy", "proud", "excited", "pleased", "glad"], "emotions"),
    ("he was [MASK] when his team lost",
     ["sad", "angry", "disappointed", "upset", "unhappy"], "emotions"),
    ("the children were [MASK] to see their grandmother",
     ["happy", "excited", "glad", "pleased"], "emotions"),
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
    ("he covered his [MASK] with both hands",
     ["face", "eyes", "head", "mouth", "ears"], "body"),
    ("she brushed her long [MASK] every night",
     ["hair", "teeth"], "body"),
    ("the quick brown [MASK] jumped over the lazy dog",
     ["fox"], "pangram_baseline"),
]


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="MagneticLM Dual-Map Runner (Pulse Training)")

    g = ap.add_argument_group("Data")
    g.add_argument("--data-dir", default="data/wt103")
    g.add_argument("--train-lines", type=int, default=100000,
                   help="0 = all lines")
    g.add_argument("--test-lines", type=int, default=0,
                   help="0 = all lines")

    g = ap.add_argument_group("Contextual map")
    g.add_argument("--max-ctx-children", type=int, default=500)

    g = ap.add_argument_group("Semantic training")
    g.add_argument("--semantic-window", type=int, default=2)
    g.add_argument("--penalty-threshold", type=float, default=0.5)
    g.add_argument("--neighbor-top-k", type=int, default=20)
    g.add_argument("--transitive-interval", type=int, default=200)
    g.add_argument("--transitive-decay", type=float, default=0.5)
    g.add_argument("--semantic-threshold", type=float, default=0.1)

    g = ap.add_argument_group("Five forces")
    g.add_argument("--k-spring", type=float, default=2.0,
                   help="Spring: co-occurrence attraction strength")
    g.add_argument("--k-repulsion", type=float, default=0.3,
                   help="Repulsion: high-degree node push-apart")
    g.add_argument("--k-attraction", type=float, default=0.5,
                   help="Far-field: prevent graph fragmentation")
    g.add_argument("--k-gravity", type=float, default=0.01,
                   help="Gravity: pull weights toward zero")
    g.add_argument("--damping", type=float, default=0.15,
                   help="Velocity damping (0=unstable, 1=frozen)")
    g.add_argument("--force-lr", type=float, default=0.02,
                   help="Integration learning rate")
    g.add_argument("--optimal-weight", type=float, default=3.0,
                   help="Target weight for far-field attraction")

    g = ap.add_argument_group("Inference")
    g.add_argument("--adoption-neighbors", type=int, default=10)
    g.add_argument("--adoption-min-weight", type=float, default=0.2)
    g.add_argument("--alpha-direct", type=float, default=0.7)
    g.add_argument("--beta-adopt", type=float, default=0.2)
    g.add_argument("--gamma-unigram", type=float, default=0.1)

    g = ap.add_argument_group("Evaluation")
    g.add_argument("--eval-batch-size", type=int, default=16384)
    g.add_argument("--run-ppl", action="store_true",
                   help="WT103 perplexity (dual-map)")
    g.add_argument("--run-hit-rate", action="store_true",
                   help="Candidate hit rate diagnostic")
    g.add_argument("--run-hit-rate-tokens", type=int, default=100000)
    g.add_argument("--run-ood-cloze", action="store_true",
                   help="OOD cloze evaluation")
    g.add_argument("--run-all", action="store_true")

    g = ap.add_argument_group("Other")
    g.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def build_config(args) -> PulseConfig:
    cfg = PulseConfig()
    cfg.max_ctx_children = args.max_ctx_children
    cfg.semantic_window = args.semantic_window
    cfg.penalty_threshold = args.penalty_threshold
    cfg.neighbor_top_k = args.neighbor_top_k
    cfg.transitive_interval = args.transitive_interval
    cfg.transitive_decay = args.transitive_decay
    cfg.semantic_threshold = args.semantic_threshold
    # Five forces
    cfg.K_spring = args.k_spring
    cfg.K_repulsion = args.k_repulsion
    cfg.K_attraction = args.k_attraction
    cfg.K_gravity = args.k_gravity
    cfg.damping = args.damping
    cfg.force_lr = args.force_lr
    cfg.optimal_weight = args.optimal_weight
    # Inference
    cfg.adoption_neighbors = args.adoption_neighbors
    cfg.adoption_min_weight = args.adoption_min_weight
    cfg.alpha_direct = args.alpha_direct
    cfg.beta_adopt = args.beta_adopt
    cfg.gamma_unigram = args.gamma_unigram
    cfg.eval_batch_size = args.eval_batch_size
    cfg.seed = args.seed
    return cfg


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.", file=sys.stderr)
        sys.exit(2)

    cfg = build_config(args)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(cfg.seed)

    n_dev = torch.cuda.device_count()
    for i in range(n_dev):
        props = torch.cuda.get_device_properties(i)
        print("Device %d: %s (%.1f GB)" %
              (i, torch.cuda.get_device_name(i),
               props.total_memory / 1024 ** 3))

    print("\n" + "=" * 64)
    print("  MagneticLM Dual-Map (Pulse Training)")
    print("  " + "-" * 60)
    print("  Contextual: max_children=%d" % cfg.max_ctx_children)
    print("  Semantic:   window=%d  threshold=%.2f" %
          (cfg.semantic_window, cfg.penalty_threshold))
    print("  Forces:     spring=%.2f  repulsion=%.2f  attraction=%.2f" %
          (cfg.K_spring, cfg.K_repulsion, cfg.K_attraction))
    print("              gravity=%.3f  damping=%.2f  lr=%.3f" %
          (cfg.K_gravity, cfg.damping, cfg.force_lr))
    print("  Transitive: interval=%d  decay=%.2f" %
          (cfg.transitive_interval, cfg.transitive_decay))
    print("  Inference:  alpha=%.2f  beta=%.2f  gamma=%.2f" %
          (cfg.alpha_direct, cfg.beta_adopt, cfg.gamma_unigram))
    print("  Adoption:   neighbors=%d  min_weight=%.2f" %
          (cfg.adoption_neighbors, cfg.adoption_min_weight))
    print("=" * 64)

    # ---- Load data ----
    train_path, test_path = ensure_wt103(args.data_dir)
    limit = args.train_lines if args.train_lines > 0 else None
    print("\nLoading train (limit=%s)..." %
          ("all" if limit is None else str(limit)))
    train = load_lines(train_path, limit)
    print("Loaded %d train lines" % len(train))

    test_limit = args.test_lines if args.test_lines > 0 else None
    test = load_lines(test_path, test_limit)
    print("Loaded %d test lines" % len(test))

    # ---- Train ----
    model = PulseModel(cfg, device=device)
    t0 = time.time()
    model.train(train)
    train_time = time.time() - t0
    print("  Memory: %s" % model.memory_summary())

    # ---- Evaluate ----
    run_ppl = args.run_ppl or args.run_all
    run_hit = args.run_hit_rate or args.run_all
    run_ood = args.run_ood_cloze or args.run_all

    if not any([run_ppl, run_hit, run_ood]):
        print("\nNo evaluation requested. Use --run-all or:")
        print("  --run-ppl         WT103 perplexity (dual-map)")
        print("  --run-hit-rate    Candidate hit rate")
        print("  --run-ood-cloze   OOD cloze evaluation")
        print("  --run-all         All of the above")
        return

    evaluator = DualMapEvaluator(model)
    ppl = hit = ood = None

    if run_ppl:
        print("\n" + "=" * 64)
        print("  WT103 Perplexity (dual-map scoring)")
        print("=" * 64)
        t0 = time.time()
        ppl = evaluator.wt103_ppl(test, batch_size=cfg.eval_batch_size)
        print("  PPL: %.2f (%.0fs)" % (ppl, time.time() - t0))

    if run_hit:
        print("\n" + "=" * 64)
        print("  Candidate Hit Rate")
        print("=" * 64)
        t0 = time.time()
        hit = evaluator.candidate_hit_rate(
            test, max_tokens=args.run_hit_rate_tokens)
        print("  Checked: %d tokens" % hit["total_checked"])
        print("  Direct hits:  %d (%.1f%%)" %
              (hit["direct_hits"], 100 * hit["direct_hit_rate"]))
        print("  Adopted hits: %d (%.1f%%)" %
              (hit["adopted_hits"], 100 * hit["adopted_hit_rate"]))
        print("  Total hits:   %.1f%%" % (100 * hit["total_hit_rate"]))
        print("  Misses:       %d (%.1f%%)" %
              (hit["misses"], 100 * hit["miss_rate"]))
        print("  Avg candidates: %.0f" % hit["avg_candidates"])
        print("  Time: %.0fs" % (time.time() - t0))

    if run_ood:
        print("\n" + "=" * 64)
        print("  OOD Cloze (hand-designed sentences)")
        print("=" * 64)
        t0 = time.time()
        ood = evaluator.ood_cloze(OOD_TESTS, top_k=20, verbose=True)
        N = ood["n"]
        if N > 0:
            print("\n  Overall (N=%d):" % N)
            print("    top-1  = %d/%d (%.0f%%)" %
                  (ood["top1"], N, 100 * ood["top1"] / N))
            print("    top-5  = %d/%d (%.0f%%)" %
                  (ood["top5"], N, 100 * ood["top5"] / N))
            print("    top-10 = %d/%d (%.0f%%)" %
                  (ood["top10"], N, 100 * ood["top10"] / N))
            print("\n  By category:")
            for cat, s in sorted(ood["by_category"].items()):
                cn = s["n"]
                if cn > 0:
                    print("    %-20s t1=%d/%d(%2.0f%%) "
                          "t5=%d/%d(%2.0f%%) "
                          "t10=%d/%d(%2.0f%%)" %
                          (cat,
                           s["top1"], cn, 100 * s["top1"] / cn,
                           s["top5"], cn, 100 * s["top5"] / cn,
                           s["top10"], cn, 100 * s["top10"] / cn))
        print("  Time: %.0fs" % (time.time() - t0))

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print("  Train: %d lines, %d vocab" %
          (len(train), len(model.vocab)))
    sem = model.semantic_map.stats()
    ctx = model.ctx_map.stats()
    print("  Contextual map: %d bigrams" % ctx["pairs"])
    print("  Semantic map: %d edges (%d pos / %d neg)" %
          (sem["total_edges"], sem["positive_edges"], sem["negative_edges"]))
    if ppl is not None:
        print("  PPL: %.2f" % ppl)
    if hit is not None:
        print("  Hit rate: %.1f%% (direct %.1f%% + adopted %.1f%%)" %
              (100 * hit["total_hit_rate"],
               100 * hit["direct_hit_rate"],
               100 * hit["adopted_hit_rate"]))
    if ood is not None and ood["n"] > 0:
        print("  OOD cloze: top-10 = %d/%d (%.0f%%)" %
              (ood["top10"], ood["n"], 100 * ood["top10"] / ood["n"]))
    print("  Training time: %.0fs" % train_time)
    print("  Memory: %s" % model.memory_summary())


if __name__ == "__main__":
    main()
