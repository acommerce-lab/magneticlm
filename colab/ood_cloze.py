#!/usr/bin/env python3
# ood_cloze.py - out-of-distribution cloze evaluation for MagneticLM.
#
# Purpose: distinguish memorization from compositional generalization.
#
# The random-training-sentence cloze in generator.py hits 90% top-1 at
# 100k training lines. Impressive, BUT in-distribution - every sentence
# we mask is a sentence the model saw verbatim during training. The
# danger: most of that 90% could be memorization of specific n-grams.
#
# This script tests on HAND-DESIGNED sentences that:
#   1. Use simple, common English that is trivial for a real language
#      model to complete.
#   2. Are very unlikely to appear verbatim in WikiText-103 (which is
#      Wikipedia articles, not everyday English).
#   3. Still use words that ARE in the WT103 vocabulary, so the test
#      is fair - we don't fail because of OOV words.
#
# Example:
#     "she drinks her [MASK] every morning"   -> coffee / tea / water
#     "the doctor examined the [MASK]"         -> patient / wound
#     "he opened the [MASK] and started reading" -> book / letter
#
# Interpretation of the final top-10 number:
#   ~90%  -> the model really generalizes, and 90% in-dist was largely
#            real signal. Big green light to keep building.
#   ~50%  -> partial generalization. Clear gap to close.
#   ~10%  -> almost all in-dist success was memorization. Architectural
#            rethink required.
#
# This script does NOT modify any training or model code. It only adds
# a new evaluation path on the same MagneticLMGPU class, reusing
# cloze_topk from generator.py.

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required (pip install torch).", file=sys.stderr)
    sys.exit(1)

from MagneticLMFastRunner import MagneticLMGPU, ensure_wt103
from generator import cloze_topk, _text_to_ids


# ---------------------------------------------------------------------------
# Hand-designed OOD test set
# ---------------------------------------------------------------------------
# Format: (sentence_with_[MASK], list_of_acceptable_answers, category)
#
# Design rules:
#  - Sentences should be natural spoken/written English, NOT encyclopedic.
#  - Every word (except the masked one) must be a plain common English
#    word likely in WT103's vocab.
#  - Acceptable answers are LIBERAL: any word that a competent speaker
#    would accept counts as a hit. We are not testing the model's
#    preferences; we are testing whether its top-10 contains ANY
#    reasonable completion.
#  - Categories let us see which kinds of knowledge generalize and
#    which do not.

OOD_TESTS = [
    # -------- daily_actions --------
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

    # -------- simple_factual --------
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

    # -------- adjectives / descriptors --------
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

    # -------- animals --------
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

    # -------- relationships --------
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

    # -------- emotions --------
    ("she felt very [MASK] after winning the prize",
     ["happy", "proud", "excited", "pleased", "glad"], "emotions"),
    ("he was [MASK] when his team lost",
     ["sad", "angry", "disappointed", "upset", "unhappy"], "emotions"),
    ("the children were [MASK] to see their grandmother",
     ["happy", "excited", "glad", "pleased"], "emotions"),

    # -------- common_objects --------
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

    # -------- body / self --------
    ("he covered his [MASK] with both hands",
     ["face", "eyes", "head", "mouth", "ears"], "body"),
    ("she brushed her long [MASK] every night",
     ["hair", "teeth"], "body"),

    # -------- pangram (known-hard baseline) --------
    ("the quick brown [MASK] jumped over the lazy dog",
     ["fox"], "pangram_baseline"),
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_one(model, sentence, acceptable, top_k=20, position_weight=0.15):
    """Run one OOD cloze test. Returns a dict with rank and top-k preds."""
    if "[MASK]" not in sentence:
        return None
    before, after = sentence.split("[MASK]", 1)
    left_ids, left_oov = _text_to_ids(model, before)
    right_ids, right_oov = _text_to_ids(model, after)

    # Resolve acceptable answers to vocab ids
    accept_ids = set()
    missing = []
    for w in acceptable:
        tid = model.word2id.get(w.lower(), -1)
        if tid < 0:
            missing.append(w)
        else:
            accept_ids.add(tid)

    top = cloze_topk(
        model, left_ids, right_ids,
        k=top_k, position_weight=position_weight)

    best_rank = None
    for r, (tid, _p) in enumerate(top, 1):
        if tid in accept_ids:
            best_rank = r
            break

    return {
        "sentence": sentence,
        "acceptable": acceptable,
        "dropped_ctx": left_oov + right_oov,
        "missing_acceptable": missing,
        "top_k_pred": [
            (model.id2word[t] if 0 <= t < len(model.id2word) else "<OOV>", p)
            for t, p in top],
        "best_rank": best_rank,
        "all_acceptable_oov": len(accept_ids) == 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Out-of-distribution cloze evaluation for MagneticLM")
    ap.add_argument("--train-lines", type=int, default=100000)
    ap.add_argument("--physics-iters", type=int, default=100)
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--data-dir", default="data/wt103")
    ap.add_argument("--position-weight", type=float, default=0.15,
                    help="Weight on bidirectional position similarity "
                         "(higher = more weight on right-side evidence)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="How deep to search in the candidate list")
    ap.add_argument("--sweep-position-weight", action="store_true",
                    help="Also run with weights 0.05/0.15/0.30/0.50 and "
                         "report how OOD accuracy responds.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    train_path, _ = ensure_wt103(args.data_dir)

    print("Loading up to %d training lines..." % args.train_lines)
    train = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            train.append(s)
            if len(train) >= args.train_lines:
                break
    print("Loaded %d lines" % len(train))

    print("Building model (max_order=%d, physics_iters=%d)..." %
          (args.max_order, args.physics_iters))
    t0 = time.time()
    model = MagneticLMGPU(
        device=device, max_order=args.max_order, multi_gpu=False)
    model.train_gpu(train)
    model.build(physics_iters=args.physics_iters)
    print("Build time: %.1fs" % (time.time() - t0))
    print("Vocab size: %d" % len(model.id2word))

    weights_to_try = [args.position_weight]
    if args.sweep_position_weight:
        weights_to_try = [0.05, 0.15, 0.30, 0.50]

    for w in weights_to_try:
        print("\n" + "=" * 72)
        print("  OOD CLOZE EVALUATION   position_weight=%.2f   top_k=%d" %
              (w, args.top_k))
        print("=" * 72)
        print("  Hand-designed simple English sentences unlikely to appear")
        print("  verbatim in WT103. High accuracy = real generalization;")
        print("  low accuracy = in-distribution memorization only.")
        print()

        results = []
        cat_stats = {}

        for i, (sent, acceptable, cat) in enumerate(OOD_TESTS, 1):
            res = run_one(
                model, sent, acceptable,
                top_k=args.top_k, position_weight=w)
            if res is None:
                continue
            results.append((cat, res))

            print("  [%2d] %s" % (i, sent))
            print("       category: %s" % cat)
            if res["dropped_ctx"]:
                print("       WARN: OOV in context -> %s" %
                      res["dropped_ctx"])
            if res["missing_acceptable"]:
                print("       WARN: acceptable words not in vocab -> %s" %
                      res["missing_acceptable"])
            if res["all_acceptable_oov"]:
                print("       SKIP: all acceptable answers are OOV "
                      "(test not scorable)")
            rank_str = (str(res["best_rank"])
                        if res["best_rank"] is not None else ">%d" % args.top_k)
            print("       accept: %s" % res["acceptable"])
            print("       rank  : %s" % rank_str)
            top5 = ", ".join(
                "%s(%.2e)" % (w_, p) for w_, p in res["top_k_pred"][:5])
            print("       top-5 : %s" % top5)
            print()

            # Track stats (skip tests where all acceptable are OOV)
            if res["all_acceptable_oov"]:
                continue
            if cat not in cat_stats:
                cat_stats[cat] = {
                    "total": 0, "top1": 0, "top5": 0, "top10": 0}
            cat_stats[cat]["total"] += 1
            if res["best_rank"] is not None:
                if res["best_rank"] <= 1:
                    cat_stats[cat]["top1"] += 1
                if res["best_rank"] <= 5:
                    cat_stats[cat]["top5"] += 1
                if res["best_rank"] <= 10:
                    cat_stats[cat]["top10"] += 1

        # Overall
        scorable = [r for _c, r in results if not r["all_acceptable_oov"]]
        N = len(scorable)
        top1 = sum(1 for r in scorable
                   if r["best_rank"] is not None and r["best_rank"] <= 1)
        top5 = sum(1 for r in scorable
                   if r["best_rank"] is not None and r["best_rank"] <= 5)
        top10 = sum(1 for r in scorable
                    if r["best_rank"] is not None and r["best_rank"] <= 10)

        print("-" * 72)
        print("  SUMMARY (position_weight=%.2f)" % w)
        print("-" * 72)
        print("  Total scorable tests: %d" % N)
        if N:
            print("    top-1  = %d/%d (%.0f%%)" %
                  (top1, N, 100.0 * top1 / N))
            print("    top-5  = %d/%d (%.0f%%)" %
                  (top5, N, 100.0 * top5 / N))
            print("    top-10 = %d/%d (%.0f%%)" %
                  (top10, N, 100.0 * top10 / N))
        print()
        print("  By category:")
        print("    %-22s  %6s  %5s  %5s  %6s" %
              ("category", "total", "top-1", "top-5", "top-10"))
        for cat in sorted(cat_stats):
            s = cat_stats[cat]
            print("    %-22s  %6d  %5d  %5d  %6d" %
                  (cat, s["total"], s["top1"], s["top5"], s["top10"]))

    # Comparison anchor
    print()
    print("=" * 72)
    print("  REFERENCE: in-distribution cloze from generator.py")
    print("=" * 72)
    print("  100k training lines, 10 random training sentences, "
          "middle-word mask:")
    print("    top-1 = 90 %   top-5 = 90 %   top-10 = 90 %")
    print()
    print("  Interpretation:")
    print("    OOD top-10 close to 90 %  -> model really generalizes")
    print("    OOD top-10 near 50 %      -> partial generalization")
    print("    OOD top-10 below 20 %     -> in-dist win was memorization")


if __name__ == "__main__":
    main()
