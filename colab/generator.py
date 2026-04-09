#!/usr/bin/env python3
# generator.py - experimental text generation + cloze evaluation for
# MagneticLM. Research script, not part of the production WT103 eval.
#
# Purpose (research diary, 2026-04)
# =================================
# Before adding new layers to MagneticLM (spreading activation, PPMI
# semantic layer, hierarchical circles...), we need to SEE what the
# current model actually produces when asked to generate text, and
# whether it can use both-side context at all. The 14.20 WT103 PPL
# number is a left-only next-word prediction score - it does NOT tell us
# whether the model can:
#
#   1. Generate a coherent continuation from a prompt (unidirectional).
#   2. Fill in a missing word given BOTH left and right context
#      (the symmetric position layer naturally supports this even though
#      KN is unidirectional).
#
# This script exercises both paths so we can diagnose what is actually
# missing before designing fixes.
#
# Usage
# =====
#   # Fast iteration: 10k training lines, 100 physics iters
#   python generator.py --train-lines 10000 --physics-iters 100 \
#       --prompt "the cat sat on the" --max-tokens 20 --greedy
#
#   # Cloze on 5 random training sentences (default)
#   python generator.py --train-lines 10000 --physics-iters 100
#
#   # Explicit cloze sentence (use [MASK] to mark the blank)
#   python generator.py --train-lines 10000 \
#       --cloze-text "the quick brown [MASK] jumped over the lazy dog"
#
# Notes
# =====
# - Scoring candidates is done in chunks of 32k so large vocabularies
#   stay within T4 memory limits.
# - The generator is **not** integrated into MagneticLMFastRunner.py on
#   purpose - it lives in its own file so the production eval path is
#   untouched. If an experiment here proves valuable, THEN promote.

import argparse
import random
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required (pip install torch).", file=sys.stderr)
    sys.exit(1)

from MagneticLMFastRunner import MagneticLMGPU, tokenize, ensure_wt103


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _score_left_only(model, left_tokens, chunk=32768):
    """KN + position probabilities for all V candidates given LEFT context
    only. This mirrors the per-token scoring inside eval_full_wt103 but
    returns a full (V,) vector instead of the single mixed probability for
    the actual next word."""
    V = len(model.id2word)
    K = model.max_order
    dev = model.device

    ctx_len = min(len(left_tokens), K)
    ctx_row = torch.full((K,), -1, dtype=torch.int64, device=dev)
    if ctx_len > 0:
        tail = torch.tensor(
            left_tokens[-ctx_len:], dtype=torch.int64, device=dev)
        ctx_row[-ctx_len:] = tail

    out = torch.empty(V, dtype=torch.float32, device=dev)
    pos = model.positions
    imp = model.importance

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        B = end - start
        ctx = ctx_row.unsqueeze(0).expand(B, -1).contiguous()
        nxt = torch.arange(start, end, dtype=torch.int64, device=dev)

        kn = model.kn_batch(ctx, nxt)

        safe_ctx = ctx.clamp_min(0)
        safe_nxt = nxt.clamp_min(0)
        ctx_pos = pos[safe_ctx]
        nxt_pos = pos[safe_nxt].unsqueeze(1)
        dot = (ctx_pos * nxt_pos).sum(-1)
        ctx_norm = ctx_pos.norm(dim=-1)
        nxt_norm = nxt_pos.norm(dim=-1)
        denom = (ctx_norm * nxt_norm).clamp_min(1e-6)
        sim = (dot / denom).clamp(-1.0, 1.0)
        valid = (ctx >= 0) & (sim > 0.05)
        sim = torch.where(valid, sim, torch.zeros_like(sim))
        ctx_imp = imp[safe_ctx]
        boost_imp = 1.0 + ctx_imp * 0.05
        contrib = sim * boost_imp
        pos_count = valid.sum(dim=1).clamp_min(1)
        has_any = valid.any(dim=1)
        pos_score = contrib.sum(dim=1)
        pos_prob = (pos_score /
                    (pos_count.to(pos_score.dtype) * 3.0)).clamp_max(0.3)
        pos_prob = torch.where(has_any, pos_prob, torch.zeros_like(pos_prob))

        band = torch.where(
            kn > 0.05, torch.tensor(0.02, device=dev),
            torch.where(kn > 0.005, torch.tensor(0.06, device=dev),
                        torch.tensor(0.12, device=dev)))
        kn_l = 1.0 - band
        mixed = (kn_l * kn + band * pos_prob).clamp(1e-10, 0.999)
        out[start:end] = mixed

    return out


def _score_bidirectional(model, left_tokens, right_tokens,
                         position_weight=0.15, chunk=32768):
    """Cloze scoring: KN uses LEFT context only (KN is unidirectional),
    but position similarity uses BOTH left and right window tokens
    because 3D cosine similarity is symmetric. A fixed position_weight
    (default 0.15) is used instead of the adaptive band because the
    adaptive band down-weights positions whenever KN is strong, which
    defeats the whole point of bringing right-side evidence in."""
    V = len(model.id2word)
    K = model.max_order
    dev = model.device

    ctx_len = min(len(left_tokens), K)
    ctx_row = torch.full((K,), -1, dtype=torch.int64, device=dev)
    if ctx_len > 0:
        tail = torch.tensor(
            left_tokens[-ctx_len:], dtype=torch.int64, device=dev)
        ctx_row[-ctx_len:] = tail

    left_tail = list(left_tokens[-K:]) if left_tokens else []
    right_head = list(right_tokens[:K]) if right_tokens else []
    window = [t for t in (left_tail + right_head) if t >= 0]

    pos = model.positions
    imp = model.importance

    if window:
        win_ids = torch.tensor(window, dtype=torch.int64, device=dev)
        win_pos = pos[win_ids]
        win_imp = imp[win_ids]
        win_norm = win_pos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        win_u = win_pos / win_norm
    else:
        win_ids = torch.empty(0, dtype=torch.int64, device=dev)

    out = torch.empty(V, dtype=torch.float32, device=dev)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        B = end - start
        ctx = ctx_row.unsqueeze(0).expand(B, -1).contiguous()
        nxt = torch.arange(start, end, dtype=torch.int64, device=dev)

        kn = model.kn_batch(ctx, nxt)

        if win_ids.numel() > 0:
            cand_pos = pos[nxt]
            cand_norm = cand_pos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cand_u = cand_pos / cand_norm
            sim = (cand_u @ win_u.T).clamp(-1.0, 1.0)
            valid = sim > 0.05
            sim = torch.where(valid, sim, torch.zeros_like(sim))
            boost = (1.0 + win_imp * 0.05).unsqueeze(0)
            contrib = sim * boost
            pos_count = valid.sum(dim=1).clamp_min(1)
            has_any = valid.any(dim=1)
            pos_score = contrib.sum(dim=1)
            pos_prob = (pos_score /
                        (pos_count.to(pos_score.dtype) * 3.0)).clamp_max(0.3)
            pos_prob = torch.where(
                has_any, pos_prob, torch.zeros_like(pos_prob))
        else:
            pos_prob = torch.zeros(B, dtype=torch.float32, device=dev)

        w = float(position_weight)
        mixed = ((1.0 - w) * kn + w * pos_prob).clamp(1e-10, 0.999)
        out[start:end] = mixed

    return out


# ---------------------------------------------------------------------------
# High-level modes
# ---------------------------------------------------------------------------

def generate(model, prompt_ids, max_new_tokens=30, greedy=True,
             temperature=1.0, top_k=40):
    """Autoregressive left-only generation. At each step we score all V
    candidates and pick the next token (greedy argmax or top-k sampling).
    The generated tokens are appended to `out` and used as left context
    for the next step."""
    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        probs = _score_left_only(model, out)
        if greedy:
            nxt_id = int(torch.argmax(probs).item())
        else:
            logits = torch.log(probs.clamp_min(1e-12))
            if 0 < top_k < logits.numel():
                topv, topi = torch.topk(logits, top_k)
                masked = torch.full_like(logits, float("-inf"))
                masked[topi] = topv
                logits = masked
            logits = logits / max(temperature, 1e-6)
            p = torch.softmax(logits, dim=-1)
            nxt_id = int(torch.multinomial(p, 1).item())
        out.append(nxt_id)
    return out


def cloze_topk(model, left_tokens, right_tokens, k=10, position_weight=0.15):
    probs = _score_bidirectional(
        model, left_tokens, right_tokens, position_weight=position_weight)
    topv, topi = torch.topk(probs, k)
    return [(int(i), float(v)) for i, v in zip(topi.tolist(), topv.tolist())]


def _ids_to_words(model, ids):
    return [model.id2word[t] if 0 <= t < len(model.id2word) else "<OOV>"
            for t in ids]


def _text_to_ids(model, text):
    words = tokenize(text)
    ids, unknown = [], []
    for w in words:
        tid = model.word2id.get(w, -1)
        if tid < 0:
            unknown.append(w)
        else:
            ids.append(tid)
    return ids, unknown


def _pick_cloze_samples(train_lines, model, n=5, min_len=12, max_len=30,
                        seed=42):
    """Sample n training sentences of reasonable length, all words in
    vocab. We mask the middle token so there is non-trivial left AND
    right context on both sides."""
    rng = random.Random(seed)
    candidates = []
    for line in train_lines:
        ws = tokenize(line)
        if not (min_len <= len(ws) <= max_len):
            continue
        ids = []
        ok = True
        for w in ws:
            tid = model.word2id.get(w, -1)
            if tid < 0:
                ok = False
                break
            ids.append(tid)
        if ok:
            candidates.append(ids)
        if len(candidates) >= 5000:
            break
    rng.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MagneticLM generator + cloze tester (experimental)")
    ap.add_argument("--train-lines", type=int, default=10000)
    ap.add_argument("--physics-iters", type=int, default=100)
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--data-dir", default="data/wt103")

    ap.add_argument("--prompt", type=str, default=None,
                    help="If set, run autoregressive generation from this "
                         "prompt.")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--greedy", action="store_true",
                    help="Greedy decoding (default is top-k sampling).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)

    ap.add_argument("--cloze-random", type=int, default=5,
                    help="Number of random training sentences to cloze-test "
                         "by masking the middle word. Set to 0 to skip.")
    ap.add_argument("--cloze-text", type=str, default=None,
                    help='Explicit cloze sentence containing "[MASK]".')
    ap.add_argument("--cloze-topk", type=int, default=10)
    ap.add_argument("--cloze-position-weight", type=float, default=0.15,
                    help="Fixed mixing weight for position similarity in "
                         "cloze scoring. Higher = more weight on the "
                         "bidirectional signal.")

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
    model = MagneticLMGPU(device=device, max_order=args.max_order,
                          multi_gpu=False)
    model.train_gpu(train)
    model.build(physics_iters=args.physics_iters)
    print("Build time: %.1fs" % (time.time() - t0))
    print("Vocab size: %d" % len(model.id2word))

    # -------------------------------------------------------------------
    # Mode 1: autoregressive generation
    # -------------------------------------------------------------------
    if args.prompt:
        print("\n" + "=" * 64)
        print("  GENERATION  (left-only, autoregressive)")
        print("=" * 64)
        prompt_ids, unknown = _text_to_ids(model, args.prompt)
        if unknown:
            print("  WARNING: %d OOV words dropped: %s" %
                  (len(unknown), unknown[:10]))
        if not prompt_ids:
            print("  ERROR: prompt has zero in-vocab tokens")
        else:
            print("  Prompt   : %s" %
                  " ".join(_ids_to_words(model, prompt_ids)))
            t0 = time.time()
            full_ids = generate(
                model, prompt_ids,
                max_new_tokens=args.max_tokens,
                greedy=args.greedy,
                temperature=args.temperature,
                top_k=args.top_k)
            gen_t = time.time() - t0
            new_ids = full_ids[len(prompt_ids):]
            mode = ("greedy" if args.greedy
                    else "top%d T=%.2f" % (args.top_k, args.temperature))
            print("  Mode     : %s, %.1fs (%.0f ms/tok)" %
                  (mode, gen_t, gen_t / max(1, args.max_tokens) * 1000))
            print("  Generated: %s" %
                  " ".join(_ids_to_words(model, new_ids)))
            print("  Full     : %s" %
                  " ".join(_ids_to_words(model, full_ids)))

    # -------------------------------------------------------------------
    # Mode 2: explicit cloze sentence with [MASK]
    # -------------------------------------------------------------------
    if args.cloze_text:
        print("\n" + "=" * 64)
        print("  CLOZE  (explicit sentence with [MASK])")
        print("=" * 64)
        if "[MASK]" not in args.cloze_text:
            print("  ERROR: --cloze-text must contain [MASK]")
        else:
            before, after = args.cloze_text.split("[MASK]", 1)
            left_ids, lu = _text_to_ids(model, before)
            right_ids, ru = _text_to_ids(model, after)
            if lu or ru:
                print("  WARNING: dropped OOV left=%s right=%s" %
                      (lu[:5], ru[:5]))
            top = cloze_topk(
                model, left_ids, right_ids,
                k=args.cloze_topk,
                position_weight=args.cloze_position_weight)
            print("  Left  : %s" %
                  " ".join(_ids_to_words(model, left_ids)))
            print("  Right : %s" %
                  " ".join(_ids_to_words(model, right_ids)))
            print("  Top-%d candidates (position_weight=%.2f):" %
                  (args.cloze_topk, args.cloze_position_weight))
            for i, (tid, p) in enumerate(top, 1):
                word = (model.id2word[tid]
                        if 0 <= tid < len(model.id2word) else "<OOV>")
                print("    %2d. %-20s p=%.4e" % (i, word, p))

    # -------------------------------------------------------------------
    # Mode 3: random cloze tests from training (mask middle word)
    # -------------------------------------------------------------------
    if args.cloze_random > 0:
        print("\n" + "=" * 64)
        print("  CLOZE  (%d random training sentences, mask middle word)" %
              args.cloze_random)
        print("=" * 64)
        print("  Scoring: KN on left context + symmetric position similarity")
        print("           on BOTH left and right windows (weight=%.2f)" %
              args.cloze_position_weight)
        sentences = _pick_cloze_samples(
            train, model, n=args.cloze_random,
            min_len=12, max_len=30, seed=42)
        if not sentences:
            print("  No training sentences matched length constraints.")
        hits_top1 = 0
        hits_top5 = 0
        hits_top10 = 0
        for idx, sent_ids in enumerate(sentences, 1):
            mid = len(sent_ids) // 2
            target_id = sent_ids[mid]
            left_ids = sent_ids[:mid]
            right_ids = sent_ids[mid + 1:]

            top = cloze_topk(
                model, left_ids, right_ids,
                k=max(10, args.cloze_topk),
                position_weight=args.cloze_position_weight)
            top_ids = [t[0] for t in top]
            hit_rank = None
            for r, tid in enumerate(top_ids, 1):
                if tid == target_id:
                    hit_rank = r
                    break
            if hit_rank == 1:
                hits_top1 += 1
            if hit_rank is not None and hit_rank <= 5:
                hits_top5 += 1
            if hit_rank is not None and hit_rank <= 10:
                hits_top10 += 1

            display = _ids_to_words(model, sent_ids)
            display[mid] = "[%s]" % display[mid]  # mark the blank
            target_word = model.id2word[target_id]
            print("\n  [%d] %s" % (idx, " ".join(display)))
            print("      target='%s'  hit_rank=%s" %
                  (target_word, hit_rank if hit_rank else ">10"))
            top5_str = ", ".join(
                "%s(%.2e)" % (model.id2word[t], p) for t, p in top[:5])
            print("      top-5: %s" % top5_str)

        N = len(sentences)
        if N:
            print("\n  Accuracy on %d cloze tests:" % N)
            print("    top-1  = %d/%d (%.0f%%)" %
                  (hits_top1, N, 100.0 * hits_top1 / N))
            print("    top-5  = %d/%d (%.0f%%)" %
                  (hits_top5, N, 100.0 * hits_top5 / N))
            print("    top-10 = %d/%d (%.0f%%)" %
                  (hits_top10, N, 100.0 * hits_top10 / N))

    print()


if __name__ == "__main__":
    main()
