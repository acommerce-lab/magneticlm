# magnetic/evaluator.py
#
# Evaluation harness for MagneticLM. Three evaluations:
#
#   1. WT103 perplexity in full mode (no cache). Uses the KN layer
#      only - this is the comparison point against the 14.20 PPL
#      baseline in MagneticLMFastRunner.py. It does NOT use the
#      excitation engine, because WT103 PPL is a per-token left-only
#      task where each token is scored independently. Mixing the
#      excitation engine in here would be a different benchmark.
#
#   2. In-distribution cloze. Picks random training sentences, masks
#      the middle word, and asks the generator's cloze scorer to
#      recover it.
#
#   3. Out-of-distribution cloze. Same as (2) but on a fixed
#      hand-designed list of simple English sentences that are
#      unlikely to appear verbatim in WT103. This measures real
#      compositional generalisation.
#
# The OOD test set lives alongside ood_cloze.py (colab/ood_cloze.py)
# so one source of truth. We re-import it here when available.

import array
import math
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from .model import MagneticModel
from .generator import MagneticGenerator
from .tokenizer import tokenize


class Evaluator:
    def __init__(self, model: MagneticModel):
        self.model = model
        self.generator = MagneticGenerator(model)

    # -------------------------------------------------------------------
    # 1. WT103 perplexity (KN only, batched)
    # -------------------------------------------------------------------
    def wt103_perplexity(
        self,
        test_lines: List[str],
        batch_size: int = 16384,
    ) -> float:
        """Full-mode no-cache WT103 perplexity using only the n-gram
        layer (Modified KN-5 with the full lambda)."""
        model = self.model
        dev = model.device
        K = model.config.max_ngram_order

        w2i = model.vocab.word2id
        toks = array.array('i')
        boundaries = array.array('i')
        boundaries.append(0)
        for line in test_lines:
            ws = tokenize(line)
            if len(ws) < 2:
                boundaries.append(len(toks))
                continue
            for w in ws:
                toks.append(w2i.get(w, -1))
            boundaries.append(len(toks))

        T = len(toks)
        if T < 2:
            return float("inf")
        np_toks = np.frombuffer(toks, dtype=np.int32)
        toks_gpu = torch.from_numpy(np_toks).to(
            device=dev, dtype=torch.int64)
        del toks, np_toks

        np_bnd = np.frombuffer(boundaries, dtype=np.int32)
        anchor = np.empty(T, dtype=np.int32)
        for i in range(len(np_bnd) - 1):
            s = int(np_bnd[i])
            e = int(np_bnd[i + 1])
            if e > s:
                anchor[s:e] = s
        anchor_gpu = torch.from_numpy(anchor).to(
            device=dev, dtype=torch.int64)

        total_logp = torch.zeros((), dtype=torch.float64, device=dev)
        total_tok = 0
        t0 = time.time()
        pos_index_all = torch.arange(1, T, dtype=torch.int64, device=dev)

        for start in range(0, pos_index_all.numel(), batch_size):
            end = min(start + batch_size, pos_index_all.numel())
            pidx = pos_index_all[start:end]
            nxt = toks_gpu[pidx]
            anch = anchor_gpu[pidx]
            ctx_start = torch.maximum(anch, pidx - K)
            j_range = torch.arange(K, dtype=torch.int64, device=dev)
            avail = (pidx - ctx_start)
            left_pad = (K - avail).clamp_min(0)
            col = j_range.unsqueeze(0) - left_pad.unsqueeze(1)
            src = ctx_start.unsqueeze(1) + col
            mask_pad = col < 0
            src_clamped = src.clamp(min=0)
            ctx_batch = toks_gpu[src_clamped]
            ctx_batch = torch.where(
                mask_pad, torch.full_like(ctx_batch, -1), ctx_batch)

            kn = model.ngram.kn_score_batch(ctx_batch, nxt)
            kn = kn.clamp(1e-10, 0.999)
            total_logp += torch.log(kn).to(torch.float64).sum()
            total_tok += kn.numel()

            if (start // batch_size) % 20 == 0:
                print("\r  WT103 eval: %d/%d (%.0fs)" %
                      (end, pos_index_all.numel(), time.time() - t0),
                      end="", flush=True)
        print()
        del toks_gpu, anchor_gpu
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if total_tok == 0:
            return float("inf")
        return math.exp(-float(total_logp.item()) / total_tok)

    # -------------------------------------------------------------------
    # 2. In-distribution cloze
    # -------------------------------------------------------------------
    def indist_cloze(
        self,
        train_lines: List[str],
        n: int = 10,
        min_len: int = 12,
        max_len: int = 30,
        top_k: int = 20,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """Pick n training sentences, mask the middle token, predict
        it with the full multi-force generator (bidirectional cloze
        scorer). Return a dict with hit rates."""
        model = self.model
        rng = random.Random(seed)
        candidates = []
        for line in train_lines:
            ws = tokenize(line)
            if not (min_len <= len(ws) <= max_len):
                continue
            ids = []
            ok = True
            for w in ws:
                tid = model.vocab.get(w, -1)
                if tid < 0:
                    ok = False
                    break
                ids.append(tid)
            if ok:
                candidates.append(ids)
            if len(candidates) >= 5000:
                break
        rng.shuffle(candidates)
        sentences = candidates[:n]

        top1 = top5 = top10 = 0
        per_sentence = []
        for sent_ids in sentences:
            mid = len(sent_ids) // 2
            target_id = sent_ids[mid]
            left = sent_ids[:mid]
            right = sent_ids[mid + 1:]

            scores = self.generator.score_cloze(left, right)
            topv, topi = torch.topk(scores, max(10, top_k))
            top_ids = topi.tolist()

            best_rank = None
            for r, tid in enumerate(top_ids, 1):
                if tid == target_id:
                    best_rank = r
                    break
            if best_rank == 1:
                top1 += 1
            if best_rank is not None and best_rank <= 5:
                top5 += 1
            if best_rank is not None and best_rank <= 10:
                top10 += 1

            if verbose:
                disp = self.model.vocab.lookup(sent_ids)
                disp[mid] = "[%s]" % disp[mid]
                tw = self.model.vocab.id2word[target_id]
                tops = ", ".join(
                    "%s(%.2e)" % (self.model.vocab.id2word[tid], float(p))
                    for tid, p in zip(topi.tolist()[:5], topv.tolist()[:5]))
                rank_str = str(best_rank) if best_rank is not None else ">top_k"
                print("  %s" % " ".join(disp))
                print("    target='%s' rank=%s  top-5: %s" %
                      (tw, rank_str, tops))

        N = len(sentences)
        return {
            "n": N,
            "top1": top1,
            "top5": top5,
            "top10": top10,
        }

    # -------------------------------------------------------------------
    # 3. Out-of-distribution cloze (hand-designed sentences)
    # -------------------------------------------------------------------
    def ood_cloze(
        self,
        tests: List[Tuple[str, List[str], str]],
        top_k: int = 20,
        verbose: bool = True,
    ) -> dict:
        """Run OOD cloze tests. Each entry is
        (sentence_with_[MASK], acceptable_answers, category)."""
        model = self.model
        per_cat = {}
        results = []

        for sent, acceptable, cat in tests:
            if "[MASK]" not in sent:
                continue
            before, after = sent.split("[MASK]", 1)
            left_ids, _ = model.vocab.tokenize_text(before)
            right_ids, _ = model.vocab.tokenize_text(after)

            accept_ids = set()
            for w in acceptable:
                tid = model.vocab.get(w.lower(), -1)
                if tid >= 0:
                    accept_ids.add(tid)
            if not accept_ids:
                continue  # can't score

            scores = self.generator.score_cloze(left_ids, right_ids)
            topv, topi = torch.topk(scores, max(10, top_k))
            top_ids = topi.tolist()

            best_rank = None
            for r, tid in enumerate(top_ids, 1):
                if tid in accept_ids:
                    best_rank = r
                    break

            results.append((sent, acceptable, cat, best_rank,
                            topi.tolist()[:5], topv.tolist()[:5]))

            if cat not in per_cat:
                per_cat[cat] = {"n": 0, "top1": 0, "top5": 0, "top10": 0}
            per_cat[cat]["n"] += 1
            if best_rank == 1:
                per_cat[cat]["top1"] += 1
            if best_rank is not None and best_rank <= 5:
                per_cat[cat]["top5"] += 1
            if best_rank is not None and best_rank <= 10:
                per_cat[cat]["top10"] += 1

            if verbose:
                tops = ", ".join(
                    "%s(%.2e)" % (self.model.vocab.id2word[tid], float(p))
                    for tid, p in zip(topi.tolist()[:5], topv.tolist()[:5]))
                rank_str = str(best_rank) if best_rank is not None else ">%d" % top_k
                print("  %s  [%s]" % (sent, cat))
                print("    accept=%s  rank=%s  top-5: %s" %
                      (acceptable, rank_str, tops))

        N = len(results)
        top1 = sum(1 for r in results if r[3] == 1)
        top5 = sum(1 for r in results if r[3] is not None and r[3] <= 5)
        top10 = sum(1 for r in results if r[3] is not None and r[3] <= 10)
        return {
            "n": N,
            "top1": top1,
            "top5": top5,
            "top10": top10,
            "by_category": per_cat,
            "results": results,
        }
