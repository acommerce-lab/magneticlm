# pulse/evaluator.py
#
# Evaluation using ONLY the dual-map model as described:
#   - Contextual map (bigram succession)
#   - Semantic map (iterative reward/penalty)
#   - Adoption mechanism (children of semantic neighbors)
#
# No KN backoff. No physics. No position similarity.
# Scoring: P(w|ctx) = alpha * P_direct + beta * P_adopt + gamma * P_unigram
#
# Evaluations:
#   1. WT103 perplexity (dual-map scoring)
#   2. Candidate hit rate (diagnostic)
#   3. OOD cloze (masked word prediction)

import array
import math
import re
import time
from typing import List, Tuple

import numpy as np
import torch

_SPLIT_RE = re.compile(r'[.,;!?()\[\]{}"]+')


def _tokenize(line: str):
    return [w for w in _SPLIT_RE.sub(' ', line.lower()).split() if w]


class DualMapEvaluator:
    """Evaluator for the dual-map pulse-trained model."""

    def __init__(self, model):
        """model: a PulseModel with .config, .device, .vocab,
        .ctx_map, .semantic_map, .freq_gpu"""
        self.model = model

    # ==================================================================
    # 1. WT103 Perplexity (dual-map scoring)
    # ==================================================================
    def wt103_ppl(self, test_lines: List[str],
                  batch_size: int = 16384) -> float:
        """Per-token left-to-right perplexity using dual-map scoring:

        P(w_t | context) = alpha * P_ctx(w_t | w_{t-1})
                         + beta  * P_adopt(w_t | w_{t-1})
                         + gamma * P_unigram(w_t)

        Where:
          P_ctx = direct bigram probability from contextual map
          P_adopt = weighted adoption via semantic neighbors
          P_unigram = frequency-based fallback
        """
        model = self.model
        cfg = model.config
        dev = model.device
        ctx_map = model.ctx_map
        sem_map = model.semantic_map
        w2i = model.vocab.word2id

        alpha = cfg.alpha_direct
        beta = cfg.beta_adopt
        gamma = cfg.gamma_unigram

        # Tokenize test set
        toks = array.array('i')
        boundaries = array.array('i')
        boundaries.append(0)
        for line in test_lines:
            ws = _tokenize(line)
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

        # Per-token sentence boundary (to avoid cross-sentence bigrams)
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

        # Score each token position (skip position 0 = no context)
        pos_all = torch.arange(1, T, dtype=torch.int64, device=dev)

        for start in range(0, pos_all.numel(), batch_size):
            end = min(start + batch_size, pos_all.numel())
            pidx = pos_all[start:end]
            B = pidx.numel()

            target = toks_gpu[pidx]          # (B,) next word
            current = toks_gpu[pidx - 1]     # (B,) previous word

            # Skip cross-sentence boundaries
            anch = anchor_gpu[pidx]
            same_sent = (pidx - 1) >= anch
            oov = (target < 0) | (current < 0) | (~same_sent)

            safe_current = current.clamp_min(0)
            safe_target = target.clamp_min(0)

            # P_direct: bigram probability from contextual map
            p_direct = ctx_map.lookup_batch(safe_current, safe_target)

            # P_adopt: adoption probability from semantic neighbors
            p_adopt = ctx_map.adoption_batch(
                safe_current, safe_target, sem_map, cfg)

            # P_unigram: frequency-based fallback
            p_unigram = ctx_map.unigram_probs[safe_target]

            # Combined probability
            p_combined = (alpha * p_direct +
                          beta * p_adopt +
                          gamma * p_unigram).clamp(1e-10, 0.999)

            # OOV tokens get floor probability
            p_combined = torch.where(
                oov,
                torch.full_like(p_combined, 1e-10),
                p_combined)

            total_logp += torch.log(p_combined).to(torch.float64).sum()
            total_tok += B

            if (start // batch_size) % 20 == 0:
                print("\r  PPL: %d/%d (%.0fs)" %
                      (end, pos_all.numel(), time.time() - t0),
                      end="", flush=True)

        print()
        del toks_gpu, anchor_gpu
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if total_tok == 0:
            return float("inf")
        return math.exp(-float(total_logp.item()) / total_tok)

    # ==================================================================
    # 2. Candidate hit rate (diagnostic)
    # ==================================================================
    def candidate_hit_rate(self, test_lines: List[str],
                           max_tokens: int = 100000) -> dict:
        """For each bigram in the test set, check if the next word is:
          - A direct contextual child
          - An adopted child (via semantic neighbor)
          - Neither (miss)

        Also reports average candidate set size.
        """
        model = self.model
        cfg = model.config
        ctx_map = model.ctx_map
        sem_map = model.semantic_map
        w2i = model.vocab.word2id

        direct_hits = 0
        adopted_hits = 0
        misses = 0
        total_candidates = 0
        checked = 0

        t0 = time.time()
        for line in test_lines:
            ws = _tokenize(line)
            if len(ws) < 2:
                continue
            ids = [w2i.get(w, -1) for w in ws]

            for i in range(len(ids) - 1):
                if ids[i] < 0 or ids[i + 1] < 0:
                    continue

                current = ids[i]
                actual_next = ids[i + 1]
                n_cand = ctx_map.num_children(current)

                # Direct child?
                if ctx_map.has_child(current, actual_next):
                    direct_hits += 1
                    total_candidates += n_cand
                    checked += 1
                    if checked >= max_tokens:
                        break
                    continue

                # Adopted child?
                neighbors = sem_map.get_all_neighbors(current)
                neighbors = [(nid, w) for nid, w in neighbors
                             if w >= cfg.adoption_min_weight]
                neighbors = neighbors[:cfg.adoption_neighbors]

                found_adopted = False
                for sem_id, _ in neighbors:
                    n_cand += ctx_map.num_children(sem_id)
                    if ctx_map.has_child(sem_id, actual_next):
                        found_adopted = True
                        # Don't break — continue counting candidates

                if found_adopted:
                    adopted_hits += 1
                else:
                    misses += 1

                total_candidates += n_cand
                checked += 1
                if checked >= max_tokens:
                    break
            if checked >= max_tokens:
                break

        elapsed = time.time() - t0
        N = checked
        if N == 0:
            return {"error": "no valid tokens"}

        return {
            "total_checked": N,
            "direct_hits": direct_hits,
            "adopted_hits": adopted_hits,
            "misses": misses,
            "direct_hit_rate": direct_hits / N,
            "adopted_hit_rate": adopted_hits / N,
            "total_hit_rate": (direct_hits + adopted_hits) / N,
            "miss_rate": misses / N,
            "avg_candidates": total_candidates / N,
            "time_s": elapsed,
        }

    # ==================================================================
    # 3. OOD Cloze (masked word prediction)
    # ==================================================================
    def ood_cloze(self, tests: List[Tuple[str, List[str], str]],
                  top_k: int = 20,
                  verbose: bool = True) -> dict:
        """Bidirectional cloze using dual-map scoring.

        For each candidate word c:
          left_score = P_ctx(c | last_left_word) + adoption contribution
          right_score = P_ctx(first_right_word | c) + adoption contribution
          combined = left_score * (1 + right_score)

        Returns top-K and checks against acceptable answers.
        """
        model = self.model
        cfg = model.config
        dev = model.device
        ctx_map = model.ctx_map
        sem_map = model.semantic_map
        V = len(model.vocab)

        per_cat = {}
        results = []

        for sent, acceptable, cat in tests:
            if "[MASK]" not in sent:
                continue

            before, after = sent.split("[MASK]", 1)
            left_words = _tokenize(before)
            right_words = _tokenize(after)
            left_ids = [model.vocab.word2id.get(w, -1) for w in left_words]
            right_ids = [model.vocab.word2id.get(w, -1) for w in right_words]
            left_ids = [i for i in left_ids if i >= 0]
            right_ids = [i for i in right_ids if i >= 0]

            accept_ids = set()
            for w in acceptable:
                tid = model.vocab.word2id.get(w.lower(), -1)
                if tid >= 0:
                    accept_ids.add(tid)
            if not accept_ids:
                continue

            # Build candidate set from the last left word
            if left_ids:
                last_left = left_ids[-1]
                neighbors = sem_map.get_all_neighbors(last_left)
                neighbors = [(nid, w) for nid, w in neighbors
                             if w >= cfg.adoption_min_weight]
                neighbors = neighbors[:cfg.adoption_neighbors]
                cand_ids, is_direct, _ = ctx_map.get_candidate_set(
                    last_left, neighbors)
            else:
                cand_ids = torch.arange(V, dtype=torch.int64, device=dev)

            if cand_ids.numel() == 0:
                cand_ids = torch.arange(V, dtype=torch.int64, device=dev)

            # Score candidates
            # Left: P(candidate | last_left)
            if left_ids:
                last_left_t = torch.full(
                    (cand_ids.numel(),), left_ids[-1],
                    dtype=torch.int64, device=dev)
                left_score = ctx_map.lookup_batch(last_left_t, cand_ids)

                # Add adoption from left context
                left_adopt = ctx_map.adoption_batch(
                    last_left_t, cand_ids, sem_map, cfg)
                left_total = (cfg.alpha_direct * left_score +
                              cfg.beta_adopt * left_adopt +
                              cfg.gamma_unigram * ctx_map.unigram_probs[cand_ids])
            else:
                left_total = ctx_map.unigram_probs[cand_ids]

            # Right: P(first_right | candidate)
            if right_ids:
                first_right = right_ids[0]
                first_right_t = torch.full(
                    (cand_ids.numel(),), first_right,
                    dtype=torch.int64, device=dev)
                right_score = ctx_map.lookup_batch(cand_ids, first_right_t)

                right_adopt = ctx_map.adoption_batch(
                    cand_ids, first_right_t, sem_map, cfg)
                right_total = (cfg.alpha_direct * right_score +
                               cfg.beta_adopt * right_adopt +
                               cfg.gamma_unigram * ctx_map.unigram_probs[
                                   first_right_t.clamp_min(0)])
            else:
                right_total = torch.ones(cand_ids.numel(),
                                         dtype=torch.float32, device=dev)

            # Combine left and right
            combined = left_total * (1.0 + right_total)

            # Map back to full vocab for ranking
            scores = torch.full((V,), -float('inf'),
                                dtype=torch.float32, device=dev)
            scores[cand_ids] = combined

            topv, topi = torch.topk(scores, min(top_k, V))
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
                    "%s(%.2e)" % (model.vocab.id2word[tid], float(p))
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
