"""v5 Evaluation — tests each layer independently then combined."""

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .kn import score_all as kn_score

OOD_CLOZE = [
    ("the king ruled the", "kingdom"),
    ("the queen ruled the", "kingdom"),
    ("water is a", "liquid"),
    ("the sun is a", "star"),
    ("dogs bark and cats", "meow"),
    ("birds fly and fish", "swim"),
    ("summer is hot and winter is", "cold"),
    ("fire is hot and ice is", "cold"),
    ("man is to woman as king is to", "queen"),
    ("paris is the capital of", "france"),
    ("london is the capital of", "england"),
    ("tokyo is the capital of", "japan"),
    ("apples are red and bananas are", "yellow"),
    ("grass is green and sky is", "blue"),
]


def _decay_boost(history: List[int], V: int, device: torch.device) -> torch.Tensor:
    """Logarithmic decay cache boost. Returns [V] distribution."""
    b = torch.zeros(V, dtype=torch.float32, device=device)
    if len(history) < 2:
        return b
    n = len(history)
    for i, tok in enumerate(history):
        age = n - 1 - i
        b[tok] += 1.0 / math.log(2.0 + age)
    s = b.sum()
    return b / s if s > 1e-9 else b


def _adoption_dist(kn: Dict, subs: Dict, current: int, context: List[int], V: int, device: torch.device) -> torch.Tensor:
    """Build adoption distribution via sparse bigram matrix — no Python loops.

    Uses pre-built bg_trans[V,V]: one sparse matmul instead of K calls to score_all.
    """
    bg = kn.get("bg_trans")
    if bg is None:
        return torch.zeros(V, dtype=torch.float32, device=device)

    sub_ids = subs["succ_ids"][current]  # [K]
    sub_wts = subs["succ_wts"][current]  # [K]

    # Find active substitutes
    active = sub_wts > 1e-6
    if not active.any():
        return torch.zeros(V, dtype=torch.float32, device=device)

    a_ids = sub_ids[active]   # [A]
    a_wts = sub_wts[active]   # [A]

    # Selector vector: one-hot weighted by substitute weights
    selector = torch.zeros(V, dtype=torch.float32, device=device)
    selector.scatter_add_(0, a_ids.long(), a_wts)

    # adoption = selector @ bg_trans.T = weighted sum of substitute bigram rows
    adoption = torch.sparse.mm(bg.t(), selector.unsqueeze(1)).squeeze(1)

    s = adoption.sum()
    if s > 1e-9:
        adoption = adoption / s
    return adoption


def eval_kn_layers(
    kn: Dict, subs: Dict, encoded: List[np.ndarray],
    cfg, device: torch.device, max_tokens: int = 50000,
) -> Dict:
    """Test each layer independently then combined."""
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return {}
    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, max_tokens)
    V = kn["V"]
    lam_s = cfg.stat_cache_lambda
    lam_a = cfg.adoption_lambda
    results = {}

    def _run(name, score_fn):
        nll, n, h1, h5 = 0.0, 0, 0, 0
        history = []
        for i in range(n_eval):
            ctx = flat[max(0, i - kn["max_order"]):i + 1].tolist()
            tgt = int(flat[i + 1])
            dist = score_fn(ctx, history, int(flat[i]))
            if cfg.mask_unk and hasattr(cfg, 'unk_id'):
                dist[cfg.unk_id] = 0.0
                dist = dist / dist.sum().clamp(min=1e-9)
            p = float(dist[tgt].item())
            nll += -math.log(max(p, 1e-12))
            _, top = torch.topk(dist, 50)
            tl = top.tolist()
            if tgt in tl[:1]: h1 += 1
            if tgt in tl[:5]: h5 += 1
            n += 1
            history.append(int(flat[i]))
            if len(history) > cfg.stat_cache_window:
                history = history[-cfg.stat_cache_window:]
        ppl = math.exp(nll / max(n, 1))
        r = {"ppl": ppl, "hit@1": h1 / max(n, 1), "hit@5": h5 / max(n, 1)}
        results[name] = r
        print(f"    [{name:25s}] PPL={ppl:.2f}  hit@1={r['hit@1']:.4f}  hit@5={r['hit@5']:.4f}")

    # 0. Direct bigram matrix (sanity check — bypasses hash scoring)
    bg = kn.get("bg_trans")
    if bg is not None:
        def _bg_score(ctx, hist, cur):
            selector = torch.zeros(V, dtype=torch.float32, device=device)
            selector[cur] = 1.0
            dist = torch.sparse.mm(bg.t(), selector.unsqueeze(1)).squeeze(1)
            eps = 0.05
            dist = (1 - eps) * dist + eps * (1.0 / V)
            return dist / dist.sum().clamp(min=1e-9)
        _run("bigram (sparse matrix)", _bg_score)

    # 1. KN only
    _run("KN-5gram", lambda ctx, hist, cur: kn_score(kn, ctx))

    # 2. KN + decay cache
    _run("KN + stat_cache", lambda ctx, hist, cur:
         (1 - lam_s) * kn_score(kn, ctx) + lam_s * _decay_boost(hist, V, device))

    # 3. KN + adoption
    _run("KN + adoption", lambda ctx, hist, cur:
         (1 - lam_a) * kn_score(kn, ctx) + lam_a * _adoption_dist(kn, subs, cur, ctx, V, device))

    # 4. KN + cache + adoption
    def full_score(ctx, hist, cur):
        base = kn_score(kn, ctx)
        cache = _decay_boost(hist, V, device)
        adopt = _adoption_dist(kn, subs, cur, ctx, V, device)
        return (1 - lam_s - lam_a) * base + lam_s * cache + lam_a * adopt
    _run("KN + cache + adoption", full_score)

    return results


def eval_ood(kn: Dict, subs: Dict, vocab, cfg, device: torch.device) -> Dict:
    V = kn["V"]
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []
    lam_s = cfg.stat_cache_lambda
    lam_a = cfg.adoption_lambda

    for context, answer in OOD_CLOZE:
        toks = vocab.encode_line(context)
        if not toks:
            continue
        tgt_ids = vocab.encode_line(answer)
        if not tgt_ids:
            continue
        tgt = tgt_ids[0]
        if tgt == vocab.unk_id:
            details.append({"context": context, "answer": answer, "skipped": True})
            continue

        base = kn_score(kn, toks)
        cache = _decay_boost(toks, V, device)
        cur = toks[-1]
        adopt = _adoption_dist(kn, subs, cur, toks, V, device)
        dist = (1 - lam_s - lam_a) * base + lam_s * cache + lam_a * adopt
        dist[vocab.unk_id] = 0.0
        dist = dist / dist.sum().clamp(min=1e-9)

        _, idx = torch.topk(dist, 50)
        top = idx.tolist()
        rank = top.index(tgt) + 1 if tgt in top else None
        if rank is not None:
            if rank <= 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            if rank <= 10: hits[10] += 1
        total += 1
        top5 = [vocab.itos[i] if i < len(vocab.itos) else "?" for i in top[:5]]
        details.append({"context": context, "answer": answer, "rank": rank, "top5_words": top5})

    recall = {f"ood_hit@{k}": hits[k] / max(total, 1) for k in [1, 5, 10]}
    recall["ood_total"] = total
    recall["ood_details"] = details
    n_skip = sum(1 for d in details if d.get("skipped"))
    print(f"    ood_hit@1={recall['ood_hit@1']:.3f}  hit@5={recall['ood_hit@5']:.3f}  "
          f"hit@10={recall['ood_hit@10']:.3f}  (tested={total}, skipped={n_skip})")
    for d in details:
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED")
            continue
        r = str(d["rank"]) if d["rank"] else ">50"
        t5 = " ".join(d.get("top5_words", []))
        print(f"      [{d['context']!r}] -> rank={r}  top5=[{t5}]")
    return recall


def eval_subs_quality(subs: Dict, vocab, device: torch.device) -> Dict:
    """Test if substitution tables contain meaningful semantic links."""
    cases = OOD_CLOZE
    V = subs["V"]
    found = 0
    total = 0
    details = []

    for context, answer in cases:
        ctx_ids = vocab.encode_line(context)
        ans_ids = vocab.encode_line(answer)
        if not ctx_ids or not ans_ids:
            continue
        tgt = ans_ids[0]
        if tgt == vocab.unk_id:
            details.append({"context": context, "answer": answer, "skipped": True})
            continue

        total += 1
        best_rank = None
        best_word = None
        best_type = None

        for w in set(ctx_ids):
            if w == vocab.unk_id:
                continue
            # Check successor-subs
            for j in range(subs["K"]):
                if subs["succ_ids"][w, j] == tgt:
                    r = j + 1
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_word = vocab.itos[w]
                        best_type = "succ"
                    break
            # Check predecessor-subs
            for j in range(subs["K"]):
                if subs["pred_ids"][w, j] == tgt:
                    r = j + 1
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_word = vocab.itos[w]
                        best_type = "pred"
                    break

        if best_rank is not None and best_rank <= 50:
            found += 1
        details.append({
            "context": context, "answer": answer,
            "sub_rank": best_rank, "via_word": best_word, "type": best_type,
        })

    recall = found / max(total, 1)
    print(f"  [subs_quality] Recall@50: {found}/{total} = {recall:.3f}")
    for d in details:
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED")
            continue
        r = str(d["sub_rank"]) if d["sub_rank"] else ">50"
        print(f"      [{d['context']!r}] -> '{d['answer']}' rank={r} via='{d.get('via_word')}'({d.get('type')})")
    return {"recall": recall, "found": found, "total": total, "details": details}
