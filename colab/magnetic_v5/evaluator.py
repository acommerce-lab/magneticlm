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
    """Logarithmic decay cache boost — vectorized. Returns [V] distribution."""
    if len(history) < 2:
        return torch.zeros(V, dtype=torch.float32, device=device)
    h = torch.tensor(history, dtype=torch.int64, device=device)
    n = h.numel()
    ages = torch.arange(n - 1, -1, -1, dtype=torch.float32, device=device)
    weights = 1.0 / torch.log(2.0 + ages)
    b = torch.zeros(V, dtype=torch.float32, device=device)
    b.scatter_add_(0, h, weights)
    s = b.sum()
    return b / s if s > 1e-9 else b


def _adoption_dist(kn, subs, current, context, V, device, n_hops=3, damping=0.7):
    """Multi-hop PPMI diffusion: seed from context, spread through semantic graph."""
    ppmi = subs.get("ppmi_matrix")
    if ppmi is None:
        return torch.zeros(V, dtype=torch.float32, device=device)

    v = torch.zeros(V, dtype=torch.float32, device=device)
    ctx_words = context[-5:] if context else []
    w = 1.0
    for t in reversed(ctx_words):
        if 0 <= t < V:
            v[t] += w
        w *= 0.7
    if v.sum() < 1e-9:
        return torch.zeros(V, dtype=torch.float32, device=device)
    v = v / v.sum()

    accumulated = torch.zeros(V, dtype=torch.float32, device=device)
    for hop in range(n_hops):
        v = torch.sparse.mm(ppmi.t(), v.unsqueeze(1)).squeeze(1)
        v = v * damping
        accumulated = accumulated + v * (damping ** hop)

    glow = subs.get("glow")
    if glow is not None:
        accumulated = accumulated * (1.0 + glow)

    for t in ctx_words:
        if 0 <= t < V:
            accumulated[t] = 0.0

    s = accumulated.sum()
    if s > 1e-9:
        accumulated = accumulated / s
    return accumulated


def eval_kn_layers(
    kn: Dict, subs: Dict, encoded: List[np.ndarray],
    cfg, device: torch.device, max_tokens: int = 5000,
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
        unk_id = getattr(cfg, 'unk_id', -1)
        for i in range(n_eval):
            ctx = flat[max(0, i - kn["max_order"]):i + 1].tolist()
            tgt = int(flat[i + 1])
            dist = score_fn(ctx, history, int(flat[i]))
            # Do NOT mask unk for PPL — masking makes unk targets get p=0
            p = float(dist[tgt].item())
            nll += -math.log(max(p, 1e-12))
            # For hit@k, mask unk so it doesn't steal top positions
            dist_masked = dist.clone()
            if unk_id >= 0:
                dist_masked[unk_id] = 0.0
            _, top = torch.topk(dist_masked, 50)
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
    bg = subs.get("bg_trans") if "bg_trans" in subs else kn.get("bg_trans")
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

    # 4. KN + cache + diffusion (weighted sum — adoption is now multi-hop)
    def full_score(ctx, hist, cur):
        base = kn_score(kn, ctx)
        cache = _decay_boost(hist, V, device)
        adopt = _adoption_dist(kn, subs, cur, ctx, V, device)
        return (1 - lam_s - lam_a) * base + lam_s * cache + lam_a * adopt
    _run("KN + cache + diffusion", full_score)

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

        # Test THREE scoring modes for each OOD case:
        base = kn_score(kn, toks)
        cache = _decay_boost(toks, V, device)
        cur = toks[-1]
        adopt = _adoption_dist(kn, subs, cur, toks, V, device)

        # Mode 1: KN+cache only (statistical)
        dist_kn = (1 - lam_s) * base + lam_s * cache
        dist_kn[vocab.unk_id] = 0.0
        dist_kn = dist_kn / dist_kn.sum().clamp(min=1e-9)

        # Mode 2: Diffusion only (semantic — isolated concept test)
        dist_diff = adopt.clone()
        dist_diff[vocab.unk_id] = 0.0
        s = dist_diff.sum()
        if s > 1e-9:
            dist_diff = dist_diff / s

        # Mode 3: Combined
        dist = (1 - lam_s - lam_a) * base + lam_s * cache + lam_a * adopt
        dist[vocab.unk_id] = 0.0
        dist = dist / dist.sum().clamp(min=1e-9)

        # Rank in each mode
        def _rank(d, t):
            _, idx = torch.topk(d, 50)
            top = idx.tolist()
            return top.index(t) + 1 if t in top else None, [vocab.itos[i] if i < len(vocab.itos) else "?" for i in top[:5]]

        rank_kn, top5_kn = _rank(dist_kn, tgt)
        rank_diff, top5_diff = _rank(dist_diff, tgt)
        rank, top5 = _rank(dist, tgt)

        if rank is not None:
            if rank <= 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            if rank <= 10: hits[10] += 1
        total += 1
        details.append({
            "context": context, "answer": answer,
            "rank_kn": rank_kn, "top5_kn": top5_kn,
            "rank_diff": rank_diff, "top5_diff": top5_diff,
            "rank": rank, "top5_words": top5,
        })

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
        rk = str(d.get("rank_kn") or ">50")
        rd = str(d.get("rank_diff") or ">50")
        rc = str(d.get("rank") or ">50")
        t5d = " ".join(d.get("top5_diff", []))
        t5c = " ".join(d.get("top5_words", []))
        print(f"      [{d['context']!r}] -> '{d['answer']}' KN={rk} DIFF={rd} MIX={rc}  diff_top5=[{t5d}]")
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
            # Check substitutes
            for j in range(subs["K"]):
                if subs["sub_ids"][w, j] == tgt:
                    r = j + 1
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_word = vocab.itos[w]
                        best_type = "cosine"
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
