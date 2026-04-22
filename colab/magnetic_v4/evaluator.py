"""Evaluation metrics: PPL, hit-rate, OOD cloze, generation.

Single responsibility: score the model against held-out/synthetic data.
Never touches training, never rebuilds the graph.

All forward passes go through wave.propagate_batch where possible to
keep GPU busy.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .graph import Graph
from .scorer import score as score_fn
from .tokenizer import Vocab
from .wave import propagate, propagate_batch
from .kn import KNModel, score_next_token as kn_score_next
from .cache import DecayCache, ConceptCache, DirectionalSubs, concept_boost


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


def _one_hot(tokens: List[int], V: int, device: torch.device) -> torch.Tensor:
    """Build impulse focusing on the LAST token (most relevant for bigram).

    Also gives diminishing weight to prior context for multi-hop PPR walks.
    """
    x = torch.zeros(V, dtype=torch.float32, device=device)
    if not tokens:
        return x
    # Last token gets dominant weight
    x[int(tokens[-1])] += 1.0
    # Prior context gets small diminishing weight (helps multi-hop)
    w = 0.3
    for t in reversed(tokens[-8:-1]):
        x[int(t)] += w
        w *= 0.5
    s = x.sum().clamp(min=1e-9)
    return x / s


def _build_batch_impulses(contexts: List[List[int]], V: int, device: torch.device) -> torch.Tensor:
    """Stack impulses for batched propagation. Shape [B, V]."""
    B = len(contexts)
    imp = torch.zeros(B, V, dtype=torch.float32, device=device)
    for i, ctx in enumerate(contexts):
        if not ctx:
            continue
        imp[i, int(ctx[-1])] += 1.0
        w = 0.3
        for t in reversed(ctx[-8:-1]):
            imp[i, int(t)] += w
            w *= 0.5
        s = imp[i].sum().clamp(min=1e-9)
        imp[i] = imp[i] / s
    return imp


def evaluate_ppl(
    graph: Graph,
    encoded: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
    max_tokens: Optional[int] = None,
) -> Dict:
    """Batched PPL via propagate_batch. Context = sliding window up to 8 tokens."""
    max_tokens = max_tokens or cfg.eval_max_tokens
    V = graph.vocab_size
    B = max(1, int(cfg.eval_batch_size))

    # Flatten valid arrays
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return {"ppl": float("inf"), "n_tokens": 0}
    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, max_tokens)

    t_start = time.time()
    nll_sum = 0.0
    n = 0

    # Build batches of (context_window, target_token) pairs
    i = 0
    while i < n_eval:
        j = min(i + B, n_eval)
        contexts = [flat[max(0, k - 8):k + 1].tolist() for k in range(i, j)]
        targets = flat[i + 1 : j + 1]

        imp = _build_batch_impulses(contexts, V, device)
        z = propagate_batch(graph, imp, cfg, device)   # [B, V]
        dist = score_fn(z, cfg, unk_id=vocab.unk_id if cfg.mask_unk_in_eval else -1)
        # gather target probs
        tgt = torch.from_numpy(targets).to(device).long()
        p = dist.gather(1, tgt.unsqueeze(1)).squeeze(1).clamp(min=1e-12)
        nll_sum += float((-torch.log(p)).sum().item())
        n += int(tgt.numel())
        i = j

    return {
        "ppl": math.exp(nll_sum / max(n, 1)),
        "n_tokens": n,
        "time_s": time.time() - t_start,
    }


def evaluate_hit_rate(
    graph: Graph,
    encoded: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
    top_k_list: Tuple[int, ...] = (1, 5, 10, 50),
    max_tokens: Optional[int] = None,
) -> Dict:
    max_tokens = max_tokens or cfg.eval_max_tokens
    V = graph.vocab_size
    B = max(1, int(cfg.eval_batch_size))
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return {f"hit@{k}": 0.0 for k in top_k_list}
    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, max_tokens)

    hits = {k: 0 for k in top_k_list}
    n = 0
    max_k = max(top_k_list)

    i = 0
    while i < n_eval:
        j = min(i + B, n_eval)
        contexts = [flat[max(0, k - 8):k + 1].tolist() for k in range(i, j)]
        targets = flat[i + 1 : j + 1]

        imp = _build_batch_impulses(contexts, V, device)
        z = propagate_batch(graph, imp, cfg, device)
        dist = score_fn(z, cfg, unk_id=vocab.unk_id if cfg.mask_unk_in_eval else -1)
        _, top_idx = torch.topk(dist, max_k, dim=-1)
        tgt = torch.from_numpy(targets).to(device).long()
        y_exp = tgt.unsqueeze(1)
        for k in top_k_list:
            hits[k] += int((top_idx[:, :k] == y_exp).any(dim=-1).sum().item())
        n += int(tgt.numel())
        i = j

    return {f"hit@{k}": hits[k] / max(n, 1) for k in top_k_list}


def evaluate_ood_cloze(
    graph: Graph,
    vocab: Vocab,
    cfg,
    device: torch.device,
    cases: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    cases = cases or OOD_CLOZE
    V = graph.vocab_size
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []

    for context, answer in cases:
        toks = vocab.encode_line(context)
        if not toks:
            continue
        tgt_ids = vocab.encode_line(answer)
        if not tgt_ids:
            continue
        tgt = tgt_ids[0]
        is_unk = (tgt == vocab.unk_id)
        if is_unk:
            details.append({
                "context": context, "answer": answer,
                "rank": None, "top5_words": [], "skipped": True,
            })
            continue

        imp = _one_hot(toks, V, device)
        z = propagate(graph, imp, cfg, device).z
        dist = score_fn(z, cfg, unk_id=vocab.unk_id)

        _, idx = torch.topk(dist, 50)
        top = idx.tolist()
        rank = top.index(tgt) + 1 if tgt in top else None
        if rank is not None:
            if rank <= 1: hits[1] += 1
            if rank <= 5: hits[5] += 1
            if rank <= 10: hits[10] += 1
        total += 1
        top5 = [vocab.itos[i] if i < len(vocab.itos) else "?" for i in top[:5]]
        details.append({
            "context": context, "answer": answer,
            "rank": rank, "top5_words": top5,
        })

    return {
        "ood_hit@1": hits[1] / max(total, 1),
        "ood_hit@5": hits[5] / max(total, 1),
        "ood_hit@10": hits[10] / max(total, 1),
        "ood_total": total,
        "ood_details": details,
    }


def evaluate_generation(
    graph: Graph,
    vocab: Vocab,
    cfg,
    device: torch.device,
    prompts: List[str],
) -> List[Dict]:
    V = graph.vocab_size
    out = []
    for prompt in prompts:
        toks = vocab.encode_line(prompt)
        if not toks:
            continue
        ids = list(toks)
        for _ in range(cfg.gen_length):
            imp = _one_hot(ids, V, device)
            z = propagate(graph, imp, cfg, device).z
            dist = score_fn(z, cfg, unk_id=vocab.unk_id)
            if cfg.gen_temperature != 1.0:
                dist = torch.pow(dist.clamp(min=1e-12), 1.0 / max(cfg.gen_temperature, 1e-6))
                dist = dist / dist.sum().clamp(min=1e-9)
            if cfg.gen_top_k > 0:
                vals, idx = torch.topk(dist, min(cfg.gen_top_k, dist.numel()))
                probs = vals / vals.sum().clamp(min=1e-9)
                if probs.sum() < 1e-9:
                    pick = int(idx[0].item())
                else:
                    pick = int(idx[torch.multinomial(probs, 1).item()].item())
            else:
                if dist.sum() < 1e-9:
                    pick = int(dist.argmax().item())
                else:
                    pick = int(torch.multinomial(dist, 1).item())
            ids.append(pick)
        text = " ".join(vocab.itos[i] for i in ids)
        out.append({"prompt": prompt, "generated": text})
    return out


def evaluate_graph_concepts(
    graph: Graph,
    vocab: Vocab,
    device: torch.device,
    cases: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 50,
) -> Dict:
    """Graph Neighbor Recall: tests if the raw graph contains conceptual links.

    For each OOD case (context, answer), checks whether 'answer' is among
    the top-K graph neighbors of ANY context word. This measures GRAPH QUALITY
    independently of scoring/propagation — if the answer isn't even a neighbor,
    no scoring formula can save it.

    This answers: "does the graph know that king→kingdom, paris→france?"
    """
    cases = cases or OOD_CLOZE
    fwd = graph.sem_fwd.coalesce()
    bwd = graph.sem_bwd.coalesce()
    V = graph.vocab_size

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
            details.append({
                "context": context, "answer": answer,
                "skipped": True, "reason": "unk",
            })
            continue

        total += 1
        best_rank = None
        best_word = None
        best_dir = None

        for w in set(ctx_ids):
            if w == vocab.unk_id:
                continue
            # Check forward neighbors of w (transpose: source→target)
            row = torch.zeros(V, device=device)
            row[w] = 1.0
            fwd_scores = torch.sparse.mm(fwd.t(), row.unsqueeze(1)).squeeze(1)
            _, fwd_top = torch.topk(fwd_scores, min(top_k, V))
            fwd_list = fwd_top.tolist()
            if tgt in fwd_list:
                r = fwd_list.index(tgt) + 1
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_word = vocab.itos[w]
                    best_dir = "fwd"
            # Check backward neighbors of w
            bwd_scores = torch.sparse.mm(bwd.t(), row.unsqueeze(1)).squeeze(1)
            _, bwd_top = torch.topk(bwd_scores, min(top_k, V))
            bwd_list = bwd_top.tolist()
            if tgt in bwd_list:
                r = bwd_list.index(tgt) + 1
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_word = vocab.itos[w]
                    best_dir = "bwd"

        if best_rank is not None and best_rank <= top_k:
            found += 1
        details.append({
            "context": context, "answer": answer,
            "graph_rank": best_rank, "via_word": best_word,
            "direction": best_dir,
        })

    recall = found / max(total, 1)
    print(f"  [concept] Graph Neighbor Recall@{top_k}: {found}/{total} = {recall:.3f}")
    for d in details:
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED ({d['reason']})")
            continue
        r = d["graph_rank"]
        r_str = str(r) if r else f">{top_k}"
        via = d.get("via_word", "?")
        dir_ = d.get("direction", "?")
        print(f"      [{d['context']!r}] -> '{d['answer']}' rank={r_str} via='{via}'({dir_})")

    return {
        "graph_recall": recall,
        "found": found,
        "total": total,
        "top_k": top_k,
        "details": details,
    }


def evaluate_kn_diagnostics(
    kn_model,
    ccache,
    encoded: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
    max_tokens: int = 50000,
    dir_subs: Optional["DirectionalSubs"] = None,
) -> Dict:
    """Test KN layer: without cache, with stat cache, with concept cache."""
    arrs = [a for a in encoded if a.size > 1]
    if not arrs or kn_model is None:
        return {}

    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, max_tokens)
    V = kn_model.vocab_size
    results = {}

    # --- KN-5gram alone ---
    nll, n, hits1, hits5 = 0.0, 0, 0, 0
    for i in range(n_eval):
        ctx_start = max(0, i - kn_model.max_order)
        ctx = flat[ctx_start:i + 1].tolist()
        tgt = int(flat[i + 1])
        dist = kn_score_next(kn_model, ctx, V)
        p = float(dist[tgt].item())
        nll += -math.log(max(p, 1e-12))
        _, top = torch.topk(dist, 50)
        top_list = top.tolist()
        if tgt in top_list[:1]: hits1 += 1
        if tgt in top_list[:5]: hits5 += 1
        n += 1
    kn_ppl = math.exp(nll / max(n, 1))
    results["kn_only"] = {"ppl": kn_ppl, "hit@1": hits1/max(n,1), "hit@5": hits5/max(n,1)}
    print(f"    [KN-5gram only]     PPL={kn_ppl:.2f}  hit@1={hits1/max(n,1):.4f}  hit@5={hits5/max(n,1):.4f}")

    # --- KN + stat cache (decay) ---
    nll, n, hits1, hits5 = 0.0, 0, 0, 0
    cache = DecayCache(window=cfg.stat_cache_window)
    lam = cfg.stat_cache_lambda
    for i in range(n_eval):
        ctx_start = max(0, i - kn_model.max_order)
        ctx = flat[ctx_start:i + 1].tolist()
        tgt = int(flat[i + 1])
        kn_dist = kn_score_next(kn_model, ctx, V)
        cache_dist = cache.get_boost(V, device)
        dist = (1 - lam) * kn_dist + lam * cache_dist
        dist = dist / dist.sum().clamp(min=1e-9)
        p = float(dist[tgt].item())
        nll += -math.log(max(p, 1e-12))
        _, top = torch.topk(dist, 50)
        top_list = top.tolist()
        if tgt in top_list[:1]: hits1 += 1
        if tgt in top_list[:5]: hits5 += 1
        n += 1
        cache.observe(int(flat[i]))
    kn_cache_ppl = math.exp(nll / max(n, 1))
    results["kn_stat_cache"] = {"ppl": kn_cache_ppl, "hit@1": hits1/max(n,1), "hit@5": hits5/max(n,1)}
    print(f"    [KN + stat cache]   PPL={kn_cache_ppl:.2f}  hit@1={hits1/max(n,1):.4f}  hit@5={hits5/max(n,1):.4f}")

    # --- KN + concept cache ---
    if ccache is not None:
        nll, n, hits1, hits5 = 0.0, 0, 0, 0
        clam = cfg.concept_cache_lambda
        for i in range(n_eval):
            ctx_start = max(0, i - kn_model.max_order)
            ctx = flat[ctx_start:i + 1].tolist()
            tgt = int(flat[i + 1])
            kn_dist = kn_score_next(kn_model, ctx, V)
            cboost = concept_boost(ccache, ctx[-8:], device)
            dist = (1 - clam) * kn_dist + clam * cboost
            dist = dist / dist.sum().clamp(min=1e-9)
            p = float(dist[tgt].item())
            nll += -math.log(max(p, 1e-12))
            _, top = torch.topk(dist, 50)
            top_list = top.tolist()
            if tgt in top_list[:1]: hits1 += 1
            if tgt in top_list[:5]: hits5 += 1
            n += 1
        kn_concept_ppl = math.exp(nll / max(n, 1))
        results["kn_concept_cache"] = {"ppl": kn_concept_ppl, "hit@1": hits1/max(n,1), "hit@5": hits5/max(n,1)}
        print(f"    [KN + concept cache] PPL={kn_concept_ppl:.2f}  hit@1={hits1/max(n,1):.4f}  hit@5={hits5/max(n,1):.4f}")

    # --- KN + both caches ---
    if ccache is not None:
        nll, n, hits1, hits5 = 0.0, 0, 0, 0
        cache2 = DecayCache(window=cfg.stat_cache_window)
        for i in range(n_eval):
            ctx_start = max(0, i - kn_model.max_order)
            ctx = flat[ctx_start:i + 1].tolist()
            tgt = int(flat[i + 1])
            kn_dist = kn_score_next(kn_model, ctx, V)
            s_boost = cache2.get_boost(V, device)
            c_boost = concept_boost(ccache, ctx[-8:], device)
            dist = (1 - lam - clam) * kn_dist + lam * s_boost + clam * c_boost
            dist = dist / dist.sum().clamp(min=1e-9)
            p = float(dist[tgt].item())
            nll += -math.log(max(p, 1e-12))
            _, top = torch.topk(dist, 50)
            top_list = top.tolist()
            if tgt in top_list[:1]: hits1 += 1
            if tgt in top_list[:5]: hits5 += 1
            n += 1
            cache2.observe(int(flat[i]))
        full_ppl = math.exp(nll / max(n, 1))
        results["kn_both_caches"] = {"ppl": full_ppl, "hit@1": hits1/max(n,1), "hit@5": hits5/max(n,1)}
        print(f"    [KN + both caches]  PPL={full_ppl:.2f}  hit@1={hits1/max(n,1):.4f}  hit@5={hits5/max(n,1):.4f}")

    # --- KN + directional adoption (successor-subs expand candidates) ---
    if dir_subs is not None:
        nll, n, hits1, hits5 = 0.0, 0, 0, 0
        cache3 = DecayCache(window=cfg.stat_cache_window)
        adopt_boost = 0.3
        for i in range(n_eval):
            ctx_start = max(0, i - kn_model.max_order)
            ctx = flat[ctx_start:i + 1].tolist()
            tgt = int(flat[i + 1])
            current = int(flat[i])

            kn_dist = kn_score_next(kn_model, ctx, V)
            s_boost = cache3.get_boost(V, device)

            # Adoption: successor-subs of current word → their KN children
            adoption = torch.zeros(V, dtype=torch.float32, device=device)
            sub_ids = dir_subs.succ_ids[current]  # [K]
            sub_wts = dir_subs.succ_weights[current]  # [K]
            for j in range(dir_subs.K):
                sw = float(sub_wts[j].item())
                if sw < 1e-6:
                    break
                sub_w = int(sub_ids[j].item())
                sub_dist = kn_score_next(kn_model, ctx[:-1] + [sub_w], V)
                # Check predecessor compatibility
                pred_sim = float(dir_subs.pred_weights[tgt].sum().item()) if tgt < V else 0
                adoption = adoption + sw * sub_dist

            if adoption.sum() > 1e-9:
                adoption = adoption / adoption.sum().clamp(min=1e-9)

            dist = (1 - lam - adopt_boost) * kn_dist + lam * s_boost + adopt_boost * adoption
            dist = dist / dist.sum().clamp(min=1e-9)
            p = float(dist[tgt].item())
            nll += -math.log(max(p, 1e-12))
            _, top = torch.topk(dist, 50)
            top_list = top.tolist()
            if tgt in top_list[:1]: hits1 += 1
            if tgt in top_list[:5]: hits5 += 1
            n += 1
            cache3.observe(int(flat[i]))
        adopt_ppl = math.exp(nll / max(n, 1))
        results["kn_adoption"] = {"ppl": adopt_ppl, "hit@1": hits1/max(n,1), "hit@5": hits5/max(n,1)}
        print(f"    [KN + cache + adopt] PPL={adopt_ppl:.2f}  hit@1={hits1/max(n,1):.4f}  hit@5={hits5/max(n,1):.4f}")

    return results


def evaluate_layer_diagnostics(
    graph: Graph,
    encoded: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
) -> Dict:
    """Isolated eval of each layer: syntax-only PPL + semantic-only PPL.

    Tests each channel independently so we know WHERE the problem is:
      - Syntax PPL: score using only Re (grammar graph) — should approach bigram PPL
      - Semantic PPL: score using only Im (concept graph) — baseline for concept quality
    """
    from copy import copy

    results = {}

    # --- Syntax-only: context_weight=1, concept_weight=0, reflection=0 ---
    cfg_syn = copy(cfg)
    cfg_syn.context_weight = 1.0
    cfg_syn.concept_weight = 0.0
    cfg_syn.reflection_coef = 0.0
    cfg_syn.scoring_method = "projection"

    arrs = [a for a in encoded if a.size > 1]
    if arrs:
        ppl_syn = evaluate_ppl(graph, encoded, vocab, cfg_syn, device, max_tokens=50000)
        hr_syn = evaluate_hit_rate(graph, encoded, vocab, cfg_syn, device,
                                   max_tokens=50000)
        results["syntax_only"] = {"ppl": ppl_syn, "hit_rate": hr_syn}
        print(f"    [syntax-only]  PPL={ppl_syn['ppl']:.2f}  "
              f"hit@1={hr_syn.get('hit@1', 0):.4f}  "
              f"hit@5={hr_syn.get('hit@5', 0):.4f}")

    # --- Semantic-only: context_weight=0, concept_weight=1, reflection=0 ---
    cfg_sem = copy(cfg)
    cfg_sem.context_weight = 0.0
    cfg_sem.concept_weight = 1.0
    cfg_sem.reflection_coef = 0.0
    cfg_sem.scoring_method = "projection"

    if arrs:
        ppl_sem = evaluate_ppl(graph, encoded, vocab, cfg_sem, device, max_tokens=50000)
        hr_sem = evaluate_hit_rate(graph, encoded, vocab, cfg_sem, device,
                                   max_tokens=50000)
        results["semantic_only"] = {"ppl": ppl_sem, "hit_rate": hr_sem}
        print(f"    [semantic-only] PPL={ppl_sem['ppl']:.2f}  "
              f"hit@1={hr_sem.get('hit@1', 0):.4f}  "
              f"hit@5={hr_sem.get('hit@5', 0):.4f}")

    return results


def run_full_eval(
    graph: Graph,
    encoded_valid: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
    kn_model=None,
    ccache=None,
    dir_subs=None,
) -> Dict:
    results: Dict = {}

    # --- KN diagnostics: before/after each cache ---
    if kn_model is not None:
        print("  [eval] KN-5gram diagnostics (before/after caches)...")
        results["kn_diagnostics"] = evaluate_kn_diagnostics(
            kn_model, ccache, encoded_valid, vocab, cfg, device,
            dir_subs=dir_subs,
        )

    # --- Layer diagnostics: syntax-only and semantic-only ---
    print("  [eval] Wave layer diagnostics (isolated channels)...")
    results["layer_diagnostics"] = evaluate_layer_diagnostics(
        graph, encoded_valid, vocab, cfg, device,
    )

    if cfg.eval_ppl:
        print("  [eval] Combined PPL...")
        r = evaluate_ppl(graph, encoded_valid, vocab, cfg, device)
        results["ppl"] = r
        print(f"    PPL = {r['ppl']:.2f} on {r['n_tokens']} tokens (in {r.get('time_s', 0):.1f}s)")

    if cfg.eval_hit_rate:
        print("  [eval] hit rate...")
        r = evaluate_hit_rate(graph, encoded_valid, vocab, cfg, device)
        results["hit_rate"] = r
        for k, v in r.items():
            print(f"    {k} = {v:.4f}")

    if cfg.eval_ood_cloze:
        print("  [eval] Graph concept recall (raw graph quality)...")
        results["graph_concepts"] = evaluate_graph_concepts(
            graph, vocab, device,
        )
        print("  [eval] OOD cloze (end-to-end scoring)...")
        r = evaluate_ood_cloze(graph, vocab, cfg, device)
        results["ood"] = r
        n_skipped = sum(1 for d in r.get("ood_details", []) if d.get("skipped"))
        print(f"    ood_hit@1 = {r['ood_hit@1']:.3f}  hit@5 = {r['ood_hit@5']:.3f}  "
              f"hit@10 = {r['ood_hit@10']:.3f}  (tested={r['ood_total']}, skipped_unk={n_skipped})")
        for d in r.get("ood_details", []):
            if d.get("skipped"):
                print(f"      [{d['context']!r}] -> SKIPPED ('{d['answer']}' is <unk>)")
                continue
            rank_str = str(d["rank"]) if d["rank"] else ">50"
            top_str = " ".join(d.get("top5_words", []))
            print(f"      [{d['context']!r}] -> rank={rank_str}  top5=[{top_str}]")

    if cfg.eval_generation:
        print("  [eval] generation...")
        prompts = ["the king", "water is", "in the morning", "she said"][: cfg.gen_samples]
        r = evaluate_generation(graph, vocab, cfg, device, prompts)
        results["generation"] = r
        for item in r:
            print(f"    [{item['prompt']!r}]")
            print(f"      {item['generated'][:200]}")

    return results
