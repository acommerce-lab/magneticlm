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
    """Build an impulse that weights the most recent context more heavily."""
    x = torch.zeros(V, dtype=torch.float32, device=device)
    if not tokens:
        return x
    # Exponential recency weighting: most recent = 1.0, each earlier ×0.7
    w = 1.0
    for t in reversed(tokens[-8:]):   # cap context length
        x[int(t)] += w
        w *= 0.7
    # Normalize
    s = x.sum().clamp(min=1e-9)
    return x / s


def _build_batch_impulses(contexts: List[List[int]], V: int, device: torch.device) -> torch.Tensor:
    """Stack impulses for batched propagation. Shape [B, V]."""
    B = len(contexts)
    imp = torch.zeros(B, V, dtype=torch.float32, device=device)
    for i, ctx in enumerate(contexts):
        if not ctx:
            continue
        w = 1.0
        for t in reversed(ctx[-8:]):
            imp[i, int(t)] += w
            w *= 0.7
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
                pick = int(idx[torch.multinomial(probs, 1).item()].item())
            else:
                pick = int(torch.multinomial(dist, 1).item())
            ids.append(pick)
        text = " ".join(vocab.itos[i] for i in ids)
        out.append({"prompt": prompt, "generated": text})
    return out


def run_full_eval(
    graph: Graph,
    encoded_valid: List[np.ndarray],
    vocab: Vocab,
    cfg,
    device: torch.device,
) -> Dict:
    results: Dict = {}

    if cfg.eval_ppl:
        print("  [eval] PPL...")
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
        print("  [eval] OOD cloze...")
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
