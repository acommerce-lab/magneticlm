"""Evaluation: PPL, hit-rate, OOD cloze, generation."""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .inference import InferenceEngine
from .tokenizer import Vocab


# Out-of-distribution cloze tests (semantic generalization)
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


def evaluate_ppl(
    engine: InferenceEngine,
    encoded: List[np.ndarray],
    vocab: Vocab,
    max_tokens: int = 200000,
) -> Dict[str, float]:
    """Perplexity on direct next-token prediction."""
    V = engine.stats.vocab_size
    device = engine.device

    nll = 0.0
    n = 0
    start = time.time()
    for arr in encoded:
        if arr.size < 2:
            continue
        for i in range(arr.size - 1):
            cur = int(arr[i])
            tgt = int(arr[i + 1])
            dist = engine.score_next_token(cur)
            p = float(dist[tgt].item()) + 1e-12
            nll += -math.log(p)
            n += 1
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    if n == 0:
        return {"ppl": float("inf"), "n_tokens": 0}
    ppl = math.exp(nll / n)
    return {"ppl": ppl, "n_tokens": n, "time_s": time.time() - start}


def evaluate_hit_rate(
    engine: InferenceEngine,
    encoded: List[np.ndarray],
    top_k_list: Tuple[int, ...] = (1, 5, 10, 50),
    max_tokens: int = 100000,
) -> Dict[str, float]:
    """Top-k accuracy at predicting next token."""
    hits = {k: 0 for k in top_k_list}
    n = 0
    for arr in encoded:
        if arr.size < 2:
            continue
        for i in range(arr.size - 1):
            cur = int(arr[i])
            tgt = int(arr[i + 1])
            dist = engine.score_next_token(cur)
            _, top_idx = torch.topk(dist, max(top_k_list))
            top_idx = top_idx.tolist()
            for k in top_k_list:
                if tgt in top_idx[:k]:
                    hits[k] += 1
            n += 1
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    if n == 0:
        return {f"hit@{k}": 0.0 for k in top_k_list}
    return {f"hit@{k}": hits[k] / n for k in top_k_list}


def evaluate_ood_cloze(
    engine: InferenceEngine,
    vocab: Vocab,
    cases: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    cases = cases or OOD_CLOZE
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []
    for context, answer in cases:
        toks = vocab.encode_line(context)
        if not toks:
            continue
        cur = toks[-1]
        tgt_ids = vocab.encode_line(answer)
        if not tgt_ids:
            continue
        tgt = tgt_ids[0]
        dist = engine.score_next_token(cur)
        _, idx = torch.topk(dist, 10)
        top = idx.tolist()
        rank = top.index(tgt) + 1 if tgt in top else None
        if rank is not None:
            if rank <= 1:
                hits[1] += 1
            if rank <= 5:
                hits[5] += 1
            if rank <= 10:
                hits[10] += 1
        total += 1
        details.append({"context": context, "answer": answer, "rank": rank, "top3": top[:3]})
    return {
        "ood_hit@1": hits[1] / max(total, 1),
        "ood_hit@5": hits[5] / max(total, 1),
        "ood_hit@10": hits[10] / max(total, 1),
        "ood_total": total,
        "ood_details": details,
    }


def evaluate_generation(
    engine: InferenceEngine,
    vocab: Vocab,
    prompts: List[str],
    length: int = 50,
    top_k: int = 40,
    temperature: float = 1.0,
) -> List[Dict]:
    out = []
    for prompt in prompts:
        seed = vocab.encode_line(prompt)
        gen_ids = engine.generate(seed, length=length, top_k=top_k, temperature=temperature)
        gen_text = " ".join(vocab.itos[i] for i in gen_ids)
        out.append({"prompt": prompt, "generated": gen_text})
    return out


def run_full_eval(
    engine: InferenceEngine,
    encoded_valid: List[np.ndarray],
    vocab: Vocab,
    cfg,
) -> Dict:
    results: Dict = {}
    if cfg.eval_ppl:
        print("  [eval] PPL...")
        results["ppl"] = evaluate_ppl(engine, encoded_valid, vocab)
        print(f"    PPL = {results['ppl']['ppl']:.2f} on {results['ppl']['n_tokens']} tokens")
    if cfg.eval_hit_rate:
        print("  [eval] hit rate...")
        results["hit_rate"] = evaluate_hit_rate(engine, encoded_valid)
        for k, v in results["hit_rate"].items():
            print(f"    {k} = {v:.4f}")
    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = evaluate_ood_cloze(engine, vocab)
        print(f"    ood_hit@1 = {results['ood']['ood_hit@1']:.3f}")
        print(f"    ood_hit@5 = {results['ood']['ood_hit@5']:.3f}")
        print(f"    ood_hit@10 = {results['ood']['ood_hit@10']:.3f}")
    if cfg.eval_generation:
        print("  [eval] generation...")
        prompts = ["the king", "water is", "in the morning", "she said"][: cfg.gen_samples]
        results["generation"] = evaluate_generation(
            engine, vocab, prompts,
            length=cfg.gen_length,
            top_k=cfg.gen_top_k,
            temperature=cfg.gen_temperature,
        )
        for item in results["generation"]:
            print(f"    [{item['prompt']!r}] -> {item['generated'][:120]}...")
    return results
