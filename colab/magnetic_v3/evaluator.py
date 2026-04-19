"""Evaluation: PPL, hit-rate, OOD cloze, generation with glow centers.

PPL/hit-rate: stateless (no field) by default, or with field if eval_use_field=True.
OOD cloze: uses session to feed context words (field accumulates from context).
Generation: always uses session (field + glow centers active).
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .inference import InferenceEngine, InferenceSession
from .tokenizer import Vocab


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
    use_field: bool = False,
) -> Dict[str, float]:
    nll = 0.0
    n = 0
    start = time.time()
    for arr in encoded:
        if arr.size < 2:
            continue
        if use_field:
            session = engine.create_session()
        for i in range(arr.size - 1):
            cur = int(arr[i])
            tgt = int(arr[i + 1])
            if use_field:
                session.observe(cur)
                dist = session.score_next(cur)
            else:
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
    use_field: bool = False,
) -> Dict[str, float]:
    hits = {k: 0 for k in top_k_list}
    n = 0
    for arr in encoded:
        if arr.size < 2:
            continue
        if use_field:
            session = engine.create_session()
        for i in range(arr.size - 1):
            cur = int(arr[i])
            tgt = int(arr[i + 1])
            if use_field:
                session.observe(cur)
                dist = session.score_next(cur)
            else:
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
    """OOD cloze uses session: context words are observed, building field."""
    cases = cases or OOD_CLOZE
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

        session = engine.create_session()
        for t in toks:
            session.observe(t)
        dist = session.score_next(toks[-1])

        _, idx = torch.topk(dist, 50)
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
        glow = session.get_glow_centers().tolist()
        top3_words = [vocab.itos[i] if i < len(vocab.itos) else "?" for i in top[:5]]
        is_unk = (tgt == vocab.unk_id)
        details.append({
            "context": context, "answer": answer,
            "rank": rank, "top5_words": top3_words,
            "glow_centers": len(glow), "answer_is_unk": is_unk,
        })
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
    """Generation always uses session (field + glow centers)."""
    out = []
    for prompt in prompts:
        seed = vocab.encode_line(prompt)
        session = engine.create_session()
        gen_ids = session.generate(seed, length=length, top_k=top_k, temperature=temperature)
        gen_text = " ".join(vocab.itos[i] for i in gen_ids)
        glow = session.get_glow_centers().tolist()
        out.append({
            "prompt": prompt,
            "generated": gen_text,
            "glow_centers_active": len(glow),
        })
    return out


def run_full_eval(
    engine: InferenceEngine,
    encoded_valid: List[np.ndarray],
    vocab: Vocab,
    cfg,
) -> Dict:
    results: Dict = {}
    use_field = getattr(cfg, "eval_use_field", False)

    if cfg.eval_ppl:
        mode = "with field" if use_field else "stateless"
        print(f"  [eval] PPL ({mode})...")
        results["ppl"] = evaluate_ppl(engine, encoded_valid, vocab, use_field=use_field)
        print(f"    PPL = {results['ppl']['ppl']:.2f} on {results['ppl']['n_tokens']} tokens")

    if cfg.eval_hit_rate:
        print(f"  [eval] hit rate ({mode})...")
        results["hit_rate"] = evaluate_hit_rate(engine, encoded_valid, use_field=use_field)
        for k, v in results["hit_rate"].items():
            print(f"    {k} = {v:.4f}")

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze (with session + glow)...")
        results["ood"] = evaluate_ood_cloze(engine, vocab)
        print(f"    ood_hit@1 = {results['ood']['ood_hit@1']:.3f}")
        print(f"    ood_hit@5 = {results['ood']['ood_hit@5']:.3f}")
        print(f"    ood_hit@10 = {results['ood']['ood_hit@10']:.3f}")
        for d in results["ood"].get("ood_details", []):
            ans_info = f"unk={d.get('answer_is_unk')}" if d.get("answer_is_unk") else ""
            rank_str = str(d["rank"]) if d["rank"] else ">50"
            top_str = " ".join(d.get("top5_words", []))
            print(f"      [{d['context']!r}] -> rank={rank_str}  top5=[{top_str}]  glow={d['glow_centers']}  {ans_info}")

    if cfg.eval_generation:
        print("  [eval] generation (with session + glow)...")
        prompts = ["the king", "water is", "in the morning", "she said"][: cfg.gen_samples]
        results["generation"] = evaluate_generation(
            engine, vocab, prompts,
            length=cfg.gen_length,
            top_k=cfg.gen_top_k,
            temperature=cfg.gen_temperature,
        )
        for item in results["generation"]:
            print(f"    [{item['prompt']!r}] glow={item['glow_centers_active']}")
            print(f"      {item['generated'][:200]}")

    return results
