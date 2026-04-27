"""v7 Evaluation — Transformer + Cache."""

import math, time
import numpy as np
import torch

from .model import decay_cache as _decay_cache

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


def eval_layers(transformer, V, encoded, cfg, device):
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return {}
    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, cfg.eval_max_tokens)
    unk_id = getattr(cfg, "unk_id", -1)
    cache_lam = 0.12
    results = {}

    contexts = [flat[max(0, i - cfg.context_len):i + 1].tolist() for i in range(n_eval)]
    targets = [int(flat[i + 1]) for i in range(n_eval)]
    currents = [int(flat[i]) for i in range(n_eval)]

    print("    pre-computing StatTransformer (batch)...")
    t0 = time.time()
    tf_dists = transformer.score_batch(contexts)
    print(f"    done in {time.time()-t0:.1f}s")

    # StatTransformer (pure)
    t_s = time.time()
    nll, n, h1, h5 = 0.0, 0, 0, 0
    for i in range(n_eval):
        dist = tf_dists[i]
        tgt = targets[i]
        nll += -math.log(max(float(dist[tgt].item()), 1e-12))
        d = dist.clone()
        if unk_id >= 0: d[unk_id] = 0.0
        _, top = torch.topk(d, 50)
        tl = top.tolist()
        if tgt in tl[:1]: h1 += 1
        if tgt in tl[:5]: h5 += 1
        n += 1
    ppl = math.exp(nll / max(n, 1))
    results["StatTransformer"] = {"ppl": ppl, "hit@1": h1/max(n,1), "hit@5": h5/max(n,1)}
    print(f"    [StatTransformer      ] PPL={ppl:.2f}  h@1={h1/max(n,1):.4f}  h@5={h5/max(n,1):.4f}  ({time.time()-t_s:.1f}s)")

    # StatTransformer + Cache
    t_s = time.time()
    nll, n, h1, h5 = 0.0, 0, 0, 0
    history = []
    for i in range(n_eval):
        cache = _decay_cache(history, V, device)
        dist = (1 - cache_lam) * tf_dists[i] + cache_lam * cache
        tgt = targets[i]
        nll += -math.log(max(float(dist[tgt].item()), 1e-12))
        d = dist.clone()
        if unk_id >= 0: d[unk_id] = 0.0
        _, top = torch.topk(d, 50)
        tl = top.tolist()
        if tgt in tl[:1]: h1 += 1
        if tgt in tl[:5]: h5 += 1
        n += 1
        history.append(currents[i])
        if len(history) > 3000:
            history = history[-3000:]
    ppl_c = math.exp(nll / max(n, 1))
    results["StatTransformer+Cache"] = {"ppl": ppl_c, "hit@1": h1/max(n,1), "hit@5": h5/max(n,1)}
    print(f"    [StatTransformer+Cache] PPL={ppl_c:.2f}  h@1={h1/max(n,1):.4f}  h@5={h5/max(n,1):.4f}  ({time.time()-t_s:.1f}s)")

    return results


def eval_ood(transformer, V, vocab, cfg, device):
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []

    for context, answer in OOD_CLOZE:
        toks = vocab.encode_line(context)
        if not toks: continue
        tgt_ids = vocab.encode_line(answer)
        if not tgt_ids: continue
        tgt = tgt_ids[0]
        if tgt == vocab.unk_id:
            details.append({"context": context, "answer": answer, "skipped": True})
            continue

        dist = transformer.score_single(toks)
        d = dist.clone()
        d[vocab.unk_id] = 0.0
        d = d / d.sum().clamp(min=1e-9)
        _, idx = torch.topk(d, 50)
        top = idx.tolist()
        rt = top.index(tgt) + 1 if tgt in top else None
        t5 = [vocab.itos[i] for i in top[:5]]

        if rt and rt <= 1: hits[1] += 1
        if rt and rt <= 5: hits[5] += 1
        if rt and rt <= 10: hits[10] += 1
        total += 1
        details.append({"context": context, "answer": answer,
                        "rank": rt or ">50", "top5": t5})

    recall = {f"ood_hit@{k}": hits[k] / max(total, 1) for k in [1, 5, 10]}
    recall["ood_total"] = total
    n_skip = sum(1 for d in details if d.get("skipped"))
    print(f"    ood h@1={recall['ood_hit@1']:.3f} h@5={recall['ood_hit@5']:.3f} "
          f"h@10={recall['ood_hit@10']:.3f} (tested={total}, skip={n_skip})")
    for d in details:
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED")
            continue
        t5 = " ".join(d.get("top5", []))
        print(f"      [{d['context']!r}] -> '{d['answer']}' rank={d['rank']}  top5=[{t5}]")
    recall["details"] = details
    return recall
