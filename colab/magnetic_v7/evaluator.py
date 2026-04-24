"""v7 Evaluation — Statistical Transformer channels."""

import math, time
from typing import Dict, List
import numpy as np
import torch

from .model import kn_score

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


def _decay_boost(history, V, device):
    if len(history) < 2:
        return torch.zeros(V, dtype=torch.float32, device=device)
    h = torch.tensor(history, dtype=torch.int64, device=device)
    ages = torch.arange(h.numel() - 1, -1, -1, dtype=torch.float32, device=device)
    b = torch.zeros(V, dtype=torch.float32, device=device)
    b.scatter_add_(0, h, 1.0 / torch.log(2.0 + ages))
    s = b.sum()
    return b / s if s > 1e-9 else b


def eval_layers(transformer, kn, encoded, cfg, device):
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return {}
    flat = np.concatenate(arrs).astype(np.int64)
    n_eval = min(len(flat) - 1, cfg.eval_max_tokens)
    V = kn["V"]
    lam_c = cfg.cache_lambda
    unk_id = getattr(cfg, "unk_id", -1)
    results = {}

    contexts = [flat[max(0, i - cfg.context_len):i + 1].tolist() for i in range(n_eval)]
    targets = [int(flat[i + 1]) for i in range(n_eval)]
    currents = [int(flat[i]) for i in range(n_eval)]

    # Pre-compute KN
    print("    pre-computing KN...")
    t0 = time.time()
    kn_dists = [kn_score(kn, ctx) for ctx in contexts]
    print(f"    KN done in {time.time()-t0:.1f}s")

    # Pre-compute StatTransformer (batched)
    print("    pre-computing StatTransformer (batch)...")
    t0 = time.time()
    tf_dists = transformer.score_batch(contexts)
    print(f"    StatTransformer batch done in {time.time()-t0:.1f}s")

    def _eval(name, make_dist_fn):
        t_s = time.time()
        nll, n, h1, h5 = 0.0, 0, 0, 0
        history = []
        for i in range(n_eval):
            dist = make_dist_fn(i, history)
            tgt = targets[i]
            nll += -math.log(max(float(dist[tgt].item()), 1e-12))
            d = dist.clone()
            if unk_id >= 0:
                d[unk_id] = 0.0
            _, top = torch.topk(d, 50)
            tl = top.tolist()
            if tgt in tl[:1]:
                h1 += 1
            if tgt in tl[:5]:
                h5 += 1
            n += 1
            history.append(currents[i])
            if len(history) > cfg.stat_cache_window:
                history = history[-cfg.stat_cache_window:]
        ppl = math.exp(nll / max(n, 1))
        r = {"ppl": ppl, "hit@1": h1 / max(n, 1), "hit@5": h5 / max(n, 1)}
        results[name] = r
        print(f"    [{name:35s}] PPL={ppl:.2f}  h@1={r['hit@1']:.4f}  h@5={r['hit@5']:.4f}  ({time.time()-t_s:.1f}s)")

    # Individual channels
    _eval("KN bigram", lambda i, h: kn_dists[i])
    _eval("KN + cache",
          lambda i, h: (1 - lam_c) * kn_dists[i] + lam_c * _decay_boost(h, V, device))
    _eval("StatTransformer (pure)", lambda i, h: tf_dists[i])
    _eval("StatTransformer + cache",
          lambda i, h: (1 - lam_c) * tf_dists[i] + lam_c * _decay_boost(h, V, device))
    _eval("KN + StatTransformer (50/50)",
          lambda i, h: 0.5 * kn_dists[i] + 0.5 * tf_dists[i])

    return results


def eval_ood(transformer, kn, vocab, cfg, device):
    V = kn["V"]
    lam_c = cfg.cache_lambda
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []

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

        kn_d = kn_score(kn, toks)
        tf_d = transformer.score_single(toks)
        cache_d = _decay_boost(toks, V, device)

        def _rank(d, t):
            d = d.clone()
            d[vocab.unk_id] = 0.0
            d = d / d.sum().clamp(min=1e-9)
            _, idx = torch.topk(d, 50)
            top = idx.tolist()
            return top.index(t) + 1 if t in top else None, [vocab.itos[i] for i in top[:5]]

        rk, t5k = _rank(kn_d, tgt)
        rt, t5t = _rank(tf_d, tgt)

        base = (1 - lam_c) * kn_d + lam_c * cache_d
        conf = tf_d.max()
        gate = torch.sigmoid(20.0 * (conf - 0.02))
        combined = (1 - gate) * base + gate * tf_d
        rc, t5c = _rank(combined, tgt)

        if rc and rc <= 1:
            hits[1] += 1
        if rc and rc <= 5:
            hits[5] += 1
        if rc and rc <= 10:
            hits[10] += 1
        total += 1
        details.append({
            "context": context, "answer": answer,
            "kn": rk or ">50", "tf": rt or ">50", "mix": rc or ">50",
            "gate": f"{float(gate):.3f}",
            "tf_top5": t5t,
        })

    recall = {f"ood_hit@{k}": hits[k] / max(total, 1) for k in [1, 5, 10]}
    recall["ood_total"] = total
    n_skip = sum(1 for d in details if d.get("skipped"))
    print(f"    ood h@1={recall['ood_hit@1']:.3f} h@5={recall['ood_hit@5']:.3f} "
          f"h@10={recall['ood_hit@10']:.3f} (tested={total}, skip={n_skip})")
    for d in details:
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED")
            continue
        t5 = " ".join(d.get("tf_top5", []))
        print(f"      [{d['context']!r}] -> '{d['answer']}' KN={d['kn']} TF={d['tf']} MIX={d['mix']} gate={d['gate']}  tf5=[{t5}]")
    recall["details"] = details
    return recall
