"""v8 Pipeline: data -> knowledge -> spectrum -> SPIM -> eval."""
import gc, json, math, os, time
from typing import Dict, List
import numpy as np, torch

from .config import Config
from .data import load_dataset
from .spectrum import build_spectrum
from .path_model import SPIMModel
from .evaluator import eval_ppl, eval_ood
from .resources import Monitor, detect, setup_cuda_tuning
from .stats import build_stats
from .tokenizer import build_vocab, encode_stream


def _measure_knowledge(encoded: List[np.ndarray], V: int) -> Dict:
    pair_counts = {}
    word_counts = np.zeros(V, dtype=np.float64)
    total = 0
    for arr in encoded:
        if arr.size < 2:
            continue
        for i in range(len(arr) - 1):
            w, w_next = int(arr[i]), int(arr[i + 1])
            word_counts[w] += 1
            key = w * V + w_next
            pair_counts[key] = pair_counts.get(key, 0) + 1
            total += 1
    if total == 0:
        return {"K": 0, "H_conditional": math.log2(V), "H_max": math.log2(V),
                "ppl_bound": V, "E": 0, "N": 0}
    H_cond = 0.0
    for key, count in pair_counts.items():
        w = key // V
        p_w = word_counts[w] / total
        p_next_given_w = count / word_counts[w]
        H_cond -= p_w * p_next_given_w * math.log2(p_next_given_w + 1e-30)
    H_max = math.log2(V)
    K = 1.0 - H_cond / H_max if H_max > 0 else 0.0
    return {"K": K, "H_conditional": H_cond, "H_max": H_max,
            "ppl_bound": 2.0 ** H_cond, "E": len(pair_counts), "N": total}


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v8 — SPIM (Spectral Path Integral Model)")
    print("  Words are transitions. Context is a path. Prediction is integration.")
    print("=" * 72)

    setup_cuda_tuning()
    res = detect(cfg)
    print(f"Resources: {res}")
    print("-" * 72)
    mon = Monitor(cfg)
    mon.snapshot("startup")

    # Data
    print("Loading dataset...")
    t0 = time.time()
    train_lines, valid_lines = load_dataset(cfg)
    print(f"  train={len(train_lines)} valid={len(valid_lines)} ({time.time()-t0:.1f}s)")

    print("Building vocabulary...")
    t0 = time.time()
    vocab = build_vocab(train_lines, max_vocab=cfg.max_vocab, min_count=cfg.min_count)
    V = vocab.size
    print(f"  vocab={V} ({time.time()-t0:.1f}s)")

    print("Encoding...")
    t0 = time.time()
    enc_train = encode_stream(train_lines, vocab)
    enc_valid = encode_stream(valid_lines, vocab)
    del train_lines, valid_lines
    gc.collect()
    print(f"  encoded in {time.time()-t0:.1f}s")
    mon.snapshot("after-encode")

    # Knowledge
    print("Measuring knowledge potential...")
    t0 = time.time()
    knowledge = _measure_knowledge(enc_train, V)
    K = knowledge["K"]
    ppl_bound = knowledge["ppl_bound"]
    H = knowledge["H_conditional"]
    H_max = knowledge["H_max"]
    N = knowledge["N"]
    cover_threshold = V * math.log(V) if V > 1 else 1
    coverage = N / cover_threshold

    print(f"  H(next|prev) = {H:.2f} bits")
    print(f"  Knowledge K = {K:.4f} ({K*100:.1f}%)")
    print(f"  PPL bound = {ppl_bound:.1f}")
    print(f"  Coverage = {coverage:.2f}")
    if coverage < 1.0:
        print(f"  ⚠ INSUFFICIENT DATA")
    print(f"  ({time.time()-t0:.1f}s)")
    print("-" * 72)

    # Spectrum
    print("Building spectrum...")
    t0 = time.time()
    stats = build_stats(enc_train, V, cfg, res.primary_device)
    embeddings, d_schedule, S_levels, d = build_spectrum(
        stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
        stats.unigram_counts, V, cfg.min_ppmi,
        res.primary_device, cfg.var_target,
    )
    cfg.unk_id = vocab.unk_id
    print(f"  spectrum -> d={d} in {time.time()-t0:.1f}s")
    del stats
    gc.collect()
    mon.snapshot("after-spectrum")

    # SPIM Model
    print(f"Assembling SPIM (d={d}, levels={len(d_schedule)})...")
    model = SPIMModel(
        embeddings=embeddings,
        d_schedule=d_schedule,
        S_levels=S_levels,
        pos_decay=cfg.pos_decay,
    )

    # Eval
    print("Running evaluation...")
    t_eval = time.time()
    results: Dict = {
        "knowledge": knowledge,
        "d": d,
        "n_levels": len(d_schedule),
        "d_schedule": d_schedule,
        "ppl_bound": ppl_bound,
        "coverage": coverage,
    }

    print("  [eval] PPL...")
    results["layers"] = eval_ppl(model, V, enc_valid, cfg, res.primary_device)

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = eval_ood(model, V, vocab, cfg, res.primary_device)

    if "SPIM" in results.get("layers", {}):
        ppl_actual = results["layers"]["SPIM"]["ppl"]
        efficiency = ppl_bound / ppl_actual if ppl_actual > 0 else 0
        print(f"  ── Knowledge Report ──")
        print(f"  PPL bound  = {ppl_bound:.1f}")
        print(f"  PPL actual = {ppl_actual:.1f}")
        print(f"  Efficiency = {efficiency:.4f} ({efficiency*100:.2f}%)")
        results["efficiency"] = efficiency

    print(f"  Total eval: {time.time()-t_eval:.1f}s")
    mon.snapshot("post-eval")

    os.makedirs(cfg.save_dir, exist_ok=True)
    out = os.path.join(cfg.save_dir, "v8_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved -> {out}")
    print("Done.")
    return results
