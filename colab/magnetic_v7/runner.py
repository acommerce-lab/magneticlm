"""v7 Pipeline: measure knowledge -> ONE SVD -> transformer -> eval.
Genetic architecture: single threshold controls everything."""
import gc, json, math, os, time
from typing import Dict, List
import numpy as np, torch

from .config import Config
from .data import load_dataset
from .model import build_all_from_spectrum, StatTransformer
from .evaluator import eval_layers, eval_ood
from .resources import Monitor, detect, setup_cuda_tuning
from .stats import build_stats
from .tokenizer import build_vocab, encode_stream


def _measure_knowledge(encoded: List[np.ndarray], V: int) -> Dict:
    """Measure intrinsic knowledge capacity from raw bigram statistics.

    Computes H(next|prev) from bigram counts — no PPMI, no SVD, no model.
    This is the fundamental limit: no model can achieve PPL < exp(H).

    Returns dict with K (knowledge fraction), H, PPL bound.
    """
    # Count bigrams
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

    # H(next|prev) = -Σ P(w) Σ P(w'|w) log2 P(w'|w)
    H_cond = 0.0
    for key, count in pair_counts.items():
        w = key // V
        p_w = word_counts[w] / total
        p_next_given_w = count / word_counts[w]
        H_cond -= p_w * p_next_given_w * math.log2(p_next_given_w + 1e-30)

    H_max = math.log2(V)
    K = 1.0 - H_cond / H_max if H_max > 0 else 0.0
    ppl_bound = 2.0 ** H_cond
    E = len(pair_counts)

    return {
        "K": K,
        "H_conditional": H_cond,
        "H_max": H_max,
        "ppl_bound": ppl_bound,
        "E": E,
        "N": total,
    }


def _build_bigram_trans(encoded, V, device):
    bg_r, bg_c = [], []
    for arr in encoded:
        if arr.size < 2:
            continue
        bg_r.append(arr[:-1])
        bg_c.append(arr[1:])
    all_r = np.concatenate(bg_r).astype(np.int64)
    all_c = np.concatenate(bg_c).astype(np.int64)
    pair_keys = all_r * V + all_c
    uniq, counts = np.unique(pair_keys, return_counts=True)
    r_u = (uniq // V).astype(np.int64)
    c_u = (uniq % V).astype(np.int64)
    r_t = torch.from_numpy(r_u).to(device)
    c_t = torch.from_numpy(c_u).to(device)
    cnt_t = torch.from_numpy(counts.astype(np.float32)).to(device)
    row_sums = torch.zeros(V, dtype=torch.float32, device=device)
    row_sums.scatter_add_(0, r_t, cnt_t)
    w = cnt_t / row_sums[r_t].clamp(min=1.0)
    bg_trans = torch.sparse_coo_tensor(
        torch.stack([r_t, c_t]), w, (V, V)
    ).coalesce()
    print(f"    {int(cnt_t.numel()):,} bigram pairs")
    return bg_trans


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v7 — Genetic Statistical Transformer")
    print("=" * 72)

    setup_cuda_tuning()
    res = detect(cfg)
    print(f"Resources: {res}")
    print("Config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k} = {v}")
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

    # ══════════════════════════════════════════════════════════════
    # STEP 0: Measure knowledge potential (before any model building)
    # ══════════════════════════════════════════════════════════════
    print("Measuring knowledge potential...")
    t0 = time.time()
    knowledge = _measure_knowledge(enc_train, V)
    K = knowledge["K"]
    ppl_bound = knowledge["ppl_bound"]
    H = knowledge["H_conditional"]
    print(f"  H(next|prev) = {H:.2f} bits")
    print(f"  H_max = log2({V}) = {knowledge['H_max']:.2f} bits")
    print(f"  Knowledge K = {K:.4f} ({K*100:.1f}%)")
    print(f"  PPL lower bound = {ppl_bound:.1f} (no model can beat this)")

    # Derive number of layers: L = ceil(log2(H_max / H))
    H_max = knowledge["H_max"]
    if H > 0.1:
        n_layers = max(1, math.ceil(math.log2(H_max / H)))
    else:
        n_layers = max(1, math.ceil(math.log2(H_max)))
    print(f"  Layers L = ceil(log2({H_max:.1f}/{H:.1f})) = {n_layers}")
    print(f"  Measured in {time.time()-t0:.1f}s")
    print("-" * 72)

    # Stats
    print("Building co-occurrence statistics...")
    t0 = time.time()
    stats = build_stats(enc_train, V, cfg, res.primary_device)
    print(f"  stats in {time.time()-t0:.1f}s")

    print("Building bigram transitions...")
    t0 = time.time()
    bg_trans = _build_bigram_trans(enc_train, V, res.primary_device)
    cfg.unk_id = vocab.unk_id
    print(f"  bigrams in {time.time()-t0:.1f}s")
    mon.snapshot("after-stats")

    # ONE SVD → everything
    print(f"Building from spectrum (threshold={cfg.spectral_threshold})...")
    t0 = time.time()
    embeddings, Wq, Wk, Wv, S_raw, idf, d = build_all_from_spectrum(
        stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
        stats.unigram_counts, bg_trans, V,
        cfg.spectral_threshold, cfg.min_ppmi,
        res.primary_device,
    )
    print(f"  spectrum -> d={d} in {time.time()-t0:.1f}s")
    del stats, bg_trans
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mon.snapshot("after-spectrum")

    # Build transformer
    devices = [res.primary_device]
    if res.multi_gpu and len(res.gpu_ids) > 1:
        devices = [torch.device(f"cuda:{i}") for i in res.gpu_ids]

    print(f"Assembling StatTransformer (d={d}, layers={n_layers}, devices={len(devices)})...")
    transformer = StatTransformer(
        embeddings=embeddings,
        Wq=Wq, Wk=Wk, Wv=Wv,
        S_raw=S_raw,
        idf=idf,
        n_layers=n_layers,
        context_len=cfg.context_len,
        pos_decay=cfg.pos_decay,
        devices=devices,
    )

    # Eval
    print("Running evaluation...")
    t_eval = time.time()
    results: Dict = {
        "knowledge": knowledge,
        "d": d,
        "n_layers": n_layers,
        "spectral_threshold": cfg.spectral_threshold,
        "ppl_bound": ppl_bound,
    }

    print("  [eval] Layer diagnostics...")
    results["layers"] = eval_layers(transformer, V, enc_valid, cfg, res.primary_device)

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = eval_ood(transformer, V, vocab, cfg, res.primary_device)

    # Efficiency report
    if "StatTransformer" in results.get("layers", {}):
        ppl_actual = results["layers"]["StatTransformer"]["ppl"]
        efficiency = ppl_bound / ppl_actual if ppl_actual > 0 else 0
        print(f"  ── Knowledge Report ──")
        print(f"  PPL bound (theoretical) = {ppl_bound:.1f}")
        print(f"  PPL actual (model)      = {ppl_actual:.1f}")
        print(f"  Efficiency              = {efficiency:.4f} ({efficiency*100:.2f}%)")
        results["efficiency"] = efficiency

    print(f"  Total eval: {time.time()-t_eval:.1f}s")
    mon.snapshot("post-eval")

    os.makedirs(cfg.save_dir, exist_ok=True)
    out = os.path.join(cfg.save_dir, "v7_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved -> {out}")
    print("Done.")
    return results
