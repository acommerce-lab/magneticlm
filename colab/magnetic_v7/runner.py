"""v7 Pipeline: data -> knowledge -> spectrum -> triangle transformer -> eval.
ZERO hyperparameters. Everything derived from data.
Caches expensive computations (SVD, stats, vocab) for reuse."""
import gc, hashlib, json, math, os, time
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
    """Measure intrinsic knowledge from raw bigram counts."""
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
    ppl_bound = 2.0 ** H_cond
    E = len(pair_counts)

    return {"K": K, "H_conditional": H_cond, "H_max": H_max,
            "ppl_bound": ppl_bound, "E": E, "N": total}


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


def _cache_key(cfg, n_train, V):
    """Deterministic cache key from data parameters."""
    sig = f"{cfg.dataset}|{n_train}|V{V}|ppmi{cfg.min_ppmi}|var{cfg.var_target}"
    sig += f"|sw{cfg.stat_window}|bw{cfg.bigram_window}|mpc{cfg.min_pair_count}"
    sig += f"|mv{cfg.max_vocab}|mc{cfg.min_count}|seed{cfg.seed}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def _try_load_cache(cache_dir, key):
    """Load cached spectrum if available."""
    path = os.path.join(cache_dir, f"spectrum_{key}.pt")
    if os.path.exists(path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            print(f"  ✓ Loaded from cache: {path}")
            return data
        except Exception as e:
            print(f"  ⚠ Cache corrupt, rebuilding: {e}")
    return None


def _save_cache(cache_dir, key, data):
    """Save spectrum to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"spectrum_{key}.pt")
    torch.save(data, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✓ Saved to cache: {path} ({size_mb:.1f}MB)")


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v7 — Genetic Triangle Transformer")
    print("  Zero hyperparameters. Everything from data.")
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
    n_train = len(train_lines)
    print(f"  train={n_train} valid={len(valid_lines)} ({time.time()-t0:.1f}s)")

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
    # KNOWLEDGE MEASUREMENT
    # ══════════════════════════════════════════════════════════════
    print("Measuring knowledge potential...")
    t0 = time.time()
    knowledge = _measure_knowledge(enc_train, V)
    K = knowledge["K"]
    ppl_bound = knowledge["ppl_bound"]
    H = knowledge["H_conditional"]
    H_max = knowledge["H_max"]

    if H > 0.1:
        n_layers = max(1, math.ceil(math.log2(H_max / H)))
    else:
        n_layers = max(1, math.ceil(math.log2(H_max)))

    print(f"  H(next|prev) = {H:.2f} bits")
    print(f"  H_max = log2({V}) = {H_max:.2f} bits")
    print(f"  Knowledge K = {K:.4f} ({K*100:.1f}%)")
    print(f"  PPL bound = {ppl_bound:.1f}")
    N = knowledge["N"]
    cover_threshold = V * math.log(V) if V > 1 else 1
    coverage = N / cover_threshold
    print(f"  Coverage: N={N:,} / V·ln(V)={cover_threshold:,.0f} = {coverage:.2f}")
    if coverage < 1.0:
        print(f"  ⚠ INSUFFICIENT DATA: need {cover_threshold:,.0f} tokens, have {N:,}")
    else:
        print(f"  ✓ Data sufficient for stable spectral extraction")
    print(f"  Layers L = ceil(log2({H_max:.1f}/{H:.1f})) = {n_layers}")
    print(f"  ({time.time()-t0:.1f}s)")
    print("-" * 72)

    # ══════════════════════════════════════════════════════════════
    # BUILD SPECTRUM (with caching)
    # ══════════════════════════════════════════════════════════════
    cfg.unk_id = vocab.unk_id
    key = _cache_key(cfg, n_train, V)
    cached = _try_load_cache(cfg.cache_dir, key)

    if cached is not None:
        embeddings = cached["embeddings"].to(res.primary_device)
        Wq = cached["Wq"].to(res.primary_device)
        Wk = cached["Wk"].to(res.primary_device)
        d = int(cached["d"])
        n_heads = int(cached.get("n_heads", 4))
        print(f"  d={d}, n_heads={n_heads} (from cache)")
        mon.snapshot("after-spectrum")
    else:
        print("Building co-occurrence statistics...")
        t0 = time.time()
        stats = build_stats(enc_train, V, cfg, res.primary_device)
        print(f"  stats in {time.time()-t0:.1f}s")

        print("Building bigram transitions...")
        t0 = time.time()
        bg_trans = _build_bigram_trans(enc_train, V, res.primary_device)
        print(f"  bigrams in {time.time()-t0:.1f}s")
        mon.snapshot("after-stats")

        print("Building from spectrum (auto threshold)...")
        t0 = time.time()
        embeddings, Wq, Wk, d, n_heads = build_all_from_spectrum(
            stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
            stats.unigram_counts, bg_trans, V, cfg.min_ppmi,
            res.primary_device, cfg.var_target,
        )
        print(f"  spectrum -> d={d} in {time.time()-t0:.1f}s")

        _save_cache(cfg.cache_dir, key, {
            "embeddings": embeddings.cpu(),
            "Wq": Wq.cpu(), "Wk": Wk.cpu(),
            "d": d, "n_heads": n_heads, "V": V, "n_train": n_train,
        })

        del stats, bg_trans
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mon.snapshot("after-spectrum")

    # ══════════════════════════════════════════════════════════════
    # ASSEMBLE TRANSFORMER
    # ══════════════════════════════════════════════════════════════
    print(f"Assembling StatTransformer (d={d}, L={n_layers})...")
    transformer = StatTransformer(
        embeddings=embeddings,
        Wq_init=Wq, Wk_init=Wk,
        n_heads=n_heads,
        n_layers=n_layers,
        context_len=cfg.context_len,
        pos_decay=cfg.pos_decay,
    )

    # Optional refinement (early stopping)
    if cfg.refine:
        print(f"Refining ALL weights (early stopping, patience=3)...")
        t0 = time.time()
        transformer.refine(enc_train)
        print(f"  refined in {time.time()-t0:.1f}s")
        print(f"  refined in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # EVALUATION
    # ══════════════════════════════════════════════════════════════
    print("Running evaluation...")
    t_eval = time.time()
    results: Dict = {
        "knowledge": knowledge,
        "d": d,
        "n_layers": n_layers,
        "ppl_bound": ppl_bound,
        "coverage": coverage,
    }

    print("  [eval] Layer diagnostics...")
    results["layers"] = eval_layers(transformer, V, enc_valid, cfg, res.primary_device)

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = eval_ood(transformer, V, vocab, cfg, res.primary_device)

    layers = results.get("layers", {})
    if layers:
        ppl_pure = layers.get("StatTransformer", {}).get("ppl", float('inf'))
        ppl_cache = layers.get("StatTransformer+Cache", {}).get("ppl", float('inf'))
        ppl_actual = min(ppl_pure, ppl_cache)
        best_name = "StatTransformer+Cache" if ppl_cache < ppl_pure else "StatTransformer"
        efficiency = ppl_bound / ppl_actual if ppl_actual > 0 else 0
        print(f"  ── Knowledge Report ──")
        print(f"  PPL bound (theoretical)  = {ppl_bound:.1f}")
        print(f"  PPL pure                 = {ppl_pure:.1f}")
        print(f"  PPL +cache               = {ppl_cache:.1f}")
        print(f"  Best ({best_name:20s}) = {ppl_actual:.1f}")
        print(f"  Efficiency               = {efficiency:.4f} ({efficiency*100:.2f}%)")
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
