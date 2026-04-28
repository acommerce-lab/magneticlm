"""v7.5 Pipeline with caching and Accelerate."""
import gc, hashlib, json, math, os, time
from typing import Dict, List
import numpy as np, torch

from .config import Config
from .data import load_dataset
from .model import build_spectrum, StatTransformer, train_model
from .evaluator import eval_layers, eval_ood
from .resources import Monitor, detect, setup_cuda_tuning
from .stats import build_stats
from .tokenizer import build_vocab, encode_stream


def _measure_knowledge(encoded, V):
    pair_counts = {}
    word_counts = np.zeros(V, dtype=np.float64)
    total = 0
    for arr in encoded:
        if arr.size < 2: continue
        for i in range(len(arr)-1):
            w, wn = int(arr[i]), int(arr[i+1])
            word_counts[w] += 1
            pair_counts[w*V+wn] = pair_counts.get(w*V+wn, 0) + 1
            total += 1
    if total == 0:
        return {"K":0, "H_conditional":math.log2(V), "H_max":math.log2(V), "ppl_bound":V, "E":0, "N":0}
    H = 0.0
    for key, count in pair_counts.items():
        w = key // V
        pw = word_counts[w] / total
        pn = count / word_counts[w]
        H -= pw * pn * math.log2(pn + 1e-30)
    Hm = math.log2(V)
    return {"K": 1-H/Hm, "H_conditional": H, "H_max": Hm,
            "ppl_bound": 2**H, "E": len(pair_counts), "N": total}


def _build_bigram_trans(encoded, V, device):
    bg_r, bg_c = [], []
    for arr in encoded:
        if arr.size < 2: continue
        bg_r.append(arr[:-1]); bg_c.append(arr[1:])
    ar = np.concatenate(bg_r).astype(np.int64)
    ac = np.concatenate(bg_c).astype(np.int64)
    pk = ar * V + ac
    uniq, counts = np.unique(pk, return_counts=True)
    rt = torch.from_numpy((uniq//V).astype(np.int64)).to(device)
    ct = torch.from_numpy((uniq%V).astype(np.int64)).to(device)
    cnt = torch.from_numpy(counts.astype(np.float32)).to(device)
    rs = torch.zeros(V, dtype=torch.float32, device=device)
    rs.scatter_add_(0, rt, cnt)
    w = cnt / rs[rt].clamp(min=1.0)
    return torch.sparse_coo_tensor(torch.stack([rt,ct]), w, (V,V)).coalesce()


def _cache_key(cfg, n_train, V):
    sig = f"{cfg.dataset}|{n_train}|V{V}|ppmi{cfg.min_ppmi}|var{cfg.var_target}"
    sig += f"|sw{cfg.stat_window}|bw{cfg.bigram_window}|mpc{cfg.min_pair_count}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def _try_load(cache_dir, key):
    fname = f"spectrum_{key}.pt"
    for d in [os.path.dirname(os.path.abspath(__file__)), cache_dir]:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                print(f"  ✓ Loaded: {p}")
                return data
            except: pass
        p2 = os.path.join(d, "cache", fname)
        if os.path.exists(p2):
            try:
                data = torch.load(p2, map_location="cpu", weights_only=False)
                print(f"  ✓ Loaded: {p2}")
                return data
            except: pass
    return None


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print("=" * 72)
    print("MagneticLM v7.5 — Distributed Statistical Transformer")
    print("=" * 72)
    setup_cuda_tuning()
    res = detect(cfg)
    print(f"Resources: {res}")
    print("-" * 72)
    mon = Monitor(cfg); mon.snapshot("startup")

    # Data
    t0 = time.time()
    train_lines, valid_lines = load_dataset(cfg)
    n_train = len(train_lines)
    print(f"  train={n_train} valid={len(valid_lines)} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    vocab = build_vocab(train_lines, max_vocab=cfg.max_vocab, min_count=cfg.min_count)
    V = vocab.size; print(f"  vocab={V} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    enc_train = encode_stream(train_lines, vocab)
    enc_valid = encode_stream(valid_lines, vocab)
    del train_lines, valid_lines; gc.collect()
    print(f"  encoded in {time.time()-t0:.1f}s")
    mon.snapshot("after-encode")

    # Knowledge
    t0 = time.time()
    knowledge = _measure_knowledge(enc_train, V)
    K, ppl_bound = knowledge["K"], knowledge["ppl_bound"]
    H, H_max = knowledge["H_conditional"], knowledge["H_max"]
    N = knowledge["N"]
    coverage = N / (V * math.log(V)) if V > 1 else 1
    cf = min(4, max(1, int(math.log2(coverage+1))))
    n_layers = max(2, min(6, math.ceil(math.log2(H_max/H))+cf-1)) if H > 0.1 else 2
    print(f"  K={K:.4f} PPL_bound={ppl_bound:.1f} Coverage={coverage:.2f} L={n_layers} ({time.time()-t0:.1f}s)")
    print("-" * 72)

    # Spectrum (cached)
    cfg.unk_id = vocab.unk_id
    key = _cache_key(cfg, n_train, V)
    cached = _try_load(cfg.cache_dir, key)
    if cached:
        embeddings = cached["embeddings"]; Wq = cached["Wq"]; Wk = cached["Wk"]
        d = int(cached["d"]); n_heads = int(cached.get("n_heads", 4))
        print(f"  d={d}, n_heads={n_heads}")
    else:
        t0 = time.time()
        stats = build_stats(enc_train, V, cfg, torch.device("cpu"))
        bg_trans = _build_bigram_trans(enc_train, V, torch.device("cpu"))
        embeddings, Wq, Wk, d, n_heads = build_spectrum(
            stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
            stats.unigram_counts, bg_trans, V, cfg.min_ppmi, cfg.var_target)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        torch.save({"embeddings": embeddings, "Wq": Wq, "Wk": Wk,
                     "d": d, "n_heads": n_heads}, os.path.join(cfg.cache_dir, f"spectrum_{key}.pt"))
        print(f"  spectrum in {time.time()-t0:.1f}s")
        del stats, bg_trans; gc.collect()
    mon.snapshot("after-spectrum")

    # Model
    device = res.primary_device
    model = StatTransformer(V, d, n_heads, n_layers, cfg.context_len, cfg.pos_decay)
    model.init_from_spectrum(embeddings, Wq, Wk)
    model = model.to(device)

    if cfg.refine:
        print(f"Refining (L={n_layers})...")
        t0 = time.time()
        model = train_model(model, enc_train, enc_valid, cfg.context_len)
        print(f"  refined in {time.time()-t0:.1f}s")

    # Unwrap DataParallel for eval
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()

    # Eval
    print("Evaluating...")
    t_eval = time.time()
    results = {"knowledge": knowledge, "d": d, "n_layers": n_layers, "ppl_bound": ppl_bound, "coverage": coverage}
    results["layers"] = eval_layers(eval_model, V, enc_valid, cfg, device)
    if cfg.eval_ood_cloze:
        results["ood"] = eval_ood(eval_model, V, vocab, cfg, device)

    layers = results.get("layers", {})
    if layers:
        ppl_pure = layers.get("StatTransformer", {}).get("ppl", float('inf'))
        ppl_cache = layers.get("StatTransformer+Cache", {}).get("ppl", float('inf'))
        best = min(ppl_pure, ppl_cache)
        eff = ppl_bound / best if best > 0 else 0
        bn = "StatTransformer+Cache" if ppl_cache < ppl_pure else "StatTransformer"
        print(f"  ── Knowledge Report ──")
        print(f"  PPL bound  = {ppl_bound:.1f}")
        print(f"  PPL pure   = {ppl_pure:.1f}")
        print(f"  PPL +cache = {ppl_cache:.1f}")
        print(f"  Best ({bn}) = {best:.1f}")
        print(f"  Efficiency = {eff:.4f} ({eff*100:.2f}%)")
        results["efficiency"] = eff

    print(f"  Total eval: {time.time()-t_eval:.1f}s")
    mon.snapshot("post-eval")
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "v7_5_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Done.")
    return results
