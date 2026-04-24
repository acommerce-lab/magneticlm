"""v7 Pipeline: stats -> embeddings -> projections -> KN -> transformer -> eval."""
import gc, json, os, time
from typing import Dict
import numpy as np, torch

from .config import Config
from .data import load_dataset
from .model import build_embeddings, build_projections, build_kn_simple, StatTransformer
from .evaluator import eval_layers, eval_ood
from .resources import Monitor, detect, setup_cuda_tuning
from .stats import build_stats
from .tokenizer import build_vocab, encode_stream


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v7 — Interpretable Statistical Transformer")
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

    # Stats
    print("Building statistics...")
    t0 = time.time()
    stats = build_stats(enc_train, V, cfg, res.primary_device)
    print(f"  stats in {time.time()-t0:.1f}s")
    mon.snapshot("after-stats")

    # Embeddings
    print("Building embeddings (SVD on PPMI)...")
    t0 = time.time()
    embeddings, idf = build_embeddings(
        stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
        stats.unigram_counts, V, cfg.embed_dim, cfg.min_ppmi,
        res.primary_device,
    )
    print(f"  embeddings in {time.time()-t0:.1f}s")
    del stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mon.snapshot("after-embed")

    # KN
    print("Building KN...")
    t0 = time.time()
    kn = build_kn_simple(enc_train, V, cfg.kn_max_order, res.primary_device)
    cfg.unk_id = vocab.unk_id
    print(f"  KN in {time.time()-t0:.1f}s")
    mon.snapshot("after-kn")

    # Q/K Projections (from bigram transitions)
    print("Building Q/K projections (from bigram transitions)...")
    t0 = time.time()
    q_fwd, k_embed = build_projections(
        embeddings, kn["bg_trans"], V, cfg.embed_dim, res.primary_device,
    )
    print(f"  projections in {time.time()-t0:.1f}s")
    mon.snapshot("after-proj")

    # Build Statistical Transformer
    print(f"Assembling StatTransformer (d={cfg.embed_dim}, heads={cfg.n_heads}, layers={cfg.n_layers})...")
    transformer = StatTransformer(
        embeddings=embeddings,
        q_fwd=q_fwd,
        k_embed=k_embed,
        idf=idf,
        unigram_prob=kn["uni_prob"],
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_len=cfg.context_len,
        pos_decay=cfg.pos_decay,
    )

    # Eval
    print("Running evaluation...")
    t_eval = time.time()
    results: Dict = {}

    print("  [eval] Layer diagnostics...")
    results["layers"] = eval_layers(transformer, kn, enc_valid, cfg, res.primary_device)

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = eval_ood(transformer, kn, vocab, cfg, res.primary_device)

    print(f"  Total eval: {time.time()-t_eval:.1f}s")
    mon.snapshot("post-eval")

    os.makedirs(cfg.save_dir, exist_ok=True)
    out = os.path.join(cfg.save_dir, "v7_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved -> {out}")
    print("Done.")
    return results
