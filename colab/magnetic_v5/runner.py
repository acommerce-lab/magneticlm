"""v5 Pipeline: stats → KN → subs → eval. No wave, no graph."""

import gc, json, os, time
from typing import Dict
import numpy as np, torch

from .config import Config
from .data import load_dataset
from .kn import build as build_kn
from .subs import build as build_subs
from .evaluator import eval_kn_layers, eval_ood, eval_subs_quality
from .resources import Monitor, detect, setup_cuda_tuning
from .stats import build_stats
from .tokenizer import build_vocab, encode_stream


def run_pipeline(cfg: Config) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print("MagneticLM v5 — KN + distributional adoption")
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
    del train_lines, valid_lines; gc.collect()
    print(f"  encoded in {time.time()-t0:.1f}s")
    mon.snapshot("after-encode")

    print("Building statistics...")
    t0 = time.time()
    stats = build_stats(enc_train, V, cfg, res.primary_device)
    print(f"  stats in {time.time()-t0:.1f}s  ctx={int(stats.ctx_counts.numel()):,}  bg={int(stats.bg_counts.numel()):,}")
    mon.snapshot("after-stats")

    print("Building KN-5gram...")
    t0 = time.time()
    kn = build_kn(enc_train, V, cfg.kn_max_order, res.primary_device)
    # Store unk_id for eval
    cfg.unk_id = vocab.unk_id
    print(f"  KN in {time.time()-t0:.1f}s")
    mon.snapshot("after-kn")

    print("Building substitution tables...")
    t0 = time.time()
    subs = build_subs(
        stats.bg_rows, stats.bg_cols,
        stats.ctx_rows, stats.ctx_cols, stats.ctx_counts,
        stats.unigram_counts, V,
        K=cfg.sub_k, min_ppmi=cfg.min_ppmi,
        block_size=cfg.sub_block_size, device=res.primary_device,
    )
    print(f"  subs in {time.time()-t0:.1f}s")
    del stats; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    mon.snapshot("after-subs")

    print("Running evaluation...")
    results: Dict = {}

    print("  [eval] KN layer diagnostics...")
    results["kn_layers"] = eval_kn_layers(kn, subs, enc_valid, cfg, res.primary_device)

    print("  [eval] Substitution quality...")
    results["subs_quality"] = eval_subs_quality(subs, vocab, res.primary_device)

    if cfg.eval_ood_cloze:
        print("  [eval] OOD cloze...")
        results["ood"] = eval_ood(kn, subs, vocab, cfg, res.primary_device)

    mon.snapshot("post-eval")

    os.makedirs(cfg.save_dir, exist_ok=True)
    out = os.path.join(cfg.save_dir, "v5_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved -> {out}")
    print("Done.")
    return results
