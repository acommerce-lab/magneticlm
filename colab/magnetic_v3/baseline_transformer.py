#!/usr/bin/env python3
"""Baseline transformer (decoder-only) for comparison with MagneticLM.

Independent script: same data, same vocab, same eval metrics.
Run once per scale, save results, compare across runs.

Usage:
    python baseline_transformer.py --max_train_lines 10000 --epochs 10
    python baseline_transformer.py --max_train_lines 100000 --epochs 5 --d_model 256

Architecture: GPT-style decoder-only transformer.
  Default: 4 layers, 4 heads, d_model=128, d_ff=512, ~2-3M params.
  Uses the same word-level tokenizer as MagneticLM v3.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Bootstrap: make magnetic_v3 importable regardless of folder name
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

_pkg_name = "magnetic_v3"
if _pkg_name not in sys.modules:
    import importlib, types
    pkg = types.ModuleType(_pkg_name)
    pkg.__path__ = [_HERE]
    pkg.__package__ = _pkg_name
    sys.modules[_pkg_name] = pkg

from magnetic_v3.data import load_dataset
from magnetic_v3.tokenizer import build_vocab, encode_stream, tokenize


# ---------------------------------------------------------------------------
# Hardware setup: maximize GPU utilization
# ---------------------------------------------------------------------------

def setup_hardware():
    """Enable cudnn benchmark, TF32, and report device capabilities."""
    if not torch.cuda.is_available():
        print("Hardware: CPU only")
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"Hardware: {n} GPU(s): {names}")
    print(f"  cudnn.benchmark=True  TF32=enabled")


# ---------------------------------------------------------------------------
# Memory monitoring (RAM + GPU)
# ---------------------------------------------------------------------------

def ram_status() -> Dict:
    """Return dict with RAM usage info."""
    if not _HAS_PSUTIL:
        return {"available": False}
    m = psutil.virtual_memory()
    p = psutil.Process()
    return {
        "available": True,
        "rss_gb": p.memory_info().rss / 1e9,
        "sys_used_gb": (m.total - m.available) / 1e9,
        "sys_available_gb": m.available / 1e9,
        "sys_total_gb": m.total / 1e9,
        "percent": m.percent,
    }


def gpu_status() -> List[Dict]:
    """Return list of GPU memory usage dicts."""
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append({
            "gpu": i,
            "used_gb": (total - free) / 1e9,
            "total_gb": total / 1e9,
            "percent": (1.0 - free / total) * 100.0,
        })
    return out


def print_mem_snapshot(tag: str = ""):
    """One-line memory snapshot for diagnostics."""
    parts = []
    r = ram_status()
    if r.get("available"):
        parts.append(
            f"RAM={r['rss_gb']:.1f}GB "
            f"sys={r['sys_used_gb']:.1f}/{r['sys_total_gb']:.1f}GB ({r['percent']:.0f}%)"
        )
    for g in gpu_status():
        parts.append(
            f"GPU{g['gpu']}={g['used_gb']:.1f}/{g['total_gb']:.1f}GB ({g['percent']:.0f}%)"
        )
    if parts:
        print(f"  [mem{(' '+tag) if tag else ''}] " + "  ".join(parts))


class MemoryGuard:
    """Periodic memory check during training; triggers GC if near limit."""

    def __init__(self, log_every: int = 100, warn_percent: float = 85.0):
        self.log_every = max(1, log_every)
        self.warn = warn_percent
        self.step = 0
        self.peak_rss = 0.0

    def check(self, force: bool = False):
        self.step += 1
        if self.step % self.log_every != 0 and not force:
            return
        r = ram_status()
        if not r.get("available"):
            return
        self.peak_rss = max(self.peak_rss, r["rss_gb"])
        gs = gpu_status()
        gpu_str = "  ".join(
            f"GPU{g['gpu']}={g['percent']:.0f}%" for g in gs
        )
        print(
            f"  [mem step={self.step}] "
            f"rss={r['rss_gb']:.1f}GB  sys={r['percent']:.0f}%  "
            f"{gpu_str}"
        )
        if r["percent"] > self.warn:
            print(f"  [mem] WARNING sys mem {r['percent']:.0f}% — running gc/empty_cache")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# OOD cloze (same cases as magnetic_v3 evaluator)
# ---------------------------------------------------------------------------
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


# ======================================================================
# Model
# ======================================================================

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            h = layer(h, mask)
        h = self.ln_f(h)
        return self.head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + self.drop(a)
        x = x + self.ff(self.ln2(x))
        return x


# ======================================================================
# Dataset: disk-backed memmap option for large corpora
# ======================================================================

def encode_to_memmap(
    encoded: List[np.ndarray],
    out_path: str,
    dtype=np.int32,
) -> np.memmap:
    """Flatten a list of encoded arrays into a disk-backed memmap.

    Keeps RAM flat for 100K-1M+ line corpora. The OS pages chunks in/out
    as the DataLoader reads from disk.
    """
    total = sum(int(a.size) for a in encoded if a.size > 0)
    if total == 0:
        raise ValueError("no tokens to encode")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total,))
    pos = 0
    for a in encoded:
        if a.size == 0:
            continue
        arr[pos:pos + a.size] = a.astype(dtype, copy=False)
        pos += int(a.size)
    arr.flush()
    print(f"  [memmap] wrote {total:,} tokens to {out_path} "
          f"({os.path.getsize(out_path) / 1e9:.2f} GB)")
    return arr


def estimate_ram_need_gb(n_lines: int, avg_tokens_per_line: float = 20.0) -> float:
    """Rough estimate for encoded corpus RAM (int64 -> 8 bytes/token)."""
    return n_lines * avg_tokens_per_line * 8 / 1e9


class LMDataset(Dataset):
    """Language model dataset over a flat token stream.

    Accepts either in-memory numpy array or a memmap handle. Returns
    (x, y) pairs where y is x shifted by one position.
    """

    def __init__(
        self,
        flat: np.ndarray,
        seq_len: int,
    ):
        n_total = int(flat.shape[0])
        n_aligned = ((n_total - 1) // seq_len) * seq_len
        self.flat = flat  # keep as numpy / memmap; __getitem__ slices lazily
        self.seq_len = seq_len
        self.n_examples = n_aligned // seq_len
        self.limit = n_aligned + 1  # safe bound for [start+seq_len+1]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = np.asarray(self.flat[s : s + self.seq_len], dtype=np.int64)
        y = np.asarray(self.flat[s + 1 : s + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


# ======================================================================
# Training
# ======================================================================

def train_model(
    model,
    train_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    warmup_steps: int = 200,
    use_amp: bool = True,
    grad_accum: int = 1,
    mem_log_every: int = 100,
    log_batches_every: int = 50,
):
    """Training loop with AMP, cosine LR, grad clipping, memory monitoring."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = max(1, epochs * len(train_loader) // max(grad_accum, 1))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast_dtype = torch.float16  # T4 supports fp16, not bf16

    guard = MemoryGuard(log_every=mem_log_every, warn_percent=88.0)
    step = 0
    micro_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss_for_backward = loss / max(grad_accum, 1)

            scaler.scale(loss_for_backward).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

            total_loss += float(loss.item())
            n_batches += 1
            guard.check()

            if (batch_idx + 1) % log_batches_every == 0:
                rate = (batch_idx + 1) / max(time.time() - t_epoch, 1e-6)
                print(f"    batch {batch_idx+1}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  rate={rate:.1f} batch/s")

        avg = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(
            f"  epoch {epoch}/{epochs}  loss={avg:.4f}  "
            f"ppl={math.exp(min(avg, 20)):.1f}  "
            f"time_epoch={time.time()-t_epoch:.1f}s  total={elapsed:.1f}s"
        )
        print_mem_snapshot(f"end-epoch-{epoch}")

    print(f"  peak RSS during training: {guard.peak_rss:.2f} GB")


# ======================================================================
# Evaluation
# ======================================================================

def _flatten_valid(encoded: List[np.ndarray]) -> np.ndarray:
    arrs = [a for a in encoded if a.size > 1]
    if not arrs:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(arrs).astype(np.int64)


@torch.no_grad()
def evaluate_ppl(model, encoded, vocab, device, max_tokens=200000,
                 seq_len=128, batch_size=32, use_amp=True):
    """Batched PPL: flattens stream, chunks into seq_len windows, processes B at once."""
    model.eval()
    flat = _flatten_valid(encoded)
    n_tokens_available = max(0, len(flat) - 1)
    if n_tokens_available == 0:
        return {"ppl": float("inf"), "n_tokens": 0}

    t = torch.from_numpy(flat).to(device)
    n_windows = (len(flat) - 1) // seq_len
    if n_windows == 0:
        return {"ppl": float("inf"), "n_tokens": 0}

    amp_enabled = use_amp and device.type == "cuda"
    nll_sum = 0.0
    n = 0
    t_start = time.time()

    for b_start in range(0, n_windows, batch_size):
        b_end = min(b_start + batch_size, n_windows)
        B = b_end - b_start
        idx = torch.arange(b_start, b_end, device=device) * seq_len
        offsets = torch.arange(seq_len, device=device).unsqueeze(0)
        x = t[idx.unsqueeze(1) + offsets]
        y = t[idx.unsqueeze(1) + offsets + 1]

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)  # [B, T]
        nll_sum += float((-gathered).sum().item())
        n += int(gathered.numel())
        if n >= max_tokens:
            break

    return {
        "ppl": math.exp(nll_sum / max(n, 1)),
        "n_tokens": n,
        "time_s": time.time() - t_start,
    }


@torch.no_grad()
def evaluate_hit_rate(model, encoded, device, top_k_list=(1, 5, 10, 50),
                      max_tokens=100000, seq_len=128, batch_size=32, use_amp=True):
    """Batched hit-rate: fully vectorized over all (B, T) positions per batch."""
    model.eval()
    flat = _flatten_valid(encoded)
    if len(flat) < 2:
        return {f"hit@{k}": 0.0 for k in top_k_list}

    t = torch.from_numpy(flat).to(device)
    n_windows = (len(flat) - 1) // seq_len
    if n_windows == 0:
        return {f"hit@{k}": 0.0 for k in top_k_list}

    amp_enabled = use_amp and device.type == "cuda"
    max_k = max(top_k_list)
    hits = {k: 0 for k in top_k_list}
    n = 0

    for b_start in range(0, n_windows, batch_size):
        b_end = min(b_start + batch_size, n_windows)
        idx = torch.arange(b_start, b_end, device=device) * seq_len
        offsets = torch.arange(seq_len, device=device).unsqueeze(0)
        x = t[idx.unsqueeze(1) + offsets]
        y = t[idx.unsqueeze(1) + offsets + 1]

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            logits = model(x)
        _, top_idx = torch.topk(logits, max_k, dim=-1)  # [B, T, max_k]

        # Vectorized hit computation for each k
        y_expanded = y.unsqueeze(-1)  # [B, T, 1]
        for k in top_k_list:
            hit_mask = (top_idx[..., :k] == y_expanded).any(dim=-1)  # [B, T]
            hits[k] += int(hit_mask.sum().item())
        n += int(y.numel())
        if n >= max_tokens:
            break

    return {f"hit@{k}": hits[k] / max(n, 1) for k in top_k_list}


@torch.no_grad()
def evaluate_ood_cloze(model, vocab, device, seq_len=128):
    model.eval()
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    details = []
    unk_id = vocab.unk_id

    for context, answer in OOD_CLOZE:
        toks = vocab.encode_line(context)
        if not toks:
            continue
        tgt_ids = vocab.encode_line(answer)
        if not tgt_ids:
            continue
        tgt = tgt_ids[0]

        if tgt == unk_id:
            details.append({
                "context": context, "answer": answer,
                "rank": None, "top5_words": [], "answer_is_unk": True,
                "skipped": True,
            })
            continue

        x = torch.tensor(toks, dtype=torch.int64, device=device).unsqueeze(0)
        if x.size(1) > seq_len:
            x = x[:, -seq_len:]
        logits = model(x)
        last_logits = logits[0, -1]  # last position
        last_logits[unk_id] = -float("inf")

        _, idx = torch.topk(last_logits, 50)
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
        top5_words = [vocab.itos[i] if i < len(vocab.itos) else "?" for i in top[:5]]
        details.append({
            "context": context, "answer": answer,
            "rank": rank, "top5_words": top5_words,
            "answer_is_unk": False,
        })

    return {
        "ood_hit@1": hits[1] / max(total, 1),
        "ood_hit@5": hits[5] / max(total, 1),
        "ood_hit@10": hits[10] / max(total, 1),
        "ood_total": total,
        "ood_details": details,
    }


@torch.no_grad()
def generate(model, vocab, prompt, length=30, top_k=40, temperature=1.0,
             device="cpu", max_len=128):
    model.eval()
    toks = vocab.encode_line(prompt)
    ids = torch.tensor(toks, dtype=torch.int64, device=device).unsqueeze(0)
    for _ in range(length):
        x = ids[:, -max_len:]
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        logits[vocab.unk_id] = -float("inf")
        if top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.numel()))
            probs = F.softmax(vals, dim=-1)
            pick = idx[torch.multinomial(probs, 1).item()]
        else:
            probs = F.softmax(logits, dim=-1)
            pick = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, pick.view(1, 1)], dim=1)
    return [int(i) for i in ids[0].tolist()]


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline Transformer LM")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--max_train_lines", type=int, default=10000)
    parser.add_argument("--max_valid_lines", type=int, default=100)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_count", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=1)

    # Hardware knobs
    parser.add_argument("--amp", default="auto",
                        help="auto|on|off — mixed precision (fp16 on T4)")
    parser.add_argument("--multi_gpu", default="auto",
                        help="auto|on|off — DataParallel across all GPUs")
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="DataLoader workers; -1=auto based on cpu_count()")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (PyTorch 2.x)")

    # Memory management
    parser.add_argument("--memmap", default="auto",
                        help="auto|on|off — disk-backed token stream")
    parser.add_argument("--memmap_threshold_gb", type=float, default=2.0,
                        help="encoded corpus size above which memmap auto-enables")
    parser.add_argument("--mem_log_every", type=int, default=100,
                        help="log memory status every N training batches")

    parser.add_argument("--eval_generation", action="store_true")
    parser.add_argument("--gen_length", type=int, default=30)
    parser.add_argument("--gen_samples", type=int, default=3)
    parser.add_argument("--gen_top_k", type=int, default=40)

    parser.add_argument("--save_dir", default="./outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Baseline Transformer LM")
    print("=" * 70)
    setup_hardware()
    print(f"Primary device: {device}")
    print_mem_snapshot("startup")

    # --- Config shim so data.py works ---
    class CfgShim:
        dataset = "wikitext-103"
        data_dir = args.data_dir
        max_train_lines = args.max_train_lines
        max_valid_lines = args.max_valid_lines

    print("Loading dataset...")
    train_lines, valid_lines = load_dataset(CfgShim())
    print(f"  train={len(train_lines)} lines  valid={len(valid_lines)} lines")

    print("Building vocabulary...")
    vocab = build_vocab(train_lines, max_vocab=args.max_vocab, min_count=args.min_count)
    V = vocab.size
    print(f"  vocab_size={V}")

    print("Encoding...")
    encoded_train = encode_stream(train_lines, vocab)
    encoded_valid = encode_stream(valid_lines, vocab)
    # Free the Python line lists — they can be large for 1M corpora
    del train_lines
    del valid_lines
    gc.collect()

    # ---- Decide memmap vs in-memory for training stream ----
    est_gb = estimate_ram_need_gb(args.max_train_lines if args.max_train_lines > 0 else len(encoded_train))
    use_memmap = {"on": True, "off": False}.get(
        args.memmap, est_gb > args.memmap_threshold_gb
    )
    os.makedirs(args.save_dir, exist_ok=True)
    if use_memmap:
        memmap_path = os.path.join(args.save_dir, "train_tokens.bin")
        train_flat = encode_to_memmap(encoded_train, memmap_path, dtype=np.int32)
        del encoded_train
        gc.collect()
        print(f"  [memmap] training stream on disk (estimated in-RAM: {est_gb:.2f} GB)")
    else:
        arrs = [a for a in encoded_train if a.size > 1]
        train_flat = np.concatenate(arrs).astype(np.int32) if arrs else np.empty(0, dtype=np.int32)
        del encoded_train
        gc.collect()
        print(f"  [memory] training stream in RAM (est {est_gb:.2f} GB)")
    print_mem_snapshot("after-encode")

    train_ds = LMDataset(train_flat, args.seq_len)
    if len(train_ds) == 0:
        print("ERROR: no training examples after chunking. Increase --max_train_lines "
              "or decrease --seq_len.")
        return

    # DataLoader worker selection
    if args.num_workers < 0:
        cpu_count = os.cpu_count() or 1
        n_workers = max(0, min(4, cpu_count - 1))
    else:
        n_workers = args.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
        prefetch_factor=(4 if n_workers > 0 else None),
    )

    model = TransformerLM(
        vocab_size=V,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    # Multi-GPU wrap
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mg_flag = args.multi_gpu
    do_multi = (
        (mg_flag == "on") or (mg_flag == "auto" and n_gpus >= 2)
    ) and mg_flag != "off" and n_gpus >= 2
    if do_multi:
        model = nn.DataParallel(model)
        print(f"Multi-GPU: DataParallel across {n_gpus} GPUs")
    else:
        print(f"Multi-GPU: disabled (n_gpus={n_gpus})")

    if args.compile and hasattr(torch, "compile"):
        print("torch.compile: enabled")
        model = torch.compile(model)

    # AMP decision
    amp_flag = args.amp
    use_amp = (amp_flag == "on") or (amp_flag == "auto" and device.type == "cuda")
    if amp_flag == "off":
        use_amp = False
    print(f"AMP (fp16): {'on' if use_amp else 'off'}")
    print(f"DataLoader: num_workers={n_workers}  pin_memory={device.type=='cuda'}")

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    n_params = base_model.count_params() if hasattr(base_model, "count_params") \
        else sum(p.numel() for p in model.parameters())
    print(f"Model: {args.n_layers}L/{args.n_heads}H/d={args.d_model}  "
          f"params={n_params:,}  seq_len={args.seq_len}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size} "
          f"(grad_accum={args.grad_accum}, effective={args.batch_size*args.grad_accum}), "
          f"lr={args.lr}, dataset={len(train_ds)} examples")
    print("-" * 70)

    t0 = time.time()
    train_model(
        model, train_loader, args.epochs, args.lr, device,
        use_amp=use_amp,
        grad_accum=args.grad_accum,
        mem_log_every=args.mem_log_every,
    )
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")
    print_mem_snapshot("post-train")

    print("-" * 70)
    print("Evaluation...")
    # Unwrap DataParallel for single-sample eval (avoids unnecessary scatter/gather)
    eval_model = model.module if isinstance(model, nn.DataParallel) else model

    results: Dict = {
        "model": "transformer_baseline",
        "params": n_params,
        "train_lines": args.max_train_lines,
        "train_time_s": train_time,
        "hardware": {
            "n_gpus": n_gpus,
            "multi_gpu": do_multi,
            "amp": use_amp,
            "memmap": use_memmap,
            "num_workers": n_workers,
        },
    }

    print("  [eval] PPL...")
    ppl_res = evaluate_ppl(
        eval_model, encoded_valid, vocab, device,
        seq_len=args.seq_len, batch_size=args.eval_batch_size, use_amp=use_amp,
    )
    results["ppl"] = ppl_res
    print(f"    PPL = {ppl_res['ppl']:.2f} on {ppl_res['n_tokens']} tokens "
          f"(in {ppl_res.get('time_s', 0):.1f}s)")

    print("  [eval] hit rate...")
    hr_res = evaluate_hit_rate(
        eval_model, encoded_valid, device,
        seq_len=args.seq_len, batch_size=args.eval_batch_size, use_amp=use_amp,
    )
    results["hit_rate"] = hr_res
    for k, v in hr_res.items():
        print(f"    {k} = {v:.4f}")

    print("  [eval] OOD cloze...")
    ood_res = evaluate_ood_cloze(eval_model, vocab, device, args.seq_len)
    results["ood"] = ood_res
    n_skipped = sum(1 for d in ood_res.get("ood_details", []) if d.get("skipped"))
    print(f"    ood_hit@1 = {ood_res['ood_hit@1']:.3f}  "
          f"ood_hit@5 = {ood_res['ood_hit@5']:.3f}  "
          f"ood_hit@10 = {ood_res['ood_hit@10']:.3f}  "
          f"(tested={ood_res['ood_total']}, skipped_unk={n_skipped})")
    for d in ood_res.get("ood_details", []):
        if d.get("skipped"):
            print(f"      [{d['context']!r}] -> SKIPPED ('{d['answer']}' is <unk>)")
            continue
        rank_str = str(d["rank"]) if d["rank"] else ">50"
        top_str = " ".join(d.get("top5_words", []))
        print(f"      [{d['context']!r}] -> rank={rank_str}  top5=[{top_str}]")

    if args.eval_generation:
        print("  [eval] generation...")
        prompts = ["the king", "water is", "in the morning", "she said"][:args.gen_samples]
        gen_results = []
        for prompt in prompts:
            gen_ids = generate(
                eval_model, vocab, prompt,
                length=args.gen_length,
                top_k=args.gen_top_k,
                device=device,
                max_len=args.seq_len,
            )
            gen_text = " ".join(vocab.itos[i] for i in gen_ids)
            gen_results.append({"prompt": prompt, "generated": gen_text})
            print(f"    [{prompt!r}]")
            print(f"      {gen_text[:200]}")
        results["generation"] = gen_results

    print_mem_snapshot("post-eval")
    peak_gpu = 0.0
    if torch.cuda.is_available():
        peak_gpu = max(
            (torch.cuda.max_memory_allocated(i) / 1e9
             for i in range(torch.cuda.device_count())),
            default=0.0,
        )
    results["peak_gpu_alloc_gb"] = peak_gpu

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "baseline_transformer_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(f"Peak GPU alloc: {peak_gpu:.2f} GB")
    print("Done.")


if __name__ == "__main__":
    main()
