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
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# Dataset
# ======================================================================

class LMDataset(Dataset):
    def __init__(self, encoded: List[np.ndarray], seq_len: int):
        flat = np.concatenate([a for a in encoded if a.size > 1])
        n = (len(flat) - 1) // seq_len * seq_len
        self.data = torch.from_numpy(flat[:n + 1].astype(np.int64))
        self.seq_len = seq_len
        self.n_examples = n // seq_len

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ======================================================================
# Training
# ======================================================================

def train_model(model, train_loader, epochs, lr, device, warmup_steps=200):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1

        avg = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  epoch {epoch}/{epochs}  loss={avg:.4f}  "
              f"ppl={math.exp(min(avg, 20)):.1f}  time={elapsed:.1f}s")


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate_ppl(model, encoded, vocab, device, max_tokens=200000, seq_len=128):
    model.eval()
    nll = 0.0
    n = 0
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        for start in range(0, len(t) - 1, seq_len):
            end = min(start + seq_len, len(t) - 1)
            x = t[start:end].unsqueeze(0)
            y = t[start + 1:end + 1]
            logits = model(x).squeeze(0)
            log_probs = F.log_softmax(logits, dim=-1)
            for i in range(len(y)):
                nll += -log_probs[i, y[i]].item()
                n += 1
                if n >= max_tokens:
                    break
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    if n == 0:
        return {"ppl": float("inf"), "n_tokens": 0}
    return {"ppl": math.exp(nll / n), "n_tokens": n}


@torch.no_grad()
def evaluate_hit_rate(model, encoded, device, top_k_list=(1, 5, 10, 50),
                      max_tokens=100000, seq_len=128):
    model.eval()
    hits = {k: 0 for k in top_k_list}
    n = 0
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        for start in range(0, len(t) - 1, seq_len):
            end = min(start + seq_len, len(t) - 1)
            x = t[start:end].unsqueeze(0)
            y = t[start + 1:end + 1]
            logits = model(x).squeeze(0)
            _, top_idx = torch.topk(logits, max(top_k_list), dim=-1)
            for i in range(len(y)):
                top_i = top_idx[i].tolist()
                for k in top_k_list:
                    if int(y[i].item()) in top_i[:k]:
                        hits[k] += 1
                n += 1
                if n >= max_tokens:
                    break
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break
    if n == 0:
        return {f"hit@{k}": 0.0 for k in top_k_list}
    return {f"hit@{k}": hits[k] / n for k in top_k_list}


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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)

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
    print(f"Device: {device}")

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

    train_ds = LMDataset(encoded_train, args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
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

    n_params = model.count_params()
    print(f"Model: {args.n_layers}L/{args.n_heads}H/d={args.d_model}  "
          f"params={n_params:,}  seq_len={args.seq_len}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, dataset={len(train_ds)} examples")
    print("-" * 70)

    t0 = time.time()
    train_model(model, train_loader, args.epochs, args.lr, device)
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    print("-" * 70)
    print("Evaluation...")

    results: Dict = {"model": "transformer_baseline", "params": n_params,
                     "train_lines": args.max_train_lines, "train_time_s": train_time}

    print("  [eval] PPL...")
    ppl_res = evaluate_ppl(model, encoded_valid, vocab, device)
    results["ppl"] = ppl_res
    print(f"    PPL = {ppl_res['ppl']:.2f} on {ppl_res['n_tokens']} tokens")

    print("  [eval] hit rate...")
    hr_res = evaluate_hit_rate(model, encoded_valid, device)
    results["hit_rate"] = hr_res
    for k, v in hr_res.items():
        print(f"    {k} = {v:.4f}")

    print("  [eval] OOD cloze...")
    ood_res = evaluate_ood_cloze(model, vocab, device, args.seq_len)
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
                model, vocab, prompt,
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

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "baseline_transformer_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
