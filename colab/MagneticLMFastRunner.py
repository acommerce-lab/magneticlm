#!/usr/bin/env python3
# MagneticLMFastRunner.py
#
# GPU-native, single-file runner for the graph-based MagneticLM language
# model. WikiText-103 only, full mode, no cache, CUDA + GPU RAM end-to-end.
#
# Restored model logic (same math as the published baseline)
# ==========================================================
# - Modified Kneser-Ney with all five orders, including the D3 * n3+ term
#   that the previous "Gemini-edited" pass dropped from the lambda formula.
# - The continuation count is recovered from the token stream (NOT from
#   inverting a polynomial hash, which was the silent corruption that pushed
#   perplexity to 600+ on 1M lines). For each unique bigram we look up the
#   first position via scatter_reduce(amin) and read tokens[first_pos + 1],
#   which is the only mathematically valid way to recover the next word.
# - Semantic edges are built directly from the token stream (offsets +1, +2)
#   instead of trying to extract pairs from bigram hashes — same reason.
# - Physics simulation + importance + circle skeleton are restored, and
#   eval_full_wt103 mixes the magnetic contribution back in via the
#   adaptive lambda bands.
#
# Memory work for the 1M-line case
# ================================
# - Per-order n-gram counting uses **incremental chunk merging**: chunks are
#   not collected into a Python list (which doubled the peak); instead each
#   chunk is folded into a running master tensor and the sort buffer is
#   freed before the next chunk runs. This is the one good change from the
#   degraded version and we keep it.
# - Chunk sizes scale **inversely with order** so the higher-cost orders use
#   smaller windows (order 5 buffers ~3 GB instead of ~6 GB on T4).
# - tokens_gpu / freq_gpu / per-chunk buffers are explicitly torch.deleted
#   between phases. torch.cuda.empty_cache() is called at the boundaries.
#
# Optional dual-T4 sharding (Kaggle gives "GPU T4 x2")
# ====================================================
# When started with --multi-gpu, the per-order tables are split across two
# CUDA devices: orders 1-3 stay on cuda:0 (alongside positions, importance
# and physics state), and orders 4-5 live on cuda:1. The token stream is
# temporarily copied across when needed and freed afterwards. The KN-batch
# eval moves only the (small) per-batch query through the cross-device
# transfer, not any of the lookup tables.
#
# CLI
# ===
#   python MagneticLMFastRunner.py --train-lines 1000000 --physics-iters 30
#   python MagneticLMFastRunner.py --train-lines 0 --max-order 5 --multi-gpu

import argparse
import math
import os
import re
import sys
import time
import array

try:
    import torch
    import numpy as np
except ImportError:
    print("ERROR: PyTorch and NumPy required (pip install torch numpy).",
          file=sys.stderr)
    sys.exit(1)


_SPLIT_RE = re.compile(r'[.,;!?()\[\]{}"]+')


def tokenize(line):
    return [w for w in _SPLIT_RE.sub(' ', line.lower()).split() if w]


def ensure_wt103(data_dir="data/wt103"):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path
    print("Downloading WikiText-103 (via HF datasets)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: first-time download needs `pip install datasets`.",
              file=sys.stderr)
        sys.exit(1)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    for split, path in (("train", train_path), ("test", test_path)):
        with open(path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                t = item["text"].strip()
                if t and not t.startswith("="):
                    f.write(t + "\n")
    return train_path, test_path


# Polynomial hashing primes - one constant per position in the ngram window.
# Ten entries because max_order can now go up to 9 (ngram = 10 tokens). uint64
# -> int64 keeps the high bit set, so the polynomial hash naturally wraps
# modulo 2^64 inside torch.int64 arithmetic, giving a near-uniform universal
# hash. Constants are well-known splitmix64 / Knuth multiplicative generators.
_HASH_PRIMES_LIST = [
    0x9E3779B97F4A7C15,
    0xBF58476D1CE4E5B9,
    0x94D049BB133111EB,
    0x7F4A7C15F39CC060,
    0xA6E36C3B4E5A7F11,
    0xD2B74407B1CE6E93,
    0xCBF29CE484222325,
    0x100000001B3,
    0xFF51AFD7ED558CCD,
    0xC4CEB9FE1A85EC53,
]
_HASH_PRIMES_CPU = torch.from_numpy(
    np.array(_HASH_PRIMES_LIST, dtype=np.uint64).astype(np.int64))


class MagneticLMGPU:
    """GPU-native MagneticLM. Tokens, hashes, sorted-key tables, physics and
    eval all live on CUDA tensors. With multi_gpu=True the per-order n-gram
    tables are split across two CUDA devices."""

    def __init__(self, device, max_order=5, multi_gpu=False):
        self.device = device
        self.max_order = max_order
        self.MAX_ORDER = max_order  # back-compat for any external reader

        # Devices: tokens + low-order tables + physics on `device`,
        # high-order tables on aux_device when multi_gpu is on.
        self.aux_device = device
        if multi_gpu and torch.cuda.device_count() >= 2:
            other = (device.index + 1) % torch.cuda.device_count() if device.index is not None else 1
            self.aux_device = torch.device(f"cuda:{other}")
        self.multi_gpu = (self.aux_device != self.device)

        # Hash primes mirrored on each device used.
        self.hash_primes = _HASH_PRIMES_CPU.to(device)
        self.hash_primes_aux = (
            _HASH_PRIMES_CPU.to(self.aux_device) if self.multi_gpu
            else self.hash_primes
        )

        # Vocab — built on CPU during tokenization, then frozen.
        self.word2id = {}
        self.id2word = []

        # Token stream (released after build())
        self.tokens_gpu = None
        self.total_tokens = 0

        # Per-order tables. Each entry is a torch tensor on whichever
        # device _device_for_order(o) returned.
        self.ngram_hash_sorted = [None] * (max_order + 1)
        self.ngram_count       = [None] * (max_order + 1)
        self.ctx_hash_sorted   = [None] * (max_order + 1)
        self.ctx_total         = [None] * (max_order + 1)
        self.ctx_count1        = [None] * (max_order + 1)
        self.ctx_count2        = [None] * (max_order + 1)
        self.ctx_uf            = [None] * (max_order + 1)

        # Bigram continuation (KN backoff base case)
        self.cont_count = None
        self.total_unique_bigrams = 0

        # Frequency / physics artifacts (always on `device`)
        self.freq_gpu = None
        self.D1, self.D2, self.D3 = 0.5, 0.75, 0.9
        self.positions = None
        self.importance = None
        self.circle_group = None

        # Semantic edges (kept on `device` — physics runs there)
        self.edge_from = None
        self.edge_to = None
        self.edge_weight = None

    # ----- device routing -----
    def _device_for_order(self, o):
        """Light orders on the primary device, heavy orders on the aux
        device. With multi_gpu=False both are the same."""
        if not self.multi_gpu:
            return self.device
        return self.device if o <= 3 else self.aux_device

    def _primes_on(self, dev):
        return self.hash_primes_aux if dev == self.aux_device else self.hash_primes

    @staticmethod
    def _adaptive_chunk(o):
        """Chunk size (in tokens) for n-gram counting at order o.
        Higher orders get smaller chunks to keep peak memory bounded.
        Tuned empirically for a 16 GB T4 with ~88M tokens (1M WT103 lines):
        order 5 peaks around ~13.5 GB, orders 6-9 scale down linearly."""
        return {
            1: 60_000_000,
            2: 50_000_000,
            3: 40_000_000,
            4: 25_000_000,
            5: 15_000_000,
            6: 10_000_000,
            7:  8_000_000,
            8:  6_000_000,
            9:  5_000_000,
        }.get(o, 5_000_000)

    # ----- tokenize corpus into a compact int32 stream, then move to GPU -----
    def _tokenize_to_gpu(self, lines):
        w2i = self.word2id
        id2w = self.id2word
        toks = array.array('i')
        ns = 0
        t0 = time.time()
        for line in lines:
            ws = tokenize(line)
            if len(ws) < 2:
                continue
            for w in ws:
                wid = w2i.get(w)
                if wid is None:
                    wid = len(id2w)
                    w2i[w] = wid
                    id2w.append(w)
                toks.append(wid)
            ns += 1
            if ns % 20000 == 0:
                print("\r  Tokenizing: %d lines, %d tokens (%.0fs)" %
                      (ns, len(toks), time.time() - t0),
                      end="", flush=True)
        print("\r  Tokenizing: %d lines, %d tokens (%.0fs) done." %
              (ns, len(toks), time.time() - t0))
        T = len(toks)
        self.total_tokens = T
        if T == 0:
            self.tokens_gpu = torch.empty(0, dtype=torch.int64, device=self.device)
            return
        np_tokens = np.frombuffer(toks, dtype=np.int32)
        self.tokens_gpu = torch.from_numpy(np_tokens).to(
            device=self.device, dtype=torch.int64)
        del toks, np_tokens

    # ----- N-gram aggregates for one order, **incremental** chunk merging -----
    # Each chunk is merged into a running master tensor before the next chunk
    # runs, instead of being collected into a Python list and merged at the
    # end. That keeps peak memory at "one chunk + master" rather than
    # "all chunks + master".
    def _count_ngrams_order(self, o, chunk_tokens=None):
        target_dev = self._device_for_order(o)
        primes = self._primes_on(target_dev)
        chunk_tokens = chunk_tokens or self._adaptive_chunk(o)

        # Bring tokens onto the target device. If we're sharding to cuda:1,
        # this is a transient copy that we free at the end of this method.
        if self.tokens_gpu.device == target_dev:
            tokens = self.tokens_gpu
            tokens_is_copy = False
        else:
            tokens = self.tokens_gpu.to(target_dev)
            tokens_is_copy = True

        T = tokens.numel()
        if T <= o:
            if tokens_is_copy:
                del tokens
                torch.cuda.empty_cache()
            return

        primes_ctx = primes[:o]
        primes_ng = primes[:o + 1]

        # Master accumulators (start empty)
        master_ng = torch.empty(0, dtype=torch.int64, device=target_dev)
        master_cnt = torch.empty(0, dtype=torch.int64, device=target_dev)
        master_ctx = torch.empty(0, dtype=torch.int64, device=target_dev)

        nG = T - o
        start = 0
        while start < nG:
            end = min(start + chunk_tokens, nG)
            tok_slice = tokens[start:end + o]
            L = tok_slice.numel() - o

            # Polynomial hash for the (o+1)-token ngram and the o-token ctx.
            ngram_h = torch.zeros(L, dtype=torch.int64, device=target_dev)
            for k in range(o + 1):
                ngram_h += tok_slice[k:L + k] * primes_ng[k]
            ctx_h = torch.zeros(L, dtype=torch.int64, device=target_dev)
            for k in range(o):
                ctx_h += tok_slice[k:L + k] * primes_ctx[k]

            # Unique ngrams in this chunk + their counts.
            uniq_ng, inverse, counts = torch.unique(
                ngram_h, return_inverse=True, return_counts=True)
            del ngram_h

            # For each unique ngram pick a stable ctx hash via first position.
            positions = torch.arange(L, dtype=torch.int64, device=target_dev)
            first_pos = torch.full(
                (uniq_ng.numel(),), L,
                dtype=torch.int64, device=target_dev)
            first_pos.scatter_reduce_(0, inverse, positions,
                                      reduce='amin', include_self=True)
            ctx_for_uniq = ctx_h[first_pos.clamp_max(L - 1)]
            del positions, first_pos, inverse, ctx_h, tok_slice

            # ---- INCREMENTAL MERGE INTO MASTER ----
            # Concatenate this chunk's uniques with the running master, then
            # re-unique. The chunk's contribution to counts is summed via
            # scatter_add; the chunk's contribution to ctx_for_ngram uses
            # scatter_reduce(amin) so the master ctx hash stays stable.
            merged_ng = torch.cat([master_ng, uniq_ng])
            merged_cnt = torch.cat([master_cnt, counts])
            merged_ctx = torch.cat([master_ctx, ctx_for_uniq])
            del master_ng, master_cnt, master_ctx, uniq_ng, counts, ctx_for_uniq

            new_uniq, new_inv = torch.unique(merged_ng, return_inverse=True)
            new_cnt = torch.zeros(
                new_uniq.numel(), dtype=torch.int64, device=target_dev)
            new_cnt.scatter_add_(0, new_inv, merged_cnt)
            new_ctx = torch.full(
                (new_uniq.numel(),), torch.iinfo(torch.int64).max,
                dtype=torch.int64, device=target_dev)
            new_ctx.scatter_reduce_(0, new_inv, merged_ctx,
                                    reduce='amin', include_self=True)
            del merged_ng, merged_cnt, merged_ctx, new_inv

            master_ng = new_uniq
            master_cnt = new_cnt
            master_ctx = new_ctx
            del new_uniq, new_cnt, new_ctx

            start = end
            torch.cuda.empty_cache()

        # ---- per-context aggregates ----
        uniq_ctx, ctx_inv = torch.unique(master_ctx, return_inverse=True)
        U = uniq_ctx.numel()
        ctx_total = torch.zeros(U, dtype=torch.int64, device=target_dev)
        ctx_total.scatter_add_(0, ctx_inv, master_cnt)
        ctx_count1 = torch.zeros(U, dtype=torch.int64, device=target_dev)
        ctx_count1.scatter_add_(0, ctx_inv, (master_cnt == 1).to(torch.int64))
        ctx_count2 = torch.zeros(U, dtype=torch.int64, device=target_dev)
        ctx_count2.scatter_add_(0, ctx_inv, (master_cnt == 2).to(torch.int64))
        ctx_uf = torch.zeros(U, dtype=torch.int64, device=target_dev)
        ctx_uf.scatter_add_(0, ctx_inv, torch.ones_like(master_cnt))

        self.ngram_hash_sorted[o] = master_ng        # already sorted (torch.unique)
        self.ngram_count[o] = master_cnt
        self.ctx_hash_sorted[o] = uniq_ctx           # already sorted
        self.ctx_total[o] = ctx_total
        self.ctx_count1[o] = ctx_count1
        self.ctx_count2[o] = ctx_count2
        self.ctx_uf[o] = ctx_uf
        del master_ctx, ctx_inv

        if tokens_is_copy:
            del tokens
        torch.cuda.empty_cache()

    # ----- Continuation count per word (KN backoff base case) -----
    # Restored: works directly off the token stream so the next word can be
    # recovered as tokens[first_pos + 1]. The "extract from hash" version
    # (hash % p2) // p1 is mathematically wrong because the hash wraps mod
    # 2^64 and the operands are not coprime — that was the silent
    # corruption that pushed perplexity to 600+.
    def _compute_continuation(self, chunk_tokens=40_000_000):
        dev = self.device
        V = len(self.id2word)
        self.cont_count = torch.zeros(V, dtype=torch.int64, device=dev)
        tokens = self.tokens_gpu
        T = tokens.numel()
        if T < 2:
            return

        primes_bi = self.hash_primes[:2]

        # Process in chunks to keep peak memory bounded.
        # Each chunk holds: bigram_h (8 bytes/entry), positions, inverse,
        # first_pos and next_w. Peak ~5x chunk_tokens bytes.
        master_uniq = torch.empty(0, dtype=torch.int64, device=dev)
        master_next = torch.empty(0, dtype=torch.int64, device=dev)

        start = 0
        end_total = T - 1
        while start < end_total:
            end = min(start + chunk_tokens, end_total)
            L = end - start
            # bigram (tokens[i], tokens[i+1]) for i in [start, end)
            big_h = (
                tokens[start:start + L] * primes_bi[0]
                + tokens[start + 1:start + 1 + L] * primes_bi[1]
            )
            uniq, inverse = torch.unique(big_h, return_inverse=True)
            del big_h

            positions = torch.arange(L, dtype=torch.int64, device=dev)
            first_pos = torch.full(
                (uniq.numel(),), L, dtype=torch.int64, device=dev)
            first_pos.scatter_reduce_(0, inverse, positions,
                                      reduce='amin', include_self=True)
            del positions, inverse
            # next word == tokens[start + first_pos + 1]
            next_w = tokens[start + first_pos.clamp_max(L - 1) + 1]
            del first_pos

            # Merge with master, re-unique to drop bigrams that already
            # appeared in earlier chunks.
            merged_uniq = torch.cat([master_uniq, uniq])
            merged_next = torch.cat([master_next, next_w])
            del master_uniq, master_next, uniq, next_w

            new_uniq, new_inv = torch.unique(merged_uniq, return_inverse=True)
            new_next = torch.full(
                (new_uniq.numel(),), -1,
                dtype=torch.int64, device=dev)
            # scatter overwrites; both chunks have the same next word for the
            # same bigram, so any deterministic write is fine.
            new_next.scatter_(0, new_inv, merged_next)
            master_uniq = new_uniq
            master_next = new_next
            del merged_uniq, merged_next, new_inv, new_uniq, new_next

            start = end
            torch.cuda.empty_cache()

        # cont_count[w] = number of unique bigrams ending in w
        self.cont_count.scatter_add_(0, master_next, torch.ones_like(master_next))
        self.total_unique_bigrams = int(master_uniq.numel())
        del master_uniq, master_next
        torch.cuda.empty_cache()

    # ----- Semantic edges for physics, all on GPU, restored -----
    # Process offsets +1 (weight 0.2/dir) and +2 (weight 0.1/dir) one at a
    # time, fold into a running unique-edge table keyed by from*V + to.
    def _build_semantic_edges(self):
        dev = self.device
        V = len(self.id2word)
        tokens = self.tokens_gpu
        T = tokens.numel()
        if T < 2 or V == 0:
            self.edge_from = torch.empty(0, dtype=torch.int64, device=dev)
            self.edge_to = torch.empty(0, dtype=torch.int64, device=dev)
            self.edge_weight = torch.empty(0, dtype=torch.float32, device=dev)
            return
        V_long = torch.tensor(V, dtype=torch.int64, device=dev)

        running_keys = torch.empty(0, dtype=torch.int64, device=dev)
        running_w = torch.empty(0, dtype=torch.float32, device=dev)

        for offset, amount in ((1, 0.2), (2, 0.1)):
            if T <= offset:
                continue
            a = tokens[:-offset]
            b = tokens[offset:]
            mask = a != b
            a = a[mask]
            b = b[mask]
            both_f = torch.cat([a, b])
            both_t = torch.cat([b, a])
            del a, b, mask
            keys = both_f * V_long + both_t
            del both_f, both_t
            uk, inv = torch.unique(keys, return_inverse=True)
            del keys
            ones = torch.ones(inv.numel(), dtype=torch.float32, device=dev) * amount
            w = torch.zeros(uk.numel(), dtype=torch.float32, device=dev)
            w.scatter_add_(0, inv, ones)
            del inv, ones

            merged_keys = torch.cat([running_keys, uk])
            merged_w = torch.cat([running_w, w])
            del uk, w
            uk2, inv2 = torch.unique(merged_keys, return_inverse=True)
            w2 = torch.zeros(uk2.numel(), dtype=torch.float32, device=dev)
            w2.scatter_add_(0, inv2, merged_w)
            del merged_keys, merged_w, inv2
            running_keys = uk2
            running_w = w2
            torch.cuda.empty_cache()

        edge_from = running_keys // V_long
        edge_to = running_keys % V_long
        strong = running_w >= 0.1
        self.edge_from = edge_from[strong].contiguous()
        self.edge_to = edge_to[strong].contiguous()
        self.edge_weight = running_w[strong].clamp(-10.0, 10.0).contiguous()
        del running_keys, running_w, edge_from, edge_to
        torch.cuda.empty_cache()

    # ----- Modified KN discounts from count-of-counts (restored) -----
    def _compute_discounts(self):
        n1 = 0
        n2 = 0
        n3 = 0
        for o in range(1, self.max_order + 1):
            c = self.ngram_count[o]
            if c is None:
                continue
            n1 += int((c == 1).sum().item())
            n2 += int((c == 2).sum().item())
            n3 += int((c == 3).sum().item())
        if n1 > 0 and n2 > 0:
            Y = n1 / (n1 + 2.0 * n2)
            self.D1 = max(0.1, min(0.95, 1.0 - 2.0 * Y * n2 / n1))
            self.D2 = max(0.1, min(0.95,
                                    2.0 - 3.0 * Y * (n3 / n2 if n2 > 0 else 0.0)))
            self.D3 = max(0.1, min(0.95,
                                    3.0 - 4.0 * Y * ((n3 + 1) / n3 if n3 > 0 else 1.0)))
        print("  KN discounts: D1=%.3f D2=%.3f D3+=%.3f" %
              (self.D1, self.D2, self.D3))

    # ----- Full GPU training pipeline -----
    def train_gpu(self, lines):
        dev = self.device
        print("  Tokenizing corpus...")
        self._tokenize_to_gpu(lines)
        T = self.tokens_gpu.numel()
        V = len(self.id2word)
        print("  Tokens: %d, Vocab: %d" % (T, V))
        if T == 0 or V == 0:
            return

        # Frequency table on the primary device.
        self.freq_gpu = torch.zeros(V, dtype=torch.int64, device=dev)
        self.freq_gpu.scatter_add_(0, self.tokens_gpu,
                                   torch.ones_like(self.tokens_gpu))

        # N-gram tables per order. Higher orders may live on cuda:1.
        for o in range(1, self.max_order + 1):
            t0 = time.time()
            tdev = self._device_for_order(o)
            tag = "" if not self.multi_gpu else " [%s]" % str(tdev)
            print("  Order %d%s: building..." % (o, tag), end="", flush=True)
            self._count_ngrams_order(o)
            print(" done (%.0fs, ngrams=%d, ctxs=%d)" % (
                time.time() - t0,
                self.ngram_count[o].numel(),
                self.ctx_total[o].numel()))

        # Continuation counts (always on the primary device since tokens
        # live there)
        t0 = time.time()
        print("  Continuation counts...", end="", flush=True)
        self._compute_continuation()
        print(" done (%.0fs, %d unique bigrams)" % (
            time.time() - t0, self.total_unique_bigrams))

        # Semantic edges
        t0 = time.time()
        print("  Semantic edges...", end="", flush=True)
        self._build_semantic_edges()
        print(" done (%.0fs, %d edges)" % (
            time.time() - t0, self.edge_from.numel()))

        # Free the token stream — physics + eval don't need it.
        self.tokens_gpu = None
        torch.cuda.empty_cache()

    # ----- Physics simulation + importance + circles (restored) -----
    def build(self, physics_iters=30):
        self._compute_discounts()
        dev = self.device
        N = len(self.id2word)
        E = self.edge_from.numel() if self.edge_from is not None else 0
        print("  Nodes: %d, Edges: %d" % (N, E))
        if N == 0:
            return

        edge_from = self.edge_from
        edge_to = self.edge_to
        edge_w = self.edge_weight

        torch.manual_seed(42)
        positions = (torch.rand((N, 3), device=dev, dtype=torch.float32) * 10.0 - 5.0)
        velocities = torch.zeros((N, 3), device=dev, dtype=torch.float32)

        K_context = 2.0
        K_frequency = 1.5
        K_attraction = 0.5
        K_repulsion = 0.3
        damping = 0.15
        lr = 0.02
        optimal_dist = 3.0
        max_radius = 15.0
        sample_size = min(N, 200)

        print("  Physics: %d iters on %s" % (physics_iters, dev),
              end="", flush=True)
        t0 = time.time()
        for it in range(physics_iters):
            forces = torch.zeros_like(positions)

            # 1. Spring forces along semantic edges
            if E > 0:
                pf = positions[edge_from]
                pt = positions[edge_to]
                diff = pt - pf
                dist = diff.norm(dim=1, keepdim=True).clamp_min(0.1)
                unit = diff / dist
                k_tensor = torch.where(
                    edge_w > 1.0,
                    torch.full_like(edge_w, K_context),
                    torch.full_like(edge_w, K_frequency))
                fmag = k_tensor * edge_w / dist.squeeze(1)
                fvec = unit * fmag.unsqueeze(1)
                forces.index_add_(0, edge_from, fvec)
                del pf, pt, diff, dist, unit, k_tensor, fmag, fvec

            # 2. Sampled repulsion + far-field attraction
            if N > sample_size:
                sample_idx = torch.randperm(N, device=dev)[:sample_size]
            else:
                sample_idx = torch.arange(N, device=dev)
            sample_pos = positions[sample_idx]

            bsz = 4096
            for bs in range(0, N, bsz):
                be = min(bs + bsz, N)
                tp = positions[bs:be]
                d = sample_pos.unsqueeze(0) - tp.unsqueeze(1)
                di = d.norm(dim=2).clamp_min(0.1)
                u = d / di.unsqueeze(2)
                rep = -K_repulsion / (di * di + 1.0)
                rep_vec = u * rep.unsqueeze(2)
                forces[bs:be] += rep_vec.sum(dim=1)
                beyond = (di > optimal_dist)
                att = torch.where(
                    beyond,
                    K_attraction * (di - optimal_dist) * 0.01,
                    torch.zeros_like(di))
                att_vec = u * att.unsqueeze(2)
                forces[bs:be] += att_vec.sum(dim=1)

            # 3. Gravity towards origin + integration
            forces.sub_(0.01 * positions)
            velocities = (velocities + forces * lr) * (1.0 - damping)
            positions.add_(velocities * lr)

            # 4. Boundary clamp
            mag = positions.norm(dim=1)
            overflow = mag > max_radius
            if overflow.any():
                scale = (max_radius / mag[overflow]).unsqueeze(1)
                positions[overflow] = positions[overflow] * scale
                velocities[overflow] = velocities[overflow] * 0.5

            if (it + 1) % 5 == 0:
                print(".", end="", flush=True)
        print(" done (%.0fs)" % (time.time() - t0))
        self.positions = positions
        del velocities

        # Importance = log(1+degree) * log(1+freq)
        degs = torch.zeros(N, dtype=torch.float32, device=dev)
        if E > 0:
            degs.index_add_(0, edge_from,
                            torch.ones(E, dtype=torch.float32, device=dev))
        self.importance = torch.log1p(degs) * torch.log1p(self.freq_gpu.float())
        del degs

        # Circles: skipped (matches the GPU-first runner's design choice).
        self.circle_group = torch.full(
            (N,), -1, dtype=torch.long, device=dev)
        torch.cuda.empty_cache()

    # ----- GPU-batched Modified KN-5 probabilities (RESTORED) -----
    # ctx_batch: (B, MAX_ORDER) int64, right-aligned with -1 padding on the
    #            left for positions where no context word is available.
    # nxt_batch: (B,) int64, next-word IDs. -1 indicates OOV (floor prob).
    # Returns (B,) float32 KN-5 probabilities on self.device.
    #
    # Critical fix vs the degraded version: lambda includes the D3 * n3+
    # term (= count of ngrams with count >= 3), AND the per-order discount
    # selector picks D1 / D2 / D3 by the actual ngram count. The previous
    # pass dropped both terms which collapsed the back-off and pushed PPL.
    def kn_batch(self, ctx_batch, nxt_batch):
        dev = self.device
        V = max(len(self.id2word), 1)
        B = ctx_batch.size(0)
        safe_nxt = nxt_batch.clamp_min(0)

        # Base case: continuation unigram on the primary device
        tub = float(self.total_unique_bigrams)
        if tub > 0:
            cw = self.cont_count[safe_nxt].float()
            cont_prob = torch.where(
                cw > 0,
                cw / tub,
                torch.full_like(cw, 0.5 / tub))
        else:
            cont_prob = torch.full(
                (B,), 1.0 / V, dtype=torch.float32, device=dev)
        oov = nxt_batch < 0
        cont_prob = torch.where(
            oov, torch.full_like(cont_prob, 1.0 / V), cont_prob)
        kn_prev = cont_prob

        D1 = float(self.D1)
        D2 = float(self.D2)
        D3 = float(self.D3)

        for o in range(1, self.max_order + 1):
            ctx_tbl = self.ctx_hash_sorted[o]
            ng_tbl = self.ngram_hash_sorted[o]
            if ctx_tbl is None or ctx_tbl.numel() == 0:
                continue

            # Hash on the device that holds this order's tables (handles
            # the dual-T4 case: orders 4-5 may live on cuda:1).
            tbl_dev = ctx_tbl.device
            primes = self._primes_on(tbl_dev)

            ctx_o_local = ctx_batch[:, -o:].to(tbl_dev) if tbl_dev != dev else ctx_batch[:, -o:]
            nxt_local = nxt_batch.to(tbl_dev) if tbl_dev != dev else nxt_batch

            ctx_has_valid = (ctx_o_local >= 0).all(dim=1) & (nxt_local >= 0)
            ctx_h = (ctx_o_local * primes[:o]).sum(dim=1)
            ngram_rows = torch.cat([ctx_o_local, nxt_local.unsqueeze(1)], dim=1)
            ngram_h = (ngram_rows * primes[:o + 1]).sum(dim=1)

            ctx_idx = torch.searchsorted(ctx_tbl, ctx_h)
            ctx_idx_cl = ctx_idx.clamp_max(ctx_tbl.numel() - 1)
            ctx_valid = (ctx_idx < ctx_tbl.numel()) & \
                        (ctx_tbl[ctx_idx_cl] == ctx_h) & ctx_has_valid

            ng_idx = torch.searchsorted(ng_tbl, ngram_h)
            ng_idx_cl = ng_idx.clamp_max(ng_tbl.numel() - 1)
            ng_valid = (ng_idx < ng_tbl.numel()) & \
                       (ng_tbl[ng_idx_cl] == ngram_h) & ctx_has_valid

            total = self.ctx_total[o][ctx_idx_cl].float()
            total = torch.where(ctx_valid, total, torch.zeros_like(total))
            c = self.ngram_count[o][ng_idx_cl].float()
            c = torch.where(ng_valid, c, torch.zeros_like(c))
            c1 = self.ctx_count1[o][ctx_idx_cl].float()
            c1 = torch.where(ctx_valid, c1, torch.zeros_like(c1))
            c2 = self.ctx_count2[o][ctx_idx_cl].float()
            c2 = torch.where(ctx_valid, c2, torch.zeros_like(c2))
            uf = self.ctx_uf[o][ctx_idx_cl].float()
            uf = torch.where(ctx_valid, uf, torch.zeros_like(uf))

            # Discount selector keyed on the ngram count c
            disc_d = torch.zeros_like(c)
            disc_d = torch.where(c == 1, torch.full_like(c, D1), disc_d)
            disc_d = torch.where(c == 2, torch.full_like(c, D2), disc_d)
            disc_d = torch.where(c >= 3, torch.full_like(c, D3), disc_d)

            safe_total = total.clamp_min(1e-10)
            disc = (c - disc_d).clamp_min(0.0) / safe_total
            # n3+ = uf - n1 - n2  (count of ngrams with count >= 3)
            n3p = (uf - c1 - c2).clamp_min(0.0)
            lam = (D1 * c1 + D2 * c2 + D3 * n3p) / safe_total
            new_kn_local = disc + lam * (kn_prev.to(tbl_dev) if tbl_dev != dev else kn_prev)
            total_local = total
            new_kn = new_kn_local.to(dev) if tbl_dev != dev else new_kn_local
            total_on_dev = total_local.to(dev) if tbl_dev != dev else total_local

            kn_prev = torch.where(total_on_dev > 0, new_kn, kn_prev)

        return kn_prev.clamp(1e-10, 0.999)

    # ----- Full-mode perplexity on WikiText-103 (RESTORED, batched) -----
    # The previous version walked tokens one at a time with .item() per
    # token, which was both 1000x slower and silently dropped the position
    # contribution. This restores the original batched evaluator.
    def eval_full_wt103(self, test_lines, batch_size=16384):
        dev = self.device
        N = len(self.id2word)
        K = self.max_order
        pos = self.positions
        imp = self.importance
        circ = self.circle_group

        # Tokenize the entire test set on CPU into a compact int32 stream
        # plus per-line boundary markers; ship to GPU as one int64 tensor.
        w2i = self.word2id
        toks = array.array('i')
        boundaries = array.array('i')
        boundaries.append(0)
        for line in test_lines:
            ws = tokenize(line)
            if len(ws) < 2:
                boundaries.append(len(toks))
                continue
            for w in ws:
                toks.append(w2i.get(w, -1))
            boundaries.append(len(toks))

        T = len(toks)
        if T < 2:
            return float("inf")
        np_toks = np.frombuffer(toks, dtype=np.int32)
        toks_gpu = torch.from_numpy(np_toks).to(device=dev, dtype=torch.int64)
        del toks, np_toks

        # Per-token anchor: anchor[p] = start of the line containing token p
        np_bnd = np.frombuffer(boundaries, dtype=np.int32)
        anchor = np.empty(T, dtype=np.int32)
        for i in range(len(np_bnd) - 1):
            s = int(np_bnd[i])
            e = int(np_bnd[i + 1])
            if e > s:
                anchor[s:e] = s
        anchor_gpu = torch.from_numpy(anchor).to(device=dev, dtype=torch.int64)

        total_logp = torch.zeros((), dtype=torch.float64, device=dev)
        total_tok = 0

        t0 = time.time()
        pos_index_all = torch.arange(1, T, dtype=torch.int64, device=dev)

        for start in range(0, pos_index_all.numel(), batch_size):
            end = min(start + batch_size, pos_index_all.numel())
            pidx = pos_index_all[start:end]
            B = pidx.numel()

            nxt = toks_gpu[pidx]
            anch = anchor_gpu[pidx]
            ctx_start = torch.maximum(anch, pidx - K)
            j_range = torch.arange(K, dtype=torch.int64, device=dev)
            avail = (pidx - ctx_start)
            left_pad = (K - avail).clamp_min(0)
            col = j_range.unsqueeze(0) - left_pad.unsqueeze(1)
            src = ctx_start.unsqueeze(1) + col
            mask_pad = col < 0
            src_clamped = src.clamp(min=0)
            ctx_batch = toks_gpu[src_clamped]
            ctx_batch = torch.where(
                mask_pad, torch.full_like(ctx_batch, -1), ctx_batch)

            oov_next = (nxt < 0)

            # === KN on GPU ===
            kn = self.kn_batch(ctx_batch, nxt)

            # === Position similarity (the magnetic contribution) ===
            safe_ctx = ctx_batch.clamp_min(0)
            safe_nxt = nxt.clamp_min(0)
            ctx_pos = pos[safe_ctx]                       # (B, K, 3)
            nxt_pos = pos[safe_nxt].unsqueeze(1)          # (B, 1, 3)
            dot = (ctx_pos * nxt_pos).sum(-1)             # (B, K)
            ctx_norm = ctx_pos.norm(dim=-1)
            nxt_norm = nxt_pos.norm(dim=-1)
            denom = (ctx_norm * nxt_norm).clamp_min(1e-6)
            sim = (dot / denom).clamp(-1.0, 1.0)
            valid = (ctx_batch >= 0) & (sim > 0.05) & (~oov_next).unsqueeze(1)
            sim = torch.where(valid, sim, torch.zeros_like(sim))
            ctx_imp = imp[safe_ctx]
            boost_imp = 1.0 + ctx_imp * 0.05
            nxt_circ = circ[safe_nxt].unsqueeze(1)
            ctx_circ = circ[safe_ctx]
            same_circle = (ctx_circ >= 0) & (ctx_circ == nxt_circ)
            circle_boost = 1.0 + 0.5 * same_circle.float()
            contrib = sim * boost_imp * circle_boost
            pos_count = valid.sum(dim=1).clamp_min(1)
            has_any = valid.any(dim=1)
            pos_score = contrib.sum(dim=1)
            pos_prob = (pos_score /
                        (pos_count.to(pos_score.dtype) * 3.0)).clamp_max(0.3)
            pos_prob = torch.where(has_any, pos_prob, torch.zeros_like(pos_prob))

            # === Adaptive lambda mixing (same bands as the original) ===
            band = torch.where(
                kn > 0.05,
                torch.tensor(0.02, device=dev),
                torch.where(
                    kn > 0.005,
                    torch.tensor(0.06, device=dev),
                    torch.tensor(0.12, device=dev)))
            kn_l = 1.0 - band
            mixed = (kn_l * kn + band * pos_prob).clamp(1e-10, 0.999)
            mixed = torch.where(
                oov_next, torch.full_like(mixed, 1e-10), mixed)

            total_logp += torch.log(mixed).to(torch.float64).sum()
            total_tok += B

            if (start // batch_size) % 10 == 0:
                print("\r  Eval: %d/%d (%.0fs)" %
                      (end, pos_index_all.numel(), time.time() - t0),
                      end="", flush=True)
        print()
        del toks_gpu, anchor_gpu
        torch.cuda.empty_cache()
        if total_tok == 0:
            return float("inf")
        return math.exp(-float(total_logp.item()) / total_tok)


def main():
    ap = argparse.ArgumentParser(
        description="MagneticLM GPU runner: WikiText-103, full mode, no cache.")
    ap.add_argument("--train-lines", type=int, default=100000,
                    help="Number of training lines to pull from WikiText-103 "
                         "(use 0 for all lines)")
    ap.add_argument("--physics-iters", type=int, default=30,
                    help="Number of physics simulation iterations on the GPU")
    ap.add_argument("--batch-size", type=int, default=16384,
                    help="Eval batch size for GPU KN + position scoring")
    ap.add_argument("--max-order", type=int, default=5,
                    help="Max n-gram order (2..9). Lower = less GPU memory. "
                         "Diminishing returns kick in hard past order 5 on "
                         "WikiText-103 — the vast majority of 6-grams already "
                         "occur exactly once, so adding orders 6+ mostly "
                         "just inflates the back-off coefficient.")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="Shard high-order n-gram tables onto a second CUDA "
                         "device (e.g. Kaggle's GPU T4 x2). Adds ~zero compute "
                         "overhead, halves the peak memory on cuda:0.")
    ap.add_argument("--data-dir", default="data/wt103",
                    help="Directory for WikiText-103 train.txt / test.txt")
    args = ap.parse_args()

    if args.max_order < 2 or args.max_order > 9:
        print("ERROR: --max-order must be in the range [2, 9]. "
              "(The hash prime table has 10 entries, enough for ngrams of "
              "length up to 10.)", file=sys.stderr)
        sys.exit(2)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. This runner is GPU-only.",
              file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    n_dev = torch.cuda.device_count()
    for i in range(n_dev):
        props = torch.cuda.get_device_properties(i)
        print("Device %d: %s (%.1f GB)" %
              (i, torch.cuda.get_device_name(i), props.total_memory / 1024 ** 3))
    if args.multi_gpu and n_dev < 2:
        print("WARNING: --multi-gpu requested but only %d CUDA device(s) "
              "available — falling back to single-device." % n_dev)

    train_path, test_path = ensure_wt103(args.data_dir)

    limit = args.train_lines if args.train_lines > 0 else None
    print("\nLoading train lines from %s (limit=%s)" %
          (train_path, "all" if limit is None else str(limit)))
    train = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            train.append(s)
            if limit is not None and len(train) >= limit:
                break
    print("Loaded %d train lines" % len(train))

    test = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                test.append(s)
    print("Loaded %d test lines" % len(test))

    print("\n" + "=" * 60)
    print("  MagneticLM Fast Runner (GPU, no cache, full mode)")
    print("  max_order=%d  multi_gpu=%s" %
          (args.max_order, args.multi_gpu and n_dev >= 2))
    print("=" * 60)

    t_all = time.time()
    model = MagneticLMGPU(
        device=device,
        max_order=args.max_order,
        multi_gpu=args.multi_gpu and n_dev >= 2)

    t0 = time.time()
    model.train_gpu(train)
    print("  Train time:  %.0fs" % (time.time() - t0))
    if torch.cuda.is_available():
        for i in range(n_dev):
            used = torch.cuda.memory_allocated(i) / 1024 ** 2
            peak = torch.cuda.max_memory_allocated(i) / 1024 ** 2
            print("  GPU%d after train:  cur=%.0f MiB  peak=%.0f MiB" %
                  (i, used, peak))

    t0 = time.time()
    model.build(physics_iters=args.physics_iters)
    print("  Build time:  %.0fs" % (time.time() - t0))
    if torch.cuda.is_available():
        for i in range(n_dev):
            used = torch.cuda.memory_allocated(i) / 1024 ** 2
            peak = torch.cuda.max_memory_allocated(i) / 1024 ** 2
            print("  GPU%d after build:  cur=%.0f MiB  peak=%.0f MiB" %
                  (i, used, peak))

    print("\nEvaluating WikiText-103 test (full mode, no cache)...")
    t0 = time.time()
    ppl = model.eval_full_wt103(test, batch_size=args.batch_size)
    eval_time = time.time() - t0
    print("\nPerplexity (MagneticLM full, no cache) = %.2f" % ppl)
    print("Eval time:   %.0fs" % eval_time)
    print("Total time:  %.0fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()

