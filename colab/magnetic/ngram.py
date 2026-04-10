# magnetic/ngram.py
#
# Modified Kneser-Ney n-gram tables on GPU. Handles:
#   - polynomial hashing of n-gram windows
#   - per-order count tables with incremental chunk merging
#   - per-context aggregates (total, count1, count2, unique_followers)
#   - bigram continuation count for the KN base case
#   - D1/D2/D3 discounts computed from count-of-counts
#   - batched KN-5 scoring with the full lambda including the D3*n3+ term
#   - optional multi-GPU sharding: low-order tables on one device,
#     high-order tables on another. Only the per-batch query moves
#     across the device boundary, not the lookup tables.
#
# The core math and the incremental-merge memory pattern are inherited
# from MagneticLMFastRunner.py; the logic was ported here intact,
# re-organised into a single class, and wrapped with docstrings.

import sys
import time
from typing import Optional

try:
    import numpy as np
    import torch
except ImportError:
    print("ERROR: PyTorch and NumPy required.", file=sys.stderr)
    sys.exit(1)


# ----- Hash primes --------------------------------------------------------
# 10 well-known splitmix64 / Knuth multiplicative constants. The int64
# tensor wraps modulo 2^64 under torch arithmetic, giving a universal
# polynomial hash over the ngram window. Keep at least max_order+1
# entries in this list.
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


# ----- Adaptive chunk size by order ---------------------------------------
# Higher-order n-grams eat more memory per chunk (one int64 per hash,
# one int64 per context hash, etc.). Inversely scale the chunk size so
# peak memory stays bounded.
def _adaptive_chunk(order: int) -> int:
    if order <= 1:
        return 60_000_000
    if order == 2:
        return 30_000_000
    if order == 3:
        return 20_000_000
    if order == 4:
        return 12_000_000
    if order == 5:
        return 9_000_000
    if order == 6:
        return 7_000_000
    if order == 7:
        return 6_000_000
    if order == 8:
        return 5_500_000
    return 5_000_000


class NgramTables:
    """GPU-resident Modified KN-5 n-gram tables. One instance manages
    all orders 1..max_order, optionally sharded across two CUDA
    devices."""

    def __init__(self, config, device, aux_device=None):
        self.config = config
        self.max_order = config.max_ngram_order
        self.device = device
        self.aux_device = aux_device or device
        self.multi_gpu = (self.aux_device != self.device)
        self.low_order_max = config.low_order_max

        if self.max_order > len(_HASH_PRIMES_LIST) - 1:
            raise ValueError(
                "max_order too high for hash prime table "
                "(max %d, got %d)" %
                (len(_HASH_PRIMES_LIST) - 1, self.max_order))

        # Hash primes mirrored on each device used.
        self.hash_primes = _HASH_PRIMES_CPU.to(device)
        self.hash_primes_aux = (
            _HASH_PRIMES_CPU.to(self.aux_device) if self.multi_gpu
            else self.hash_primes)

        # Per-order tables.
        self.ngram_hash_sorted = [None] * (self.max_order + 1)
        self.ngram_count       = [None] * (self.max_order + 1)
        self.ctx_hash_sorted   = [None] * (self.max_order + 1)
        self.ctx_total         = [None] * (self.max_order + 1)
        self.ctx_count1        = [None] * (self.max_order + 1)
        self.ctx_count2        = [None] * (self.max_order + 1)
        self.ctx_uf            = [None] * (self.max_order + 1)

        # KN continuation (unigram base case)
        self.cont_count: Optional[torch.Tensor] = None
        self.total_unique_bigrams: int = 0

        # KN discounts (derived from count-of-counts after build).
        self.D1: float = 0.5
        self.D2: float = 0.75
        self.D3: float = 0.9

    # -------------------------------------------------------------------
    # Device routing
    # -------------------------------------------------------------------
    def device_for_order(self, o: int) -> torch.device:
        if not self.multi_gpu:
            return self.device
        return self.device if o <= self.low_order_max else self.aux_device

    def primes_on(self, dev: torch.device) -> torch.Tensor:
        if dev == self.device:
            return self.hash_primes
        return self.hash_primes_aux

    # -------------------------------------------------------------------
    # Training: build all tables from a token stream
    # -------------------------------------------------------------------
    def build(self, tokens_gpu: torch.Tensor, vocab_size: int, verbose: bool = True):
        """Build n-gram tables from the token stream. tokens_gpu lives
        on the primary device; high-order tables may be placed on the
        aux device."""
        T = tokens_gpu.numel()
        if T == 0 or vocab_size == 0:
            return

        for o in range(1, self.max_order + 1):
            t0 = time.time()
            tdev = self.device_for_order(o)
            tag = ("" if not self.multi_gpu
                   else " [%s]" % str(tdev))
            if verbose:
                print("  Order %d%s: building..." % (o, tag),
                      end="", flush=True)
            self._count_order(tokens_gpu, o, tdev)
            if verbose:
                print(" done (%.0fs, ngrams=%d, ctxs=%d)" % (
                    time.time() - t0,
                    self.ngram_count[o].numel(),
                    self.ctx_total[o].numel()))

        # Continuation counts and KN base unigram (always on the
        # primary device because the token stream lives there).
        if verbose:
            print("  Continuation counts...", end="", flush=True)
        t0 = time.time()
        self._compute_continuation(tokens_gpu, vocab_size)
        if verbose:
            print(" done (%.0fs, %d unique bigrams)" % (
                time.time() - t0, self.total_unique_bigrams))

        # Discounts from count-of-counts.
        self._compute_discounts()
        if verbose:
            print("  KN discounts: D1=%.3f D2=%.3f D3+=%.3f" %
                  (self.D1, self.D2, self.D3))

    # -------------------------------------------------------------------
    # Per-order counting with incremental chunk merging
    # -------------------------------------------------------------------
    def _count_order(self, tokens_gpu: torch.Tensor, order: int, tdev: torch.device):
        """Count n-grams of size (order+1): context of `order` tokens
        plus one next word. Faithful port of MagneticLMFastRunner.py
        _count_ngrams_order (lines 251-362), the tested implementation
        that achieves 14.20 PPL."""
        src_dev = tokens_gpu.device
        T = tokens_gpu.numel()
        if T <= order:
            empty_i = torch.empty(0, dtype=torch.int64, device=tdev)
            empty_l = torch.empty(0, dtype=torch.int64, device=tdev)
            self.ngram_hash_sorted[order] = empty_i
            self.ngram_count[order] = empty_l
            self.ctx_hash_sorted[order] = empty_i
            self.ctx_total[order] = empty_l
            self.ctx_count1[order] = empty_l
            self.ctx_count2[order] = empty_l
            self.ctx_uf[order] = empty_l
            return

        # Move tokens to the target device if needed (multi-GPU: high
        # orders may live on cuda:1).
        if src_dev == tdev:
            tokens = tokens_gpu
            tokens_is_copy = False
        else:
            tokens = tokens_gpu.to(tdev)
            tokens_is_copy = True

        primes = self.primes_on(tdev)
        primes_ctx = primes[:order]      # hash first `order` tokens
        primes_ng = primes[:order + 1]   # hash all `order+1` tokens

        # Master accumulators (start empty).
        master_ng = torch.empty(0, dtype=torch.int64, device=tdev)
        master_cnt = torch.empty(0, dtype=torch.int64, device=tdev)
        master_ctx = torch.empty(0, dtype=torch.int64, device=tdev)

        chunk = _adaptive_chunk(order)
        nG = T - order   # number of valid (order+1)-token windows

        start = 0
        while start < nG:
            end = min(start + chunk, nG)
            # tok_slice has (end - start + order) tokens so the sliding
            # window tok_slice[k : L+k] works for k in 0..order.
            tok_slice = tokens[start:end + order]
            L = tok_slice.numel() - order

            # Polynomial hash for the (order+1)-token n-gram.
            ngram_h = torch.zeros(L, dtype=torch.int64, device=tdev)
            for k in range(order + 1):
                ngram_h += tok_slice[k:L + k] * primes_ng[k]

            # Polynomial hash for the order-token context.
            ctx_h = torch.zeros(L, dtype=torch.int64, device=tdev)
            for k in range(order):
                ctx_h += tok_slice[k:L + k] * primes_ctx[k]

            # Unique n-grams in this chunk + their counts.
            uniq_ng, inverse, counts = torch.unique(
                ngram_h, return_inverse=True, return_counts=True)
            del ngram_h

            # For each unique n-gram pick a stable ctx hash via
            # the first position where it appeared (scatter_reduce amin).
            positions = torch.arange(L, dtype=torch.int64, device=tdev)
            first_pos = torch.full(
                (uniq_ng.numel(),), L,
                dtype=torch.int64, device=tdev)
            first_pos.scatter_reduce_(
                0, inverse, positions,
                reduce='amin', include_self=True)
            ctx_for_uniq = ctx_h[first_pos.clamp_max(L - 1)]
            del positions, first_pos, inverse, ctx_h, tok_slice

            # ---- Incremental merge into master tables ----
            merged_ng = torch.cat([master_ng, uniq_ng])
            merged_cnt = torch.cat([master_cnt, counts])
            merged_ctx = torch.cat([master_ctx, ctx_for_uniq])
            del master_ng, master_cnt, master_ctx
            del uniq_ng, counts, ctx_for_uniq

            new_uniq, new_inv = torch.unique(
                merged_ng, return_inverse=True)
            new_cnt = torch.zeros(
                new_uniq.numel(), dtype=torch.int64, device=tdev)
            new_cnt.scatter_add_(0, new_inv, merged_cnt)
            new_ctx = torch.full(
                (new_uniq.numel(),), torch.iinfo(torch.int64).max,
                dtype=torch.int64, device=tdev)
            new_ctx.scatter_reduce_(
                0, new_inv, merged_ctx,
                reduce='amin', include_self=True)
            del merged_ng, merged_cnt, merged_ctx, new_inv

            master_ng = new_uniq
            master_cnt = new_cnt
            master_ctx = new_ctx
            del new_uniq, new_cnt, new_ctx

            start = end
            if tdev.type == "cuda":
                torch.cuda.empty_cache()

        # ---- Per-context aggregates ----
        if master_ng.numel() == 0:
            empty_i = torch.empty(0, dtype=torch.int64, device=tdev)
            empty_l = torch.empty(0, dtype=torch.int64, device=tdev)
            self.ngram_hash_sorted[order] = empty_i
            self.ngram_count[order] = empty_l
            self.ctx_hash_sorted[order] = empty_i
            self.ctx_total[order] = empty_l
            self.ctx_count1[order] = empty_l
            self.ctx_count2[order] = empty_l
            self.ctx_uf[order] = empty_l
            if tokens_is_copy:
                del tokens
            return

        uniq_ctx, ctx_inv = torch.unique(master_ctx, return_inverse=True)
        U = uniq_ctx.numel()
        ctx_total = torch.zeros(U, dtype=torch.int64, device=tdev)
        ctx_total.scatter_add_(0, ctx_inv, master_cnt)
        ctx_count1 = torch.zeros(U, dtype=torch.int64, device=tdev)
        ctx_count1.scatter_add_(0, ctx_inv, (master_cnt == 1).to(torch.int64))
        ctx_count2 = torch.zeros(U, dtype=torch.int64, device=tdev)
        ctx_count2.scatter_add_(0, ctx_inv, (master_cnt == 2).to(torch.int64))
        ctx_uf = torch.zeros(U, dtype=torch.int64, device=tdev)
        ctx_uf.scatter_add_(0, ctx_inv, torch.ones_like(master_cnt))

        # master_ng and uniq_ctx are already sorted (torch.unique output).
        self.ngram_hash_sorted[order] = master_ng
        self.ngram_count[order] = master_cnt
        self.ctx_hash_sorted[order] = uniq_ctx
        self.ctx_total[order] = ctx_total
        self.ctx_count1[order] = ctx_count1
        self.ctx_count2[order] = ctx_count2
        self.ctx_uf[order] = ctx_uf

        del master_ctx, ctx_inv
        if tokens_is_copy:
            del tokens
        if tdev.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Continuation counts (unique contexts following each word)
    # -------------------------------------------------------------------
    def _compute_continuation(self, tokens_gpu: torch.Tensor, vocab_size: int):
        dev = self.device
        T = tokens_gpu.numel()
        if T < 2:
            self.cont_count = torch.zeros(vocab_size, dtype=torch.int64, device=dev)
            self.total_unique_bigrams = 0
            return

        a = tokens_gpu[:-1]
        b = tokens_gpu[1:]
        pair_key = a * vocab_size + b

        # Unique bigrams and, for each, the first position where they
        # appeared. We then look up tokens[first_pos + 1] to recover
        # the next word id without trying to invert the hash (the
        # inversion bug cost us 600 PPL once - do not try it).
        uk, inv = torch.unique(pair_key, return_inverse=True)
        first_pos = torch.full(
            (uk.numel(),), pair_key.numel(), dtype=torch.int64, device=dev)
        arange = torch.arange(pair_key.numel(), dtype=torch.int64, device=dev)
        # scatter_reduce with amin gives the smallest index per bigram.
        first_pos.scatter_reduce_(0, inv, arange, reduce="amin", include_self=True)
        del arange, inv

        # next_word[i] = tokens_gpu[first_pos[i] + 1]; since each pair
        # starts with tokens_gpu[first_pos[i]] = uk // V, this is well
        # defined. Clamp first_pos so +1 stays in range.
        first_pos = first_pos.clamp_max(pair_key.numel() - 1)
        next_word = tokens_gpu[(first_pos + 1).clamp_max(tokens_gpu.numel() - 1)]

        # cont_count[w] = number of unique bigrams whose next word is w.
        self.cont_count = torch.zeros(vocab_size, dtype=torch.int64, device=dev)
        ones = torch.ones(next_word.numel(), dtype=torch.int64, device=dev)
        self.cont_count.scatter_add_(0, next_word, ones)
        self.total_unique_bigrams = int(uk.numel())

    # -------------------------------------------------------------------
    # Modified KN discounts from count-of-counts
    # -------------------------------------------------------------------
    def _compute_discounts(self):
        n1 = n2 = n3 = 0
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

    # -------------------------------------------------------------------
    # Modified KN-5 batch scoring
    # -------------------------------------------------------------------
    # ctx_batch: (B, max_order) int64, right-aligned with -1 padding on
    #            the left for positions where no context word is available.
    # nxt_batch: (B,) int64, next-word ids. -1 indicates OOV.
    # Returns (B,) float32 on self.device, clamped to [1e-10, 0.999].
    def kn_score_batch(self, ctx_batch: torch.Tensor, nxt_batch: torch.Tensor) -> torch.Tensor:
        dev = self.device
        V = max(self.cont_count.numel() if self.cont_count is not None else 1, 1)
        B = ctx_batch.size(0)
        safe_nxt = nxt_batch.clamp_min(0)

        # Base case: continuation unigram on the primary device.
        tub = float(self.total_unique_bigrams)
        if tub > 0 and self.cont_count is not None:
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

            tbl_dev = ctx_tbl.device
            primes = self.primes_on(tbl_dev)

            ctx_o_local = (ctx_batch[:, -o:].to(tbl_dev)
                           if tbl_dev != dev else ctx_batch[:, -o:])
            nxt_local = (nxt_batch.to(tbl_dev)
                         if tbl_dev != dev else nxt_batch)

            ctx_has_valid = (ctx_o_local >= 0).all(dim=1) & (nxt_local >= 0)
            ctx_h = (ctx_o_local * primes[:o]).sum(dim=1)
            ngram_rows = torch.cat(
                [ctx_o_local, nxt_local.unsqueeze(1)], dim=1)
            ngram_h = (ngram_rows * primes[:o + 1]).sum(dim=1)

            ctx_idx = torch.searchsorted(ctx_tbl, ctx_h)
            ctx_idx_cl = ctx_idx.clamp_max(ctx_tbl.numel() - 1)
            ctx_valid = ((ctx_idx < ctx_tbl.numel()) &
                         (ctx_tbl[ctx_idx_cl] == ctx_h) &
                         ctx_has_valid)

            ng_idx = torch.searchsorted(ng_tbl, ngram_h)
            ng_idx_cl = ng_idx.clamp_max(ng_tbl.numel() - 1)
            ng_valid = ((ng_idx < ng_tbl.numel()) &
                        (ng_tbl[ng_idx_cl] == ngram_h) &
                        ctx_has_valid)

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
            # n3+ = uf - n1 - n2 (count of ngrams with count >= 3)
            n3p = (uf - c1 - c2).clamp_min(0.0)
            lam = (D1 * c1 + D2 * c2 + D3 * n3p) / safe_total

            new_kn_local = disc + lam * (
                kn_prev.to(tbl_dev) if tbl_dev != dev else kn_prev)
            new_kn = (new_kn_local.to(dev)
                      if tbl_dev != dev else new_kn_local)
            total_on_dev = (total.to(dev)
                            if tbl_dev != dev else total)

            kn_prev = torch.where(total_on_dev > 0, new_kn, kn_prev)

        return kn_prev.clamp(1e-10, 0.999)
