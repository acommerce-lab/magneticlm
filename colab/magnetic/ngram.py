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
        src_dev = tokens_gpu.device
        T = tokens_gpu.numel()
        if T < order:
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

        primes = self.primes_on(tdev)

        master_ng: Optional[torch.Tensor] = None
        master_cnt: Optional[torch.Tensor] = None
        master_ctx: Optional[torch.Tensor] = None
        master_ctx_cnt: Optional[torch.Tensor] = None

        chunk = _adaptive_chunk(order)
        # Number of n-gram starting positions is T - order + 1.
        total_pos = T - order + 1

        for start in range(0, total_pos, chunk):
            end = min(start + chunk, total_pos)

            # Build the (B, order) window of tokens for this chunk on the
            # source device, then move to tdev (which may differ for
            # high-order tables in multi-GPU mode).
            idx = torch.arange(start, end, dtype=torch.int64, device=src_dev)
            win = torch.stack(
                [tokens_gpu[idx + j] for j in range(order)], dim=1)
            if tdev != src_dev:
                win = win.to(tdev)

            # N-gram hash (full window)
            ngram_h = (win * primes[:order]).sum(dim=1)
            # Context hash (first order-1 tokens; context for this order)
            if order >= 2:
                ctx_h = (win[:, :-1] * primes[:order - 1]).sum(dim=1)
            else:
                ctx_h = torch.zeros(win.size(0), dtype=torch.int64, device=tdev)

            del win, idx

            # Unique n-grams in this chunk and their counts.
            uh, inv = torch.unique(ngram_h, return_inverse=True)
            cnt = torch.zeros(uh.numel(), dtype=torch.int64, device=tdev)
            ones = torch.ones(ngram_h.numel(), dtype=torch.int64, device=tdev)
            cnt.scatter_add_(0, inv, ones)
            del ones, inv, ngram_h

            # Unique contexts for this chunk and their counts.
            uctx, cinv = torch.unique(ctx_h, return_inverse=True)
            cctx = torch.zeros(uctx.numel(), dtype=torch.int64, device=tdev)
            ones = torch.ones(ctx_h.numel(), dtype=torch.int64, device=tdev)
            cctx.scatter_add_(0, cinv, ones)
            del ones, cinv, ctx_h

            # Merge into the running master tables (for both ngrams
            # and contexts). This keeps peak memory at ~2x per-chunk
            # instead of collecting a Python list.
            if master_ng is None:
                master_ng = uh
                master_cnt = cnt
                master_ctx = uctx
                master_ctx_cnt = cctx
            else:
                merged_h = torch.cat([master_ng, uh])
                merged_c = torch.cat([master_cnt, cnt])
                uh2, inv2 = torch.unique(merged_h, return_inverse=True)
                cnt2 = torch.zeros(
                    uh2.numel(), dtype=torch.int64, device=tdev)
                cnt2.scatter_add_(0, inv2, merged_c)
                master_ng = uh2
                master_cnt = cnt2
                del merged_h, merged_c, inv2, uh, cnt

                merged_cx = torch.cat([master_ctx, uctx])
                merged_cc = torch.cat([master_ctx_cnt, cctx])
                uc2, cinv2 = torch.unique(merged_cx, return_inverse=True)
                cc2 = torch.zeros(
                    uc2.numel(), dtype=torch.int64, device=tdev)
                cc2.scatter_add_(0, cinv2, merged_cc)
                master_ctx = uc2
                master_ctx_cnt = cc2
                del merged_cx, merged_cc, cinv2, uctx, cctx

            if tdev.type == "cuda":
                torch.cuda.empty_cache()

        # Derive per-context aggregates (count1, count2, unique
        # followers) from the master n-gram table.
        if master_ng is None or master_ng.numel() == 0:
            empty_l = torch.empty(0, dtype=torch.int64, device=tdev)
            self.ngram_hash_sorted[order] = torch.empty(0, dtype=torch.int64, device=tdev)
            self.ngram_count[order] = empty_l
            self.ctx_hash_sorted[order] = torch.empty(0, dtype=torch.int64, device=tdev)
            self.ctx_total[order] = empty_l
            self.ctx_count1[order] = empty_l
            self.ctx_count2[order] = empty_l
            self.ctx_uf[order] = empty_l
            return

        # Re-derive the context hash for each master ngram, then scatter
        # aggregates into positions aligned with the master context table.
        # This is cheap because master_ng is small compared to T.
        if order >= 2:
            # Recover the window from the hash? Not invertible. Instead
            # we recompute per-n-gram context hashes directly by taking
            # the first order-1 primes, but since we don't have the
            # original tokens here we need to pass them through. The
            # easiest solution is: we do one more pass over the chunks
            # to collect (ctx_hash, ngram_count) tuples. Use a small
            # scan: build a mapping from master_ng -> its context hash
            # by walking the chunks a second time.
            ctx_for_ng = torch.empty_like(master_ng)
            # The master_ng tensor is sorted by torch.unique so we can
            # binary search it.
            master_ng_sorted, sort_perm = torch.sort(master_ng)
            inv_perm = torch.empty_like(sort_perm)
            inv_perm[sort_perm] = torch.arange(
                master_ng.numel(), device=tdev)

            for start in range(0, total_pos, chunk):
                end = min(start + chunk, total_pos)
                idx = torch.arange(
                    start, end, dtype=torch.int64, device=src_dev)
                win = torch.stack(
                    [tokens_gpu[idx + j] for j in range(order)], dim=1)
                if tdev != src_dev:
                    win = win.to(tdev)
                ng_h = (win * primes[:order]).sum(dim=1)
                ct_h = (win[:, :-1] * primes[:order - 1]).sum(dim=1)
                del win, idx

                # Find each chunk n-gram in the sorted master table.
                pos = torch.searchsorted(master_ng_sorted, ng_h)
                # In this construction the n-grams are guaranteed to
                # match; still clamp defensively.
                pos = pos.clamp_max(master_ng_sorted.numel() - 1)
                orig_idx = sort_perm[pos]
                ctx_for_ng[orig_idx] = ct_h
                del ng_h, ct_h, pos, orig_idx
                if tdev.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Order 1: context is the empty sequence, always hashed to 0.
            ctx_for_ng = torch.zeros_like(master_ng)

        # Sort master_ng for searchsorted lookups during eval.
        ng_sorted, ng_perm = torch.sort(master_ng)
        ng_count_sorted = master_cnt[ng_perm]
        ctx_for_ng_sorted = ctx_for_ng[ng_perm]
        del master_ng, master_cnt, ctx_for_ng, ng_perm

        # Sort master_ctx for searchsorted lookups during eval.
        ctx_sorted, ctx_perm = torch.sort(master_ctx)
        ctx_count_sorted = master_ctx_cnt[ctx_perm]
        del master_ctx, master_ctx_cnt, ctx_perm

        # Map each n-gram's context to its index in the sorted context
        # table so we can fold count1 / count2 / unique_followers.
        ctx_idx = torch.searchsorted(ctx_sorted, ctx_for_ng_sorted)
        ctx_idx = ctx_idx.clamp_max(ctx_sorted.numel() - 1)

        ctx_count1 = torch.zeros(ctx_sorted.numel(), dtype=torch.int64, device=tdev)
        ctx_count2 = torch.zeros(ctx_sorted.numel(), dtype=torch.int64, device=tdev)
        ctx_uf     = torch.zeros(ctx_sorted.numel(), dtype=torch.int64, device=tdev)

        mask1 = (ng_count_sorted == 1)
        mask2 = (ng_count_sorted == 2)
        ones_i64 = torch.ones(ng_count_sorted.numel(), dtype=torch.int64, device=tdev)

        ctx_count1.scatter_add_(0, ctx_idx, mask1.to(torch.int64))
        ctx_count2.scatter_add_(0, ctx_idx, mask2.to(torch.int64))
        ctx_uf.scatter_add_(0, ctx_idx, ones_i64)

        del mask1, mask2, ones_i64, ctx_for_ng_sorted, ctx_idx

        self.ngram_hash_sorted[order] = ng_sorted
        self.ngram_count[order]       = ng_count_sorted
        self.ctx_hash_sorted[order]   = ctx_sorted
        self.ctx_total[order]         = ctx_count_sorted
        self.ctx_count1[order]        = ctx_count1
        self.ctx_count2[order]        = ctx_count2
        self.ctx_uf[order]            = ctx_uf

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
