# pulse/contextual_map.py
#
# Contextual Map: "A يتبعها B" — which words directly follow which.
#
# This is the MANDATORY map. No word enters the candidate set unless
# it is a contextual child of someone:
#   - Direct child of the current word, OR
#   - Child of a semantic neighbor of the current word (adoption)
#
# Storage: two GPU-friendly formats:
#   1. CSR (offsets/targets/counts/probs) for per-word child enumeration
#   2. Sorted pair-keys (src*V+dst → prob) for O(log E) batch lookup
#
# Built from the token stream in one GPU pass.

import torch
import time


class ContextualMap:
    """Direct successor map on GPU.

    For each word w, ctx_children(w) = words that followed w in the
    training corpus, with counts and probabilities.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.V = 0

        # CSR format (for enumerating children of a word)
        self.offsets = None    # (V+1,) int64
        self.targets = None    # (E,) int64
        self.counts = None     # (E,) float32
        self.probs = None      # (E,) float32

        # Sorted pair-key format (for batch lookup)
        self.pair_keys = None  # (E,) int64, sorted — src * V + dst
        self.pair_probs = None # (E,) float32

        # Unigram
        self.unigram_probs = None  # (V,) float32
        self.total_tokens = 0

    @staticmethod
    def build_from_tokens(tokens_gpu: torch.Tensor, V: int,
                          freq_gpu: torch.Tensor,
                          device: torch.device,
                          max_children: int = 0,
                          verbose: bool = True) -> 'ContextualMap':
        """Build the contextual map from a token stream on GPU.

        Args:
            tokens_gpu: (T,) int64 token IDs
            V: vocabulary size
            freq_gpu: (V,) int64 unigram frequency counts
            device: target device
            max_children: if > 0, keep only top-K children per word
            verbose: print progress
        """
        t0 = time.time()
        T = tokens_gpu.numel()
        cm = ContextualMap(device)
        cm.V = V
        cm.total_tokens = T

        # Unigram probabilities
        cm.unigram_probs = freq_gpu.float() / max(T, 1)

        if T < 2:
            cm.offsets = torch.zeros(V + 1, dtype=torch.int64, device=device)
            cm.targets = torch.empty(0, dtype=torch.int64, device=device)
            cm.counts = torch.empty(0, dtype=torch.float32, device=device)
            cm.probs = torch.empty(0, dtype=torch.float32, device=device)
            cm.pair_keys = torch.empty(0, dtype=torch.int64, device=device)
            cm.pair_probs = torch.empty(0, dtype=torch.float32, device=device)
            return cm

        # All bigrams: (src, dst)
        src = tokens_gpu[:-1]
        dst = tokens_gpu[1:]

        # Encode as pair key: src * V + dst
        V_long = torch.tensor(V, dtype=torch.int64, device=device)
        pair_keys_all = src * V_long + dst

        # Count unique pairs
        uniq_keys, pair_counts = torch.unique(
            pair_keys_all, return_counts=True)
        del pair_keys_all

        # Decode
        pair_src = uniq_keys // V_long
        pair_dst = uniq_keys % V_long
        counts_f = pair_counts.float()
        del pair_counts

        # Sort by source word (for CSR)
        sort_idx = pair_src.argsort()
        pair_src = pair_src[sort_idx]
        pair_dst = pair_dst[sort_idx]
        counts_f = counts_f[sort_idx]
        uniq_keys_sorted = uniq_keys[sort_idx]
        del sort_idx, uniq_keys

        # Optional: keep only top-K children per source word
        if max_children > 0:
            pair_src, pair_dst, counts_f, uniq_keys_sorted = \
                _topk_per_source(pair_src, pair_dst, counts_f,
                                 uniq_keys_sorted, V, max_children, device)

        # Build CSR offsets
        E = pair_src.numel()
        offsets = torch.zeros(V + 1, dtype=torch.int64, device=device)
        if E > 0:
            ones = torch.ones(E, dtype=torch.int64, device=device)
            offsets[1:].scatter_add_(0, pair_src, ones)
            del ones
        offsets = offsets.cumsum(0)

        # Per-source totals for probabilities
        totals = torch.zeros(V, dtype=torch.float32, device=device)
        if E > 0:
            totals.scatter_add_(0, pair_src, counts_f)
        src_totals = totals[pair_src]
        probs = counts_f / src_totals.clamp_min(1e-10)

        cm.offsets = offsets
        cm.targets = pair_dst.contiguous()
        cm.counts = counts_f.contiguous()
        cm.probs = probs.contiguous()

        # Sorted pair-key format for batch lookup
        pk_sort = uniq_keys_sorted.argsort()
        cm.pair_keys = uniq_keys_sorted[pk_sort].contiguous()
        cm.pair_probs = probs[pk_sort].contiguous()
        del pk_sort, uniq_keys_sorted

        if verbose:
            avg = E / max(V, 1)
            print("  Contextual map: %d unique bigrams, "
                  "avg %.1f children/word (%.0fs)" % (E, avg, time.time() - t0))

        return cm

    # ------------------------------------------------------------------
    # Batch lookup: P_ctx(target | current) for a batch of pairs
    # ------------------------------------------------------------------
    def lookup_batch(self, current: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """Look up contextual bigram probabilities for a batch.

        Args:
            current: (B,) int64 — current word IDs
            target: (B,) int64 — target word IDs

        Returns:
            (B,) float32 — P_ctx(target | current), 0 if not a child
        """
        V = self.V
        query = current * V + target
        idx = torch.searchsorted(self.pair_keys, query)
        idx_cl = idx.clamp_max(self.pair_keys.numel() - 1)
        found = (idx < self.pair_keys.numel()) & \
                (self.pair_keys[idx_cl] == query)
        prob = torch.where(found, self.pair_probs[idx_cl],
                           torch.zeros_like(self.pair_probs[0:1]).expand(query.numel()))
        return prob

    # ------------------------------------------------------------------
    # Batch adoption lookup: for each (current, target), check if
    # target is a child of any semantic neighbor of current
    # ------------------------------------------------------------------
    def adoption_batch(self, current: torch.Tensor,
                       target: torch.Tensor,
                       sem_map,  # SemanticMap
                       cfg) -> torch.Tensor:
        """Compute adoption probability for a batch.

        For each position: P_adopt = Σ_s norm_w(s) * P_ctx(target | s)
        where s are semantic neighbors of current, and norm_w is the
        normalized semantic weight.

        Args:
            current: (B,) int64
            target: (B,) int64
            sem_map: SemanticMap instance
            cfg: PulseConfig

        Returns:
            (B,) float32 adoption probabilities
        """
        dev = self.device
        B = current.numel()
        adopt_prob = torch.zeros(B, dtype=torch.float32, device=dev)

        # Process by unique current words for efficiency
        uniq_current, inverse = torch.unique(current, return_inverse=True)

        for uidx in range(uniq_current.numel()):
            cw = int(uniq_current[uidx].item())
            if cw < 0:
                continue

            # Get semantic neighbors with positive weight
            neighbors = sem_map.get_all_neighbors(cw)
            neighbors = [(nid, w) for nid, w in neighbors
                         if w >= cfg.adoption_min_weight]
            neighbors = neighbors[:cfg.adoption_neighbors]
            if not neighbors:
                continue

            # Positions in batch with this current word
            mask = (inverse == uidx)
            batch_targets = target[mask]

            # Normalize semantic weights
            total_sem_w = sum(w for _, w in neighbors)
            if total_sem_w <= 0:
                continue

            # For each neighbor, look up P_ctx(target | neighbor)
            neighbor_contrib = torch.zeros(batch_targets.numel(),
                                           dtype=torch.float32, device=dev)
            for sem_id, sem_w in neighbors:
                norm_w = sem_w / total_sem_w
                sem_current = torch.full_like(batch_targets, sem_id)
                p = self.lookup_batch(sem_current, batch_targets)
                neighbor_contrib += norm_w * p

            adopt_prob[mask] = neighbor_contrib

        return adopt_prob

    # ------------------------------------------------------------------
    # Per-word child enumeration (for candidate set construction)
    # ------------------------------------------------------------------
    def children_of(self, word_id: int):
        """Get (target_ids, probs) for a single word."""
        start = int(self.offsets[word_id].item())
        end = int(self.offsets[word_id + 1].item())
        if start == end:
            return (torch.empty(0, dtype=torch.int64, device=self.device),
                    torch.empty(0, dtype=torch.float32, device=self.device))
        return self.targets[start:end], self.probs[start:end]

    def num_children(self, word_id: int) -> int:
        return int(self.offsets[word_id + 1].item() -
                   self.offsets[word_id].item())

    def has_child(self, word_id: int, child_id: int) -> bool:
        start = int(self.offsets[word_id].item())
        end = int(self.offsets[word_id + 1].item())
        if start == end:
            return False
        return (self.targets[start:end] == child_id).any().item()

    def get_candidate_set(self, word_id: int,
                          semantic_neighbors: list):
        """Build candidate set: direct children + adopted children.
        Returns (candidate_ids, is_direct, max_sem_weight)."""
        dev = self.device
        all_ids = []
        all_direct = []
        all_sem_w = []

        # Direct children
        d_ids, _ = self.children_of(word_id)
        if d_ids.numel() > 0:
            all_ids.append(d_ids)
            all_direct.append(torch.ones(d_ids.numel(), dtype=torch.bool,
                                         device=dev))
            all_sem_w.append(torch.ones(d_ids.numel(), dtype=torch.float32,
                                        device=dev))

        # Adopted children
        for sem_id, sem_w in semantic_neighbors:
            a_ids, _ = self.children_of(sem_id)
            if a_ids.numel() > 0:
                all_ids.append(a_ids)
                all_direct.append(torch.zeros(a_ids.numel(), dtype=torch.bool,
                                              device=dev))
                all_sem_w.append(torch.full((a_ids.numel(),), sem_w,
                                            dtype=torch.float32, device=dev))

        if not all_ids:
            return (torch.empty(0, dtype=torch.int64, device=dev),
                    torch.empty(0, dtype=torch.bool, device=dev),
                    torch.empty(0, dtype=torch.float32, device=dev))

        cat_ids = torch.cat(all_ids)
        cat_direct = torch.cat(all_direct)
        cat_sem_w = torch.cat(all_sem_w)

        # Deduplicate
        uniq_ids, inv = torch.unique(cat_ids, return_inverse=True)
        C = uniq_ids.numel()
        is_direct = torch.zeros(C, dtype=torch.bool, device=dev)
        is_direct.scatter_(0, inv[cat_direct],
                           torch.ones(int(cat_direct.sum().item()),
                                      dtype=torch.bool, device=dev))
        max_w = torch.zeros(C, dtype=torch.float32, device=dev)
        max_w.scatter_reduce_(0, inv, cat_sem_w,
                              reduce='amax', include_self=True)

        return uniq_ids, is_direct, max_w

    def stats(self) -> dict:
        E = self.targets.numel() if self.targets is not None else 0
        if E == 0:
            return {"pairs": 0, "avg_children": 0, "max_children": 0,
                    "words_with_children": 0}
        lengths = self.offsets[1:] - self.offsets[:-1]
        return {
            "pairs": E,
            "avg_children": float(lengths.float().mean().item()),
            "max_children": int(lengths.max().item()),
            "words_with_children": int((lengths > 0).sum().item()),
        }


def _topk_per_source(pair_src, pair_dst, counts_f, pair_keys,
                     V, K, device):
    """Keep top-K children per source word by count."""
    if pair_src.numel() == 0:
        return pair_src, pair_dst, counts_f, pair_keys

    changes = torch.cat([
        torch.tensor([True], device=device),
        pair_src[1:] != pair_src[:-1]
    ])
    group_starts = changes.nonzero(as_tuple=True)[0]
    group_ends = torch.cat([group_starts[1:],
                            torch.tensor([pair_src.numel()], device=device)])

    keep_mask = torch.ones(pair_src.numel(), dtype=torch.bool, device=device)
    for gs, ge in zip(group_starts, group_ends):
        gs_i = int(gs.item())
        ge_i = int(ge.item())
        if ge_i - gs_i > K:
            group_counts = counts_f[gs_i:ge_i]
            _, topk_idx = group_counts.topk(K)
            keep_mask[gs_i:ge_i] = False
            keep_mask[gs_i + topk_idx] = True

    return (pair_src[keep_mask], pair_dst[keep_mask],
            counts_f[keep_mask], pair_keys[keep_mask])
