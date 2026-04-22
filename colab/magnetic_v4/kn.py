"""Modified Kneser-Ney 5-gram on GPU — correct backoff weights.

Fixes from v1: proper per-context aggregates (c1, c2, uf) and
λ(ctx) = (D1·c1 + D2·c2 + D3·(uf-c1-c2)) / total.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PRIMES = [1, 10007, 100003, 1000003, 10000019, 100000007,
           1000000007, 10000000019, 100000000003]


@dataclass
class KNModel:
    max_order: int
    vocab_size: int
    device: torch.device
    ng_keys: List[Optional[torch.Tensor]]
    ng_counts: List[Optional[torch.Tensor]]
    ctx_keys: List[Optional[torch.Tensor]]
    ctx_totals: List[Optional[torch.Tensor]]
    ctx_c1: List[Optional[torch.Tensor]]
    ctx_c2: List[Optional[torch.Tensor]]
    ctx_uf: List[Optional[torch.Tensor]]
    D1: float
    D2: float
    D3: float
    cont_count: torch.Tensor
    total_unique_bigrams: int


def _collect_order(
    encoded: List[np.ndarray],
    order: int,
    primes_ng: torch.Tensor,
    primes_ctx: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Collect n-gram and context counts for one order. Returns (ng_dict, ng_to_ctx_dict)."""
    ng: Dict[int, int] = {}
    ng_ctx: Dict[int, int] = {}  # ngram_hash → context_hash (for grouping)

    for arr in encoded:
        if arr.size < order + 1:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        n = t.numel()
        L = n - order

        ng_h = torch.zeros(L, dtype=torch.int64, device=device)
        for k in range(order + 1):
            ng_h = ng_h + t[k:L + k] * primes_ng[k]

        ctx_h = torch.zeros(L, dtype=torch.int64, device=device)
        for k in range(order):
            ctx_h = ctx_h + t[k:L + k] * primes_ctx[k]

        ng_cpu = ng_h.cpu().numpy()
        ctx_cpu = ctx_h.cpu().numpy()
        for i in range(L):
            nh = int(ng_cpu[i])
            ch = int(ctx_cpu[i])
            ng[nh] = ng.get(nh, 0) + 1
            ng_ctx[nh] = ch

    return ng, ng_ctx


def build_kn(
    encoded: List[np.ndarray],
    V: int,
    cfg,
    device: torch.device,
) -> KNModel:
    max_order = int(cfg.kn_max_order)
    primes_ng = torch.tensor(_PRIMES[:max_order + 1], dtype=torch.int64, device=device)
    primes_ctx = torch.tensor(_PRIMES[:max_order], dtype=torch.int64, device=device)

    ng_keys_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ng_counts_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ctx_keys_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ctx_totals_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ctx_c1_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ctx_c2_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)
    ctx_uf_l: List[Optional[torch.Tensor]] = [None] * (max_order + 1)

    n1_all = n2_all = n3_all = 0

    for order in range(1, max_order + 1):
        ng_dict, ng_ctx = _collect_order(encoded, order, primes_ng, primes_ctx, device)
        if not ng_dict:
            continue

        # Build sorted ngram table
        keys = np.fromiter(ng_dict.keys(), dtype=np.int64, count=len(ng_dict))
        counts = np.fromiter(ng_dict.values(), dtype=np.int64, count=len(ng_dict))
        sidx = np.argsort(keys)
        ng_keys_l[order] = torch.from_numpy(keys[sidx]).to(device)
        ng_counts_l[order] = torch.from_numpy(counts[sidx]).to(device)

        n1_all += int((counts == 1).sum())
        n2_all += int((counts == 2).sum())
        n3_all += int((counts >= 3).sum())

        # Group ngrams by context → compute per-context aggregates
        ctx_groups: Dict[int, List[int]] = {}
        for nh, c in ng_dict.items():
            ch = ng_ctx.get(nh)
            if ch is None:
                continue
            ctx_groups.setdefault(ch, []).append(c)

        n_ctx = len(ctx_groups)
        c_keys = np.empty(n_ctx, dtype=np.int64)
        c_totals = np.empty(n_ctx, dtype=np.int64)
        c_c1 = np.empty(n_ctx, dtype=np.int64)
        c_c2 = np.empty(n_ctx, dtype=np.int64)
        c_uf = np.empty(n_ctx, dtype=np.int64)

        for i, (ch, count_list) in enumerate(ctx_groups.items()):
            c_keys[i] = ch
            c_totals[i] = sum(count_list)
            c_c1[i] = sum(1 for c in count_list if c == 1)
            c_c2[i] = sum(1 for c in count_list if c == 2)
            c_uf[i] = len(count_list)

        csort = np.argsort(c_keys)
        ctx_keys_l[order] = torch.from_numpy(c_keys[csort]).to(device)
        ctx_totals_l[order] = torch.from_numpy(c_totals[csort]).to(torch.float32).to(device)
        ctx_c1_l[order] = torch.from_numpy(c_c1[csort]).to(torch.float32).to(device)
        ctx_c2_l[order] = torch.from_numpy(c_c2[csort]).to(torch.float32).to(device)
        ctx_uf_l[order] = torch.from_numpy(c_uf[csort]).to(torch.float32).to(device)

    # Discounts
    if n1_all > 0 and n2_all > 0:
        Y = n1_all / (n1_all + 2.0 * n2_all)
        D1 = max(0.1, min(0.95, 1.0 - 2.0 * Y * n2_all / max(n1_all, 1)))
        D2 = max(0.1, min(0.95, 2.0 - 3.0 * Y * (n3_all / max(n2_all, 1))))
        D3 = max(0.1, min(0.95, 3.0 - 4.0 * Y * ((n3_all + 1) / max(n3_all, 1))))
    else:
        D1, D2, D3 = 0.5, 0.75, 0.9

    # Continuation counts
    cont_count = torch.ones(V, dtype=torch.float32, device=device)
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        pairs = t[:-1] * V + t[1:]
        uniq = torch.unique(pairs)
        b_ids = (uniq % V).to(torch.int64)
        cont_count.scatter_add_(0, b_ids, torch.ones_like(b_ids, dtype=torch.float32))
    total_ub = int(cont_count.sum().item())

    print(f"  KN-{max_order}gram: D1={D1:.3f} D2={D2:.3f} D3={D3:.3f}  "
          f"n1={n1_all:,} n2={n2_all:,} n3+={n3_all:,}")

    return KNModel(
        max_order=max_order, vocab_size=V, device=device,
        ng_keys=ng_keys_l, ng_counts=ng_counts_l,
        ctx_keys=ctx_keys_l, ctx_totals=ctx_totals_l,
        ctx_c1=ctx_c1_l, ctx_c2=ctx_c2_l, ctx_uf=ctx_uf_l,
        D1=D1, D2=D2, D3=D3,
        cont_count=cont_count, total_unique_bigrams=total_ub,
    )


def _lookup(sk: torch.Tensor, sv: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if sk.numel() == 0:
        return torch.zeros_like(q, dtype=torch.float32)
    pos = torch.searchsorted(sk, q)
    safe = pos.clamp(max=sk.numel() - 1)
    hit = (pos < sk.numel()) & (sk[safe] == q)
    return torch.where(hit, sv[safe].to(torch.float32), torch.zeros_like(q, dtype=torch.float32))


def score_batch(
    kn: KNModel,
    context: torch.Tensor,  # [B, max_order]
    target: torch.Tensor,   # [B]
) -> torch.Tensor:
    """Score P(target | context) with proper MKN backoff. Returns [B] log-probs."""
    B = target.shape[0]
    device = kn.device
    primes = torch.tensor(_PRIMES[:kn.max_order + 1], dtype=torch.int64, device=device)

    # Base: continuation probability
    p = kn.cont_count[target] / max(kn.total_unique_bigrams, 1)
    p = p.clamp(min=1e-10)

    for order in range(1, kn.max_order + 1):
        if kn.ng_keys[order] is None or kn.ctx_keys[order] is None:
            continue
        ctx_end = context.shape[1]
        ctx_start = max(0, ctx_end - order)
        if ctx_end - ctx_start < order:
            continue

        ctx_slice = context[:, ctx_start:ctx_end]

        ng_h = torch.zeros(B, dtype=torch.int64, device=device)
        for k in range(order):
            ng_h = ng_h + ctx_slice[:, k] * primes[k]
        ng_h = ng_h + target * primes[order]

        ctx_h = torch.zeros(B, dtype=torch.int64, device=device)
        for k in range(order):
            ctx_h = ctx_h + ctx_slice[:, k] * primes[k]

        c = _lookup(kn.ng_keys[order], kn.ng_counts[order], ng_h)
        ctx_total = _lookup(kn.ctx_keys[order], kn.ctx_totals[order], ctx_h)
        c1 = _lookup(kn.ctx_keys[order], kn.ctx_c1[order], ctx_h)
        c2 = _lookup(kn.ctx_keys[order], kn.ctx_c2[order], ctx_h)
        uf = _lookup(kn.ctx_keys[order], kn.ctx_uf[order], ctx_h)

        # Discount selection per n-gram count
        D = torch.where(c >= 3, torch.full_like(c, kn.D3),
            torch.where(c >= 2, torch.full_like(c, kn.D2),
            torch.where(c >= 1, torch.full_like(c, kn.D1),
            torch.zeros_like(c))))

        safe_total = ctx_total.clamp(min=1.0)
        disc = (c - D).clamp(min=0.0) / safe_total

        # Proper backoff weight: λ = (D1·c1 + D2·c2 + D3·(uf-c1-c2)) / total
        n3p = (uf - c1 - c2).clamp(min=0.0)
        lam = (kn.D1 * c1 + kn.D2 * c2 + kn.D3 * n3p) / safe_total

        has_ctx = ctx_total > 0
        p = torch.where(has_ctx, disc + lam * p, p)

    return torch.log(p.clamp(min=1e-10))


def score_next_token(kn: KNModel, context_ids: List[int], V: int) -> torch.Tensor:
    """Score all V next tokens for a single context. Returns [V] probs."""
    device = kn.device
    max_ctx = min(len(context_ids), kn.max_order)
    ctx = context_ids[-max_ctx:] if max_ctx > 0 else []
    padded = [0] * (kn.max_order - len(ctx)) + ctx
    ctx_t = torch.tensor([padded], dtype=torch.int64, device=device).expand(V, -1)
    targets = torch.arange(V, dtype=torch.int64, device=device)
    log_p = score_batch(kn, ctx_t, targets)
    p = torch.exp(log_p)
    return p / p.sum().clamp(min=1e-9)
