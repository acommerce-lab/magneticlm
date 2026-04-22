"""KN-5gram — fully GPU-native: hash on GPU, sort, unique, scatter. No Python dicts."""

from typing import Dict, List
import numpy as np
import torch

_PRIMES = [1, 10007, 100003, 1000003, 10000019, 100000007,
           1000000007, 10000000019, 100000000003]


def _collect_order(encoded, order, primes_ng, primes_ctx, device):
    """Collect all n-gram/context hashes for one order on GPU. No Python dicts."""
    all_ng, all_ctx = [], []
    for arr in encoded:
        if arr.size < order + 1:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        L = t.numel() - order
        ng_h = sum(t[k:L + k] * primes_ng[k] for k in range(order + 1))
        ctx_h = sum(t[k:L + k] * primes_ctx[k] for k in range(order))
        all_ng.append(ng_h)
        all_ctx.append(ctx_h)
    if not all_ng:
        return None, None, None, None, None, None, None, 0, 0, 0

    all_ng_t = torch.cat(all_ng)
    all_ctx_t = torch.cat(all_ctx)

    # Sort by ngram hash → unique with counts
    sort_idx = torch.argsort(all_ng_t)
    ng_sorted = all_ng_t[sort_idx]
    ctx_sorted = all_ctx_t[sort_idx]
    uniq_ng, ng_counts = torch.unique_consecutive(ng_sorted, return_counts=True)

    n1 = int((ng_counts == 1).sum().item())
    n2 = int((ng_counts == 2).sum().item())
    n3 = int((ng_counts == 3).sum().item())

    # Context hash for each unique n-gram: first occurrence
    first_idx = torch.zeros(uniq_ng.numel(), dtype=torch.int64, device=device)
    first_idx[1:] = torch.cumsum(ng_counts[:-1], dim=0)
    uniq_ctx_per_ng = ctx_sorted[first_idx]

    # Per-context aggregates via scatter
    ctx_sort = torch.argsort(uniq_ctx_per_ng)
    ctx_s = uniq_ctx_per_ng[ctx_sort]
    counts_s = ng_counts[ctx_sort].float()

    uniq_ctx, inverse = torch.unique_consecutive(ctx_s, return_inverse=True)
    nc = uniq_ctx.numel()
    totals = torch.zeros(nc, dtype=torch.float32, device=device)
    totals.scatter_add_(0, inverse, counts_s)
    c1 = torch.zeros(nc, dtype=torch.float32, device=device)
    c1.scatter_add_(0, inverse, (counts_s == 1).float())
    c2 = torch.zeros(nc, dtype=torch.float32, device=device)
    c2.scatter_add_(0, inverse, (counts_s == 2).float())
    uf = torch.zeros(nc, dtype=torch.float32, device=device)
    uf.scatter_add_(0, inverse, torch.ones(counts_s.numel(), dtype=torch.float32, device=device))

    return uniq_ng, ng_counts, uniq_ctx, totals, c1, c2, uf, n1, n2, n3


def build(encoded: List[np.ndarray], V: int, max_order: int, device: torch.device) -> Dict:
    primes_ng = torch.tensor(_PRIMES[:max_order + 1], dtype=torch.int64, device=device)
    primes_ctx = torch.tensor(_PRIMES[:max_order], dtype=torch.int64, device=device)

    ng_keys = [None] * (max_order + 1)
    ng_counts = [None] * (max_order + 1)
    ctx_keys = [None] * (max_order + 1)
    ctx_totals = [None] * (max_order + 1)
    ctx_c1 = [None] * (max_order + 1)
    ctx_c2 = [None] * (max_order + 1)
    ctx_uf = [None] * (max_order + 1)

    n1_all = n2_all = n3_all = 0

    for order in range(1, max_order + 1):
        r = _collect_order(encoded, order, primes_ng, primes_ctx, device)
        u_ng, u_cnt, u_ctx, u_tot, u_c1, u_c2, u_uf, n1, n2, n3 = r
        if u_ng is None:
            continue
        ng_keys[order] = u_ng
        ng_counts[order] = u_cnt
        ctx_keys[order] = u_ctx
        ctx_totals[order] = u_tot
        ctx_c1[order] = u_c1
        ctx_c2[order] = u_c2
        ctx_uf[order] = u_uf
        n1_all += n1; n2_all += n2; n3_all += n3

    if n1_all > 0 and n2_all > 0:
        Y = n1_all / (n1_all + 2.0 * n2_all)
        D1 = max(0.1, min(0.95, 1.0 - 2.0 * Y * n2_all / max(n1_all, 1)))
        D2 = max(0.1, min(0.95, 2.0 - 3.0 * Y * (n3_all / max(n2_all, 1))))
        D3 = max(0.1, min(0.95, 3.0 - 4.0 * Y * ((n3_all + 1) / max(n3_all, 1))))
    else:
        D1, D2, D3 = 0.5, 0.75, 0.9

    cont = torch.ones(V, dtype=torch.float32, device=device)
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uniq = torch.unique(t[:-1] * V + t[1:])
        cont.scatter_add_(0, (uniq % V).long(),
                          torch.ones(uniq.numel(), dtype=torch.float32, device=device))

    # Sparse bigram transition matrix for fast adoption
    bg_rows_all, bg_cols_all = [], []
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        bg_rows_all.append(t[:-1])
        bg_cols_all.append(t[1:])
    bg_trans = None
    if bg_rows_all:
        r_all = torch.cat(bg_rows_all)
        c_all = torch.cat(bg_cols_all)
        pk = r_all * V + c_all
        u_pk, u_cnt = torch.unique(pk, return_counts=True)
        r_u = (u_pk // V).long()
        c_u = (u_pk % V).long()
        rs = torch.zeros(V, dtype=torch.float32, device=device)
        rs.scatter_add_(0, r_u, u_cnt.float())
        w = u_cnt.float() / rs[r_u].clamp(min=1.0)
        bg_trans = torch.sparse_coo_tensor(torch.stack([r_u, c_u]), w, (V, V)).coalesce()

    print(f"  KN-{max_order}: D1={D1:.3f} D2={D2:.3f} D3={D3:.3f}")
    return dict(
        max_order=max_order, V=V, device=device,
        ng_keys=ng_keys, ng_counts=ng_counts,
        ctx_keys=ctx_keys, ctx_totals=ctx_totals,
        ctx_c1=ctx_c1, ctx_c2=ctx_c2, ctx_uf=ctx_uf,
        D1=D1, D2=D2, D3=D3,
        cont=cont, total_ub=int(cont.sum().item()),
        primes=torch.tensor(_PRIMES[:max_order + 1], dtype=torch.int64, device=device),
        bg_trans=bg_trans,
    )


def _lookup(sk, sv, q):
    if sk is None or sk.numel() == 0:
        return torch.zeros_like(q, dtype=torch.float32)
    pos = torch.searchsorted(sk, q)
    safe = pos.clamp(max=sk.numel() - 1)
    hit = (pos < sk.numel()) & (sk[safe] == q)
    return torch.where(hit, sv[safe].float(), torch.zeros_like(q, dtype=torch.float32))


def score_all(kn: Dict, context_ids: List[int]) -> torch.Tensor:
    """Score all V next tokens. Returns [V] probability distribution."""
    V, device = kn["V"], kn["device"]
    mo = kn["max_order"]
    primes = kn["primes"]
    targets = torch.arange(V, dtype=torch.int64, device=device)

    ctx = context_ids[-(mo):] if len(context_ids) >= mo else context_ids
    padded = [0] * (mo - len(ctx)) + list(ctx)

    p = kn["cont"][targets] / max(kn["total_ub"], 1)
    p = p.clamp(min=1e-10)

    for order in range(1, mo + 1):
        if kn["ng_keys"][order] is None:
            continue
        cs = max(0, len(padded) - order)
        if len(padded) - cs < order:
            continue
        ctx_slice = padded[cs:cs + order]

        ng_h = torch.zeros(V, dtype=torch.int64, device=device)
        ctx_h_val = 0
        for k in range(order):
            ng_h = ng_h + ctx_slice[k] * primes[k]
            ctx_h_val += ctx_slice[k] * int(primes[k].item())
        ng_h = ng_h + targets * primes[order]

        c = _lookup(kn["ng_keys"][order], kn["ng_counts"][order], ng_h)
        ctx_q = torch.tensor([ctx_h_val], dtype=torch.int64, device=device)
        ct = _lookup(kn["ctx_keys"][order], kn["ctx_totals"][order], ctx_q).item()
        c1v = _lookup(kn["ctx_keys"][order], kn["ctx_c1"][order], ctx_q).item()
        c2v = _lookup(kn["ctx_keys"][order], kn["ctx_c2"][order], ctx_q).item()
        ufv = _lookup(kn["ctx_keys"][order], kn["ctx_uf"][order], ctx_q).item()

        if ct < 1:
            continue
        D = torch.where(c >= 3, kn["D3"], torch.where(c >= 2, kn["D2"],
            torch.where(c >= 1, kn["D1"], 0.0)))
        disc = (c - D).clamp(min=0.0) / ct
        n3p = max(ufv - c1v - c2v, 0)
        lam = (kn["D1"] * c1v + kn["D2"] * c2v + kn["D3"] * n3p) / ct
        p = disc + lam * p

    return p / p.sum().clamp(min=1e-9)
