"""KN-5gram — chunked GPU build to handle 1M+ lines without OOM."""

from typing import Dict, List
import numpy as np
import torch

_PRIMES = [1, 10007, 100003, 1000003, 10000019, 100000007,
           1000000007, 10000000019, 100000000003]


def _collect_order_chunked(encoded, order, primes_ng, primes_ctx, device, chunk_tokens=4_000_000):
    """Chunked collection: concat sentences, compute hashes, reduce per chunk, merge."""
    valid = [a for a in encoded if a.size >= order + 1]
    if not valid:
        return None, None, None, None, None, None, None, 0, 0, 0

    # Build chunks by token count
    chunks = []
    cur, cur_sz = [], 0
    for a in valid:
        cur.append(a)
        cur_sz += a.size
        if cur_sz >= chunk_tokens:
            chunks.append(cur); cur = []; cur_sz = 0
    if cur:
        chunks.append(cur)

    reduced_ng_keys = []
    reduced_ng_counts = []
    reduced_ctx_per_ng = []

    for chunk in chunks:
        flat = np.concatenate(chunk).astype(np.int64)
        sent_ids = np.concatenate([
            np.full(a.size, i, dtype=np.int32) for i, a in enumerate(chunk)
        ])
        t = torch.from_numpy(flat).to(device)
        sid = torch.from_numpy(sent_ids).to(device)
        n = t.numel()
        L = n - order

        # Check sentence boundaries: all order+1 tokens must be same sentence
        same = torch.ones(L, dtype=torch.bool, device=device)
        for d in range(1, order + 1):
            same = same & (sid[:L] == sid[d:d + L])

        # Compute hashes (vectorized — one op per order, not per sentence)
        ng_h = sum(t[k:L + k] * primes_ng[k] for k in range(order + 1))
        ctx_h = sum(t[k:L + k] * primes_ctx[k] for k in range(order))

        # Filter cross-sentence n-grams
        ng_h = ng_h[same]
        ctx_h = ctx_h[same]

        if ng_h.numel() == 0:
            continue

        # Sort and reduce this chunk
        sort_idx = torch.argsort(ng_h)
        ng_sorted = ng_h[sort_idx]
        ctx_sorted = ctx_h[sort_idx]
        uniq, counts = torch.unique_consecutive(ng_sorted, return_counts=True)
        # First context per unique n-gram
        first_idx = torch.zeros(uniq.numel(), dtype=torch.int64, device=device)
        first_idx[1:] = torch.cumsum(counts[:-1], dim=0)
        ctx_per = ctx_sorted[first_idx]

        reduced_ng_keys.append(uniq.cpu())
        reduced_ng_counts.append(counts.cpu())
        reduced_ctx_per_ng.append(ctx_per.cpu())
        del t, sid, ng_h, ctx_h, sort_idx, ng_sorted, ctx_sorted
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    if not reduced_ng_keys:
        return None, None, None, None, None, None, None, 0, 0, 0

    # Merge across chunks
    # Merge on CPU to avoid GPU OOM on large tensors (60M+ at order 4-5)
    all_keys = torch.cat(reduced_ng_keys)  # already on CPU
    all_counts = torch.cat(reduced_ng_counts)
    all_ctx = torch.cat(reduced_ctx_per_ng)

    sort_idx = torch.argsort(all_keys)
    all_keys = all_keys[sort_idx]
    all_counts = all_counts[sort_idx]
    all_ctx = all_ctx[sort_idx]

    uniq_ng, inverse = torch.unique_consecutive(all_keys, return_inverse=True)
    final_counts = torch.zeros(uniq_ng.numel(), dtype=torch.int64)
    final_counts.scatter_add_(0, inverse, all_counts)

    boundaries = torch.ones(all_keys.numel(), dtype=torch.bool)
    boundaries[1:] = all_keys[1:] != all_keys[:-1]
    first_positions = boundaries.nonzero(as_tuple=True)[0]
    uniq_ctx_per_ng = all_ctx[first_positions]

    n1 = int((final_counts == 1).sum().item())
    n2 = int((final_counts == 2).sum().item())
    n3 = int((final_counts == 3).sum().item())

    # Per-context aggregates (CPU)
    ctx_sort = torch.argsort(uniq_ctx_per_ng)
    ctx_s = uniq_ctx_per_ng[ctx_sort]
    counts_s = final_counts[ctx_sort].float()
    uniq_ctx, inv2 = torch.unique_consecutive(ctx_s, return_inverse=True)
    nc = uniq_ctx.numel()
    totals = torch.zeros(nc, dtype=torch.float32)
    totals.scatter_add_(0, inv2, counts_s)
    c1 = torch.zeros(nc, dtype=torch.float32)
    c1.scatter_add_(0, inv2, (counts_s == 1).float())
    c2 = torch.zeros(nc, dtype=torch.float32)
    c2.scatter_add_(0, inv2, (counts_s == 2).float())
    uf = torch.zeros(nc, dtype=torch.float32)
    uf.scatter_add_(0, inv2, torch.ones(counts_s.numel(), dtype=torch.float32))

    del all_keys, all_counts, all_ctx, sort_idx

    # Move final results to GPU for scoring
    return (uniq_ng.to(device), final_counts.to(device),
            uniq_ctx.to(device), totals.to(device),
            c1.to(device), c2.to(device), uf.to(device),
            n1, n2, n3)


def build(encoded: List[np.ndarray], V: int, max_order: int, device: torch.device) -> Dict:
    import time
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
        t0 = time.time()
        r = _collect_order_chunked(encoded, order, primes_ng, primes_ctx, device)
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
        print(f"    order {order}: {u_ng.numel():,} ngrams  ({time.time()-t0:.1f}s)")

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

    bg_trans = None
    bg_r, bg_c = [], []
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        bg_r.append(t[:-1]); bg_c.append(t[1:])
    if bg_r:
        all_r = torch.cat(bg_r); all_c = torch.cat(bg_c)
        pk = all_r * V + all_c
        u, uc = torch.unique(pk, return_counts=True)
        r_u = (u // V).long(); c_u = (u % V).long()
        rs = torch.zeros(V, dtype=torch.float32, device=device)
        rs.scatter_add_(0, r_u, uc.float())
        w = uc.float() / rs[r_u].clamp(min=1.0)
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
