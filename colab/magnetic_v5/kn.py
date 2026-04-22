"""Modified KN-5gram — pure functions on GPU tensors, no classes.

All state stored in a plain dict for maximum flexibility.
All heavy ops use searchsorted / scatter_add on GPU.
"""

from typing import Dict, List
import numpy as np
import torch

_PRIMES = [1, 10007, 100003, 1000003, 10000019, 100000007,
           1000000007, 10000000019, 100000000003]


def build(encoded: List[np.ndarray], V: int, max_order: int, device: torch.device) -> Dict:
    """Build KN tables. Returns dict with all needed tensors."""
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
        ng_dict: Dict[int, int] = {}
        ng_ctx: Dict[int, int] = {}
        for arr in encoded:
            if arr.size < order + 1:
                continue
            t = torch.from_numpy(arr.astype(np.int64)).to(device)
            n = t.numel()
            L = n - order
            ng_h = sum(t[k:L + k] * primes_ng[k] for k in range(order + 1))
            ctx_h = sum(t[k:L + k] * primes_ctx[k] for k in range(order))
            ng_cpu = ng_h.cpu().numpy()
            ctx_cpu = ctx_h.cpu().numpy()
            for i in range(L):
                nh, ch = int(ng_cpu[i]), int(ctx_cpu[i])
                ng_dict[nh] = ng_dict.get(nh, 0) + 1
                ng_ctx[nh] = ch
        if not ng_dict:
            continue

        keys_np = np.fromiter(ng_dict.keys(), np.int64, len(ng_dict))
        counts_np = np.fromiter(ng_dict.values(), np.int64, len(ng_dict))
        si = np.argsort(keys_np)
        ng_keys[order] = torch.from_numpy(keys_np[si]).to(device)
        ng_counts[order] = torch.from_numpy(counts_np[si]).to(device)
        n1_all += int((counts_np == 1).sum())
        n2_all += int((counts_np == 2).sum())
        n3_all += int((counts_np == 3).sum())

        # Per-context aggregates
        grp: Dict[int, List[int]] = {}
        for nh, c in ng_dict.items():
            ch = ng_ctx.get(nh)
            if ch is not None:
                grp.setdefault(ch, []).append(c)
        nc = len(grp)
        ck = np.empty(nc, np.int64)
        ct = np.empty(nc, np.int64)
        c1a = np.empty(nc, np.int64)
        c2a = np.empty(nc, np.int64)
        ufa = np.empty(nc, np.int64)
        for i, (ch, cl) in enumerate(grp.items()):
            ck[i] = ch
            ct[i] = sum(cl)
            c1a[i] = sum(1 for c in cl if c == 1)
            c2a[i] = sum(1 for c in cl if c == 2)
            ufa[i] = len(cl)
        cs = np.argsort(ck)
        ctx_keys[order] = torch.from_numpy(ck[cs]).to(device)
        ctx_totals[order] = torch.from_numpy(ct[cs]).float().to(device)
        ctx_c1[order] = torch.from_numpy(c1a[cs]).float().to(device)
        ctx_c2[order] = torch.from_numpy(c2a[cs]).float().to(device)
        ctx_uf[order] = torch.from_numpy(ufa[cs]).float().to(device)

    # Discounts
    if n1_all > 0 and n2_all > 0:
        Y = n1_all / (n1_all + 2.0 * n2_all)
        D1 = max(0.1, min(0.95, 1.0 - 2.0 * Y * n2_all / max(n1_all, 1)))
        D2 = max(0.1, min(0.95, 2.0 - 3.0 * Y * (n3_all / max(n2_all, 1))))
        D3 = max(0.1, min(0.95, 3.0 - 4.0 * Y * ((n3_all + 1) / max(n3_all, 1))))
    else:
        D1, D2, D3 = 0.5, 0.75, 0.9

    # Continuation counts
    cont = torch.ones(V, dtype=torch.float32, device=device)
    for arr in encoded:
        if arr.size < 2:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uniq = torch.unique(t[:-1] * V + t[1:])
        cont.scatter_add_(0, (uniq % V).long(), torch.ones(uniq.numel(), dtype=torch.float32, device=device))

    # Build sparse bigram transition matrix T[V,V] for fast adoption
    bg_trans = None
    if ng_keys[1] is not None:
        idx = torch.stack([
            (ng_keys[1] // _PRIMES[1]).to(torch.int64) % V,  # approximate source
            torch.zeros_like(ng_keys[1])  # placeholder
        ])
        # Actually rebuild from raw bigrams properly
        bg_rows_list = []
        bg_cols_list = []
        bg_counts_list = []
        for arr in encoded:
            if arr.size < 2:
                continue
            t = torch.from_numpy(arr.astype(np.int64)).to(device)
            bg_rows_list.append(t[:-1])
            bg_cols_list.append(t[1:])
        if bg_rows_list:
            all_r = torch.cat(bg_rows_list)
            all_c = torch.cat(bg_cols_list)
            pair_keys = all_r * V + all_c
            uniq, counts_u = torch.unique(pair_keys, return_counts=True)
            r_u = (uniq // V).long()
            c_u = (uniq % V).long()
            w_u = counts_u.float()
            # Row-normalize
            row_sums = torch.zeros(V, dtype=torch.float32, device=device)
            row_sums.scatter_add_(0, r_u, w_u)
            w_norm = w_u / row_sums[r_u].clamp(min=1.0)
            bg_trans = torch.sparse_coo_tensor(
                torch.stack([r_u, c_u]), w_norm, (V, V)
            ).coalesce()

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

    ctx = context_ids[-(mo):] if len(context_ids) >= mo else context_ids
    padded = [0] * (mo - len(ctx)) + list(ctx)

    targets = torch.arange(V, dtype=torch.int64, device=device)
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
