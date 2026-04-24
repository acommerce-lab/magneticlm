"""Deterministic attention model — transformer-like, no backprop.

Three components:
  1. Embedding: SVD on PPMI matrix → [V, d] word vectors
  2. Multi-head attention: cosine(Q, K) → softmax → weighted V
  3. KN scoring: n-gram probabilities as the "feed-forward" layer

All weights from corpus statistics. GPU-native dense matmul.
"""

import time, math
from typing import Dict, List
import numpy as np
import torch


# ======================================================================
# 1. Embedding: PPMI → SVD → dense word vectors
# ======================================================================

def build_embeddings(ctx_rows, ctx_cols, ctx_counts, unigram, V, d, min_ppmi, device):
    """Build [V, d] word vectors via truncated SVD on PPMI matrix."""
    t0 = time.time()

    # Build PPMI
    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    # Build sparse PPMI on CPU for SVD
    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    M = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()

    # Dense for SVD (V×V can be large but fits in RAM for V≤50K)
    M_dense = M.to_dense()

    # Truncated SVD via torch.linalg.svd (or randomized for speed)
    print(f"    SVD on [{V}×{V}] PPMI matrix, d={d}...")
    try:
        # Randomized SVD: much faster for large V
        U, S, Vt = torch.svd_lowrank(M_dense, q=d, niter=5)
    except Exception:
        # Fallback: full SVD then truncate
        U, S, Vt = torch.linalg.svd(M_dense, full_matrices=False)
        U, S, Vt = U[:, :d], S[:d], Vt[:d, :]

    # Word vectors = U × sqrt(S)
    embeddings = (U * S.sqrt().unsqueeze(0)).to(device)

    # IDF weights for attention
    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))
    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    embeddings [{V}×{d}] built in {time.time()-t0:.1f}s")
    return embeddings, idf


def build_embeddings_basis(ctx_rows, ctx_cols, ctx_counts, unigram, V, k, min_ppmi, device):
    """Build [V, k] word vectors using PPMI columns from an independent
    dominating set as basis dimensions.

    Steps:
      1. Build PPMI matrix (same as SVD path).
      2. Greedy independent dominating set: pick k words that don't share
         edges but cover the vocabulary.
      3. Embedding[w] = PPMI[w, basis_words].

    Preserves raw PPMI values (no lossy compression) and keeps dimensions
    interpretable (each dim = a real word).
    """
    t0 = time.time()

    total = ctx_counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = ctx_counts / total
    p_a = unigram[ctx_rows].float() / N
    p_b = unigram[ctx_cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    rows_f, cols_f, vals_f = ctx_rows[keep], ctx_cols[keep], ppmi[keep]

    # Sparse PPMI on CPU for basis selection
    idx = torch.stack([rows_f.cpu(), cols_f.cpu()])
    M = torch.sparse_coo_tensor(idx, vals_f.cpu().float(), (V, V)).coalesce()

    # Degree per word (number of edges)
    degree = torch.zeros(V, dtype=torch.float32)
    degree.scatter_add_(0, rows_f.cpu(), torch.ones(rows_f.numel(), dtype=torch.float32))

    # Build neighbor lists (CPU) for greedy IDS
    print(f"    selecting independent dominating set (target k={k})...")
    M_csr = M.to_sparse_csr()
    crow = M_csr.crow_indices()
    cols = M_csr.col_indices()

    # Greedy: sort by degree descending, pick high-degree words that don't
    # conflict with already-picked (no shared edge). Stop when we have k.
    order = torch.argsort(degree, descending=True).tolist()
    picked = []
    blocked = torch.zeros(V, dtype=torch.bool)
    covered = torch.zeros(V, dtype=torch.bool)

    for w in order:
        if len(picked) >= k:
            break
        if blocked[w]:
            continue
        picked.append(w)
        covered[w] = True
        # Block all neighbors of w (they share an edge with w)
        start, end = int(crow[w].item()), int(crow[w + 1].item())
        neigh = cols[start:end]
        blocked[neigh] = True
        blocked[w] = True
        covered[neigh] = True

    # If we didn't reach k, fill with highest-degree remaining (drop independence)
    if len(picked) < k:
        picked_set = set(picked)
        for w in order:
            if len(picked) >= k:
                break
            if w not in picked_set:
                picked.append(w)
                picked_set.add(w)

    basis = torch.tensor(picked[:k], dtype=torch.int64)
    cov = int(covered.sum().item())
    print(f"    basis: {len(picked)} words, covered={cov}/{V} ({100*cov/V:.1f}%)")

    # Embedding = PPMI[:, basis]  →  [V, k]
    # Build by filtering the sparse triplets: keep entries whose col is in basis
    basis_set = torch.zeros(V, dtype=torch.int64) - 1
    basis_set[basis] = torch.arange(len(basis), dtype=torch.int64)
    col_idx_in_basis = basis_set[cols_f.cpu()]
    mask = col_idx_in_basis >= 0
    sel_rows = rows_f.cpu()[mask].to(torch.int64)
    sel_cols = col_idx_in_basis[mask]
    sel_vals = vals_f.cpu()[mask].float()

    embeddings = torch.zeros(V, len(basis), dtype=torch.float32)
    embeddings[sel_rows, sel_cols] = sel_vals
    embeddings = embeddings.to(device)

    idf = (1.0 / (1.0 + degree.sqrt())).to(device)
    idf = idf / idf.max().clamp(min=1e-9)

    print(f"    embeddings [{V}×{len(basis)}] (basis) built in {time.time()-t0:.1f}s")
    return embeddings, idf


# ======================================================================
# 2. Multi-Head Deterministic Attention
# ======================================================================

def attention_score(
    embeddings: torch.Tensor,  # [V, d]
    context_ids: List[int],
    idf: torch.Tensor,         # [V]
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute attention-weighted score for all V candidates.

    Q = IDF-weighted sum of context embeddings
    K = all word embeddings
    score = softmax(Q @ K.T / sqrt(d))

    Returns [V] probability distribution.
    """
    d = embeddings.shape[1]
    ctx = context_ids[-8:] if context_ids else []
    if not ctx:
        return torch.ones(V, dtype=torch.float32, device=device) / V

    # Build query: IDF-weighted context embedding
    Q = torch.zeros(d, dtype=torch.float32, device=device)
    w = 1.0
    for t in reversed(ctx):
        if 0 <= t < V:
            Q += w * idf[t] * embeddings[t]
        w *= 0.7
    Q_norm = Q / Q.norm().clamp(min=1e-9)

    # Cosine similarity with all words (K = embeddings)
    # [V] = [V, d] @ [d]
    K_norms = embeddings.norm(dim=1).clamp(min=1e-9)
    scores = (embeddings @ Q_norm) / K_norms

    # Softmax with temperature
    scores = scores / math.sqrt(d)
    # Sharpen: square before softmax for concentration
    scores = scores.clamp(min=0.0) ** 2

    # Zero context words (no echo)
    for t in ctx:
        if 0 <= t < V:
            scores[t] = 0.0

    s = scores.sum()
    return scores / s if s > 1e-9 else torch.ones(V, dtype=torch.float32, device=device) / V


def attention_batch(
    embeddings: torch.Tensor,  # [V, d]
    contexts: List[List[int]],
    idf: torch.Tensor,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Batched attention for all eval tokens. Returns [n, V]."""
    n = len(contexts)
    d = embeddings.shape[1]

    # Build Q matrix: [n, d]
    Q = torch.zeros(n, d, dtype=torch.float32, device=device)
    for i, ctx in enumerate(contexts):
        w = 1.0
        for t in reversed(ctx[-8:]):
            if 0 <= t < V:
                Q[i] += w * idf[t] * embeddings[t]
            w *= 0.7

    # Normalize Q
    Q_norms = Q.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Q_normed = Q / Q_norms

    # K = embeddings normalized
    K_norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
    K_normed = embeddings / K_norms

    # Batch cosine: [n, V] = [n, d] @ [d, V]
    scores = torch.mm(Q_normed, K_normed.t()) / math.sqrt(d)

    # Sharpen + clamp
    scores = scores.clamp(min=0.0) ** 2

    # Zero context words
    for i, ctx in enumerate(contexts):
        for t in ctx[-8:]:
            if 0 <= t < V:
                scores[i, t] = 0.0

    # Normalize rows
    s = scores.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return scores / s


# ======================================================================
# 3. KN scoring (simplified: bigram + trigram for speed)
# ======================================================================

def build_kn_simple(encoded, V, max_order, device):
    """Simplified KN: build bigram transition matrix + optional trigram."""
    import time
    t0 = time.time()

    # Bigram: collect on CPU, build sparse on device
    bg_r, bg_c = [], []
    for arr in encoded:
        if arr.size < 2:
            continue
        bg_r.append(arr[:-1])
        bg_c.append(arr[1:])

    all_r = np.concatenate(bg_r).astype(np.int64)
    all_c = np.concatenate(bg_c).astype(np.int64)

    # Count pairs on CPU
    pair_keys = all_r * V + all_c
    uniq, counts = np.unique(pair_keys, return_counts=True)
    r_u = (uniq // V).astype(np.int64)
    c_u = (uniq % V).astype(np.int64)

    # Row-normalize → transition probabilities
    r_t = torch.from_numpy(r_u).to(device)
    c_t = torch.from_numpy(c_u).to(device)
    cnt_t = torch.from_numpy(counts.astype(np.float32)).to(device)
    row_sums = torch.zeros(V, dtype=torch.float32, device=device)
    row_sums.scatter_add_(0, r_t, cnt_t)
    w = cnt_t / row_sums[r_t].clamp(min=1.0)

    bg_trans = torch.sparse_coo_tensor(
        torch.stack([r_t, c_t]), w, (V, V)
    ).coalesce()

    # Unigram
    uni = torch.zeros(V, dtype=torch.float32, device=device)
    for arr in encoded:
        if arr.size == 0:
            continue
        t = torch.from_numpy(arr.astype(np.int64)).to(device)
        uni.scatter_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    uni_prob = uni / uni.sum().clamp(min=1.0)

    print(f"    KN simple: {int(cnt_t.numel()):,} bigram pairs ({time.time()-t0:.1f}s)")
    return dict(bg_trans=bg_trans, uni_prob=uni_prob, V=V, device=device)


def kn_score(kn, context_ids):
    """Score using bigram transition + unigram backoff."""
    V, device = kn["V"], kn["device"]
    bg = kn["bg_trans"]
    uni = kn["uni_prob"]

    if not context_ids:
        return uni.clone()

    cur = context_ids[-1]
    sel = torch.zeros(V, dtype=torch.float32, device=device)
    sel[cur] = 1.0
    bigram = torch.sparse.mm(bg.t(), sel.unsqueeze(1)).squeeze(1)

    # Interpolate with unigram (KN-style backoff, simplified)
    lam = 0.3  # backoff weight
    eps = 0.02  # floor
    result = (1 - lam - eps) * bigram + lam * uni + eps * (1.0 / V)
    return result / result.sum().clamp(min=1e-9)
