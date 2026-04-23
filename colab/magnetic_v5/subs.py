"""Distributional substitution via PPMI vectors + cosine similarity.

Steps 1-5 implementation:
  1. Use context window (not bigram) for distributional similarity
  2. Build PPMI vectors (each word = row in PPMI matrix)
  3. Cosine similarity for substitutes (not binary overlap)
  4. λ^distance penalty for adoption paths
  5. Glow = graph centrality × cluster density

All matrix operations — GPU native.
"""

from typing import Dict, Tuple
import torch
import math


def _build_ppmi_matrix(
    rows: torch.Tensor, cols: torch.Tensor, counts: torch.Tensor,
    unigram: torch.Tensor, V: int, min_ppmi: float, device: torch.device,
) -> torch.Tensor:
    """Build sparse PPMI matrix [V, V]. Each row = word's PPMI vector."""
    total = counts.sum().clamp(min=1.0)
    N = unigram.float().sum().clamp(min=1.0)
    p_ab = counts / total
    p_a = unigram[rows].float() / N
    p_b = unigram[cols].float() / N
    ppmi = torch.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)).clamp(min=0.0)
    keep = ppmi >= min_ppmi
    idx = torch.stack([rows[keep], cols[keep]])
    vals = ppmi[keep]
    return torch.sparse_coo_tensor(idx, vals, (V, V)).coalesce()


def _row_norms(M: torch.Tensor, V: int, device: torch.device) -> torch.Tensor:
    """Compute L2 norm of each row of a sparse matrix."""
    M = M.coalesce()
    norms = torch.zeros(V, dtype=torch.float32, device=device)
    norms.scatter_add_(0, M.indices()[0], M.values() ** 2)
    return norms.sqrt().clamp(min=1e-9)


def _cosine_topk_blocked(
    M: torch.Tensor, V: int, K: int, block_size: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute cosine similarity in blocks. Returns (ids[V,K], weights[V,K], n_found)."""
    M = M.coalesce()
    norms = _row_norms(M, V, device)

    ids = torch.zeros(V, K, dtype=torch.int64, device=device)
    wts = torch.zeros(V, K, dtype=torch.float32, device=device)
    n_found = 0

    m_idx = M.indices()
    m_val = M.values()

    for start in range(0, V, block_size):
        end = min(start + block_size, V)
        bs = end - start

        # Extract block rows as dense [bs, V]
        block = torch.zeros(bs, V, dtype=torch.float32, device=device)
        mask = (m_idx[0] >= start) & (m_idx[0] < end)
        if mask.any():
            block[m_idx[0][mask] - start, m_idx[1][mask]] = m_val[mask]

        if block.abs().sum() == 0:
            continue

        # Normalize block rows
        block_norms = norms[start:end].unsqueeze(1)
        block_normed = block / block_norms

        # Cosine: block_normed @ M_normed.T
        # = block_normed @ (M / norms).T
        # Use sparse: M.T @ block_normed.T → [V, bs] → transpose
        raw = torch.sparse.mm(M.t(), block_normed.t()).t()  # [bs, V]
        # Divide by target norms
        raw = raw / norms.unsqueeze(0)

        # Zero self-similarity
        for i in range(bs):
            w = start + i
            if w < V:
                raw[i, w] = 0.0

        vals, idx_k = torch.topk(raw, min(K, V), dim=1)
        for i in range(bs):
            w = start + i
            if w >= V:
                break
            nv = int((vals[i] > 0).sum().item())
            nt = min(nv, K)
            if nt > 0:
                ids[w, :nt] = idx_k[i, :nt]
                wts[w, :nt] = vals[i, :nt]
                n_found += 1

    # Normalize weights per word (for mixing)
    ws = wts.sum(dim=1, keepdim=True).clamp(min=1e-9)
    wts = wts / ws
    return ids, wts, n_found


def _compute_glow(
    M: torch.Tensor, V: int, device: torch.device,
) -> torch.Tensor:
    """Glow = graph centrality × cluster coefficient (approximation).

    Centrality: degree centrality (normalized nnz per row).
    Cluster density: average PPMI of a node's edges (how tight its neighborhood is).
    Glow = centrality × density → high for words in tight semantic clusters,
    low for hub words (high centrality but low density) like "the".
    """
    M = M.coalesce()
    m_idx = M.indices()
    m_val = M.values()

    # Degree centrality: nnz per row / max_nnz
    degree = torch.zeros(V, dtype=torch.float32, device=device)
    degree.scatter_add_(0, m_idx[0], torch.ones_like(m_val))
    max_deg = degree.max().clamp(min=1.0)
    centrality = degree / max_deg

    # Cluster density: average edge weight per node
    weight_sum = torch.zeros(V, dtype=torch.float32, device=device)
    weight_sum.scatter_add_(0, m_idx[0], m_val)
    density = weight_sum / degree.clamp(min=1.0)
    max_density = density.max().clamp(min=1e-9)
    density = density / max_density

    glow = centrality * density
    glow = glow / glow.max().clamp(min=1e-9)
    return glow


def build(
    bg_rows: torch.Tensor, bg_cols: torch.Tensor,
    ctx_rows: torch.Tensor, ctx_cols: torch.Tensor,
    ctx_counts: torch.Tensor, unigram: torch.Tensor,
    V: int, K: int, min_ppmi: float, block_size: int,
    device: torch.device,
) -> Dict:
    """Build substitution tables from CONTEXT window (not bigram).

    Steps:
      1. PPMI matrix from context pairs (window=5, captures multi-depth relations)
      2. Cosine similarity for distributional substitutes
      3. Glow scores (centrality × cluster density)
      4. Bigram transition matrix for adoption scoring
    """
    # Step 1-2: PPMI vectors + cosine similarity
    print("    building PPMI matrix from context window...")
    M = _build_ppmi_matrix(ctx_rows, ctx_cols, ctx_counts, unigram, V, min_ppmi, device)
    nnz = M._nnz()
    print(f"    PPMI matrix: {nnz:,} edges")

    print("    computing cosine similarity (blocked)...")
    sub_ids, sub_wts, n_found = _cosine_topk_blocked(M, V, K, block_size, device)
    print(f"    substitutes: {n_found}/{V} words have subs")

    # Step 5: Glow scores
    print("    computing glow (centrality × density)...")
    glow = _compute_glow(M, V, device)
    n_glow = int((glow > 0.1).sum().item())
    print(f"    glow: {n_glow} words with glow > 0.1")

    # Bigram transition for adoption child lookup
    print("    building bigram transition matrix...")
    idx_bg = torch.stack([bg_rows, bg_cols])
    ones_bg = torch.ones(bg_rows.numel(), dtype=torch.float32, device=device)
    bg_raw = torch.sparse_coo_tensor(idx_bg, ones_bg, (V, V)).coalesce()
    # Row-normalize
    bg_idx = bg_raw.indices()
    bg_val = bg_raw.values()
    row_sums = torch.zeros(V, dtype=torch.float32, device=device)
    row_sums.scatter_add_(0, bg_idx[0], bg_val)
    bg_norm = bg_val / row_sums[bg_idx[0]].clamp(min=1.0)
    bg_trans = torch.sparse_coo_tensor(bg_idx, bg_norm, (V, V)).coalesce()

    return dict(
        sub_ids=sub_ids, sub_wts=sub_wts,
        glow=glow, bg_trans=bg_trans,
        ppmi_matrix=M, K=K, V=V,
    )
