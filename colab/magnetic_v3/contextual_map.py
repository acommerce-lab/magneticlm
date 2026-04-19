"""Contextual map: direct succession A -> B as CSR on GPU.

This map is mandatory and frozen after build (no pulse training).
It captures the raw successor distribution.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class ContextualMap:
    vocab_size: int
    row_ptr: torch.Tensor        # [V+1] int64
    col_idx: torch.Tensor        # [nnz] int32
    counts: torch.Tensor         # [nnz] int32
    row_total: torch.Tensor      # [V] int64
    search_key: torch.Tensor     # [nnz] int64 = row*V+col (globally sorted)

    @staticmethod
    def build(
        encoded: List[np.ndarray],
        vocab_size: int,
        max_children: int,
        device: torch.device,
    ) -> "ContextualMap":
        V = vocab_size

        pair_buf: List[torch.Tensor] = []
        for arr in encoded:
            if arr.size < 2:
                continue
            t = torch.from_numpy(arr.astype(np.int64)).to(device)
            a = t[:-1]
            b = t[1:]
            pair_buf.append(a * V + b)
        if not pair_buf:
            return ContextualMap(
                vocab_size=V,
                row_ptr=torch.zeros(V + 1, dtype=torch.int64, device=device),
                col_idx=torch.empty(0, dtype=torch.int32, device=device),
                counts=torch.empty(0, dtype=torch.int32, device=device),
                row_total=torch.zeros(V, dtype=torch.int64, device=device),
                search_key=torch.empty(0, dtype=torch.int64, device=device),
            )

        pair_ids = torch.cat(pair_buf)
        pair_ids, _ = torch.sort(pair_ids)
        uniq, counts = torch.unique_consecutive(pair_ids, return_counts=True)

        rows = (uniq // V).to(torch.int64)
        cols = (uniq % V).to(torch.int64)
        vals = counts.to(torch.int64)

        # Enforce max children per row (keep top counts)
        if max_children > 0:
            rows, cols, vals = _truncate_rows(rows, cols, vals, V, max_children)

        row_total = torch.zeros(V, dtype=torch.int64, device=device).scatter_add_(
            0, rows, vals
        )
        row_ptr = torch.zeros(V + 1, dtype=torch.int64, device=device)
        ones = torch.ones_like(rows, dtype=torch.int64)
        row_ptr.scatter_add_(0, rows + 1, ones)
        row_ptr = torch.cumsum(row_ptr, dim=0)

        search_key = rows * V + cols
        return ContextualMap(
            vocab_size=V,
            row_ptr=row_ptr,
            col_idx=cols.to(torch.int32),
            counts=vals.to(torch.int32),
            row_total=row_total,
            search_key=search_key,
        )

    def prob(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """p(b|a) from direct succession counts."""
        device = self.row_ptr.device
        a = a.to(device).to(torch.int64)
        b = b.to(device).to(torch.int64)
        V = self.vocab_size
        if self.search_key.numel() == 0:
            tot = self.row_total[a].to(torch.float32).clamp(min=1.0)
            return torch.zeros_like(tot)
        query = a * V + b
        pos = torch.searchsorted(self.search_key, query, right=False)
        nnz = self.search_key.numel()
        safe = torch.clamp(pos, max=nnz - 1)
        hit = (pos < nnz) & (self.search_key[safe] == query)
        c = torch.where(hit, self.counts[safe].to(torch.float32), torch.zeros_like(a, dtype=torch.float32))
        tot = self.row_total[a].to(torch.float32).clamp(min=1.0)
        return c / tot

    def children(self, a: int):
        s = int(self.row_ptr[a].item())
        e = int(self.row_ptr[a + 1].item())
        return self.col_idx[s:e], self.counts[s:e]


def _truncate_rows(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    V: int,
    max_children: int,
):
    """For each row keep at most `max_children` highest-count columns."""
    # Sort by row ascending, count descending
    order = torch.argsort(rows * (vals.max() + 1) - vals.to(torch.int64))
    rows_s = rows[order]
    cols_s = cols[order]
    vals_s = vals[order]

    # rank within each row
    row_changes = torch.cat(
        [torch.ones(1, dtype=torch.int64, device=rows.device), (rows_s[1:] != rows_s[:-1]).to(torch.int64)]
    )
    rank = torch.zeros_like(rows_s)
    # prefix sum within segments: use cumulative row index then compute pos - row_start
    row_idx = torch.cumsum(row_changes, dim=0) - 1
    # position = index - first_index_of_row
    idx_range = torch.arange(rows_s.numel(), device=rows_s.device)
    first_of_row = torch.zeros(row_idx.max().item() + 1, dtype=torch.int64, device=rows.device)
    first_of_row.scatter_reduce_(
        0, row_idx, idx_range, reduce="amin", include_self=False
    )
    rank = idx_range - first_of_row[row_idx]
    keep = rank < max_children
    return rows_s[keep], cols_s[keep], vals_s[keep]
