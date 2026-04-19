"""Semantic (conceptual) map with capacity-based isolation.

Each node has a capacity C(n) from the statistical layer. An edge (a,b)
exists with weight w and velocity v. When a new edge is proposed for a
node whose edges already exceed capacity, the weakest existing edge is
evicted (unless the new one would be weaker still, in which case it is
rejected).

Weights can go negative via force dynamics but the primary isolation
mechanism is the capacity cap — repulsion is optional.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SemanticMapState:
    vocab_size: int
    # Edges stored as dict[(a,b)] = (weight, velocity, last_pulse)
    # For GPU efficiency, we'll use a pair-hash dense store in later refactor.
    edges: Dict[int, float]
    velocity: Dict[int, float]
    last_pulse: Dict[int, int]
    # per-node adjacency lists (b ids in order of insertion) for capacity mgmt
    adjacency: List[List[int]]
    capacity: np.ndarray  # [V] int


def make_empty(vocab_size: int, capacity: torch.Tensor) -> SemanticMapState:
    cap_np = capacity.detach().to("cpu").numpy().astype(np.int32)
    return SemanticMapState(
        vocab_size=vocab_size,
        edges={},
        velocity={},
        last_pulse={},
        adjacency=[[] for _ in range(vocab_size)],
        capacity=cap_np,
    )


def init_from_ppmi(
    state: SemanticMapState,
    ppmi_rows: torch.Tensor,
    ppmi_cols: torch.Tensor,
    ppmi_vals: torch.Tensor,
) -> None:
    V = state.vocab_size
    rows = ppmi_rows.detach().to("cpu").numpy()
    cols = ppmi_cols.detach().to("cpu").numpy()
    vals = ppmi_vals.detach().to("cpu").numpy()
    for r, c, v in zip(rows, cols, vals):
        _insert_edge(state, int(r), int(c), float(v))


def _pair_key(a: int, b: int, V: int) -> int:
    return a * V + b


def _insert_edge(state: SemanticMapState, a: int, b: int, w: float):
    if a == b:
        return
    V = state.vocab_size
    key = _pair_key(a, b, V)
    if key in state.edges:
        state.edges[key] = w
        return
    # capacity check on `a`
    cap_a = int(state.capacity[a])
    adj_a = state.adjacency[a]
    if len(adj_a) >= cap_a and cap_a > 0:
        # find weakest edge on a
        weakest_b = min(adj_a, key=lambda bb: state.edges.get(_pair_key(a, bb, V), 0.0))
        weakest_w = state.edges.get(_pair_key(a, weakest_b, V), 0.0)
        if w <= weakest_w:
            return  # reject new edge
        # evict weakest
        wk = _pair_key(a, weakest_b, V)
        state.edges.pop(wk, None)
        state.velocity.pop(wk, None)
        state.last_pulse.pop(wk, None)
        adj_a.remove(weakest_b)
    adj_a.append(b)
    state.edges[key] = w
    state.velocity[key] = 0.0
    state.last_pulse[key] = 0


def get_weight(state: SemanticMapState, a: int, b: int) -> float:
    return state.edges.get(_pair_key(a, b, state.vocab_size), 0.0)


def set_weight(state: SemanticMapState, a: int, b: int, w: float):
    if a == b:
        return
    key = _pair_key(a, b, state.vocab_size)
    if key in state.edges:
        state.edges[key] = w
    else:
        _insert_edge(state, a, b, w)


def neighbors(state: SemanticMapState, a: int, top_k: int = -1) -> List[Tuple[int, float]]:
    V = state.vocab_size
    items = [(b, state.edges[_pair_key(a, b, V)]) for b in state.adjacency[a]]
    items.sort(key=lambda x: -x[1])
    if top_k > 0:
        items = items[:top_k]
    return items


# ------------- bulk accessors (used by inference / spreading) --------------


def to_sparse_tensor(state: SemanticMapState, device: torch.device) -> torch.sparse.Tensor:
    """Convert edges into a sparse COO tensor on device (for spreading activation)."""
    V = state.vocab_size
    if not state.edges:
        ind = torch.zeros((2, 0), dtype=torch.int64, device=device)
        val = torch.zeros(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(ind, val, (V, V)).coalesce()
    keys = np.fromiter(state.edges.keys(), dtype=np.int64, count=len(state.edges))
    vals = np.fromiter(state.edges.values(), dtype=np.float32, count=len(state.edges))
    rows = (keys // V).astype(np.int64)
    cols = (keys % V).astype(np.int64)
    ind = torch.from_numpy(np.stack([rows, cols])).to(device)
    val = torch.from_numpy(vals).to(device)
    return torch.sparse_coo_tensor(ind, val, (V, V)).coalesce()


def merge_delta(state: SemanticMapState, delta_dict: Dict[int, float]):
    """Apply a batch of weight deltas (keyed by pair_key)."""
    V = state.vocab_size
    for key, dw in delta_dict.items():
        a = key // V
        b = key % V
        cur = state.edges.get(key, 0.0)
        new = cur + dw
        if key in state.edges:
            state.edges[key] = new
        else:
            _insert_edge(state, int(a), int(b), new)
