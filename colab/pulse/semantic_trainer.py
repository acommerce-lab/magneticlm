# pulse/semantic_trainer.py
#
# Iterative semantic map with 5-force dynamics.
#
# Each sentence is a pulse. The five forces determine HOW MUCH each
# edge weight changes — not fixed amounts, but dynamic forces computed
# from the local state of the graph:
#
#   1. SPRING: co-occurring words attract. Attraction is STRONGER when
#      the current weight is WEAK (far apart → pull harder). This means
#      new co-occurrences get a big initial boost, but established
#      connections stabilize.
#
#   2. REPULSION: high-degree nodes push each other apart. If word A is
#      connected to 50k words and word B to 5, the edge A↔B gets a
#      strong repulsive force. This is what naturally devalues common
#      words ("the", "of") semantically while keeping them contextually
#      present.
#
#   3. FAR-FIELD ATTRACTION: prevents the graph from fragmenting.
#      Applies a small pull toward an optimal weight level for edges
#      that are becoming too weak.
#
#   4. GRAVITY: pulls all weights toward zero. Prevents unbounded
#      growth. Strong edges resist gravity; weak edges decay.
#
#   5. DAMPING: smooths velocity changes to prevent oscillation.
#      Each edge accumulates a velocity (rate of change). Damping
#      reduces it each step so the system converges.
#
# These same forces also govern spreading activation at inference time:
#   - Strong positive edge → high activation transfer
#   - Negative edge → damping (suppression)
#   - High-degree source → reduced spreading (common words don't
#     propagate well)

import heapq
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


class SemanticMap:
    """Iterative semantic map with 5-force dynamics."""

    def __init__(self, config):
        self.cfg = config
        # edges[w1][w2] = weight (positive = attraction, negative = repulsion)
        self.edges: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float))
        # velocity[w1][w2] = accumulated rate of change (momentum)
        self.velocity: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float))
        # degree[w] = number of active edges for word w
        self.degree: Dict[int, int] = defaultdict(int)
        self.sentences_processed = 0

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------
    def _compute_delta(self, wi: int, wj: int, event: str) -> float:
        """Compute the weight delta for edge (wi, wj) using 5 forces.

        Args:
            wi, wj: word IDs
            event: one of 'cooccurrence', 'reward', 'penalty'

        Returns:
            delta to add to edges[wi][wj]
        """
        cfg = self.cfg
        w = self.edges[wi].get(wj, 0.0)
        deg_i = self.degree.get(wi, 1)
        deg_j = self.degree.get(wj, 1)

        # --- 1. Spring force ---
        # Attraction proportional to semantic "distance" (inverse of weight).
        # Stronger pull when words are far apart (low weight).
        # For co-occurrence and reward: positive (attraction).
        # For penalty: not applied (penalty comes from repulsion).
        if event == 'penalty':
            f_spring = 0.0
        else:
            f_spring = cfg.K_spring / (1.0 + abs(w))

        # --- 2. Repulsion force ---
        # High-degree nodes repel each other. The more connections both
        # words have, the stronger the repulsion. This is what pushes
        # "the" away from everything semantically.
        # degree_ratio = min/max normalized degree overlap
        deg_min = min(deg_i, deg_j)
        deg_max = max(deg_i, deg_j)
        degree_ratio = deg_min / (deg_max + 1.0)
        f_repulsion = -cfg.K_repulsion * degree_ratio

        # For penalty events: repulsion is AMPLIFIED (wrong prediction
        # from a strong edge → push harder)
        if event == 'penalty':
            f_repulsion *= 2.0

        # --- 3. Far-field attraction ---
        # Small pull toward optimal weight to prevent fragmentation.
        # Only applies when weight is below optimal and the event is
        # not a penalty.
        if event != 'penalty' and w < cfg.optimal_weight:
            f_attraction = cfg.K_attraction * (cfg.optimal_weight - w) * 0.01
        else:
            f_attraction = 0.0

        # --- 4. Gravity ---
        # Pulls weight toward zero. Prevents unbounded growth.
        # Strong edges resist gravity (they earned their weight).
        f_gravity = -cfg.K_gravity * w

        # --- 5. Net force + velocity update with damping ---
        net_force = f_spring + f_repulsion + f_attraction + f_gravity

        # Update velocity with momentum
        v = self.velocity[wi].get(wj, 0.0)
        v = (v + net_force) * (1.0 - cfg.damping)
        self.velocity[wi][wj] = v

        # Delta = velocity * learning rate
        return v * cfg.force_lr

    def _update_edge(self, wi: int, wj: int, event: str):
        """Update edge (wi, wj) symmetrically using force dynamics."""
        delta = self._compute_delta(wi, wj, event)

        # Update weight
        old_w = self.edges[wi][wj]
        self.edges[wi][wj] += delta
        self.edges[wj][wi] += delta

        # Update velocity symmetrically
        self.velocity[wj][wi] = self.velocity[wi][wj]

        # Update degree tracking
        new_w = self.edges[wi][wj]
        threshold = self.cfg.semantic_threshold
        was_active = abs(old_w) >= threshold
        is_active = abs(new_w) >= threshold
        if not was_active and is_active:
            self.degree[wi] = self.degree.get(wi, 0) + 1
            self.degree[wj] = self.degree.get(wj, 0) + 1
        elif was_active and not is_active:
            self.degree[wi] = max(0, self.degree.get(wi, 0) - 1)
            self.degree[wj] = max(0, self.degree.get(wj, 0) - 1)

    # ------------------------------------------------------------------
    # Core: process one sentence as a training pulse
    # ------------------------------------------------------------------
    def pulse(self, token_ids: List[int]):
        """Process one sentence as a training pulse with 5-force dynamics."""
        cfg = self.cfg
        n = len(token_ids)
        if n < 2:
            return

        # --- Phase 1: window ±K co-occurrence → spring attraction ---
        for i in range(n):
            wi = token_ids[i]
            for d in range(1, cfg.semantic_window + 1):
                for j in (i - d, i + d):
                    if 0 <= j < n:
                        wj = token_ids[j]
                        if wi != wj:
                            self._update_edge(wi, wj, 'cooccurrence')

        # --- Phase 2: prediction → reward/penalty ---
        top_k = cfg.neighbor_top_k
        for i in range(n - 1):
            wi = token_ids[i]
            actual = token_ids[i + 1]

            neighbors = self._get_top_k(wi, top_k)
            if not neighbors:
                continue

            for nid, weight in neighbors:
                if nid == actual:
                    self._update_edge(wi, actual, 'reward')
                elif weight > cfg.penalty_threshold:
                    self._update_edge(wi, nid, 'penalty')

        # --- Phase 3: periodic transitive propagation ---
        self.sentences_processed += 1
        if self.sentences_processed % cfg.transitive_interval == 0:
            self._propagate_transitive(token_ids)

    # ------------------------------------------------------------------
    # Spreading activation (for inference)
    # ------------------------------------------------------------------
    def spread_activation(self, seed_ids: List[int],
                          iterations: int = 10,
                          damping: float = 0.85) -> Dict[int, float]:
        """Spread activation from seed words through the semantic graph.

        The 5 forces govern spreading:
          - Strong positive edge → high transfer
          - Negative edge → suppression
          - High-degree node → reduced propagation (common words don't
            spread well)

        Returns: {word_id: activation_level}
        """
        activation = defaultdict(float)
        # Initialize seeds with decreasing activation (recent = stronger)
        for i, wid in enumerate(seed_ids):
            activation[wid] = 0.5 + 0.5 * (i / max(len(seed_ids), 1))

        threshold = self.cfg.semantic_threshold
        for _ in range(iterations):
            new_activation = defaultdict(float)
            for wid, act in activation.items():
                if act < 0.01:
                    continue
                if wid not in self.edges:
                    continue

                # High-degree nodes spread less (repulsion effect)
                deg = self.degree.get(wid, 1)
                spread_factor = 1.0 / (1.0 + math.log1p(deg) * 0.1)

                for neighbor, weight in self.edges[wid].items():
                    if abs(weight) < threshold:
                        continue
                    if weight > 0:
                        # Positive edge: activate neighbor
                        transfer = act * weight * spread_factor * 0.1
                        new_activation[neighbor] += transfer
                    else:
                        # Negative edge: suppress neighbor
                        suppress = act * abs(weight) * spread_factor * 0.05
                        new_activation[neighbor] -= suppress

            # Merge with damping
            for wid in set(list(activation.keys()) +
                           list(new_activation.keys())):
                activation[wid] = (activation[wid] * damping +
                                   new_activation.get(wid, 0.0))

        return dict(activation)

    # ------------------------------------------------------------------
    # Neighbor lookup
    # ------------------------------------------------------------------
    def _get_top_k(self, word_id: int, top_k: int) -> List[Tuple[int, float]]:
        """Top-K neighbors with positive weight above threshold."""
        if word_id not in self.edges:
            return []
        threshold = self.cfg.semantic_threshold
        valid = [(nid, w) for nid, w in self.edges[word_id].items()
                 if w >= threshold]
        if not valid:
            return []
        if len(valid) <= top_k:
            valid.sort(key=lambda x: -x[1])
            return valid
        return heapq.nlargest(top_k, valid, key=lambda x: x[1])

    def get_all_neighbors(self, word_id: int) -> List[Tuple[int, float]]:
        """All neighbors with |weight| above threshold, sorted desc."""
        if word_id not in self.edges:
            return []
        threshold = self.cfg.semantic_threshold
        neighbors = [(nid, w) for nid, w in self.edges[word_id].items()
                     if abs(w) >= threshold]
        neighbors.sort(key=lambda x: -x[1])
        return neighbors

    # ------------------------------------------------------------------
    # Transitive propagation (2-hop)
    # ------------------------------------------------------------------
    def _propagate_transitive(self, token_ids: List[int]):
        """2-hop transitive relations using force-based update."""
        cfg = self.cfg
        threshold = cfg.semantic_threshold
        decay = cfg.transitive_decay
        top_k = cfg.transitive_top_k

        seen = set()
        for word in token_ids:
            if word in seen:
                continue
            seen.add(word)

            hop1 = self._get_top_k(word, top_k)
            for neighbor_id, w1 in hop1:
                if w1 < threshold:
                    continue
                hop2 = self._get_top_k(neighbor_id, top_k)
                for trans_id, w2 in hop2:
                    if trans_id == word or w2 < threshold:
                        continue
                    tw = w1 * w2 * decay * 0.01
                    if tw > 0.001:
                        self.edges[word][trans_id] += tw
                        self.edges[trans_id][word] += tw

    # ------------------------------------------------------------------
    # Train on full corpus
    # ------------------------------------------------------------------
    def train_corpus(self, sentences_as_ids: List[List[int]],
                     verbose: bool = True):
        t0 = time.time()
        total = len(sentences_as_ids)
        for i, sent in enumerate(sentences_as_ids):
            self.pulse(sent)
            if verbose and (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print("\r  Semantic pulse: %d/%d (%.0f sent/s, ETA %.0fs)" %
                      (i + 1, total, rate, eta),
                      end="", flush=True)
        if verbose:
            elapsed = time.time() - t0
            print("\r  Semantic pulse: %d sentences done (%.0fs)%s" %
                  (total, elapsed, " " * 30))

    # ------------------------------------------------------------------
    # Convert to GPU tensors
    # ------------------------------------------------------------------
    def to_gpu_tensors(self, device: torch.device):
        threshold = self.cfg.semantic_threshold
        froms, tos, weights = [], [], []
        for w1, neighbors in self.edges.items():
            for w2, weight in neighbors.items():
                if abs(weight) >= threshold:
                    froms.append(w1)
                    tos.append(w2)
                    weights.append(weight)
        if not froms:
            return (torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))
        return (torch.tensor(froms, dtype=torch.int64, device=device),
                torch.tensor(tos, dtype=torch.int64, device=device),
                torch.tensor(weights, dtype=torch.float32, device=device))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        total = positive = negative = 0
        max_w = 0.0
        min_w = 0.0
        words_with_edges = 0
        threshold = self.cfg.semantic_threshold

        for w1, neighbors in self.edges.items():
            has_strong = False
            for w2, w in neighbors.items():
                if abs(w) >= threshold:
                    total += 1
                    has_strong = True
                    if w > 0:
                        positive += 1
                        max_w = max(max_w, w)
                    else:
                        negative += 1
                        min_w = min(min_w, w)
            if has_strong:
                words_with_edges += 1

        return {
            "total_edges": total,
            "positive_edges": positive,
            "negative_edges": negative,
            "words_with_edges": words_with_edges,
            "max_weight": max_w,
            "min_weight": min_w,
            "sentences_processed": self.sentences_processed,
        }
