# MagneticLM v8 — Spectral Path Integral Language Model

## 1. Definition

A language model that predicts the next word by integrating context
change patterns through a pre-computed spectral cone. No training loop.
No learned parameters. All weights derived from corpus statistics in
one pass.

**Core principle**: Words are not points in space — they are
**transitions** (directional changes). A context is a **path** through
transition space. Prediction is the **integral** of that path projected
through a spectral cone.

**Name**: Spectral Path Integral Model (SPIM)
- "Spectral": structure from SVD eigenspectrum
- "Path": context = sequence of transitions, not sequence of points
- "Integral": prediction = accumulation along the path

Alternative names:
- Transition Cone Model (TCM)
- Spectral Context Integrator (SCI)
- Algebraic Path Predictor (APP)


## 2. Mathematical Framework

### 2.1 Transition Embeddings (not positional embeddings)

**Standard approach** (v7 and all transformers):
  E(w) ∈ ℝ^d — each word is a point

**New approach**:
  ΔE(w_i → w_j) = E(w_j) - E(w_i) ∈ ℝ^d — each bigram is a direction

**Justification (PROVEN)**: The bigram transition matrix T encodes
P(w_j | w_i). The embedding difference ΔE captures the same
information in continuous space. By Eckart-Young, SVD of the
transition-difference matrix gives the optimal low-rank approximation
of these directions.

**What this changes**: Instead of "where is each word?", we ask
"how does meaning CHANGE from word to word?"

### 2.2 Context as Path

A context [w_1, w_2, ..., w_n] generates a path:
  Δ_1 = ΔE(w_1 → w_2)
  Δ_2 = ΔE(w_2 → w_3)
  ...
  Δ_{n-1} = ΔE(w_{n-1} → w_n)

The context representation = accumulated path:
  C = Σ_{k=1}^{n-1} α_k · Δ_k

where α_k are decay weights (recent transitions matter more).

**Justification**: This is analogous to a line integral in vector
calculus. The path through transition space accumulates directional
information. The prediction point = where the path leads.

**Status**: HYPOTHETICAL. The line integral analogy is motivated
but not proven optimal. The decay weights α_k are a design choice.

### 2.3 Spectral Cone (pre-computed, not learned)

The PPMI matrix decomposes via SVD into spectral circles.
Each circle represents an independent pattern of co-occurrence.

The cone has L levels, determined by L = ceil(log2(H_max / H)):
  Level 0: d dimensions (all patterns)
  Level 1: d/2 dimensions (dominant patterns)
  ...
  Level L-1: d/2^{L-1} dimensions (core patterns)

**Justification (PROVEN)**: SVD gives optimal compression (Eckart-Young).
The level schedule d/2^l is HYPOTHETICAL (motivated by rate-distortion
but not proven as unique optimal).

### 2.4 Triangle Pre-computation

For each word w in vocabulary, pre-compute its "triangle":
  T(w) = [E(w)[:d], E(w)[:d/2], ..., E(w)[:d/2^{L-1}]]

This is just truncation of the embedding at each level.
No computation needed — just slicing.

For each bigram (w_i, w_j), pre-compute the transition triangle:
  ΔT(w_i→w_j) = T(w_j) - T(w_i)

**Justification (PROVEN)**: Truncation preserves the most important
SVD dimensions (by construction of SVD ordering).

### 2.5 Prediction via Path Integration

Given context [w_1, ..., w_n], predict w_{n+1}:

At each cone level l:
  1. Compute path: C_l = Σ α_k · ΔT(w_k→w_{k+1})[:d_l]
  2. Predicted embedding at level l: P_l = E(w_n)[:d_l] + C_l
  3. Score all words: scores_l(w) = P_l · E(w)[:d_l]

Final scores = weighted combination across levels:
  scores(w) = Σ_l β_l · scores_l(w)

where β_l = S_m[l] / Σ S_m (spectral weight per level).

Output: softmax(scores)

**Status**:
- Path accumulation: HYPOTHETICAL (motivated by calculus analogy)
- Multi-level scoring: HYPOTHETICAL (motivated by multi-resolution)
- Per-level weights β_l: derived from S_m (PROVEN optimal via SVD)

### 2.6 No Attention Mechanism

In v7, attention computed which context words matter. In v8,
the path integral REPLACES attention:
- Attention asks: "which context words should I focus on?"
- Path integral asks: "where does the accumulated change lead?"

The decay weights α_k serve the role of attention — recent
transitions get higher weight. But this is FIXED (not computed
from Q/K similarity).

**This is a deliberate simplification**. It means:
- No Wq, Wk matrices needed
- No softmax over context positions
- No O(seq²) attention computation
- Just O(seq × d) path accumulation

**Tradeoff**: We lose the ability to selectively attend to specific
context words. We gain simplicity and interpretability.

### 2.7 FFN: Vocabulary Lookup (same as v7)

FFN(x) = normalize(ReLU(x · E_norm^T)) · E

This is unchanged from v7. It serves as "memory retrieval" —
given a representation x, find the most similar words and combine
their embeddings.

**Status**: HYPOTHETICAL but empirically tested in v7.

### 2.8 Knowledge Measurement (same as v7)

H(next|prev) from bigram counts. K, PPL_bound, coverage check.

**Status**: PROVEN (Shannon).


## 3. Proven vs Hypothetical Summary

### Proven (mathematical theorems):
1. PPMI is the unique consistent association measure (Shannon)
2. SVD gives optimal low-rank approximation (Eckart-Young)
3. PPL_bound = 2^H is a true lower bound (source coding theorem)
4. Coverage N > V·ln(V) for stable extraction (covering lemma)
5. Truncation preserves most important dimensions (SVD ordering)
6. Per-level weights from S_m are optimal (SVD)

### Hypothetical (design choices requiring validation):
1. Transition embeddings ΔE better than positional E — motivated but unproven
2. Path integral for context — analogy to calculus, not proven optimal
3. Decay weights α_k — design choice, not derived
4. Cone level schedule d/2^l — motivated by rate-distortion
5. Multi-level score combination — reasonable but not proven
6. No attention (path replaces it) — simplification with tradeoff
7. FFN as vocabulary lookup — approximation

### From standard transformers (justified by empirical success):
1. Causal ordering (predict only forward) — standard in LMs
2. Softmax for probability distribution — standard
3. LayerNorm for stability — empirical, widely validated


## 4. Architecture Comparison

| Component          | v7 (Triangle Transformer) | v8 (SPIM)                  |
|--------------------|---------------------------|------------------------------|
| Word representation| Point E(w) ∈ ℝ^d          | Triangle T(w) across levels  |
| Context            | Sequential layer processing| Path integral Σ α·ΔT        |
| Attention          | Wq/Wk spectral projection | None (replaced by path)      |
| Prediction         | h @ E.T at last layer     | Multi-level scoring          |
| FFN                | Vocab memory lookup       | Same                         |
| Parameters         | S_m (d numbers, refined)  | α_k decay + β_l level weights|
| Computation        | O(L × seq² × d)          | O(seq × d + L × V × d_l)    |


## 5. Code Plan

### 5.1 File Structure
```
magnetic_v8/
  __init__.py          — version info
  config.py            — Config (var_target, context_len, pos_decay)
  data.py              — copy from v7
  tokenizer.py         — copy from v7
  stats.py             — copy from v7
  resources.py         — copy from v7
  spectrum.py          — build_spectrum(): PPMI → SVD → embeddings, cone
  path_model.py        — SPIMModel: path integration + multi-level scoring
  evaluator.py         — eval_ppl(), eval_ood()
  runner.py            — pipeline: data → knowledge → spectrum → model → eval
  run.py               — CLI entry
```

### 5.2 spectrum.py — Build Phase
```python
def build_spectrum(ctx_rows, ctx_cols, ctx_counts, unigram,
                   bg_trans, V, min_ppmi, device, var_target):
    """
    Returns:
      embeddings: [V, d]           — word embeddings from SVD
      d_schedule: [d, d//2, ...]   — cone level dimensions
      S_levels: [L]                — spectral weight per level
      bigram_trans: [V, V] sparse  — transition matrix
    """
```

### 5.3 path_model.py — Core Model
```python
class SPIMModel:
    def __init__(self, embeddings, d_schedule, S_levels, bg_trans,
                 context_len, pos_decay):
        # Pre-compute: E_levels[l] = embeddings[:, :d_l] for each level
        # Pre-compute: E_norm_levels[l] = normalized E_levels[l]
        pass

    def _path_integral(self, context_ids):
        """Compute accumulated transition path at each cone level.

        For context [w1, w2, ..., wn]:
          path_l = Σ_{k=1}^{n-1} decay^(n-1-k) * (E[w_{k+1}][:d_l] - E[w_k][:d_l])

        Returns: [L, d_max] — path representation per level
        """
        pass

    def _score(self, context_ids):
        """Score all vocabulary words at each level, combine.

        At level l:
          predicted_l = E[w_n][:d_l] + path_l
          scores_l = predicted_l @ E_levels[l].T

        Combined: scores = Σ β_l * scores_l
        Returns: [V] logits
        """
        pass

    def _ffn(self, x, level):
        """Vocabulary memory lookup at given cone level."""
        pass

    def score_batch(self, contexts):
        """Batch scoring for evaluation."""
        pass

    def score_single(self, context_ids):
        """Single context scoring."""
        pass
```

### 5.4 evaluator.py — Pure Evaluation
```python
def eval_ppl(model, V, encoded, cfg, device):
    """PPL on encoded validation data. No masking, no cache."""
    pass

def eval_ood(model, V, vocab, cfg, device):
    """OOD cloze test — 14 semantic questions."""
    pass
```

### 5.5 runner.py — Pipeline
```python
def run_pipeline(cfg):
    """
    1. Load data, build vocab, encode
    2. Measure knowledge (H, K, PPL_bound, coverage, L)
    3. Build spectrum (PPMI → SVD → embeddings, cone)
    4. Build SPIM model (pre-compute triangles)
    5. Evaluate (PPL + OOD)
    6. Report efficiency
    """
    pass
```

### 5.6 Testing Strategy
- Test spectrum.py: verify SVD, d_schedule, S_levels
- Test path_model.py: verify path integral shapes, scoring shapes
- Test on toy data (4 sentences): verify predictions make sense
- Scale test: 100 → 1K → 10K → 100K → 860K lines
- Compare with v7 PPL=188 baseline at 100 lines
