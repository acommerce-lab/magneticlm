# MagneticLM v7 — Genetic Triangle Transformer: Theoretical Reference

## 1. Introduction

MagneticLM v7 is a language model that replicates transformer architecture using **only corpus statistics** — no gradient descent, no learned parameters. Every weight matrix, every structural parameter (embedding dimension `d`, number of layers `L`, attention projections `Wq/Wk`), is derived deterministically from the input data through a chain of mathematically justified operations.

The core thesis: **transformers redistribute flat statistical knowledge into a multi-dimensional spectral space**. If this redistribution can be computed analytically (via SVD) rather than iteratively (via backpropagation), we obtain a model that is:
- Fully interpretable (every parameter traces to a corpus statistic)
- Deterministic (same data → same model, always)
- Theoretically grounded (each step backed by a proven theorem)

## 2. Mathematical Foundations

### 2.1 Knowledge Measurement (Shannon, 1948)

Before building any model, we measure the **intrinsic knowledge** in the data from raw bigram counts:

```
H(next|prev) = -Σ_w P(w) Σ_{w'} P(w'|w) log₂ P(w'|w)
```

This is the **conditional entropy** — how many bits are needed to predict the next word given the previous one. From this:

- **Knowledge fraction**: `K = 1 - H(next|prev) / log₂(V)` where `V` = vocabulary size
- **PPL lower bound**: `PPL_bound = 2^H(next|prev)` — no model can beat this (Shannon's source coding theorem)
- **Covering Lemma**: Stable extraction requires `N > V · ln(V)` tokens. Below this, SVD vectors are unstable.

Example from experiments: WikiText-103 with 860K lines gives `K = 53.3%`, `PPL_bound = 157.2`, coverage = 191x.

### 2.2 PPMI — Pointwise Mutual Information

**Definition**: `PPMI(a,b) = max(0, log(P(a,b) / P(a)·P(b)))`

**Justification (Shannon)**: PPMI is the **unique** measure satisfying:
1. Zero when words are independent
2. Symmetric: `I(a;b) = I(b;a)`
3. Invariant to marginal frequencies (removes frequency bias)

No other measure satisfies all three. This is proven in information theory.

**Yoneda Connection**: In category theory, the Yoneda lemma states: "An object is completely determined by its relationships with all other objects." The PPMI row for word `w` — `[PPMI(w, w₁), PPMI(w, w₂), ...]` — is exactly this: the word's identity IS its co-occurrence profile. This embedding is **faithful** (injective): different words have different profiles.

### 2.3 SVD — Singular Value Decomposition

**Eckart-Young Theorem (1936)**: For any matrix M and any target rank d, the truncated SVD `M_d = U[:,:d] @ diag(S[:d]) @ V[:d,:]` minimizes the Frobenius reconstruction error. **No other rank-d approximation is better.** This is a proven theorem, not a heuristic.

Applied to PPMI: `PPMI ≈ U @ diag(S) @ V.T`
- Each singular value `S_i` represents a **knowledge circle** — an independent pattern in the data
- `S_1` (largest): the dominant pattern (often syntactic structure)
- `S_d` (smallest kept): the weakest retained pattern
- Embeddings: `E = U[:,:d] @ diag(sqrt(S[:d]))` — each word is a point in d-dimensional spectral space

### 2.4 Dimension Selection

**Marchenko-Pastur Noise Floor**: For a random V×V matrix with E non-zero entries of variance σ²:
```
S_noise = σ_ppmi × √(2E/V)
```
Any `S_i > S_noise` is signal. Below is noise. This is proven for random matrices.

**Problem**: PPMI spectra decay gradually — Marchenko-Pastur gives `d ≈ V` (too large). The noise floor separates signal from noise, but not **useful** signal from **harmful** signal (overfitting).

**Solution: Cumulative Variance Threshold**:
```
cumsum(S²) / sum(S²) ≥ var_target → d
```
With `var_target = 0.3` (retain 30% of spectral energy), we get practical dimensions:
- 100 lines → d=21, 1K → d=104, 860K → d=186

This controls the **bias-variance tradeoff**: too high → overfitting (PPL explodes), too low → underfitting (information lost). The 30% default captures the "structural core" of the language.

### 2.5 Spectral Decomposition of Transition Operator

The bigram transition matrix `T[i,j] = P(w_j | w_i)` captures sequential patterns. We project it into embedding space:

```
M_embed = E.T @ T @ E    [d × d]
```

SVD of M_embed: `U_m @ diag(S_m) @ V_m.T`
- `U_m`: spectral rotation for queries (what a word predicts)
- `V_m`: spectral rotation for keys (what a word represents)
- `S_m`: **transition strengths** — how strongly each spectral circle connects words sequentially

**Wq and Wk construction**:
```
Wq = U_m @ diag(sqrt(S_dampened))
Wk = V_m @ diag(sqrt(S_dampened))
```

Where `S_dampened = S_m × sigmoid(α × (S_m - μ) / σ)` normalizes magnitude while preserving relative weights.

**Key insight**: `Q @ K.T = x.T @ M_embed_dampened @ x` — attention scores measure **transition-weighted similarity**, not just cosine similarity.

### 2.6 Progressive Spectral Distillation (The Cone)

Each layer applies progressively stronger dampening:
```
α_l = 2^l    (geometric: 1, 2, 4, 8, ...)
```

- Layer 0 (α=1): all spectral circles contribute (exploration)
- Layer 1 (α=2): weak circles dampened (focusing)
- Layer L-1 (α=2^{L-1}): only strong circles active (distillation)

**Number of layers**: `L = ceil(log₂(H_max / H(next|prev)))` — derived from the entropy ratio. Each layer halves the remaining entropy (binary search principle, justified by rate-distortion theory).

**Why r=2**: In information theory, the optimal compression path through information space follows a geodesic where each step removes a constant fraction of entropy. The binary fraction (r=2) is the fastest convergence rate for discrete systems.

### 2.7 Triangle Architecture

Dimensions shrink per layer: `d_l = d // 2^l`
```
Layer 0: d₀ = d      (all circles)
Layer 1: d₁ = d//2   (top half)
Layer 2: d₂ = d//4   (top quarter)
```

**Residual via truncation**: `x_new = truncate(x + attn(x), d_new)`. This is valid because SVD dimensions are ordered by importance — truncation removes the **weakest** dimensions, not random ones.

The word representation across layers is a **triangle** (decreasing width), not a rectangle (constant width). This is more efficient (smaller computation per layer) and more honest (no pretending weak dimensions exist).

## 3. Architecture

### 3.1 Full Pipeline
```
Data → Vocabulary → Encode → Knowledge(H, K, PPL_bound, Coverage)
  → PPMI → SVD(d from var_target) → Spectral(Wq, Wk from E.T@T@E)
  → Triangle Transformer(L layers, shrinking d) → Evaluation
```

### 3.2 Attention
```python
Q = x @ Wq_l    # [batch, seq, d_l]
K = x @ Wk_l    # [batch, seq, d_l]
V = x           # values are raw embeddings
scores = Q @ K.T / sqrt(d_l)
scores = masked_fill(scores, causal_mask, -inf)
output = softmax(scores) @ V
x = LayerNorm(x + output)
```

### 3.3 Statistical FFN
```python
h = ReLU(x @ E_norm.T / sqrt(d))    # word activations [V]
h = h / h.sum()                      # normalize
ffn_out = h @ E                      # weighted sum of embeddings
x = LayerNorm(x + ffn_out)
```
This is the vocabulary as memory bank — find similar words, combine their embeddings.

### 3.4 Output Head
```python
logits = x[-1] @ E.T    # raw dot product (GPT-2 style)
probs = softmax(logits)
```

## 4. What Proved and What Didn't

### 4.1 Proven (Mathematical Theorems)
| Claim | Theorem | Status |
|-------|---------|--------|
| PPMI is optimal association measure | Shannon MI uniqueness | **Proven** |
| SVD is optimal compression | Eckart-Young (1936) | **Proven** |
| PPL_bound is a true lower bound | Source coding theorem | **Proven** |
| N > V·ln(V) needed for stability | Covering Lemma | **Proven** |
| K measures extractable knowledge | Information theory | **Proven** |

### 4.2 Empirical Results
| Model | d | PPL (860K) | kingdom rank | hit@5 |
|-------|---|-----------|-------------|-------|
| v6 simple cosine | 100 | 42,345 | **1** | 0.030 |
| v7 d=4096 | 4096 | 364B | 1 | 0.056 |
| v7 d=186 | 186 | 322M | 27 | 0.035 |
| v7 d=8 | 8 | 96K | >50 | 0.010 |
| PPL_bound | — | 157 | — | — |

**Key finding**: Simpler v6 (cosine similarity + IDF + sharpening) outperforms sophisticated v7 (spectral Wq/Wk + FFN + triangle). Reason: statistical weights are not optimized for prediction.

### 4.3 Open Problems
- **var_target**: No closed-form optimal value. Controls bias-variance tradeoff.
- **Cone slope r**: r=2 justified but not proven unique optimal.
- **FFN design**: Vocabulary memory works but may not be optimal.
- **Output head**: Raw dot product has frequency bias.

## 5. Comparison with Standard Transformers

| Component | Standard Transformer | MagneticLM v7 |
|-----------|---------------------|---------------|
| Embeddings | Learned (backprop) | SVD on PPMI (Eckart-Young) |
| d (dimension) | Hyperparameter | Auto (cumulative variance) |
| n_heads | Hyperparameter | d (each dim = one head) |
| n_layers | Hyperparameter | Auto: ceil(log₂(H_max/H)) |
| Wq, Wk, Wv | Learned | From SVD of E.T@T@E |
| Attention math | softmax(QK.T/√d)V | **Identical** |
| FFN | Learned W1, W2 | E.T/E as memory |
| LayerNorm | Learned scale/bias | No learnable params |
| Residual | x + sublayer(x) | **Identical** |
| Output | h @ E.T | **Identical** |
| Training | Gradient descent (hours/days) | One-pass statistics (seconds) |

## 6. Applications to Existing Transformers

### 6.1 Spectral Analysis of Learned Weights
Given a trained transformer, extract Wq and Wk, compute SVD(Wq @ Wk.T). Compare with our statistical SVD(E.T @ T @ E). The alignment measures how much the trained model rediscovered corpus statistics vs learned something new.

### 6.2 Weight Initialization
Initialize Wq from SVD of E.T @ T @ E before training. This provides a "warm start" that already encodes bigram transition patterns.

### 6.3 Model Compression
If learned spectral circles align with statistical ones, prune the misaligned (likely noise). Use glow (S_m values) to rank circles by importance.

### 6.4 Architecture Design
Use spectral gap of the data to suggest d, n_heads, L. The data's spectral structure tells you how complex the model needs to be.

## 7. Theoretical Implications

**"Transformers are an algorithm for distributing flat statistical knowledge into multi-dimensional spectral space."** The embedding maps words from discrete tokens to continuous vectors. SVD decomposes this mapping into independent knowledge circles. Attention recombines these circles based on context. The output projects back to vocabulary space.

**The Yoneda perspective**: A word IS its relationships. The PPMI row is the Yoneda embedding. SVD finds the irreducible representations. The transformer is a functor between the category of contexts and the category of predictions.

**Knowledge as randomness reduction**: K = 1 - H/H_max. Language has structure (K > 0) because words are not random. The model's job is to capture this structure. The gap between PPL_actual and PPL_bound measures how much structure the model misses.

## 8. Limitations

1. **Statistical weights ≠ learned weights**: The fundamental gap. Our weights encode bigram statistics; trained weights encode the full training objective.
2. **Bigram bottleneck**: All knowledge comes from P(next|prev). Higher-order dependencies are missed.
3. **var_target is a free parameter**: Not fully automatic yet.
4. **Computational cost**: SVD on V×V matrix is O(V²d), expensive for large vocabularies.
5. **No compositional semantics**: "not hot" ≠ "hot" — bigrams can't capture negation.

## 9. Future Directions

1. **Higher-order transitions**: Trigram/4-gram M_embed for richer context
2. **Iterative refinement**: Multiple passes of global attention (like training epochs)
3. **Early exit**: Stop at confident layers, "I don't know" for uncertain ones
4. **Hierarchical spectral circles**: Circles of circles (multi-resolution)
5. **Category-theoretic formalization**: Full Yoneda-based proof of the model
6. **Hybrid initialization**: Use statistical weights to warm-start gradient-based training
