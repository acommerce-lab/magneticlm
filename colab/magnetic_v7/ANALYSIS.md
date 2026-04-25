# MagneticLM v7 — Mathematical Analysis

## Part 1: Formal Model Definition

### 1.1 Complete Specification

**Input**: Corpus C of tokenized sentences, vocabulary V of size |V|.

**Step 1 — Bigram Statistics**:
- Transition matrix: `T[i,j] = count(w_i, w_j) / count(w_i)`
- Unigram: `P(w) = count(w) / N`
- Knowledge: `H = -Σ P(w) Σ T[w,w'] log₂ T[w,w']`, `K = 1 - H/log₂|V|`
- Coverage check: `N > |V| · ln(|V|)`
- Layer count: `L = ceil(log₂(log₂|V| / H))`

**Step 2 — PPMI Matrix**:
- `PPMI[i,j] = max(0, log(P(i,j) / P(i)P(j)))` where `P(i,j) = count(i,j) / Σcount`
- Sparse: most entries are zero (words that never co-occur)

**Step 3 — SVD + Dimension Selection**:
- `PPMI = U @ diag(S) @ V.T` (truncated or randomized SVD)
- `cumsum(S²) / sum(S²) ≥ var_target → d`
- Embeddings: `E = U[:,:d] @ diag(√S[:d])` — shape `[|V|, d]`

**Step 4 — Spectral Transition Operator**:
- `M = E.T @ T @ E` — shape `[d, d]`
- SVD: `M = U_m @ diag(S_m) @ V_m.T`
- Per-layer construction: for layer l with `d_l = d // 2^l`:
  ```
  α_l = 2^l
  S_damp = S_m[:d_l] × sigmoid(α_l × (S_m[:d_l] - μ) / σ)
  S_unit = S_damp / ||S_damp|| × √d_l
  Wq_l = U_m[:d_l,:d_l] × diag(√S_unit)
  Wk_l = V_m[:d_l,:d_l] × diag(√S_unit)
  ```

**Step 5 — Inference** (for context `[c_1, ..., c_n]`):
```
x = E[[c_1,...,c_n]] × pos_decay    # [n, d]
for l in 0..L-1:
    d_l = d // 2^l
    x = x[:, :d_l]                   # truncate (triangle)
    Q = x @ Wq_l;  K = x @ Wk_l;  V = x
    attn = softmax(Q@K.T / √d_l, causal) @ V
    x = LayerNorm(x + attn)
    ffn = normalize(ReLU(x @ E_norm[:,:d_l].T)) @ E[:,:d_l]
    x = LayerNorm(x + ffn)
logits = x[-1] @ E[:,:d_L].T
prediction = softmax(logits)
```

### 1.2 Theorem Chain

| Step | Operation | Guaranteeing Theorem |
|------|-----------|---------------------|
| Knowledge H | Bigram entropy | Shannon source coding |
| PPMI | Mutual information | Shannon uniqueness |
| SVD truncation | Low-rank approx | Eckart-Young (1936) |
| Noise floor | Signal/noise separation | Marchenko-Pastur |
| Coverage N | Sample complexity | Coupon collector |
| Layer count L | Entropy halving | Rate-distortion |

### 1.3 Complexity

- SVD: O(|V|² × d) for randomized, O(|V|³) for full
- Per-layer attention: O(batch × seq² × d_l)
- Per-layer FFN: O(batch × seq × |V| × d_l)
- Triangle savings: layer l is 2^l times cheaper than layer 0
- Total FFN dominates: O(batch × seq × |V| × d) summed across layers

## Part 2: Minimal Ideal Example

### 2.1 Toy Corpus

```
S1: "the king ruled the kingdom"
S2: "the queen ruled the kingdom"
S3: "the sky is blue"
S4: "the grass is green"
```

**Vocabulary** (V=10): {the, king, queen, ruled, kingdom, sky, is, blue, grass, green}
Index mapping: the=0, king=1, queen=2, ruled=3, kingdom=4, sky=5, is=6, blue=7, grass=8, green=9

### 2.2 Bigram Counts

From the corpus (window=1):
```
        the  king queen ruled kingdom sky  is  blue grass green
the      0    2    1     0     0      1    0    0    1     0
king     0    0    0     1     0      0    0    0    0     0
queen    0    0    0     1     0      0    0    0    0     0
ruled    2    0    0     0     0      0    0    0    0     0
kingdom  0    0    0     0     0      0    0    0    0     0
sky      0    0    0     0     0      0    1    0    0     0
is       0    0    0     0     0      0    0    1    0     1
blue     0    0    0     0     0      0    0    0    0     0
grass    0    0    0     0     0      0    1    0    0     0
green    0    0    0     0     0      0    0    0    0     0
```

**Transition matrix T** (row-normalized):
```
the:     [0, 0.4, 0.2, 0, 0, 0.2, 0, 0, 0.2, 0]
king:    [0, 0,   0,   1, 0, 0,   0, 0, 0,   0]
queen:   [0, 0,   0,   1, 0, 0,   0, 0, 0,   0]
ruled:   [1, 0,   0,   0, 0, 0,   0, 0, 0,   0]
kingdom: [0, 0,   0,   0, 0, 0,   0, 0, 0,   0]  (sentence-final)
sky:     [0, 0,   0,   0, 0, 0,   1, 0, 0,   0]
is:      [0, 0,   0,   0, 0, 0,   0, 0.5, 0, 0.5]
blue:    [0, 0,   0,   0, 0, 0,   0, 0,   0, 0]  (sentence-final)
grass:   [0, 0,   0,   0, 0, 0,   1, 0,   0, 0]
green:   [0, 0,   0,   0, 0, 0,   0, 0,   0, 0]  (sentence-final)
```

### 2.3 Knowledge Measurement

- Total bigrams N = 16
- H(next|prev) computation:
  - "the" (P=5/16): H_the = -(0.4·log₂0.4 + 0.2·log₂0.2 × 3) = 1.92 bits
  - "king" (P=1/16): H_king = 0 (deterministic → "ruled")
  - "queen" (P=1/16): H_queen = 0
  - "ruled" (P=2/16): H_ruled = 0 (always → "the")
  - "is" (P=2/16): H_is = 1.0 bit (50/50 blue or green)
  - Others: 0 (deterministic or terminal)
- H ≈ (5/16)×1.92 + (2/16)×1.0 ≈ 0.725 bits
- K = 1 - 0.725/log₂(10) = 1 - 0.725/3.32 = **0.78 (78%)**
- PPL_bound = 2^0.725 ≈ **1.65**
- Coverage: N=16, V·ln(V)=23 → coverage=0.7 (insufficient!)

### 2.4 What the Model Should Predict

For context "the king ruled the":
- Last word = "the". T["the"] = [0, 0.4, 0.2, 0, 0, 0.2, 0, 0, 0.2, 0]
- Bigram says: next = king(40%), queen(20%), sky(20%), grass(20%)
- But context "king ruled the" should favor **kingdom** — which has P=0 in bigrams!

This reveals the **bigram bottleneck**: the model only knows P(next|prev=the). It doesn't know that "king ruled the" → kingdom. That requires at least trigram context.

### 2.5 Where the Model Succeeds and Fails

**Succeeds** (at scale, 860K lines):
- "the king ruled the" → kingdom at rank 1 (attention captures multi-word context)
- Semantic patterns emerge from SVD (similar words cluster)

**Fails**:
- PPL remains orders of magnitude above PPL_bound
- Function words (the, of, and) dominate output
- Compositional meaning not captured

**Connection to experiments**: At 860K lines, v6 found kingdom=1 because cosine similarity between context embedding and "kingdom" was high (they share SVD neighbors). But PPL=42K vs PPL_bound=157 — a 270x gap.

## Part 3: Generalization Analysis

### 3.1 When the Model Provably Works

**Theorem (Block-Diagonal Transitions)**: If T is block-diagonal (topics are separable), then SVD spectral circles exactly match topic clusters, and attention within each block is optimal.

**Theorem (High Knowledge)**: If K → 1 (language is nearly deterministic), PPL_bound → 1, and even a simple bigram model achieves low PPL. The statistical transformer, using bigram transitions, will approach PPL_bound.

**Theorem (Sufficient Coverage)**: If N > V · ln(V), SVD vectors converge to their population values (by matrix concentration inequalities). The model's predictions stabilize.

### 3.2 When the Model Provably Fails

**Random Text (K=0)**: All S_i are equal (no spectral gap), d=V (no compression possible), PPL=V. The model correctly detects "no knowledge."

**Insufficient Data (N < V·ln(V))**: SVD vectors are unstable. Small perturbations in data cause large changes in embeddings. The model warns but proceeds.

**Long-Range Dependencies**: "The man who the woman saw left" — the subject "man" is far from "left." Bigram T doesn't capture this. Context_len=8 helps but is insufficient for deeply nested structures.

**Compositional Semantics**: "not hot" should predict "cold" but bigram of "not"→"hot" doesn't encode negation. This requires learned compositional operations that statistics alone cannot provide.

### 3.3 The Fundamental Gap

A learned transformer captures `P(next | context_1, ..., context_n)` through gradient descent over many examples. Our model captures at most `P(next | prev)` through bigram transitions, enhanced by cosine similarity via SVD embeddings.

The gap = everything that backpropagation adds beyond one-pass statistics:
- Multi-word pattern recognition
- Compositional operations
- Long-range dependencies
- Task-specific optimization

This gap is **irreducible** without some form of iterative optimization.

### 3.4 Data Structure Requirements

For the model to extract ANY knowledge:
1. `T ≠ uniform` (some word pairs are more likely than others)
2. `N > V · ln(V)` (sufficient coverage for stable SVD)
3. Spectral gap exists in PPMI (d << V meaningful components)
4. `K > 0` (language has predictable structure)

These are **necessary** conditions. **Sufficient** conditions for good PPL also require that the attention mechanism effectively combines context — which depends on Wq/Wk quality.

## Part 4: Analyzing Existing Transformers

### 4.1 Spectral Interpretation Protocol

Given a trained transformer model T with weights Wq, Wk per layer:

**Step 1**: From the same training corpus, compute our statistical quantities:
- PPMI matrix → SVD → E (embeddings)
- Bigram T → M_embed = E.T @ T @ E → SVD → U_m, S_m, V_m (statistical spectral circles)

**Step 2**: From the trained model, extract:
- Wq_learned, Wk_learned for each layer
- Compute SVD(Wq_learned @ Wk_learned.T) → U_learned, S_learned, V_learned

**Step 3**: Measure alignment:
```
alignment_k = |cos(u_stat_k, u_learned_k)|    for each spectral circle k
total_alignment = Σ (S_learned_k × alignment_k) / Σ S_learned_k
```

**Interpretation**:
- `total_alignment ≈ 1`: The transformer mostly rediscovered corpus statistics. Statistical initialization would have saved training time.
- `total_alignment ≈ 0`: The transformer learned something fundamentally different from bigram statistics. Backpropagation adds substantial value.
- In practice, expect 0.3-0.7 — partial alignment.

### 4.2 Practical Applications

**Weight Initialization**: Initialize `Wq = U_m @ diag(√S_m)` before training. Research shows this can reduce training time by 10-30% for small models.

**Model Compression**: Compute alignment per spectral circle. Circles with low alignment AND low S_learned can be pruned — they're neither statistically important nor learned to be useful.

**Interpretability**: Each statistical spectral circle corresponds to identifiable linguistic patterns (syntax, semantics, topic). By measuring which circles a trained model amplifies vs dampens, we can understand what the model "focuses on."

**Architecture Search**: The data's spectral structure (spectral gap, cumulative variance curve) suggests:
- `d_recommended = d at var_target=0.3` (30% cumulative variance)
- `n_heads_recommended = spectral_gap_index`
- `L_recommended = ceil(log₂(H_max/H))`

These can serve as starting points for hyperparameter search.

### 4.3 What Transformers Learn Beyond Statistics

The gap between statistical and learned spectral circles reveals:
1. **Positional patterns**: Trained attention heads develop position-specific behaviors that pure statistics miss.
2. **Compositional operations**: "king - man + woman = queen" emerges from gradient-based optimization, not from bigram statistics.
3. **Task adaptation**: Fine-tuned models develop task-specific circles that have no statistical counterpart.
4. **Error correction**: Backpropagation adjusts weights to minimize prediction error — a feedback loop that one-pass statistics cannot replicate.

### 4.4 The Unifying View

Both statistical and trained transformers operate in the same mathematical framework:
- Embeddings define a vector space
- Attention computes weighted combinations based on Q/K similarity
- FFN provides non-linear transformation
- LayerNorm stabilizes representations

The difference is **only** in how the weight matrices are determined:
- Statistical: one-pass SVD on corpus statistics
- Trained: iterative gradient descent on prediction loss

This means: **any improvement in statistical weight estimation directly translates to better transformer initialization, compression, and interpretability.** The two approaches are not competing — they are complementary views of the same mathematical structure.
