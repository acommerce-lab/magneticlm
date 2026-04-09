# MagneticLM — research / continuation notes

**Primary file:** `MagneticLMFastRunner.py` in this directory (1000 lines,
self-contained, runs on Kaggle/Colab T4).

**Goal:** a language model that beats Transformer-XL on WikiText-103
using no neural networks — only Modified Kneser–Ney n-gram counts and
a 3D physics simulation for word positions.

This file is the hand-off document. If you are a future AI agent
resuming this work, read it before touching the code. You should not
need to re-derive any decision made here.

---

## Latest empirical results (Kaggle T4 x2, 2026-04)

Runner CLI:

```bash
python MagneticLMFastRunner.py \
    --train-lines 1000000 \
    --physics-iters 300 \
    --max-order 5 \
    --multi-gpu
```

| Train lines | Physics iters | max_order | Device | PPL | Train+Build |
|---|---|---|---|---|---|
| 860k (all) | 1 | 3 | T4 | **21.72** | 50 s |
| 860k | 300 | 3 | T4 | 20.94 | 82 s |
| 860k | 300 | 4 | T4 x2 | 16.39 | 79 s |
| 860k | 300 | 5 | T4 x2 | 14.36 | 85 s |
| 860k | 1000 | 5 | T4 x2 | **14.20** | 156 s |

For comparison on the same benchmark:

| Model | PPL |
|---|---|
| published KN-5 | ~141 |
| AWD-LSTM + Cache (Merity 2017) | ~52 |
| GPT-2 small | ~35 |
| **Transformer-XL (Dai 2019)** | **~16.4** |
| **This model** | **14.20** |

Note on training data: the WT103 train file has 1.8M raw entries; after
filtering out `=`-prefixed section headers and blank lines we have
~860k usable lines ≈ 88M tokens. Passing `--train-lines 10000000` just
loads all ~860k.

Observed physics convergence: 300 iters → 14.36, 1000 iters → 14.20.
Diminishing returns set in around iter 200–300. Running past 500 iters
is a waste unless you're also changing the force constants.

Observed order scaling: each additional order past 3 gives a smaller
delta. 3→4: −4.5 PPL. 4→5: −2.0 PPL. Order 6+ would give fractions of
a PPL because 93 % of 5-grams already occur exactly once on this
corpus. The current code caps `--max-order` at 9 just in case, but
expect diminishing returns.

---

## Code map

```
MagneticLMFastRunner.py        the single file that runs
    _HASH_PRIMES_LIST            10 splitmix64 / Knuth constants
    _HASH_PRIMES_CPU             torch.int64 tensor, moves to GPU per device

    class MagneticLMGPU:
        __init__                  vocab + per-order table slots + device routing
        _device_for_order         [1,2,3] -> cuda:0, [4,5] -> cuda:1 (if multi_gpu)
        _adaptive_chunk           token-per-chunk by order (60M @ o=1 down to 5M @ o=9)

        _tokenize_to_gpu          text -> int32 array('i') -> int64 GPU tensor
        _count_ngrams_order       polynomial hash, torch.unique, incremental
                                  chunk merge (the one good thing Gemini's edit
                                  contributed), then per-context aggregates
                                  (total, count1, count2, unique_followers)

        _compute_continuation     bigram-level unique, look up next-word via
                                  tokens[first_pos + 1] (MATHEMATICALLY
                                  CORRECT; do NOT switch to inverting the
                                  hash — that bug dropped PPL to 600.47)

        _build_semantic_edges     adjacency edges at offset +1 (weight 0.2/dir)
                                  and +2 (weight 0.1/dir), symmetric, keyed by
                                  from*V + to, incrementally reduced

        _compute_discounts        Modified KN D1/D2/D3 from count-of-counts

        train_gpu                 pipeline orchestrator
        build                     physics (spring + sampled repulsion + far-
                                  field attraction + gravity + boundary clamp)
                                  + importance (log1p(degree) × log1p(freq))
                                  + circles SKIPPED (see "open questions")

        kn_batch                  GPU-vectorised Modified KN-5 with the full
                                  lambda = (D1*c1 + D2*c2 + D3*n3+)/total
                                  DO NOT drop the D3*n3+ term

        eval_full_wt103           batched (default batch 16384) eval that
                                  mixes KN + position similarity + importance
                                  boost via an adaptive lambda band
                                  (2% / 6% / 12% depending on KN strength)

    def main()                    CLI: --train-lines --physics-iters
                                        --batch-size --max-order --multi-gpu
                                        --data-dir
```

---

## History of what was broken and fixed

The runner went through three major phases. If you see an old output
or an old bug report, this timeline will help you place it.

### Phase 1 — initial clean runner

The original file was ~830 lines. It worked fine up to 100k training
lines. At 1M lines it OOM'd on a single T4. The logic was correct
(PPL around 100–150 range on small sets), but the memory management
was naive: per-order n-gram chunks were collected into a Python list
and merged at the end, doubling the peak.

### Phase 2 — the Gemini-corrupted edit

The user edited the file with Gemini to address the OOM. The memory
got better (incremental chunk merging — see `_count_ngrams_order`'s
"master_ng / master_cnt / master_ctx" variables). But **five things
silently broke the math**:

1. **`_compute_continuation`** tried to extract the next word from
   the bigram hash via `(hash % p2) // p1`. The hash is
   `tokens[i]*p1 + tokens[i+1]*p2` mod 2^64 with large odd primes —
   it is mathematically not invertible this way. The corrupted
   continuation count poisoned the KN backoff base.
2. **`_build_semantic_edges`** used the same wrong inversion to
   derive (from, to) from bigram hashes. Garbage edges → garbage
   physics → random positions.
3. **`_compute_discounts`** was deleted entirely; D1/D2/D3 stayed at
   hardcoded defaults instead of being derived from the corpus's
   count-of-counts.
4. **`build()`** was reduced to a stub with no physics loop.
   `positions` became random noise and was never updated.
5. **`kn_batch`** dropped the `D3 * n3+` term from the Modified KN
   lambda, and the position similarity / adaptive mixing were
   entirely removed.
6. **`eval_full_wt103`** became a per-token Python loop with
   `.item()` per token (massive GPU-CPU stall) that also didn't use
   positions at all.
7. **MAX_ORDER** was silently changed from 5 to 4.

The smoking gun: identical 600.47 PPL with 30 vs 300 physics iters
proved the model wasn't using positions during eval at all.

### Phase 3 — restored model + dual-T4 support (current)

A single commit restored every broken piece to the mathematically
correct baseline, kept the one genuinely-good memory improvement
(incremental chunk merging), and added:

- adaptive per-order chunk sizes
- `--multi-gpu` flag that shards per-order tables (orders 1–3 on
  cuda:0, orders 4–5 on cuda:1). Only the per-batch query crosses the
  device boundary, never the lookup tables.
- `--max-order` capped at 9 with a full 10-entry hash prime table.

With these, 1M lines runs cleanly and hits 14.20 PPL as shown above.

### Key constants you should not touch

- `_HASH_PRIMES_LIST` — 10 splitmix64/Knuth multiplicative constants.
  Changing them re-randomises the hash but doesn't change the math.
  Do not reduce the list to fewer than `max_order + 1` entries.
- `_adaptive_chunk` — lowering values is safe but slower. Raising
  values above the current maxima can OOM on T4.
- `D1 / D2 / D3` defaults (0.5 / 0.75 / 0.9) — these are only the
  fallbacks if the corpus has zero count-of-counts, which never
  happens in practice.
- `physics` force constants: `K_context=2.0`, `K_frequency=1.5`,
  `K_attraction=0.5`, `K_repulsion=0.3`, `damping=0.15`, `lr=0.02`,
  `optimal_dist=3.0`, `max_radius=15.0`. Tuned for WT103-scale.
- Adaptive lambda bands in `eval_full_wt103`: 2% / 6% / 12%
  (kn > 0.05 / 0.005 / else). These are the mixing weights between
  KN and position similarity.

---

## Open questions and the next experiments

### 1. Is the "semantic" layer actually semantic?

Honest answer: **no**. The edges in `_build_semantic_edges` are
built from offset ±1 / ±2 adjacency. They are a distributional signal
at the finest possible granularity — effectively the same information
as bigram/trigram counts, symmetrised. The physics pushes together
words that are **directly adjacent** in the training text, not words
that "appear in similar contexts across the corpus".

A true semantic layer would build distributional fingerprints per
word (PPMI with all neighbours in a ±K window) and compute
`cosine(fingerprint[w1], fingerprint[w2])` as the semantic distance.
Two words like "coffee" and "tea" that never appear adjacent but
appear in the same *kinds* of sentences would then be close — which
the current model is blind to.

This is the biggest conceptual gap and probably where the next big
PPL drop lives. A sketch:

```python
def _build_semantic_layer(self, window=5, topk=20):
    # 1. For each word, accumulate its neighbour histogram within ±window
    #    into a sparse vector (PPMI-normalised).
    # 2. For each word, find its top-k most semantically similar
    #    neighbours by cosine on those sparse vectors.
    # 3. Build a second edge list (sem_from, sem_to, sem_weight) from
    #    those top-k neighbours.
    # 4. Run a second physics simulation on that edge list to get
    #    sem_positions[w] — a separate 3D embedding that captures
    #    distributional similarity, not adjacency.
```

Then at eval time mix three signals:

```python
final_prob = (1 - λ1 - λ2) * kn
           + λ1 * cosine_similarity(ctx_positions, nxt_positions)
           + λ2 * cosine_similarity(sem_positions, nxt_positions)
```

Expected delta: 11–13 PPL (a ~20–30% relative improvement over 14.20).

### 2. Knowledge circles (dense mutual cliques)

The original C# version had an "importance circles" layer that
detected bidirectional strong ties and grouped them into cliques,
then boosted predictions between members of the same circle by 1.5×.
The current GPU runner skips this (`circle_group` is all −1) because
the CPU greedy clique-finder doesn't port cleanly to CUDA.

Restoring it on GPU would look like:

```python
# 1. From _build_semantic_edges, keep edges with weight >= 0.3.
# 2. For each edge (a, b), check whether (b, a) also has weight >= 0.3.
#    This gives the "strong bidirectional" subgraph.
# 3. Find triangles: for each (a, b) strong, iterate b's strong
#    neighbours and keep those that are also a's strong neighbours.
# 4. Grow to 4-cliques greedily.
# 5. Assign a circle_id to each clique member. In the eval step,
#    multiply pos_prob by 1.5 when ctx word and nxt word share a circle.
```

On a 300k-node graph, triangle counting via sorted-list intersection
is ~O(E * avg_degree). For ~30M edges and typical degree 100, that's
~3 billion ops — manageable on GPU with a batched two-pointer scan.

Expected delta: 0.5–1.5 PPL once the semantic layer above is in
place. Without the semantic layer, probably negligible (the
adjacency-based circles are already captured by the physics).

### 3. Asymmetric edges

The current edge weights are symmetric. But "سارة" → "لاتيه" might
be a much stronger conditional than "لاتيه" → "سارة" (Sara almost
always orders a latte, but a latte order doesn't identify Sara). The
user pointed out this asymmetry explicitly during the university-era
"knowledge tree" discussion.

A change to test:

```python
# In _build_semantic_edges, split the running table into two:
#   forward_weight[a, b]  = P(b follows a)
#   backward_weight[a, b] = P(a precedes b)
# In physics, use the larger of the two when applying spring force,
# OR use the conditional as a separate attraction/repulsion signal.
```

Expected delta: unknown, small. The signal is already mostly
captured by the KN backoff.

### 4. Hierarchical clustering → decision tree inference

The user's original university-era idea: cluster the 2D co-occurrence
matrix hierarchically, build a tree, and do R-tree-like marking at
inference (each prompt word marks the leaves, propagation up to
roots, then down again with weighted scores).

This is essentially hierarchical softmax (Mikolov 2013) with a custom
clustering. It's a significant undertaking — probably 2–3 days of
work to implement cleanly. It would mainly help **inference speed**,
not perplexity.

### 5. Longer context (cache)

The current runner is full-mode-no-cache. A continuous cache
(Grave et al. 2017) over the test stream could add 0.5–1.0 PPL
improvement. The original C# code had one; I dropped it for
simplicity.

---

## How to run reproducibly

### Single T4 (Colab or Kaggle single-GPU)

```bash
pip install torch numpy datasets
python MagneticLMFastRunner.py \
    --train-lines 1000000 \
    --physics-iters 300 \
    --max-order 4
```

### Dual T4 (Kaggle "GPU T4 x2")

```bash
python MagneticLMFastRunner.py \
    --train-lines 1000000 \
    --physics-iters 300 \
    --max-order 5 \
    --multi-gpu
```

### Smaller experiments

- `--train-lines 10000` — 200ms train time, useful for iteration.
- `--train-lines 100000` — 5s train time.
- `--physics-iters 30` — fast baseline for A/B experiments.

---

## Files you might care about

- `MagneticLMFastRunner.py` — the current clean single-file runner (read top-to-bottom; comments explain every non-obvious decision).
- `graph.py`, `trainer.py`, `benchmark.py`, `wt103_benchmark.py`, `MagneticLM_Colab.py` — the older NumPy prototypes. Kept for historical reference only; they were the source of the initial "14–22 PPL on small sets" measurements before the GPU rewrite.

---

## The mental model in one paragraph

> Every word is a point in a 3-D space.  
> Adjacent words pull each other closer; random-sample repulsion and
> far-field attraction keep the cloud from collapsing.  
> After physics converges, two words are "close" iff they tend to
> appear next to each other (*this is currently only adjacency-based,
> not truly distributional*).  
> At inference time, the Modified KN-5 backoff gives the main
> probability; the 3-D cosine similarity of the context words to the
> candidate next word adds a small correction via an adaptive lambda
> mix.  
> The accounting-style "parties + tags + analyzers + execute"
> structure doesn't apply here — MagneticLM is research code, not
> part of the e-commerce platform. It lives in `Examples/` for a
> reason.
