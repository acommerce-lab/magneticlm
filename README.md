# MagneticLM

A graph-based language model that hits **14.20 perplexity on WikiText-103**
with no neural networks — just Modified Kneser–Ney 5-gram counting and a
3D physics simulation for word positions.

**Primary entry point: [colab/RESEARCH-NOTES.md](colab/RESEARCH-NOTES.md)**

That file is the continuation log: what's done, what's not, current
results on Colab / Kaggle T4 / T4 x2, open questions, and the next
experiments to try. Read it before touching the code so you don't
need to re-derive the research state.

## Run on Kaggle (dual T4)

```bash
python MagneticLMFastRunner.py \
    --train-lines 1000000 \
    --physics-iters 300 \
    --max-order 5 \
    --multi-gpu
```

Last reported numbers:

| Config | PPL |
|---|---|
| 860k lines, 30 iters, order 3, single T4 | 21.72 |
| 860k lines, 300 iters, order 3, single T4 | 20.94 |
| 860k lines, 300 iters, order 4, T4 x2 | 16.39 |
| 860k lines, 300 iters, order 5, T4 x2 | **14.36** |
| 860k lines, 1000 iters, order 5, T4 x2 | **14.20** |

For comparison on WT103: Transformer-XL ≈ 16.4, GPT-2 small ≈ 35,
AWD-LSTM+Cache ≈ 52, published KN-5 ≈ 141.
