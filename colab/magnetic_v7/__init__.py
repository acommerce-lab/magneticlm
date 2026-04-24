"""MagneticLM v7 — Interpretable Statistical Transformer.

Transformer-like architecture without backpropagation:
  Embedding (SVD on PPMI)
  → Q/K Projections (from bigram transitions)
  → Multi-Head Causal Self-Attention
  → Residual + LayerNorm
  → Multi-Layer stacking
  → Output Head (cosine scoring)

All weights derived from corpus statistics. No gradient descent.
Every parameter is traceable and interpretable.
"""
__version__ = "7.0.0"
