"""MagneticLM v6 — Deterministic Attention over Statistical Graph.

Transformer-like architecture without backpropagation:
  Embedding (SVD on PPMI) → Multi-Head Attention (cosine) → KN Scoring.
All weights derived from corpus statistics. No gradient descent.
"""
__version__ = "6.0.0"
