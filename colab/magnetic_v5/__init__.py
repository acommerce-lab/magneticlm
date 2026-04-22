"""MagneticLM v5 — Matrix-native KN + distributional adoption.

No wave physics, no graph propagation. Pure matrix operations:
  1. KN-5gram statistical base (GPU hashed tables + searchsorted)
  2. Directional substitution tables (B @ B.T overlap in blocks)
  3. Decay cache (recent token boost)
  4. Adoption scoring (substitute children as creative candidates)

All computation via sparse/dense matrix ops on GPU.
"""
__version__ = "5.0.0"
