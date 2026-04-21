"""MagneticLM v4 — Wave-physics language model.

Forward propagation carries a REAL (contextual) signal through the graph.
Backward propagation carries an IMAGINARY (conceptual) signal in reverse.
Interference at nodes produces glow centers naturally; reflection at
deep boundaries converts real→imaginary (conceptual leap) and vice versa.

No force-based training — the graph is built directly from directional
probabilities (cost = -log p), and inference is sparse matrix propagation.
"""

__version__ = "4.0.0"
