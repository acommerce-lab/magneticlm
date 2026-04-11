# pulse - Dual-Map MagneticLM with Pulse Training
#
# Two maps, nothing else:
#   Contextual (mandatory): word A → word B from direct succession
#   Semantic (advisory): word A ≈ word C from iterative reward/penalty
#
# Inference: candidates = ctx_children(current)
#          + ctx_children(semantic_neighbors(current))

from .config import PulseConfig
from .semantic_trainer import SemanticMap
from .contextual_map import ContextualMap
from .evaluator import DualMapEvaluator

__all__ = [
    "PulseConfig",
    "SemanticMap",
    "ContextualMap",
    "DualMapEvaluator",
]
