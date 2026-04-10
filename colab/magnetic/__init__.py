# magnetic - MagneticLM research package
#
# A clean modular reimplementation of the original C# MagneticLM
# architecture on GPU. Splits the monolithic runner into small,
# swappable components so each idea can be tested and evolved
# independently.
#
# Core architecture (from the C# original, Generator.cs + WordGraph.cs):
#
#   1. Tokenizer + Vocabulary                 -> tokenizer.py
#   2. Modified Kneser-Ney n-gram tables      -> ngram.py
#   3. Semantic edges (window +/- K + PMI)    -> edges.py
#   4. Physics-based position embedding       -> physics.py
#   5. Spreading-activation inference engine  -> excitation.py
#   6. Multi-force generator (alpha*KN +      -> generator.py
#      beta*semantic + repulsion)
#   7. Evaluators (WT103 PPL, cloze, OOD)     -> evaluator.py
#
# Extensions kept behind config flags:
#   - Higher embedding dimension (default 16)
#   - Multi-hop spreading (default 2, C# had 1)
#   - PMI weighting of semantic edges (default on)
#   - Multi-GPU sharding of high-order n-grams
#
# The runner scripts live one level up in colab/:
#   - train_magnetic.py     full training pipeline + WT103 eval
#   - generate_magnetic.py  text generation from a prompt
#   - ood_magnetic.py       out-of-distribution cloze evaluation

from .config import MagneticConfig
from .tokenizer import tokenize, Vocabulary
from .data import load_wt103_lines, ensure_wt103
from .ngram import NgramTables
from .edges import EdgeBuilder, EdgeSet
from .physics import PhysicsSimulator
from .excitation import ExcitationEngine
from .model import MagneticModel
from .generator import MagneticGenerator
from .evaluator import Evaluator

__all__ = [
    "MagneticConfig",
    "tokenize",
    "Vocabulary",
    "load_wt103_lines",
    "ensure_wt103",
    "NgramTables",
    "EdgeBuilder",
    "EdgeSet",
    "PhysicsSimulator",
    "ExcitationEngine",
    "MagneticModel",
    "MagneticGenerator",
    "Evaluator",
]
