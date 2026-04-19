"""magnetic_v3 — dual-map language model with capacity-based isolation.

Layers:
  1. Statistical foundation: unigram, bigram (KN), PPMI, entropy, capacity
  2. Contextual map: A -> B direct succession (CSR)
  3. Semantic map: capacity-limited weighted edges + force dynamics
  4. Concept layer: many-to-many word<->concept, community detection
  5. Inference: 5-component mixture + persistent activation field + glow centers

Pluggable registries:
  - forces.FORCES       (spring, decay, damping, ...)
  - transfer.TRANSFERS  (threshold_fixed, threshold_auto, sigmoid, linear)
  - statistical_layer.CAPACITY_METHODS (entropy, log_inv_freq, uniform)
"""

from .config import Config, config_from_args, add_cli_args
from .resources import Resources, detect as detect_resources
from .tokenizer import Vocab, tokenize, build_vocab, encode_stream
from .data import load_dataset
from .statistical_layer import Statistics, build_statistics
from .contextual_map import ContextualMap
from .semantic_map import SemanticMapState, make_empty, init_from_ppmi
from .concepts import ConceptLayer, discover_concepts
from .forces import register_force, FORCES
from .transfer import register_transfer, TRANSFERS
from .pulse_trainer import train_parallel
from .inference import InferenceEngine, InferenceSession
from .evaluator import run_full_eval

__all__ = [
    "Config",
    "config_from_args",
    "add_cli_args",
    "Resources",
    "detect_resources",
    "Vocab",
    "tokenize",
    "build_vocab",
    "encode_stream",
    "load_dataset",
    "Statistics",
    "build_statistics",
    "ContextualMap",
    "SemanticMapState",
    "make_empty",
    "init_from_ppmi",
    "ConceptLayer",
    "discover_concepts",
    "register_force",
    "FORCES",
    "register_transfer",
    "TRANSFERS",
    "train_parallel",
    "InferenceEngine",
    "InferenceSession",
    "run_full_eval",
]
