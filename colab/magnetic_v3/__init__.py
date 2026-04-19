"""magnetic_v3 — dual-map language model with capacity-based isolation.

Layers:
  1. Statistical foundation: unigram, bigram (KN), PPMI, entropy, capacity
  2. Contextual map: A -> B direct succession (CSR)
  3. Semantic map: capacity-limited weighted edges + force dynamics
  4. Inference: mixture scoring + spreading activation with binary transfer

Pluggable strategies (registries):
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
from .forces import register_force, FORCES
from .transfer import register_transfer, TRANSFERS
from .pulse_trainer import train_parallel
from .inference import InferenceEngine
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
    "register_force",
    "FORCES",
    "register_transfer",
    "TRANSFERS",
    "train_parallel",
    "InferenceEngine",
    "run_full_eval",
]
