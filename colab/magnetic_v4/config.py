"""Configuration dataclass — single source of truth for all knobs.

Design: ONE config object flows through the pipeline. No hidden defaults
scattered in functions. Adding a knob = adding a field here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset: str = "wikitext-103"
    data_dir: str = "./data"
    max_train_lines: int = -1
    max_valid_lines: int = -1
    max_vocab: int = 50000
    min_count: int = 2
    unk_token: str = "<unk>"

    # ------------------------------------------------------------------
    # Statistics layer
    # ------------------------------------------------------------------
    stat_window: int = 5            # co-occurrence window (symmetric)
    bigram_window: int = 1          # direct-succession window
    min_pair_count: int = 2         # drop pairs with fewer observations
    cost_smoothing: float = 1.0     # Laplace-style pseudo-count for cost calc

    # ------------------------------------------------------------------
    # Graph: directional adjacency
    # ------------------------------------------------------------------
    max_out_edges: int = 200        # per-node cap on edges (ranked by PPMI)
    min_ppmi: float = 0.5           # drop edges with PPMI below this (pure noise)
    cost_ceiling: float = 20.0      # clip costs above this (sanity)

    # ------------------------------------------------------------------
    # Wave propagation
    # ------------------------------------------------------------------
    wave_iters: int = 5             # propagation steps per inference
    wave_damping: float = 0.85      # per-step attenuation (stability)
    wave_teleport: float = 0.15     # fraction that stays at original impulse (PPR-style)
    reflection_coef: float = 0.3    # how much signal flips real<->imag at deep nodes
    reflection_depth: int = 3       # after N steps, enable reflection

    # ------------------------------------------------------------------
    # Scoring (how to convert complex field to P(next token))
    # ------------------------------------------------------------------
    scoring_method: str = "projection"     # projection | magnitude | hybrid
    context_weight: float = 0.85           # weight on Re(z) — syntactic channel
    concept_weight: float = 0.15           # weight on Im(z) — semantic channel
    mask_unk_in_eval: bool = True          # skip <unk> in generation/OOD

    # ------------------------------------------------------------------
    # KN-5gram statistical base
    # ------------------------------------------------------------------
    kn_max_order: int = 5               # up to 5-gram
    kn_chunk_size: int = 500_000        # sentences per build chunk (memory bound)

    # ------------------------------------------------------------------
    # Statistical cache (decay — boosts recent tokens)
    # ------------------------------------------------------------------
    stat_cache_window: int = 3000       # tokens to remember
    stat_cache_lambda: float = 0.15     # mix weight with KN base

    # ------------------------------------------------------------------
    # Conceptual cache (PPMI trigger — boosts semantic neighbors)
    # ------------------------------------------------------------------
    concept_cache_k: int = 30           # top-K PPMI neighbors per context word
    concept_cache_lambda: float = 0.10  # mix weight for concept boost
    eval_ppl: bool = True
    eval_hit_rate: bool = True
    eval_ood_cloze: bool = True
    eval_generation: bool = False
    eval_max_tokens: int = 200000
    eval_batch_size: int = 512
    gen_length: int = 30
    gen_samples: int = 3
    gen_top_k: int = 40
    gen_temperature: float = 1.0

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    device: str = "auto"                   # auto | cuda | cuda:0 | cpu
    multi_gpu: bool = True                 # spread stats collection across GPUs
    num_workers: int = -1                  # -1 = auto (cpu_count - 1, capped)
    memmap_threshold_gb: float = 2.0       # auto-disk when encoded corpus > this

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    mem_log_every: int = 100               # log memory every N operations
    mem_warn_percent: float = 88.0         # trigger GC above this system-memory %

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    save_dir: str = "./outputs"
    save_state: bool = True

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------
    seed: int = 42


def config_from_args(args) -> Config:
    """Merge CLI namespace into a Config instance."""
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def add_cli_args(parser):
    """Auto-generate CLI flags from Config fields."""
    cfg = Config()
    for k, v in cfg.__dict__.items():
        t = type(v)
        if t is bool:
            parser.add_argument(
                f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes", "on")
            )
        elif t in (int, float, str):
            parser.add_argument(f"--{k}", type=t)
    return parser
