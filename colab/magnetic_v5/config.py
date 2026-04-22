"""v5 Config — KN + distributional substitution + adoption. No wave physics."""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    dataset: str = "wikitext-103"
    data_dir: str = "./data"
    max_train_lines: int = -1
    max_valid_lines: int = -1
    max_vocab: int = 50000
    min_count: int = 2
    unk_token: str = "<unk>"

    # Statistics
    stat_window: int = 5
    bigram_window: int = 1
    min_pair_count: int = 2

    # KN-5gram
    kn_max_order: int = 5

    # Distributional substitution tables
    sub_k: int = 30               # top-K substitutes per word
    min_ppmi: float = 0.5         # PPMI threshold for context edges
    sub_block_size: int = 1000    # GPU block size for B@B.T

    # Scoring mix
    stat_cache_window: int = 3000
    stat_cache_lambda: float = 0.12
    adoption_lambda: float = 0.08

    # Eval
    eval_ppl: bool = True
    eval_hit_rate: bool = True
    eval_ood_cloze: bool = True
    eval_generation: bool = False
    eval_max_tokens: int = 200000
    gen_length: int = 30
    gen_samples: int = 3
    gen_top_k: int = 40
    gen_temperature: float = 1.0
    mask_unk: bool = True

    # Hardware
    device: str = "auto"
    multi_gpu: bool = True
    num_workers: int = -1
    mem_log_every: int = 100
    mem_warn_percent: float = 88.0

    # Output
    save_dir: str = "./outputs"
    seed: int = 42


def config_from_args(args) -> "Config":
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def add_cli_args(parser):
    cfg = Config()
    for k, v in cfg.__dict__.items():
        t = type(v)
        if t is bool:
            parser.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes", "on"))
        elif t in (int, float, str):
            parser.add_argument(f"--{k}", type=t)
    return parser
