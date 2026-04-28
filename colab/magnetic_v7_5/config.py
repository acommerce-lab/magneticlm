"""v7.5 Config."""
from dataclasses import dataclass

@dataclass
class Config:
    dataset: str = "wikitext-103"
    data_dir: str = "./data"
    max_train_lines: int = -1
    max_valid_lines: int = -1
    max_vocab: int = 50000
    min_count: int = 2
    unk_token: str = "<unk>"
    stat_window: int = 5
    bigram_window: int = 1
    min_pair_count: int = 2
    cost_smoothing: float = 1.0
    cost_ceiling: float = 20.0
    max_out_edges: int = 200
    var_target: float = 0.3
    refine: bool = False
    min_ppmi: float = 0.5
    context_len: int = 32
    pos_decay: float = 0.1
    eval_ppl: bool = True
    eval_ood_cloze: bool = True
    eval_max_tokens: int = 5000
    mask_unk: bool = True
    device: str = "auto"
    multi_gpu: bool = True
    num_workers: int = -1
    mem_log_every: int = 100
    mem_warn_percent: float = 88.0
    save_dir: str = "./outputs"
    cache_dir: str = "./cache"
    seed: int = 42

def config_from_args(args):
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
            parser.add_argument(f"--{k}", type=lambda s: s.lower() in ("1","true","yes","on"))
        elif t in (int, float, str):
            parser.add_argument(f"--{k}", type=t)
    return parser
