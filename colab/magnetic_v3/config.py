from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # =====================================================================
    # Pipeline toggles
    # =====================================================================
    run_stats: bool = True
    run_training: bool = True
    run_eval: bool = True
    load_stats_from: str = ""
    load_maps_from: str = ""

    # =====================================================================
    # Data
    # =====================================================================
    dataset: str = "wikitext-103"
    data_dir: str = "./data"
    max_train_lines: int = -1
    max_valid_lines: int = -1
    max_vocab: int = 50000
    min_count: int = 2
    unk_token: str = "<unk>"

    # =====================================================================
    # Statistical layer
    #   capacity_method: how to turn node-level stats into capacity C(n)
    #     - "entropy"         : C(n) = k_base * (1 - H(n)/H_max)
    #     - "log_inv_freq"    : C(n) = k_base * log(N / freq(n) + 1)
    #     - "uniform"         : C(n) = k_base
    #   k_base_method:
    #     - "median_neighbors": median |distinct_neighbors|
    #     - "fixed"           : k_base_fixed
    # =====================================================================
    stat_window: int = 5
    ppmi_min_count: int = 5
    ppmi_threshold: float = 1.0
    capacity_method: str = "entropy"
    k_base_method: str = "median_neighbors"
    k_base_fixed: int = 50
    capacity_multiplier: float = 1.0
    capacity_min: int = 4
    capacity_max: int = 2000

    # =====================================================================
    # Contextual map (A -> B direct succession, CSR on GPU)
    # =====================================================================
    max_ctx_children: int = 500

    # =====================================================================
    # Semantic map (concept edges)
    # =====================================================================
    semantic_window: int = 2
    init_from_ppmi: bool = True

    # Active forces (comma-separated registry keys)
    #   Available: spring, decay, damping
    active_forces: str = "spring,decay,damping"

    K_spring: float = 2.0
    K_decay: float = 0.001
    damping: float = 0.15
    force_lr: float = 0.02

    # =====================================================================
    # Transfer equation (binary decision at propagation)
    #   - "threshold_fixed": use theta_positive / theta_negative
    #   - "threshold_auto" : per-node mean +/- std
    #   - "sigmoid"        : continuous with soft gating
    #   - "linear"         : no gating (pass-through weight)
    # =====================================================================
    transfer_method: str = "threshold_auto"
    theta_positive: float = 0.5
    theta_negative: float = -0.3

    # =====================================================================
    # Inference scoring
    #   - "mixture": alpha*P_concept + beta*P_ppmi + gamma*P_kn
    #   - "concept_only", "stats_only"
    # =====================================================================
    scoring_method: str = "mixture"
    alpha_concept: float = 0.6
    beta_ppmi: float = 0.2
    gamma_kn: float = 0.2
    kn_discount: float = 0.75

    # Adoption: borrow children from semantic neighbors
    adoption_neighbors: int = 10
    adoption_min_weight: float = 0.2

    # Spreading activation during inference
    spreading_iters: int = 5
    spreading_damping: float = 0.85
    spreading_top_k: int = 64

    # =====================================================================
    # Training
    # =====================================================================
    epochs: int = 1
    pulse_workers: int = -1
    pulse_batch: int = 2000
    eval_batch_size: int = 16384

    # =====================================================================
    # Evaluation modes
    # =====================================================================
    eval_ppl: bool = True
    eval_hit_rate: bool = True
    eval_ood_cloze: bool = True
    eval_generation: bool = False
    gen_length: int = 50
    gen_samples: int = 5
    gen_top_k: int = 40
    gen_temperature: float = 1.0

    # =====================================================================
    # Resources (auto-detected at runtime if "auto" / -1)
    # =====================================================================
    device: str = "auto"
    multi_gpu: bool = True
    num_workers: int = -1

    # =====================================================================
    # Output
    # =====================================================================
    save_dir: str = "./outputs"
    save_stats: bool = True
    save_maps: bool = True
    log_every: int = 1000

    # =====================================================================
    # Determinism
    # =====================================================================
    seed: int = 42

    def force_list(self) -> List[str]:
        return [x.strip() for x in self.active_forces.split(",") if x.strip()]


def config_from_args(args) -> Config:
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
            parser.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif t in (int, float, str):
            parser.add_argument(f"--{k}", type=t)
    return parser
