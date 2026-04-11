# pulse/config.py
#
# All hyperparameters for the dual-map MagneticLM.
#
#   1. Contextual map: direct word succession (A → B)
#   2. Semantic map: iterative pulse training with 5-force dynamics
#   3. Inference: direct children + adopted children + spreading activation

from dataclasses import dataclass


@dataclass
class PulseConfig:
    # =================================================================
    # Contextual map (A → B from direct succession)
    # =================================================================
    max_ctx_children: int = 500

    # =================================================================
    # Semantic map: pulse training
    # =================================================================

    # Window for co-occurrence edges
    semantic_window: int = 2

    # Neighbor lookup during reward/penalty
    neighbor_top_k: int = 20

    # Penalty threshold (only penalize edges above this weight)
    penalty_threshold: float = 0.5

    # Transitive propagation
    transitive_interval: int = 200
    transitive_top_k: int = 8
    transitive_decay: float = 0.5

    # Edge threshold
    semantic_threshold: float = 0.1

    # =================================================================
    # Five forces (dynamic reward/penalty computation)
    # =================================================================

    # 1. Spring: attraction for co-occurring words.
    #    Stronger pull when words are far apart (low weight).
    K_spring: float = 2.0

    # 2. Repulsion: high-degree nodes push each other apart.
    #    Common words naturally get pushed away from everything.
    K_repulsion: float = 0.3

    # 3. Far-field attraction: prevents graph fragmentation.
    #    Small pull toward optimal_weight for weak edges.
    K_attraction: float = 0.5
    optimal_weight: float = 3.0  # target weight for healthy edges

    # 4. Gravity: pulls all weights toward zero.
    #    Prevents unbounded growth.
    K_gravity: float = 0.01

    # 5. Damping: smooths velocity to prevent oscillation.
    #    0 = no damping (unstable), 1 = full damping (no movement).
    damping: float = 0.15

    # Integration learning rate (how much velocity moves the weight)
    force_lr: float = 0.02

    # Legacy fixed amounts (ONLY used if forces are disabled for
    # comparison experiments — normally the forces compute these)
    reward_amount: float = 0.05
    penalty_amount: float = 0.02
    window_weight_d1: float = 0.1
    window_weight_d2: float = 0.05

    # =================================================================
    # Inference (dual-map candidate selection + scoring)
    # =================================================================

    # Adoption: borrow children from semantic neighbors
    adoption_neighbors: int = 10
    adoption_min_weight: float = 0.2

    # Scoring mixture:
    # P = alpha * P_direct + beta * P_adopt + gamma * P_unigram
    alpha_direct: float = 0.7
    beta_adopt: float = 0.2
    gamma_unigram: float = 0.1

    # Spreading activation (at inference time)
    spreading_iters: int = 10
    spreading_damping: float = 0.85

    # =================================================================
    # Evaluation
    # =================================================================
    eval_batch_size: int = 16384

    # =================================================================
    # Multi-GPU
    # =================================================================
    multi_gpu: bool = False

    # =================================================================
    # Determinism
    # =================================================================
    seed: int = 42
