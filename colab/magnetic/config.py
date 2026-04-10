# magnetic/config.py
#
# Single source of truth for every hyperparameter in the model.
# One dataclass, one flat namespace - so any experiment only needs to
# change the fields relevant to its question.
#
# All defaults are taken from the original C# Generator.cs and
# WordGraph.cs, except where explicitly noted as extensions.

from dataclasses import dataclass


@dataclass
class MagneticConfig:
    # =====================================================================
    # N-gram layer (Modified Kneser-Ney 5)
    # =====================================================================

    max_ngram_order: int = 5
    # Hash primes are defined in ngram.py; the table has 10 entries so the
    # max order is capped at 9 for safety.

    # =====================================================================
    # Semantic edge layer
    # =====================================================================

    # Window for building initial edges. The C# trainer used exactly
    # +/- 2 with weight 1.0 for abs(d)==1 and 0.5 for abs(d)==2, then
    # multiplied by 0.1. Keeping the same defaults.
    edge_window: int = 2
    edge_weight_schedule: tuple = (1.0, 0.5, 0.25, 0.12, 0.06)  # at dist 1..5
    edge_base_amount: float = 0.1

    # Threshold below which an edge is ignored during physics and
    # inference (kept in the tables, just filtered at use).
    semantic_threshold: float = 0.1

    # PMI re-weighting (extension, not in C#). Applied after the base
    # edge build. Each edge weight is multiplied by max(0, ppmi) /
    # ppmi_cap so rare+predictive pairs keep their strength and
    # frequent+uninformative pairs (e.g. "the", "and", "of") collapse
    # toward zero. Set use_pmi=False to reproduce the C# behaviour
    # exactly.
    use_pmi: bool = True
    pmi_cap: float = 6.0       # clip |PPMI| at this value before normalising
    pmi_floor: float = 0.0     # weights whose PPMI is below this get zeroed

    # Jaccard degree-ratio reweighting (the user's "relational
    # differentiation" idea). For each edge (a, b), multiply the weight
    # by min(degree_a, degree_b) / max(degree_a, degree_b). Edges
    # between nodes with similar connectivity keep their weight; edges
    # between a hub (degree 50k) and a specialist (degree 5) get
    # suppressed by 1000x. This is a fast O(E) approximation of full
    # Jaccard similarity |N(a)∩N(b)| / |N(a)∪N(b)| that captures the
    # key insight: common words should repel rare words because their
    # edge sets barely overlap.
    use_jaccard: bool = False

    # =====================================================================
    # Physics simulation
    # =====================================================================

    # Embedding dimension. C# used 3. The current GPU runner also used
    # 3. Experimentation showed that dim alone doesn't rescue OOD
    # performance (the bottleneck is the edge set, not the dim), but
    # higher dim doesn't hurt either as long as the init is scaled.
    # 16 is a reasonable middle ground.
    dim: int = 16

    physics_iters: int = 100

    # Force constants (tuned in C# for dim=3; init scaling in physics.py
    # keeps E[||x||] constant across dim so these do not need retuning).
    K_context: float = 2.0
    K_frequency: float = 1.5
    K_attraction: float = 0.5
    K_repulsion: float = 0.3
    damping: float = 0.15
    physics_lr: float = 0.02
    optimal_dist: float = 3.0
    max_radius: float = 15.0

    # Chunk size for the spring-force step. Needed to keep memory
    # bounded when the edge count * dim is large (D=64 on a 10M edge
    # graph OOMs on T4 without chunking).
    physics_edge_chunk: int = 1_000_000

    # Random repulsion sample per iteration.
    physics_sample_size: int = 200

    # =====================================================================
    # Inference: multi-force scoring (Generator.cs defaults)
    # =====================================================================

    alpha_contextual: float = 0.6   # weight on KN-5 score
    beta_semantic: float = 0.35     # weight on spreading-activation score
    repulsion_strength: float = 0.5 # initial repulsion of used words
    excitation_decay: float = 0.85  # per-step excitation damping
    repulsion_decay: float = 0.5    # per-step repulsion damping
    temperature: float = 0.3        # softmax temperature
    excitation_threshold: float = 0.05  # ignore excitation below this

    # Initial prompt excitation: linear ramp from ramp_low at the first
    # token to ramp_high at the last. The C# original used
    # 0.5 + 0.5 * (i / P).
    prompt_ramp_low: float = 0.5
    prompt_ramp_high: float = 1.0

    # Fraction of edge weight that transfers to a neighbour when the
    # prompt excites its semantic neighbours. C# used 0.3 for one hop.
    prompt_neighbor_share: float = 0.3

    # Spreading activation (extension). C# did one explicit hop during
    # prompt excitation and then relied on "excited words stay in the
    # candidate pool". We add an explicit multi-hop diffusion at prompt
    # time, bounded by spreading_hops and decayed by hop_decay per hop.
    # Set spreading_hops=1 to match C# exactly.
    spreading_hops: int = 2
    hop_decay: float = 0.3

    # Chunk size when scoring candidates over the full vocabulary in
    # the generator (keeps per-step memory bounded).
    candidate_chunk_size: int = 32768

    # For generation: only score the top-K KN candidates plus any
    # excited words. This eliminates the "ghost words" problem where
    # 100k rare words each get a tiny KN backoff probability and
    # collectively swamp the good candidates. Matches the C# approach
    # of `contextNeighbors.Union(excitedWords)` which was ~100-500
    # candidates. Set to 0 to disable (score all V).
    generation_topk: int = 500

    # =====================================================================
    # Multi-GPU
    # =====================================================================

    multi_gpu: bool = False
    # Which orders live on the primary device. Orders above this go on
    # the aux device. Matches the existing MagneticLMFastRunner policy.
    low_order_max: int = 3

    # =====================================================================
    # Determinism
    # =====================================================================

    seed: int = 42
