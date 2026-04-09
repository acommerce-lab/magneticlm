"""
MagneticLM Benchmark - OPTIMIZED with NumPy cache evaluation
"""

import math
import time
import numpy as np
from graph import WordGraph
from trainer import Trainer, tokenize
from typing import List


def compute_perplexity(graph: WordGraph, test_lines: List[str], mode: str = "magnetic") -> float:
    total_log_prob = 0.0
    total_tokens = 0
    floor = 1e-10

    # Cache as list (will be trimmed)
    cache = []

    for idx, line in enumerate(test_lines):
        if (idx + 1) % 500 == 0:
            print(f"\r    [{mode}] {idx+1}/{len(test_lines)}...", end="", flush=True)

        words = tokenize(line)
        if len(words) < 2:
            continue

        is_new = len(cache) < 20

        for i in range(1, len(words)):
            ctx_start = max(0, i - graph.max_order)
            context = tuple(words[ctx_start:i])

            if mode == "bigram":
                key = words[i - 1]
                total = graph.ngram_totals.get(key, 0)
                prob = graph.ngram_counts.get(key, {}).get(words[i], 0) / total if total > 0 else 0
            elif mode == "kn":
                prob = graph.kn_probability(context, words[i])
            else:  # cache or magnetic
                prob = graph.magnetic_probability(context, words[i], cache, is_new)

            prob = max(prob, floor)
            total_log_prob += math.log(prob)
            total_tokens += 1

            cache.append((words[i], context))
            if len(cache) > 4000:
                cache = cache[-4000:]

    print()
    if total_tokens == 0:
        return float('inf')
    return math.exp(-total_log_prob / total_tokens)


def run_benchmark(train_path: str, test_path: str, physics_iterations: int = 50):
    print("=" * 60)
    print("  MagneticLM v6: Modified KN + Physics + Cache + Semantic")
    print("  OPTIMIZED: NumPy vectorized physics")
    print("=" * 60)

    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"\nTrain: {len(train_lines):,} sentences")
    print(f"Test:  {len(test_lines):,} sentences")

    graph = WordGraph(max_ngram_order=5)
    trainer = Trainer(graph)

    t0 = time.time()
    trainer.train_batch(train_lines)
    graph.build_post_training(physics_iterations)
    train_time = time.time() - t0

    nodes, ngrams, semantic, circles = graph.get_stats()
    print(f"\nTraining: {train_time:.1f}s")
    print(f"Graph: {nodes:,} nodes, {ngrams:,} n-gram, {semantic:,} semantic, {circles:,} circles")
    print(f"Tokens: {graph.total_tokens:,}")

    print("\nComputing perplexity...")

    ppl_bi = compute_perplexity(graph, test_lines, "bigram")
    print(f"  Bigram:                    PPL = {ppl_bi:.1f}")

    ppl_kn = compute_perplexity(graph, test_lines, "kn")
    print(f"  Modified KN-5gram:         PPL = {ppl_kn:.1f}")

    ppl_mag = compute_perplexity(graph, test_lines, "magnetic")
    print(f"  MagneticLM (full):         PPL = {ppl_mag:.1f}")

    print(f"\n{'='*55}")
    print(f"  {'Model':<35}| Perplexity")
    print(f"  {'-'*35}+{'-'*12}")
    print(f"  {'Our Bigram':<35}| {ppl_bi:.1f}")
    print(f"  {'Our Modified KN-5gram':<35}| {ppl_kn:.1f}")
    print(f"  {'Our MagneticLM (full)':<35}| {ppl_mag:.1f}")
    print(f"  {'-'*35}+{'-'*12}")
    print(f"  {'5-gram KN (published)':<35}| ~141")
    print(f"  {'LSTM (Zaremba 2014)':<35}| ~78")
    print(f"  {'AWD-LSTM + Cache (Merity)':<35}| ~52")
    print(f"  {'Transformer-XL (Dai 2019)':<35}| ~54")
    print(f"{'='*55}")

    best = min(ppl_kn, ppl_mag)
    if best < 52: print("\n  >>> BEAT AWD-LSTM + Cache! <<<")
    elif best < 78: print("\n  >>> BEAT LSTM! <<<")
    elif best < 141: print("\n  BETTER than 5-gram KN!")

    return graph, ppl_bi, ppl_kn, ppl_mag


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run_benchmark(sys.argv[1], sys.argv[2],
                      physics_iterations=int(sys.argv[3]) if len(sys.argv) > 3 else 50)
    else:
        print("Usage: python benchmark.py <train.txt> <test.txt> [physics_iterations]")
