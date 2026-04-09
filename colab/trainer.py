"""
MagneticLM Trainer - OPTIMIZED
Training as accounting entries with batch processing.
"""

from graph import WordGraph
from typing import List
import re

# Pre-compiled regex for fast tokenization
_SPLIT_PATTERN = re.compile(r'[.,;!?؟،"()\[\]{}]+')


def tokenize(sentence: str) -> List[str]:
    """Optimized tokenizer."""
    cleaned = _SPLIT_PATTERN.sub(' ', sentence.lower())
    return [w for w in cleaned.split() if len(w) > 0]


class Trainer:
    def __init__(self, graph: WordGraph):
        self.graph = graph
        self.sentences_trained = 0

    def train_sentence(self, sentence: str):
        words = tokenize(sentence)
        if len(words) < 2:
            return

        g = self.graph  # local reference for speed

        # === 1. Register n-grams ===
        for i in range(len(words)):
            node = g.get_or_create(words[i])
            node.freq += 1
            g.total_tokens += 1

            if i < len(words) - 1:
                next_word = words[i + 1]
                for order in range(1, min(g.max_order + 1, i + 2)):
                    context = tuple(words[i + 1 - order:i + 1])
                    g.add_ngram(context, next_word)

        # === 2. Semantic (depth ±2) + Reward/Penalty ===
        semantic_add = g.add_semantic  # local ref
        for i in range(len(words)):
            w = words[i]
            for d in (-2, -1, 1, 2):
                j = i + d
                if 0 <= j < len(words):
                    semantic_add(w, words[j], (1.0 if abs(d) <= 1 else 0.5) * 0.1)

            if i < len(words) - 1:
                actual = words[i + 1]
                neighbors = list(g.semantic.get(w, {}).items())[:20]
                for neighbor, wt in neighbors:
                    if neighbor == actual:
                        semantic_add(w, actual, 0.05)
                    elif wt > 0.5:
                        semantic_add(w, neighbor, -0.02)

        # === 3. Transitive propagation ===
        self.sentences_trained += 1
        if self.sentences_trained % 200 == 0:
            self._propagate(words)
        if self.sentences_trained % 1000 == 0:
            print(f"\r  Training: {self.sentences_trained:,} sentences...", end="", flush=True)

    def _propagate(self, words: List[str]):
        threshold = self.graph.semantic_threshold
        for word in words:
            neighbors = sorted(
                ((n, w) for n, w in self.graph.semantic.get(word, {}).items() if abs(w) >= threshold),
                key=lambda x: -abs(x[1])
            )[:8]
            for neighbor, w1 in neighbors:
                for trans, w2 in sorted(
                    ((n, w) for n, w in self.graph.semantic.get(neighbor, {}).items() if abs(w) >= threshold),
                    key=lambda x: -abs(x[1])
                )[:8]:
                    if trans == word: continue
                    tw = w1 * w2 * 0.5 * 0.01
                    if tw > 0.001:
                        self.graph.add_semantic(word, trans, tw)

    def train_batch(self, sentences):
        for s in sentences:
            self.train_sentence(s)
        if self.sentences_trained >= 1000:
            print()
