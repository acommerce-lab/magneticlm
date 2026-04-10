# magnetic/tokenizer.py
#
# Minimal tokenizer + mutable vocabulary. Matches the tokenization
# used in both the C# Trainer.cs and the existing Python runner so
# vocabularies stay compatible across models trained on the same
# WikiText-103 dump.

import re
from typing import Dict, Iterable, List

# Strip the same punctuation the C# trainer strips, then lowercase +
# whitespace-split. This is deliberately simple so we do not build up
# a subword tokenizer by accident.
_SPLIT_RE = re.compile(r'[.,;!?()\[\]{}"]+')


def tokenize(line: str) -> List[str]:
    """Split a sentence into lowercased word tokens."""
    return [w for w in _SPLIT_RE.sub(' ', line.lower()).split() if w]


class Vocabulary:
    """Growing word <-> id map. Built during tokenization, frozen
    before training the n-gram tables."""

    def __init__(self):
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []

    def add(self, word: str) -> int:
        tid = self.word2id.get(word, -1)
        if tid < 0:
            tid = len(self.id2word)
            self.word2id[word] = tid
            self.id2word.append(word)
        return tid

    def get(self, word: str, default: int = -1) -> int:
        return self.word2id.get(word, default)

    def __len__(self) -> int:
        return len(self.id2word)

    def __contains__(self, word: str) -> bool:
        return word in self.word2id

    def lookup(self, ids: Iterable[int]) -> List[str]:
        """Resolve ids back to words. Unknown ids are rendered as <OOV>."""
        out = []
        for t in ids:
            if 0 <= t < len(self.id2word):
                out.append(self.id2word[t])
            else:
                out.append("<OOV>")
        return out

    def tokenize_text(self, text: str):
        """Convenience: tokenize then look up. Returns (ids, unknown_words)."""
        ids = []
        unknown = []
        for w in tokenize(text):
            tid = self.word2id.get(w, -1)
            if tid < 0:
                unknown.append(w)
            else:
                ids.append(tid)
        return ids, unknown
