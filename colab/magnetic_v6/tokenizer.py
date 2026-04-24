"""Tokenizer + vocabulary.

Single responsibility: string → token ids and back. No knowledge of
datasets, files, or the rest of the pipeline.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def tokenize(line: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(line)]


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    freq: np.ndarray
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        u = self.unk_id
        s = self.stoi
        return [s.get(t, u) for t in tokens]

    def encode_line(self, line: str) -> List[int]:
        return self.encode(tokenize(line))


def build_vocab(
    lines: Iterable[str],
    max_vocab: int = 50000,
    min_count: int = 2,
    unk: str = "<unk>",
) -> Vocab:
    counter: Counter = Counter()
    for line in lines:
        counter.update(tokenize(line))

    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab > 0:
        items = items[: max(1, max_vocab - 1)]

    itos: List[str] = [unk]
    freq_list: List[int] = [0]
    for w, c in items:
        itos.append(w)
        freq_list.append(c)

    stoi = {w: i for i, w in enumerate(itos)}
    unk_id = stoi[unk]
    seen = set(stoi.keys())
    unk_count = sum(c for w, c in counter.items() if w not in seen)
    freq_list[unk_id] = unk_count

    freq = np.asarray(freq_list, dtype=np.int64)
    return Vocab(stoi=stoi, itos=itos, freq=freq, unk_id=unk_id)


def encode_stream(lines: Iterable[str], vocab: Vocab) -> List[np.ndarray]:
    out = []
    for line in lines:
        ids = vocab.encode_line(line)
        if ids:
            out.append(np.asarray(ids, dtype=np.int32))
    return out
