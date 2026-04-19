import os
from typing import Iterator, List, Optional, Tuple


def _try_paths(*paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _locate_wikitext103(data_dir: str) -> Tuple[str, str]:
    """Return (train_path, valid_path) searching common Colab/Kaggle layouts."""
    candidates_train = [
        os.path.join(data_dir, "wiki.train.tokens"),
        os.path.join(data_dir, "wikitext-103/wiki.train.tokens"),
        os.path.join(data_dir, "wikitext-103-raw/wiki.train.tokens"),
        "/kaggle/input/wikitext-103/wiki.train.tokens",
        "/kaggle/input/wikitext-103-raw/wiki.train.tokens",
        "/content/wikitext-103/wiki.train.tokens",
    ]
    candidates_valid = [
        os.path.join(data_dir, "wiki.valid.tokens"),
        os.path.join(data_dir, "wikitext-103/wiki.valid.tokens"),
        os.path.join(data_dir, "wikitext-103-raw/wiki.valid.tokens"),
        "/kaggle/input/wikitext-103/wiki.valid.tokens",
        "/kaggle/input/wikitext-103-raw/wiki.valid.tokens",
        "/content/wikitext-103/wiki.valid.tokens",
    ]
    tr = _try_paths(*candidates_train)
    va = _try_paths(*candidates_valid)
    if not tr:
        raise FileNotFoundError(
            f"wiki.train.tokens not found. Tried: {candidates_train}"
        )
    if not va:
        raise FileNotFoundError(
            f"wiki.valid.tokens not found. Tried: {candidates_valid}"
        )
    return tr, va


def iter_lines(path: str, max_lines: int = -1) -> Iterator[str]:
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("="):
                continue
            yield line
            seen += 1
            if 0 < max_lines <= seen:
                return


def load_dataset(cfg) -> Tuple[List[str], List[str]]:
    if cfg.dataset != "wikitext-103":
        raise ValueError(f"dataset {cfg.dataset!r} not supported yet")
    tr_path, va_path = _locate_wikitext103(cfg.data_dir)
    train = list(iter_lines(tr_path, cfg.max_train_lines))
    valid = list(iter_lines(va_path, cfg.max_valid_lines))
    return train, valid
