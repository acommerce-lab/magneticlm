import glob
import os
from typing import Iterator, List, Optional, Tuple


def _try_paths(*paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# Directories we scan recursively for wiki.{train,valid}.tokens when a direct
# path miss. Kaggle datasets mount under /kaggle/input/<slug>/..., Colab under
# /content/..., and local layouts under data_dir.
_SEARCH_ROOTS = ("/kaggle/input", "/content", "/workspace", "/data")


def _glob_first(root: str, filename: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    # Cap depth to avoid scanning huge trees; wikitext nests at most a few deep.
    for depth in range(1, 6):
        pattern = os.path.join(root, *(["*"] * depth), filename)
        for hit in glob.glob(pattern):
            if os.path.isfile(hit):
                return hit
    return None


def _locate_wikitext103(data_dir: str) -> Tuple[str, str]:
    """Return (train_path, valid_path) searching common Colab/Kaggle layouts."""
    direct_train = [
        os.path.join(data_dir, "wiki.train.tokens"),
        os.path.join(data_dir, "wikitext-103/wiki.train.tokens"),
        os.path.join(data_dir, "wikitext-103-raw/wiki.train.tokens"),
    ]
    direct_valid = [
        os.path.join(data_dir, "wiki.valid.tokens"),
        os.path.join(data_dir, "wikitext-103/wiki.valid.tokens"),
        os.path.join(data_dir, "wikitext-103-raw/wiki.valid.tokens"),
    ]

    tr = _try_paths(*direct_train)
    va = _try_paths(*direct_valid)

    if not tr:
        for root in _SEARCH_ROOTS:
            tr = _glob_first(root, "wiki.train.tokens")
            if tr:
                break
    if not va:
        for root in _SEARCH_ROOTS:
            va = _glob_first(root, "wiki.valid.tokens")
            if va:
                break

    # As a last resort, if we found train, look for valid next to it.
    if tr and not va:
        sibling = os.path.join(os.path.dirname(tr), "wiki.valid.tokens")
        if os.path.exists(sibling):
            va = sibling

    if not tr:
        raise FileNotFoundError(
            "wiki.train.tokens not found. Checked direct paths "
            f"{direct_train} and recursively scanned {list(_SEARCH_ROOTS)}. "
            "Pass --data_dir /path/to/folder-containing-wiki.train.tokens."
        )
    if not va:
        raise FileNotFoundError(
            "wiki.valid.tokens not found. Checked direct paths "
            f"{direct_valid} and recursively scanned {list(_SEARCH_ROOTS)}. "
            "Pass --data_dir /path/to/folder-containing-wiki.valid.tokens."
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
    print(f"  train file: {tr_path}")
    print(f"  valid file: {va_path}")
    train = list(iter_lines(tr_path, cfg.max_train_lines))
    valid = list(iter_lines(va_path, cfg.max_valid_lines))
    return train, valid
