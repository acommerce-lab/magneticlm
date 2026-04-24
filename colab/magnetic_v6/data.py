"""Dataset loading with robust discovery + HF fallback.

Single responsibility: locate wikitext-103 on disk and iterate its lines.
No tokenization here — see tokenizer.py.

Searches:
  1. --data_dir (+ common subdirs)
  2. /kaggle/working/data, /kaggle/input, /content, /workspace, /data
  3. HuggingFace download (cached to stable path under /kaggle/working if avail)
"""

import glob
import os
from typing import Iterator, List, Optional, Tuple


_SEARCH_ROOTS = (
    "/kaggle/working/data", "/kaggle/working",
    "/kaggle/input", "/content", "/content/data",
    "/workspace", "/workspace/data", "/data",
)
_SUBDIRS = ("wikitext-103", "wikitext-103-raw", "wikitext-103-v1", "wikitext-103-raw-v1")
_TRAIN = ("wiki.train.tokens", "wiki.train.raw", "wiki.train.txt")
_VALID = ("wiki.valid.tokens", "wiki.valid.raw", "wiki.valid.txt")


def _try_paths(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _glob_first(root: str, filenames) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    for depth in range(1, 7):
        for name in filenames:
            pattern = os.path.join(root, *(["*"] * depth), name)
            for hit in glob.glob(pattern):
                if os.path.isfile(hit):
                    return hit
    return None


def _list_roots() -> str:
    lines = []
    for r in _SEARCH_ROOTS:
        if not os.path.isdir(r):
            continue
        try:
            entries = sorted(os.listdir(r))
            lines.append(f"  {r}: {entries[:10]}" + ("…" if len(entries) > 10 else ""))
        except OSError as e:
            lines.append(f"  {r}: <unreadable: {e}>")
    return "\n".join(lines) or "  (no known roots exist)"


def _hf_download(data_dir: str) -> Optional[Tuple[str, str]]:
    if os.path.isdir("/kaggle/working"):
        stable_dir = "/kaggle/working/data/wikitext-103"
    else:
        stable_dir = os.path.join(data_dir, "wikitext-103")
    target_dir = os.path.join(data_dir, "wikitext-103")

    stable_tr = os.path.join(stable_dir, "wiki.train.tokens")
    stable_va = os.path.join(stable_dir, "wiki.valid.tokens")
    if os.path.exists(stable_tr) and os.path.exists(stable_va):
        return stable_tr, stable_va

    os.makedirs(target_dir, exist_ok=True)
    train_path = os.path.join(target_dir, "wiki.train.tokens")
    valid_path = os.path.join(target_dir, "wiki.valid.tokens")
    if os.path.exists(train_path) and os.path.exists(valid_path):
        return train_path, valid_path

    print("  [hf] downloading WikiText-103...")
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        print("  [hf] missing `datasets` package")
        return None

    try:
        ds = _hf_load("Salesforce/wikitext", "wikitext-103-v1")
    except Exception as e:
        print(f"  [hf] download failed: {e}")
        return None

    write_pairs = [("train", train_path), ("validation", valid_path)]
    if stable_dir != target_dir:
        os.makedirs(stable_dir, exist_ok=True)
        write_pairs += [("train", stable_tr), ("validation", stable_va)]
    for split, path in write_pairs:
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                t = item["text"].strip()
                if t and not t.startswith("="):
                    f.write(t + "\n")
    print(f"  [hf] wrote {train_path}")
    return train_path, valid_path


def locate_wikitext(data_dir: str) -> Tuple[str, str]:
    """Return (train_path, valid_path) — raises if unreachable."""
    search_dirs = [data_dir] + [r for r in _SEARCH_ROOTS if r not in (data_dir,)]

    tr = None
    va = None
    for d in search_dirs:
        candidates_t = [os.path.join(d, n) for n in _TRAIN] + \
                       [os.path.join(d, s, n) for s in _SUBDIRS for n in _TRAIN]
        candidates_v = [os.path.join(d, n) for n in _VALID] + \
                       [os.path.join(d, s, n) for s in _SUBDIRS for n in _VALID]
        if tr is None:
            tr = _try_paths(*candidates_t)
        if va is None:
            va = _try_paths(*candidates_v)
        if tr and va:
            break

    if not tr:
        for r in _SEARCH_ROOTS:
            tr = _glob_first(r, _TRAIN)
            if tr:
                break
    if not va:
        for r in _SEARCH_ROOTS:
            va = _glob_first(r, _VALID)
            if va:
                break

    if tr and not va:
        base = os.path.dirname(tr)
        for n in _VALID:
            cand = os.path.join(base, n)
            if os.path.exists(cand):
                va = cand
                break

    if not tr or not va:
        hit = _hf_download(data_dir)
        if hit is not None:
            tr, va = hit

    if not tr or not va:
        raise FileNotFoundError(
            "wikitext-103 not found.\n"
            f"Searched: {[data_dir] + list(_SEARCH_ROOTS)}\n"
            "What's mounted:\n" + _list_roots() + "\n"
            "Fix: attach the dataset, set --data_dir, or enable internet."
        )
    return tr, va


def iter_lines(path: str, max_lines: int = -1) -> Iterator[str]:
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            yield line
            seen += 1
            if 0 < max_lines <= seen:
                return


def load_dataset(cfg) -> Tuple[List[str], List[str]]:
    if cfg.dataset != "wikitext-103":
        raise ValueError(f"dataset {cfg.dataset!r} not supported yet")
    tr, va = locate_wikitext(cfg.data_dir)
    print(f"  train file: {tr}")
    print(f"  valid file: {va}")
    train = list(iter_lines(tr, cfg.max_train_lines))
    valid = list(iter_lines(va, cfg.max_valid_lines))
    return train, valid
