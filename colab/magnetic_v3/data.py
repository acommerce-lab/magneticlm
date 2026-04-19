"""Dataset loading with robust auto-discovery + HuggingFace fallback.

Search strategy for wikitext-103:
  1. Direct paths under --data_dir (wiki.train.tokens, wikitext-103/, ...)
  2. Recursive glob under /kaggle/input, /content, /workspace, /data
     accepting wiki.train.* (.tokens, .raw, .txt) names
  3. HuggingFace `datasets` download as last resort (cached under data_dir)

On miss, prints what directories actually exist under the search roots so
the user can see what's attached to their notebook.
"""

import glob
import os
from typing import Iterator, List, Optional, Tuple


_SEARCH_ROOTS = ("/kaggle/input", "/content", "/workspace", "/data")
_TRAIN_NAMES = ("wiki.train.tokens", "wiki.train.raw", "wiki.train.txt")
_VALID_NAMES = ("wiki.valid.tokens", "wiki.valid.raw", "wiki.valid.txt")


def _try_paths(*paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _glob_first(root: str, filenames) -> Optional[str]:
    """Recursively find the first match for any of `filenames` under `root`."""
    if not os.path.isdir(root):
        return None
    for depth in range(1, 7):
        for name in filenames:
            pattern = os.path.join(root, *(["*"] * depth), name)
            for hit in glob.glob(pattern):
                if os.path.isfile(hit):
                    return hit
    return None


def _list_available_dirs() -> str:
    """Human-readable listing of what's actually mounted under search roots."""
    lines = []
    for root in _SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        try:
            entries = sorted(os.listdir(root))
        except OSError as e:
            lines.append(f"  {root}: <unreadable: {e}>")
            continue
        if not entries:
            lines.append(f"  {root}: <empty>")
        else:
            lines.append(f"  {root}: {entries}")
    return "\n".join(lines) if lines else "  (none of /kaggle/input, /content, /workspace, /data exist)"


def _hf_download_wikitext(data_dir: str) -> Optional[Tuple[str, str]]:
    """Fallback: fetch wikitext-103 via HuggingFace datasets, write .tokens files."""
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        return None

    target_dir = os.path.join(data_dir, "wikitext-103")
    train_path = os.path.join(target_dir, "wiki.train.tokens")
    valid_path = os.path.join(target_dir, "wiki.valid.tokens")
    if os.path.exists(train_path) and os.path.exists(valid_path):
        return train_path, valid_path

    print("  [hf] downloading wikitext-103 via HuggingFace datasets...")
    try:
        ds = _hf_load("wikitext", "wikitext-103-v1")
    except Exception as e:
        print(f"  [hf] download failed: {e}")
        return None

    os.makedirs(target_dir, exist_ok=True)
    for split, path in (("train", train_path), ("validation", valid_path)):
        with open(path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                text = row["text"]
                if text:
                    f.write(text if text.endswith("\n") else text + "\n")
    print(f"  [hf] wrote {train_path} and {valid_path}")
    return train_path, valid_path


def _locate_wikitext103(data_dir: str) -> Tuple[str, str]:
    """Return (train_path, valid_path)."""
    direct_train = [
        os.path.join(data_dir, name) for name in _TRAIN_NAMES
    ] + [
        os.path.join(data_dir, sub, name)
        for sub in ("wikitext-103", "wikitext-103-raw", "wikitext-103-v1", "wikitext-103-raw-v1")
        for name in _TRAIN_NAMES
    ]
    direct_valid = [
        os.path.join(data_dir, name) for name in _VALID_NAMES
    ] + [
        os.path.join(data_dir, sub, name)
        for sub in ("wikitext-103", "wikitext-103-raw", "wikitext-103-v1", "wikitext-103-raw-v1")
        for name in _VALID_NAMES
    ]

    tr = _try_paths(*direct_train)
    va = _try_paths(*direct_valid)

    if not tr:
        for root in _SEARCH_ROOTS:
            tr = _glob_first(root, _TRAIN_NAMES)
            if tr:
                break
    if not va:
        for root in _SEARCH_ROOTS:
            va = _glob_first(root, _VALID_NAMES)
            if va:
                break

    if tr and not va:
        base = os.path.dirname(tr)
        for name in _VALID_NAMES:
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                va = cand
                break

    # Last resort: HuggingFace download
    if not tr or not va:
        hf = _hf_download_wikitext(data_dir)
        if hf is not None:
            tr, va = hf

    if not tr or not va:
        listing = _list_available_dirs()
        raise FileNotFoundError(
            "wikitext-103 files not found.\n"
            f"  tried direct paths under --data_dir={data_dir}\n"
            f"  recursively scanned: {list(_SEARCH_ROOTS)}\n"
            f"  HuggingFace fallback: "
            f"{'unavailable (no `datasets` pkg or no internet)' if not tr else 'ok'}\n"
            f"What is actually mounted:\n{listing}\n"
            "Fix: either\n"
            "  (a) attach a wikitext-103 Kaggle dataset to your notebook, or\n"
            "  (b) pass --data_dir /kaggle/input/<your-wikitext-slug>, or\n"
            "  (c) enable internet in the notebook so HuggingFace download works."
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
