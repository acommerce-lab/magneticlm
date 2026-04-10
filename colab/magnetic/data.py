# magnetic/data.py
#
# WikiText-103 loader. Thin wrapper around the existing download logic
# in MagneticLMFastRunner.ensure_wt103 so we do not fork two different
# data paths.

import os
import sys
from typing import List, Optional


def ensure_wt103(data_dir: str = "data/wt103"):
    """Download WikiText-103 to data_dir if needed. Returns
    (train_path, test_path)."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    print("Downloading WikiText-103 (via HF datasets)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: first-time download needs `pip install datasets`.",
              file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    for split, path in (("train", train_path), ("test", test_path)):
        with open(path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                t = item["text"].strip()
                # Drop blank lines and encyclopedia section headers.
                if t and not t.startswith("="):
                    f.write(t + "\n")
    return train_path, test_path


def load_wt103_lines(path: str, limit: Optional[int] = None) -> List[str]:
    """Read raw lines (already filtered) from a WT103 text file.
    Blank lines are skipped. When limit is not None, read at most that
    many lines."""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if limit is not None and len(lines) >= limit:
                break
    return lines
