"""CLI runner for magnetic_v4.

Handles flat layouts where the folder isn't named `magnetic_v4/`:
  - If the files live in a folder named `magnetic_v4/`, regular import works.
  - Otherwise (e.g. /kaggle/input/.../wavev4/), we bootstrap `magnetic_v4`
    as an in-memory package pointing to the current directory so that
    relative imports like `from .config import Config` keep working.

Usage:
    python run.py --max_train_lines 10000 --max_valid_lines 100
    python run.py --max_train_lines 1000000 --batch_size 512 \
                  --wave_iters 5 --eval_generation on
"""

import argparse
import importlib.util
import os
import sys
import types


def _bootstrap_package(pkg_name: str = "magnetic_v4"):
    """Make `pkg_name` importable regardless of the actual folder name."""
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)

    # Case 1: already importable
    if pkg_name in sys.modules:
        return

    # Case 2: folder IS named pkg_name — add parent to sys.path
    if os.path.basename(here) == pkg_name:
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            __import__(pkg_name)
            return
        except Exception:
            pass

    # Case 3: arbitrary folder name — synthesize an in-memory package
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [here]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg


def main():
    _bootstrap_package()
    from magnetic_v4.config import Config, add_cli_args, config_from_args
    from magnetic_v4.runner import run_pipeline

    parser = argparse.ArgumentParser(description="MagneticLM v4 (wave physics)")
    add_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
