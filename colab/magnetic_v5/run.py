"""CLI entry for MagneticLM v5."""

import argparse, os, sys, types

def _bootstrap(pkg="magnetic_v5"):
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if pkg in sys.modules:
        return
    if os.path.basename(here) == pkg:
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            __import__(pkg); return
        except Exception:
            pass
    p = types.ModuleType(pkg)
    p.__path__ = [here]
    p.__package__ = pkg
    sys.modules[pkg] = p

def main():
    _bootstrap()
    from magnetic_v5.config import add_cli_args, config_from_args
    from magnetic_v5.runner import run_pipeline
    parser = argparse.ArgumentParser(description="MagneticLM v5")
    add_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
