"""CLI entry for MagneticLM v6."""
import argparse, os, sys, types

def _bootstrap(pkg="magnetic_v6"):
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if pkg in sys.modules: return
    if os.path.basename(here) == pkg:
        if parent not in sys.path: sys.path.insert(0, parent)
        try: __import__(pkg); return
        except: pass
    p = types.ModuleType(pkg)
    p.__path__ = [here]; p.__package__ = pkg
    sys.modules[pkg] = p

def main():
    _bootstrap()
    from magnetic_v6.config import add_cli_args, config_from_args
    from magnetic_v6.runner import run_pipeline
    parser = argparse.ArgumentParser(description="MagneticLM v6")
    add_cli_args(parser)
    cfg = config_from_args(parser.parse_args())
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
