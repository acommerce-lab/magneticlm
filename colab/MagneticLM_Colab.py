#!/usr/bin/env python3
"""
MagneticLM v6 - Google Colab Runner (OPTIMIZED)
=================================================
NumPy vectorized physics: 10-50x faster than pure Python.
Optimized tokenizer, batch processing, sparse operations.

Upload 4 files to Colab: graph.py, trainer.py, benchmark.py, this file.
Or: pip install numpy && python MagneticLM_Colab.py
  3. Results appear for PTB, WikiText-2, WikiText-103

Usage locally:
  python MagneticLM_Colab.py
"""

import os
import sys
import time


def download_datasets():
    """Download standard LM benchmark datasets."""

    # === PTB ===
    ptb_dir = "data/ptb"
    os.makedirs(ptb_dir, exist_ok=True)

    if not os.path.exists(f"{ptb_dir}/train.txt"):
        print("Downloading PTB...")
        os.system(f'curl -sL "https://raw.githubusercontent.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/master/data/ptb.train.txt" -o {ptb_dir}/train.txt')
        os.system(f'curl -sL "https://raw.githubusercontent.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/master/data/ptb.test.txt" -o {ptb_dir}/test.txt')
        print(f"  PTB downloaded: {os.path.getsize(f'{ptb_dir}/train.txt') // 1024}KB")

    # === WikiText-2 ===
    wt2_dir = "data/wikitext-2"
    os.makedirs(wt2_dir, exist_ok=True)

    if not os.path.exists(f"{wt2_dir}/train.txt"):
        print("Downloading WikiText-2...")
        os.system(f'curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt" -o {wt2_dir}/train.txt')
        os.system(f'curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt" -o {wt2_dir}/test.txt')
        print(f"  WT-2 downloaded: {os.path.getsize(f'{wt2_dir}/train.txt') // 1024}KB")

    # === WikiText-103 (from Hugging Face - works on Colab) ===
    wt103_dir = "data/wikitext-103"
    os.makedirs(wt103_dir, exist_ok=True)

    if not os.path.exists(f"{wt103_dir}/train.txt"):
        print("Downloading WikiText-103 (this may take a few minutes)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

            with open(f"{wt103_dir}/train.txt", 'w') as f:
                for item in ds['train']:
                    text = item['text'].strip()
                    if text and not text.startswith('='):
                        f.write(text + '\n')

            with open(f"{wt103_dir}/test.txt", 'w') as f:
                for item in ds['test']:
                    text = item['text'].strip()
                    if text and not text.startswith('='):
                        f.write(text + '\n')

            print(f"  WT-103 downloaded: {os.path.getsize(f'{wt103_dir}/train.txt') // 1024 // 1024}MB")
        except ImportError:
            print("  WikiText-103 requires 'datasets' package: pip install datasets")
            print("  Skipping WT-103. Install with: pip install datasets")
        except Exception as e:
            print(f"  WikiText-103 download failed: {e}")

    return ptb_dir, wt2_dir, wt103_dir


def clean_wikitext(path: str) -> str:
    """Clean WikiText files: remove headers and empty lines."""
    out_path = path.replace('.txt', '_clean.txt')
    if os.path.exists(out_path):
        return out_path

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    clean = [l.strip() for l in lines if l.strip() and not l.strip().startswith('=') and len(l.split()) >= 4]

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean))

    return out_path


def main():
    print("=" * 60)
    print("  MagneticLM v6 - Complete Benchmark Suite")
    print("  Physics-based Language Model with Accounting Kernel")
    print("=" * 60)

    # Download datasets
    ptb_dir, wt2_dir, wt103_dir = download_datasets()

    from benchmark import run_benchmark

    results = {}

    # === PTB ===
    if os.path.exists(f"{ptb_dir}/train.txt"):
        print("\n" + "=" * 60)
        print("  BENCHMARK 1: Penn Treebank (PTB)")
        print("=" * 60)
        _, bi, kn, cache, mag = run_benchmark(
            f"{ptb_dir}/train.txt",
            f"{ptb_dir}/test.txt",
            physics_iterations=50
        )
        results['PTB'] = {'bigram': bi, 'kn': kn, 'cache': cache, 'magnetic': mag}

    # === WikiText-2 ===
    if os.path.exists(f"{wt2_dir}/train.txt"):
        print("\n" + "=" * 60)
        print("  BENCHMARK 2: WikiText-2")
        print("=" * 60)
        train_clean = clean_wikitext(f"{wt2_dir}/train.txt")
        test_clean = clean_wikitext(f"{wt2_dir}/test.txt")
        _, bi, kn, cache, mag = run_benchmark(
            train_clean, test_clean,
            physics_iterations=50
        )
        results['WikiText-2'] = {'bigram': bi, 'kn': kn, 'cache': cache, 'magnetic': mag}

    # === WikiText-103 (THE BOSS) ===
    if os.path.exists(f"{wt103_dir}/train.txt"):
        print("\n" + "=" * 60)
        print("  BENCHMARK 3: WikiText-103 (THE FINAL BOSS)")
        print("=" * 60)

        # WT-103 is huge - use streaming approach
        # Train on first 100k lines, test on all
        train_path = f"{wt103_dir}/train.txt"
        test_path = f"{wt103_dir}/test.txt"

        # Check size
        with open(train_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        print(f"  WT-103 total train lines: {total_lines:,}")

        if total_lines > 200000:
            # Create a subset for feasibility
            subset_path = f"{wt103_dir}/train_100k.txt"
            if not os.path.exists(subset_path):
                print(f"  Creating 100k subset for training...")
                with open(train_path, 'r') as f_in, open(subset_path, 'w') as f_out:
                    for i, line in enumerate(f_in):
                        if i >= 100000:
                            break
                        f_out.write(line)
            train_path = subset_path

        _, bi, kn, cache, mag = run_benchmark(
            train_path, test_path,
            physics_iterations=30  # Less iterations for huge graph
        )
        results['WikiText-103'] = {'bigram': bi, 'kn': kn, 'cache': cache, 'magnetic': mag}

    # === Final Summary ===
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<18} {'Bigram':>8} {'KN-5':>8} {'Cache':>8} {'MagLM':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<18} {r['bigram']:>8.1f} {r['kn']:>8.1f} {r['cache']:>8.1f} {r['magnetic']:>8.1f}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'AWD-LSTM+Cache':<18} {'':>8} {'':>8} {'52.8':>8} {'52.0':>8} (PTB/WT2)")
    print(f"  {'Transformer-XL':<18} {'':>8} {'':>8} {'':>8} {'16.4':>8} (WT103)")
    print("=" * 70)


if __name__ == "__main__":
    main()
