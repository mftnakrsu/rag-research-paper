"""Run all planned experiments sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).parent / "run_experiment.py")


# Define all experiments: (method, embedding, reranker, extra_args)
TIER1_EXPERIMENTS = [
    # E1: BM25 baseline
    ("bm25", "openai_large", "none", []),
    # E2: Dense (OpenAI)
    ("dense", "openai_large", "none", []),
    # E3: Dense (Cohere)
    ("dense", "cohere", "none", []),
    # E4: ColBERT
    ("colbert", "openai_large", "none", []),
    # E5: Hybrid (BM25 + OpenAI, RRF)
    ("hybrid", "openai_large", "none", []),
    # E6: Hybrid + Cohere Reranker
    ("hybrid", "openai_large", "cohere", []),
    # E7: HyDE
    ("hyde", "openai_large", "none", []),
    # E8: Multi-Query + RRF
    ("multi_query", "openai_large", "none", []),
]

TIER2_EXPERIMENTS = [
    # E9: Dense (Voyage)
    ("dense", "voyage", "none", []),
    # E10: Dense (BGE-M3 local)
    ("dense", "bge_m3", "none", []),
    # E11: Hybrid + BGE local reranker
    ("hybrid", "openai_large", "local", []),
]


def run_experiment(method: str, embedding: str, reranker: str, extra_args: list[str]):
    """Run a single experiment as a subprocess."""
    cmd = [
        PYTHON, SCRIPT,
        "--method", method,
        "--embedding", embedding,
        "--reranker", reranker,
    ] + extra_args

    name = f"{method}_{embedding}_{reranker}"
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2],
                        help="Which tier of experiments to run")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit queries for testing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    experiments = TIER1_EXPERIMENTS if args.tier == 1 else TIER2_EXPERIMENTS

    extra = []
    if args.max_queries:
        extra = ["--max-queries", str(args.max_queries)]

    total = len(experiments)
    passed = 0
    failed = 0

    for i, (method, emb, reranker, exp_args) in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] ", end="")
        all_args = exp_args + extra

        if args.dry_run:
            name = f"{method}_{emb}_{reranker}"
            print(f"DRY RUN: {name}")
            continue

        if run_experiment(method, emb, reranker, all_args):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"DONE: {passed} passed, {failed} failed out of {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
