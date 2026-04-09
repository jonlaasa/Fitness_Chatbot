from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full local RAG pipeline in one command."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip downloading the lightweight raw datasets.",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip the final query step.",
    )
    parser.add_argument(
        "--question",
        default="Which exercises target the glutes?",
        help="Question used for the final validation query.",
    )
    return parser.parse_args()


def run_step(args: list[str], label: str) -> None:
    print(f"\n[{label}]")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable

    if not args.skip_fetch:
        run_step(
            [python_executable, "scripts/fetch_datasets.py"],
            "1/4 Downloading lightweight datasets",
        )

    run_step(
        [python_executable, "scripts/ingest_all.py"],
        "2/4 Normalizing source datasets",
    )
    run_step(
        [python_executable, "scripts/build_index.py"],
        "3/4 Building Chroma vector database",
    )

    if not args.skip_query:
        run_step(
            [python_executable, "scripts/query_rag.py", "--question", args.question],
            "4/4 Running validation query",
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
