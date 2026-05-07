from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runner import run_evaluation
from src.evaluation.ragas_runner import list_eval_metrics
from src.evaluation.variants import list_eval_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG variants with RAGAS and save a summary table."
    )
    parser.add_argument(
        "--csv-path",
        default="rag_eval.csv",
        help="Path to the CSV file with evaluation questions and references.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory where evaluation runs will be saved.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=list_eval_variants(),
        help="Optional subset of variants to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        choices=list_eval_metrics(),
        help="Optional subset of metrics to evaluate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1,
        help="Maximum number of rows from rag_eval.csv to evaluate in this run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional generation model override for the evaluated variants.",
    )
    parser.add_argument(
        "--evaluator-model",
        default="qwen3.5:4b",
        help="Local Ollama model used by RAGAS as evaluator.",
    )
    parser.add_argument(
        "--embeddings-model",
        default=None,
        help="Optional Ollama embeddings model used by RAGAS for answer relevancy. If omitted, the project embedding model is reused.",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Existing evaluation_results run directory to update instead of creating a new one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_evaluation(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        variants=args.variants,
        metrics=args.metrics,
        max_rows=args.max_rows,
        model_name=args.model,
        evaluator_model=args.evaluator_model,
        embeddings_model=args.embeddings_model,
        resume_run_dir=args.resume_run_dir,
    )
    print(f"Saved evaluation run to: {run_dir}")
    print(f"Summary CSV: {run_dir / 'ragas_summary.csv'}")
    print(f"Summary Markdown: {run_dir / 'ragas_summary.md'}")


if __name__ == "__main__":
    main()
