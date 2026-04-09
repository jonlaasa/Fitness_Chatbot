from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.prompt_strategies import list_prompt_strategies
from src.retrieval.pipeline import answer_question
from src.utils.paths import DB_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prompt engineering strategies on the same retrieved context."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to evaluate across multiple prompting strategies.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_DIR),
        help="Path to the persisted Chroma database.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model name override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = [strategy.name for strategy in list_prompt_strategies()]

    for strategy in strategies:
        result = answer_question(
            question=args.question,
            db_path=args.db_path,
            model_name=args.model,
            k=3,
            prompt_strategy=strategy,
        )
        print("=" * 80)
        print(f"Strategy: {strategy}")
        print("-" * 80)
        for index, doc in enumerate(result.retrieved_documents, start=1):
            title = doc.metadata.get("title") or doc.metadata.get("id")
            print(f"{index}. {title} | {doc.metadata.get('source')}")
        print("\nAnswer:")
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
