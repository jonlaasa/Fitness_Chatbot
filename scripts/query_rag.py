from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.prompt_strategies import list_prompt_strategies
from src.retrieval.pipeline import answer_question, get_retrieval_engine
from src.utils.paths import DB_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local RAG system backed by Chroma and an Ollama model."
    )
    parser.add_argument(
        "--question",
        help="Question to answer. If omitted, the script starts in interactive mode.",
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
    parser.add_argument(
        "--strategy",
        default="zero-shot",
        choices=[strategy.name for strategy in list_prompt_strategies()],
        help="Prompt engineering strategy to apply on top of the retrieved context.",
    )
    return parser.parse_args()


def _run_single_question(
    question: str, db_path: str, model: str | None, strategy: str
) -> None:
    result = answer_question(
        question=question,
        db_path=db_path,
        model_name=model,
        k=3,
        prompt_strategy=strategy,
    )
    print(f"\nPrompt strategy: {strategy}")
    print("\nRetrieved documents:")
    for index, doc in enumerate(result.retrieved_documents, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("id")
        print(f"{index}. {title} | {doc.metadata.get('source')}")
    print("\nAnswer:")
    print(result.answer)


def _run_interactive(db_path: str, model: str | None, strategy: str) -> None:
    print(f"Local RAG assistant ready with strategy '{strategy}'. Type 'exit' to quit.")
    engine = get_retrieval_engine(
        db_path=db_path,
        model_name=model,
        k=3,
        prompt_strategy=strategy,
    )
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = engine.answer(question)
        print(f"\nPrompt strategy: {strategy}")
        print("\nRetrieved documents:")
        for index, doc in enumerate(result.retrieved_documents, start=1):
            title = doc.metadata.get("title") or doc.metadata.get("id")
            print(f"{index}. {title} | {doc.metadata.get('source')}")
        print("\nAnswer:")
        print(result.answer)


def main() -> None:
    args = parse_args()
    if args.question:
        _run_single_question(args.question, args.db_path, args.model, args.strategy)
    else:
        _run_interactive(args.db_path, args.model, args.strategy)


if __name__ == "__main__":
    main()
