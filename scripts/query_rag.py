from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.prompt_strategies import list_prompt_strategies
from src.retrieval.pipeline import answer_question, get_retrieval_engine
from src.utils.document_display import extract_chunk_preview
from src.utils.paths import DB_DIR
from src.utils.conversation_logger import save_conversation


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
        default=None,
        choices=[strategy.name for strategy in list_prompt_strategies()],
        help="Prompt engineering strategy to apply on top of the retrieved context.",
    )
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help="Enable modular guardrails on top of the retrieval pipeline.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="similarity",
        choices=["similarity", "mmr"],
        help="Retrieval strategy to use before generation.",
    )
    parser.add_argument(
        "--save-dir",
        default="conversations",
        help="Directory where conversation evidence will be saved.",
    )
    return parser.parse_args()


def _build_doc_payload(doc) -> dict:
    title = doc.metadata.get("parent_title") or doc.metadata.get("title") or doc.metadata.get("id") or "unknown"
    metadata = {
        key: value
        for key, value in doc.metadata.items()
        if value not in ("", None)
    }
    preview = extract_chunk_preview(doc.page_content, max_chars=440)
    return {
        "title": title,
        "source": doc.metadata.get("source", "unknown"),
        "metadata": metadata,
        "preview": preview,
    }


def _print_retrieved_documents(documents: list) -> list[dict]:
    print("\nRetrieved chunks:")
    payloads: list[dict] = []
    for index, doc in enumerate(documents, start=1):
        payload = _build_doc_payload(doc)
        payloads.append(payload)
        chunk_index = payload["metadata"].get("chunk_index", index)
        chunk_count = payload["metadata"].get("chunk_count", "?")
        print(f"\nChunk {index}:")
        print(f"Parent document: {payload['title']}")
        print(f"Source: {payload['source']}")
        print(f"Chunk position: {chunk_index}/{chunk_count}")
        print(f"Metadata keys: {list(payload['metadata'].keys())}")
        print(f"Chunk preview: {payload['preview']} ...")
        print("-" * 80)
    return payloads


def _run_single_question(
    question: str,
    db_path: str,
    model: str | None,
    strategy: str | None,
    save_dir: str,
    guardrails: bool,
    retrieval_mode: str,
) -> None:
    effective_strategy = strategy or ("few-shot" if guardrails else "zero-shot")
    result = answer_question(
        question=question,
        db_path=db_path,
        model_name=model,
        k=3,
        prompt_strategy=effective_strategy,
        retrieval_mode=retrieval_mode,
        use_guardrails=guardrails,
    )
    print(f"\nPrompt strategy: {effective_strategy}")
    print(f"Guardrails: {'enabled' if guardrails else 'disabled'}")
    print(f"Retrieval mode: {result.retrieval_mode} {result.retrieval_kwargs}")
    if result.blocked_by_guardrails:
        retrieved_payloads = []
        print(f"Guardrail scope block: {result.guardrail_details.get('scope_reason', '')}")
    else:
        retrieved_payloads = _print_retrieved_documents(result.retrieved_documents)
    print("\nAnswer:")
    print(result.answer)
    save_path = save_conversation(
        conversation_type="rag",
        question=question,
        answer=result.answer,
        retrieved_documents=retrieved_payloads,
        output_dir=save_dir,
        extra={
            "strategy": effective_strategy,
            "guardrails": guardrails,
            "retrieval_mode": result.retrieval_mode,
            "retrieval_kwargs": result.retrieval_kwargs or {},
            "guardrail_details": result.guardrail_details or {},
        },
    )
    print(f"\nConversation saved to: {save_path}")


def _run_interactive(
    db_path: str,
    model: str | None,
    strategy: str | None,
    save_dir: str,
    guardrails: bool,
    retrieval_mode: str,
) -> None:
    effective_strategy = strategy or ("few-shot" if guardrails else "zero-shot")
    print(
        f"Local RAG assistant ready with strategy '{effective_strategy}' "
        f"and guardrails {'enabled' if guardrails else 'disabled'}. Type 'exit' to quit."
    )
    engine = get_retrieval_engine(
        db_path=db_path,
        model_name=model,
        k=3,
        prompt_strategy=effective_strategy,
        retrieval_mode=retrieval_mode,
        use_guardrails=guardrails,
    )
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = engine.answer(question)
        print(f"\nPrompt strategy: {effective_strategy}")
        print(f"Guardrails: {'enabled' if guardrails else 'disabled'}")
        print(f"Retrieval mode: {result.retrieval_mode} {result.retrieval_kwargs}")
        if result.blocked_by_guardrails:
            retrieved_payloads = []
            print(f"Guardrail scope block: {result.guardrail_details.get('scope_reason', '')}")
        else:
            retrieved_payloads = _print_retrieved_documents(result.retrieved_documents)
        print("\nAnswer:")
        print(result.answer)
        save_path = save_conversation(
            conversation_type="rag",
            question=question,
            answer=result.answer,
            retrieved_documents=retrieved_payloads,
            output_dir=save_dir,
            extra={
                "strategy": effective_strategy,
                "guardrails": guardrails,
                "retrieval_mode": result.retrieval_mode,
                "retrieval_kwargs": result.retrieval_kwargs or {},
                "guardrail_details": result.guardrail_details or {},
            },
        )
        print(f"\nConversation saved to: {save_path}")


def main() -> None:
    args = parse_args()
    if args.question:
        _run_single_question(
            args.question,
            args.db_path,
            args.model,
            args.strategy,
            args.save_dir,
            args.guardrails,
            args.retrieval_mode,
        )
    else:
        _run_interactive(
            args.db_path,
            args.model,
            args.strategy,
            args.save_dir,
            args.guardrails,
            args.retrieval_mode,
        )


if __name__ == "__main__":
    main()
