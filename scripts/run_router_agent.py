from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_agent.graph import build_router_graph
from src.utils.conversation_logger import save_conversation
from src.utils.document_display import extract_chunk_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LangGraph router agent for fitness vs nutrition questions."
    )
    parser.add_argument(
        "--question",
        help="Question to answer. If omitted, the script starts in interactive mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model name override.",
    )
    return parser.parse_args()


def _build_doc_payload(doc) -> dict:
    title = doc.metadata.get("parent_title") or doc.metadata.get("title") or doc.metadata.get("id") or "unknown"
    metadata = {
        key: value
        for key, value in doc.metadata.items()
        if value not in ("", None)
    }
    return {
        "title": title,
        "source": doc.metadata.get("source", "unknown"),
        "metadata": metadata,
        "preview": extract_chunk_preview(doc.page_content, max_chars=440),
    }


def _print_result(result: dict) -> list[dict]:
    print(f"Routed domain: {result['domain']}")
    if result.get("route_reason"):
        print(f"Route reason: {result['route_reason']}")
    print("\nRetrieved chunks:")
    payloads: list[dict] = []
    for index, doc in enumerate(result["documents"], start=1):
        payload = _build_doc_payload(doc)
        payloads.append(payload)
        chunk_index = payload["metadata"].get("chunk_index", index)
        chunk_count = payload["metadata"].get("chunk_count", "?")
        print(f"\nChunk {index}:")
        print(f"Parent document: {payload['title']}")
        print(f"Source: {payload['source']}")
        print(f"Chunk position: {chunk_index}/{chunk_count}")
        print(f"Chunk preview: {payload['preview']} ...")
        print("-" * 80)
    print("\nAnswer:")
    print(result["answer"])
    return payloads


def _run_once(question: str, model: str | None) -> None:
    graph = build_router_graph(model_name=model)
    result = graph.invoke({"user_query": question, "messages": []})
    retrieved_payloads = _print_result(result)
    save_path = save_conversation(
        conversation_type="router_agent",
        question=question,
        answer=result["answer"],
        retrieved_documents=retrieved_payloads,
        output_dir="conversations",
        extra={
            "domain": result["domain"],
            "route_reason": result.get("route_reason", ""),
            "architecture": "langgraph_router",
        },
    )
    print(f"\nConversation saved to: {save_path}")


def _run_interactive(model: str | None) -> None:
    print("Router agent ready. Type 'exit' to quit.")
    graph = build_router_graph(model_name=model)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = graph.invoke({"user_query": question, "messages": []})
        retrieved_payloads = _print_result(result)
        save_path = save_conversation(
            conversation_type="router_agent",
            question=question,
            answer=result["answer"],
            retrieved_documents=retrieved_payloads,
            output_dir="conversations",
            extra={
                "domain": result["domain"],
                "route_reason": result.get("route_reason", ""),
                "architecture": "langgraph_router",
            },
        )
        print(f"\nConversation saved to: {save_path}")


def main() -> None:
    args = parse_args()
    if args.question:
        _run_once(args.question, args.model)
    else:
        _run_interactive(args.model)


if __name__ == "__main__":
    main()
