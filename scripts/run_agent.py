from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.executor import build_agent_executor
from src.utils.conversation_logger import save_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple local agent with two tools for the fitness project."
    )
    parser.add_argument(
        "--question",
        help="Question to answer. If omitted, the script starts in interactive mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama chat model override.",
    )
    return parser.parse_args()


def _print_trace(result: dict) -> None:
    for msg in result["messages"]:
        kind = type(msg).__name__
        if kind == "HumanMessage":
            print(f"Human: {msg.content}\n")
        elif kind == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    print(f"AI -> Tool call: {tool_call['name']}({tool_call['args']})\n")
            else:
                print(f"AI (final): {msg.content}\n")
        elif kind == "ToolMessage":
            preview = msg.content[:320] + "..." if len(msg.content) > 320 else msg.content
            print(f"Tool result [{msg.name}]:\n{preview}\n")


def _collect_tool_usage(result: dict) -> list[dict]:
    tool_uses: list[dict] = []
    for msg in result["messages"]:
        if type(msg).__name__ == "AIMessage" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_uses.append(
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )
    return tool_uses


def _print_tool_summary(tool_uses: list[dict]) -> None:
    print("=" * 80)
    if not tool_uses:
        print("Tool usage summary: no tools were used in this run.")
        print("=" * 80)
        print()
        return

    print(f"Tool usage summary: {len(tool_uses)} tool call(s) detected.")
    for index, tool_use in enumerate(tool_uses, start=1):
        print(f"{index}. {tool_use['name']} -> args={tool_use['args']}")
    print("=" * 80)
    print()


def _run_once(question: str, model: str | None) -> None:
    agent_executor = build_agent_executor(model_name=model)
    result = agent_executor.invoke({"messages": [("human", question)]})
    _print_trace(result)
    tool_uses = _collect_tool_usage(result)
    _print_tool_summary(tool_uses)
    final_answer = ""
    for msg in result["messages"]:
        if type(msg).__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
    save_path = save_conversation(
        conversation_type="agent",
        question=question,
        answer=final_answer,
        retrieved_documents=[],
        output_dir="conversations",
        extra={"mode": "agent_tools", "tool_uses": tool_uses},
    )
    print(f"Conversation saved to: {save_path}")


def _run_interactive(model: str | None) -> None:
    print("Local agent ready. Type 'exit' to quit.")
    agent_executor = build_agent_executor(model_name=model)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent_executor.invoke({"messages": [("human", question)]})
        _print_trace(result)
        tool_uses = _collect_tool_usage(result)
        _print_tool_summary(tool_uses)
        final_answer = ""
        for msg in result["messages"]:
            if type(msg).__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
                final_answer = msg.content
        save_path = save_conversation(
            conversation_type="agent",
            question=question,
            answer=final_answer,
            retrieved_documents=[],
            output_dir="conversations",
            extra={"mode": "agent_tools", "tool_uses": tool_uses},
        )
        print(f"Conversation saved to: {save_path}")


def main() -> None:
    args = parse_args()
    if args.question:
        _run_once(args.question, args.model)
    else:
        _run_interactive(args.model)


if __name__ == "__main__":
    main()
