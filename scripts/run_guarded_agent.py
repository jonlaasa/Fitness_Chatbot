from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guarded_agent.executor import run_guarded_agent_once
from src.utils.conversation_logger import save_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guarded agent variant with tools plus guardrails."
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
    parser.add_argument(
        "--output-guardrail",
        action="store_true",
        help="Enable an additional output guardrail pass after the agent answer.",
    )
    return parser.parse_args()


def _safe_console_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _print_result(result) -> None:
    print(_safe_console_text("Guardrails: enabled"))
    print(_safe_console_text(f"Scope reason: {result.scope_reason}"))
    print(_safe_console_text(f"Output guardrail: {'enabled' if result.output_guard_enabled else 'disabled'}"))
    if result.blocked:
        print(_safe_console_text("\nAnswer:"))
        print(_safe_console_text(result.final_answer))
        return

    print(_safe_console_text("\nTool usage summary:"))
    if not result.tool_uses:
        print(_safe_console_text("- No tools were used"))
    else:
        for index, tool_use in enumerate(result.tool_uses, start=1):
            print(_safe_console_text(f"- {index}. {tool_use['name']} -> args={tool_use['args']}"))

    print(_safe_console_text("\nRaw agent answer:"))
    print(_safe_console_text(result.raw_answer))
    print(_safe_console_text("\nGuarded final answer:"))
    print(_safe_console_text(result.final_answer))


def _run_once(question: str, model: str | None, output_guardrail: bool) -> None:
    result = run_guarded_agent_once(
        question=question,
        model_name=model,
        use_output_guard=output_guardrail,
    )
    _print_result(result)
    save_path = save_conversation(
        conversation_type="guarded_agent",
        question=question,
        answer=result.final_answer,
        retrieved_documents=[],
        output_dir="conversations",
        extra={
            "mode": "guarded_agent",
            "blocked": result.blocked,
            "scope_reason": result.scope_reason,
            "output_guardrail": result.output_guard_enabled,
            "tool_uses": result.tool_uses,
            "raw_answer": result.raw_answer,
        },
    )
    print(_safe_console_text(f"\nConversation saved to: {save_path}"))


def _run_interactive(model: str | None, output_guardrail: bool) -> None:
    print("Guarded agent ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        _run_once(question, model, output_guardrail)


def main() -> None:
    args = parse_args()
    if args.question:
        _run_once(args.question, args.model, args.output_guardrail)
    else:
        _run_interactive(args.model, args.output_guardrail)


if __name__ == "__main__":
    main()
