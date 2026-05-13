from __future__ import annotations

from dataclasses import dataclass

from langchain_ollama import ChatOllama

from src.agent.executor import build_agent_executor, resolve_agent_model
from src.guardrails.schemas import GuardedAgentAnswer
from src.guardrails.service import GuardrailService


@dataclass(slots=True)
class GuardedAgentResult:
    # Resultado final del agente guardado, incluyendo trazas útiles
    # para depuración, consola y guardado de conversaciones.
    blocked: bool
    scope_reason: str
    raw_agent_result: dict | None
    tool_uses: list[dict]
    raw_answer: str
    final_answer: str
    output_guard_enabled: bool


def run_guarded_agent_once(
    question: str,
    model_name: str | None = None,
    use_output_guard: bool = False,
) -> GuardedAgentResult:
    # Flujo del agente con guardrails:
    # 1. comprobar si la pregunta está dentro del dominio
    # 2. ejecutar el agente con tools
    # 3. opcionalmente limpiar también la salida final
    resolved_model_name = model_name or resolve_agent_model()
    guardrail_service = GuardrailService(resolved_model_name)

    scope_decision = guardrail_service.check_scope(question)
    if not scope_decision.in_scope:
        blocked_answer = (
            "This question is outside the current scope of the assistant. "
            "Please ask about exercises, training, nutrition, supplements, or healthy habits."
        )
        return GuardedAgentResult(
            blocked=True,
            scope_reason=scope_decision.reason,
            raw_agent_result=None,
            tool_uses=[],
            raw_answer="",
            final_answer=blocked_answer,
            output_guard_enabled=use_output_guard,
        )

    agent_executor = build_agent_executor(model_name=resolved_model_name)
    raw_result = agent_executor.invoke({"messages": [("human", question)]})
    tool_uses = _collect_tool_usage(raw_result)
    raw_answer = _extract_final_answer(raw_result)
    final_answer = (
        _guard_agent_output(
            model_name=resolved_model_name,
            question=question,
            raw_answer=raw_answer,
        )
        if use_output_guard
        else raw_answer
    )

    return GuardedAgentResult(
        blocked=False,
        scope_reason=scope_decision.reason,
        raw_agent_result=raw_result,
        tool_uses=tool_uses,
        raw_answer=raw_answer,
        final_answer=final_answer,
        output_guard_enabled=use_output_guard,
    )


def _collect_tool_usage(result: dict) -> list[dict]:
    # Extrae qué tools llamó realmente el agente para poder enseñarlo en consola
    # y guardarlo luego como evidencia.
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


def _extract_final_answer(result: dict) -> str:
    # El último AIMessage sin tool_calls es la respuesta final del agente.
    final_answer = ""
    for msg in result["messages"]:
        if type(msg).__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
    return final_answer


def _guard_agent_output(model_name: str, question: str, raw_answer: str) -> str:
    # Segunda barrera opcional: revisar la salida del agente antes de devolverla.
    guard_model = ChatOllama(
        base_url="http://localhost:11434",
        model=model_name,
        temperature=0.1,
        num_ctx=2048,
        num_predict=300,
        seed=42,
    )
    chain = guard_model.with_structured_output(GuardedAgentAnswer)
    try:
        validated = chain.invoke(
            [
                (
                    "system",
                    "You are a lightweight output guardrail for a local fitness and nutrition agent. "
                    "Keep the final answer within scope, remove unsafe or irrelevant content, "
                    "and do not invent new facts. If the answer is not safe or not grounded enough, "
                    "replace it with a short fallback answer.",
                ),
                (
                    "human",
                    "Question:\n"
                    f"{question}\n\n"
                    "Agent answer to validate:\n"
                    f"{raw_answer}",
                ),
            ]
        )
    except Exception:
        return raw_answer
    if not validated.safe_and_in_scope:
        if validated.fallback_reason:
            return (
                "The agent answer was limited by the guardrail. "
                f"{validated.fallback_reason}"
            )
        return "The agent answer was limited by the guardrail."
    return validated.answer.strip() or raw_answer
