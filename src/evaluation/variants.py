from __future__ import annotations

from dataclasses import dataclass

from src.agent.tools import search_fitness_documents
from src.guarded_agent.executor import run_guarded_agent_once
from src.retrieval.pipeline import answer_question
from src.router_agent.graph import build_router_graph


@dataclass(slots=True)
class VariantPrediction:
    variant_name: str
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    metadata: dict


def run_variant_prediction(
    variant_name: str,
    question: str,
    reference: str,
    model_name: str | None = None,
) -> VariantPrediction:
    if variant_name == "few_shot_baseline":
        result = answer_question(
            question=question,
            model_name=model_name,
            prompt_strategy="few-shot",
            retrieval_mode="similarity",
            use_guardrails=False,
        )
        return _from_rag_result(variant_name, question, reference, result)

    if variant_name == "few_shot_guardrails":
        result = answer_question(
            question=question,
            model_name=model_name,
            prompt_strategy="few-shot",
            retrieval_mode="similarity",
            use_guardrails=True,
        )
        return _from_rag_result(variant_name, question, reference, result)

    if variant_name == "few_shot_mmr":
        result = answer_question(
            question=question,
            model_name=model_name,
            prompt_strategy="few-shot",
            retrieval_mode="mmr",
            use_guardrails=False,
        )
        return _from_rag_result(variant_name, question, reference, result)

    if variant_name == "router_agent":
        graph = build_router_graph(model_name=model_name)
        result = graph.invoke({"user_query": question, "messages": []})
        return VariantPrediction(
            variant_name=variant_name,
            user_input=question,
            response=result["answer"],
            retrieved_contexts=[doc.page_content for doc in result["documents"]],
            reference=reference,
            metadata={
                "domain": result["domain"],
                "route_reason": result.get("route_reason", ""),
            },
        )

    if variant_name == "guarded_agent":
        result = run_guarded_agent_once(
            question=question,
            model_name=model_name,
            use_output_guard=False,
        )
        search_query = _extract_search_query(result.tool_uses) or question
        contexts = []
        if search_query:
            contexts = [doc.page_content for doc in search_fitness_documents(search_query, k=3)]
        return VariantPrediction(
            variant_name=variant_name,
            user_input=question,
            response=result.final_answer,
            retrieved_contexts=contexts,
            reference=reference,
            metadata={
                "blocked": result.blocked,
                "scope_reason": result.scope_reason,
                "tool_uses": result.tool_uses,
                "output_guard_enabled": result.output_guard_enabled,
            },
        )

    raise ValueError(f"Unknown variant '{variant_name}'")


def list_eval_variants() -> list[str]:
    return [
        "few_shot_baseline",
        "few_shot_guardrails",
        "few_shot_mmr",
        "router_agent",
        "guarded_agent",
    ]


def _from_rag_result(variant_name: str, question: str, reference: str, result) -> VariantPrediction:
    return VariantPrediction(
        variant_name=variant_name,
        user_input=question,
        response=result.answer,
        retrieved_contexts=[doc.page_content for doc in result.retrieved_documents],
        reference=reference,
        metadata={
            "retrieval_mode": result.retrieval_mode,
            "retrieval_kwargs": result.retrieval_kwargs or {},
            "guardrails_enabled": result.guardrails_enabled,
            "blocked_by_guardrails": result.blocked_by_guardrails,
            "guardrail_details": result.guardrail_details or {},
        },
    )


def _extract_search_query(tool_uses: list[dict]) -> str | None:
    for tool_use in tool_uses:
        if tool_use.get("name") == "search_fitness_knowledge":
            return tool_use.get("args", {}).get("query")
    return None
