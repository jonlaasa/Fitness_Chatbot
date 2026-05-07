from __future__ import annotations

from dataclasses import dataclass

from langchain_ollama import ChatOllama

from src.guardrails.schemas import GuardedRagAnswer, ScopeDecision
from src.llm.prompt_strategies import get_prompt_strategy


DEFAULT_GUARDRAIL_API_BASE = "http://localhost:11434"
ALLOWED_KEYWORDS = {
    "exercise",
    "exercises",
    "workout",
    "training",
    "train",
    "fitness",
    "muscle",
    "glute",
    "glutes",
    "chest",
    "shoulder",
    "back",
    "nutrition",
    "diet",
    "healthy",
    "habit",
    "habits",
    "calorie",
    "calories",
    "protein",
    "carbs",
    "fat",
    "meal",
    "ingredient",
    "ingredients",
    "supplement",
    "supplements",
    "bmi",
    "routine",
    "strength",
    "cardio",
}
OBVIOUS_OUT_OF_SCOPE_TERMS = {
    "world cup",
    "football",
    "soccer",
    "president",
    "election",
    "capital of",
    "programming",
    "python code",
    "javascript",
    "movie",
    "film",
    "weather",
    "stock market",
}


@dataclass(slots=True)
class GuardrailRunResult:
    blocked: bool
    answer: str
    scope_decision: ScopeDecision | None
    validated_output: GuardedRagAnswer | None


class GuardrailService:
    def __init__(
        self,
        model_name: str,
        api_base: str = DEFAULT_GUARDRAIL_API_BASE,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.scope_model = self._build_chat_model(temperature=0)
        self.answer_model = self._build_chat_model(temperature=0.1)

    def run(
        self,
        question: str,
        retrieved_docs: list,
        prompt_strategy: str = "few-shot",
        scope_decision: ScopeDecision | None = None,
    ) -> GuardrailRunResult:
        scope_decision = scope_decision or self.check_scope(question)
        if not scope_decision.in_scope:
            blocked_answer = (
                "This question is outside the current scope of the assistant. "
                "Please ask about exercises, training, nutrition, supplements, or healthy habits."
            )
            return GuardrailRunResult(
                blocked=True,
                answer=blocked_answer,
                scope_decision=scope_decision,
                validated_output=None,
            )

        validated_output = self.generate_guarded_answer(
            question=question,
            retrieved_docs=retrieved_docs,
            prompt_strategy=prompt_strategy,
        )
        final_answer = validated_output.answer.strip()
        if not final_answer:
            final_answer = "Not enough information in the retrieved documents."
        if not validated_output.grounded_in_context:
            if validated_output.fallback_reason:
                final_answer = (
                    "Not enough information in the retrieved documents. "
                    f"{validated_output.fallback_reason}".strip()
                )
            elif not final_answer.lower().startswith("not enough information"):
                final_answer = "Not enough information in the retrieved documents."

        return GuardrailRunResult(
            blocked=False,
            answer=final_answer,
            scope_decision=scope_decision,
            validated_output=validated_output,
        )

    def check_scope(self, question: str) -> ScopeDecision:
        heuristic_decision = _heuristic_scope_decision(question)
        if heuristic_decision is not None:
            return heuristic_decision

        try:
            scope_chain = self.scope_model.with_structured_output(ScopeDecision)
            return scope_chain.invoke(
                [
                    (
                        "system",
                        "You are a scope classifier for a local academic RAG system. "
                        "The allowed scope is: exercises, training, fitness, sports science, "
                        "basic nutrition, healthy habits, and dietary supplements. "
                        "Mark the query as out of scope if it mainly asks about unrelated topics "
                        "such as politics, programming, entertainment, geography, or general chit-chat.",
                    ),
                    (
                        "human",
                        "Decide if this question is in scope for the assistant.\n"
                        f"Question: {question}",
                    ),
                ]
            )
        except Exception:
            return ScopeDecision(
                in_scope=True,
                reason="Structured scope classification failed, so the fallback allowed the query.",
            )

    def generate_guarded_answer(
        self,
        question: str,
        retrieved_docs: list,
        prompt_strategy: str = "few-shot",
    ) -> GuardedRagAnswer:
        base_prompt = get_prompt_strategy(prompt_strategy).builder(question, retrieved_docs)
        answer_chain = self.answer_model.with_structured_output(GuardedRagAnswer)
        try:
            return answer_chain.invoke(
                [
                    (
                        "system",
                        "You are a guarded local RAG assistant. "
                        "Use only the retrieved context. "
                        "Do not invent facts. "
                        "If the context is insufficient, say so in the structured output.",
                    ),
                    ("human", base_prompt),
                ]
            )
        except Exception:
            return GuardedRagAnswer(
                answer="Not enough information in the retrieved documents.",
                grounded_in_context=False,
                fallback_reason="Structured guardrail validation failed.",
            )


    def _build_chat_model(self, temperature: float) -> ChatOllama:
        return ChatOllama(
            base_url=self.api_base,
            model=self.model_name,
            temperature=temperature,
            num_ctx=2048,
            num_predict=300,
            seed=42,
        )


def _heuristic_scope_decision(question: str) -> ScopeDecision | None:
    lowered = question.lower()
    has_allowed_keyword = any(keyword in lowered for keyword in ALLOWED_KEYWORDS)
    matched_out_of_scope = [term for term in OBVIOUS_OUT_OF_SCOPE_TERMS if term in lowered]
    if matched_out_of_scope and not has_allowed_keyword:
        return ScopeDecision(
            in_scope=False,
            reason=(
                "The question matches an obvious out-of-scope topic: "
                + ", ".join(matched_out_of_scope)
            ),
        )
    return None
