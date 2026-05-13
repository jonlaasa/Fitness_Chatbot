from __future__ import annotations

from dataclasses import dataclass

from langchain_ollama import ChatOllama

from src.guardrails.schemas import GuardedRagAnswer, ScopeDecision
from src.guardrails.vendor import load_guardrails_vendor
from src.llm.prompt_strategies import get_prompt_strategy


DEFAULT_GUARDRAIL_API_BASE = "http://localhost:11434"
VALID_TOPICS = [
    "exercise",
    "training",
    "fitness",
    "sports science",
    "nutrition",
    "diet",
    "healthy habits",
    "supplements",
    "bmi",
    "body mass index",
]
INVALID_TOPICS = [
    "politics",
    "programming",
    "software development",
    "cinema",
    "movies",
    "weather",
    "finance",
    "geography",
]
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
ALLOWED_KEYWORDS = {
    "exercise",
    "exercises",
    "workout",
    "training",
    "fitness",
    "muscle",
    "glute",
    "glutes",
    "chest",
    "shoulder",
    "back",
    "nutrition",
    "diet",
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
    "body mass index",
    "routine",
    "strength",
    "cardio",
}


@dataclass(slots=True)
class GuardrailRunResult:
    # Resultado intermedio del guardrail: permite saber si hubo bloqueo
    # y qué respuesta validada terminó devolviéndose.
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
        # Usamos dos modelos ChatOllama sobre el mismo LLM:
        # uno más "frío" para clasificar alcance y otro para estructurar la salida.
        self.model_name = model_name
        self.api_base = api_base
        self.scope_model = self._build_chat_model(temperature=0)
        self.answer_model = self._build_chat_model(temperature=0.1)
        self.topic_validator = self._build_topic_validator()

    def run(
        self,
        question: str,
        retrieved_docs: list,
        prompt_strategy: str = "few-shot",
        scope_decision: ScopeDecision | None = None,
    ) -> GuardrailRunResult:
        # Punto de entrada principal del guardrail dentro del RAG:
        # primero valida dominio y luego controla la salida final.
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
        # Si la validación estructurada indica que la respuesta no está bien
        # soportada por el contexto, devolvemos una salida más conservadora.
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
        # Primero aplicamos heurísticas baratas para cortar rápido los casos claros.
        heuristic_decision = _heuristic_scope_decision(question)
        if heuristic_decision is not None:
            return heuristic_decision

        if self.topic_validator is not None:
            try:
                # RestrictToTopic es la barrera principal de entrada en esta versión.
                validation_result = self.topic_validator.validate(question, {})
                result_type = type(validation_result).__name__
                if result_type == "PassResult":
                    return ScopeDecision(
                        in_scope=True,
                        reason=(
                            "Guardrails RestrictToTopic accepted the question as part of the "
                            "fitness, training, nutrition, or supplements domain."
                        ),
                    )
                return ScopeDecision(
                    in_scope=False,
                    reason=getattr(validation_result, "error_message", "The question is outside the configured domain."),
                )
            except Exception as exc:
                # Si el validador externo falla, permitimos seguir y dejamos
                # constancia del motivo para no romper todo el pipeline.
                return ScopeDecision(
                    in_scope=True,
                    reason=(
                        "RestrictToTopic could not validate the query locally, "
                        f"so the fallback allowed it. Details: {exc}"
                    ),
                )

        try:
            # Fallback final: clasificación estructurada usando el propio LLM local.
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
        # La respuesta final se vuelve a pasar por el modelo en formato estructurado
        # para comprobar si realmente está apoyada en el contexto recuperado.
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
        # Parámetros comunes para todas las llamadas locales del guardrail.
        return ChatOllama(
            base_url=self.api_base,
            model=self.model_name,
            temperature=temperature,
            num_ctx=2048,
            num_predict=300,
            seed=42,
        )

    def _build_topic_validator(self):
        # El validador RestrictToTopic se carga desde una ruta corta externa
        # para evitar problemas de instalación en Windows + OneDrive.
        vendor_load = load_guardrails_vendor()
        if not vendor_load.available or vendor_load.restrict_to_topic_class is None:
            return None

        try:
            return vendor_load.restrict_to_topic_class(
                valid_topics=VALID_TOPICS,
                invalid_topics=INVALID_TOPICS,
                device="cpu",
                disable_classifier=True,
                disable_llm=False,
                llm_callable=self._topic_llm_callable,
                on_fail="exception",
                use_local=False,
            )
        except Exception:
            return None

    def _topic_llm_callable(self, text: str, topics: list[str]) -> list[str]:
        # RestrictToTopic permite usar un callable propio como clasificador.
        # Aquí reutilizamos el LLM local para decidir qué temas están presentes.
        result = self.scope_model.invoke(
            [
                (
                    "system",
                    "You are a topic validator used by Guardrails RestrictToTopic. "
                    "Return only the candidate topics that are clearly present in the user question. "
                    "Reply as a comma-separated list using the exact topic labels. "
                    "If none are clearly present, reply only with: none",
                ),
                (
                    "human",
                    "Candidate topics:\n"
                    f"{topics}\n\n"
                    "User question:\n"
                    f"{text}",
                ),
            ]
        )
        content = getattr(result, "content", "") or ""
        matched_topics = []
        normalized_topics = {topic.lower(): topic for topic in topics}
        raw_tokens = [
            token.strip()
            for token in content.replace("\n", ",").split(",")
            if token.strip()
        ]
        for topic in raw_tokens:
            if not topic:
                continue
            normalized = topic.lower().strip()
            if normalized == "none":
                continue
            if normalized in normalized_topics:
                matched_topics.append(normalized_topics[normalized])
        return matched_topics


def _heuristic_scope_decision(question: str) -> ScopeDecision | None:
    # Reglas rápidas para reducir latencia y evitar llamadas innecesarias al modelo.
    lowered = question.lower()
    matched_out_of_scope = [term for term in OBVIOUS_OUT_OF_SCOPE_TERMS if term in lowered]
    if matched_out_of_scope:
        return ScopeDecision(
            in_scope=False,
            reason=(
                "The question matches an obvious out-of-scope topic: "
                + ", ".join(matched_out_of_scope)
            ),
        )
    matched_allowed = [term for term in ALLOWED_KEYWORDS if term in lowered]
    if matched_allowed:
        return ScopeDecision(
            in_scope=True,
            reason=(
                "The question contains clear domain keywords related to the assistant scope: "
                + ", ".join(sorted(matched_allowed))
            ),
        )
    return None
