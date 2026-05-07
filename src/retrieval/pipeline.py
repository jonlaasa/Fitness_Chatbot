from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from src.guardrails import GuardrailService
from src.embeddings.factory import build_embedding_model
from src.llm.local_model import build_local_llm, resolve_ollama_model
from src.llm.prompt_strategies import get_prompt_strategy
from src.retrieval.vector_store import load_chroma_index
from src.utils.paths import DB_DIR


@dataclass(slots=True)
class RetrievalResult:
    question: str
    answer: str
    prompt: str
    retrieved_documents: list
    retrieval_mode: str = "similarity"
    retrieval_kwargs: dict | None = None
    guardrails_enabled: bool = False
    blocked_by_guardrails: bool = False
    guardrail_details: dict | None = None


@dataclass(slots=True)
class RetrievalEngine:
    vector_store: object
    llm: object
    top_k: int
    prompt_strategy_name: str
    retrieval_mode: str = "similarity"
    mmr_fetch_k: int = 20
    mmr_lambda_mult: float = 1.0
    guardrails_enabled: bool = False
    guardrail_service: GuardrailService | None = None

    def answer(self, question: str) -> RetrievalResult:
        if self.guardrails_enabled and self.guardrail_service is not None:
            scope_only = self.guardrail_service.check_scope(question)
            if not scope_only.in_scope:
                blocked_answer = (
                    "This question is outside the current scope of the assistant. "
                    "Please ask about exercises, training, nutrition, supplements, or healthy habits."
                )
                return RetrievalResult(
                    question=question,
                    answer=blocked_answer,
                    prompt="",
                    retrieved_documents=[],
                    retrieval_mode=self.retrieval_mode,
                    retrieval_kwargs=self._build_retrieval_kwargs(),
                    guardrails_enabled=True,
                    blocked_by_guardrails=True,
                    guardrail_details={
                        "scope_reason": scope_only.reason,
                        "scope_in_scope": scope_only.in_scope,
                    },
                )

        retrieved_documents = self._retrieve_documents(question)
        prompt_builder = get_prompt_strategy(self.prompt_strategy_name).builder
        prompt = prompt_builder(question, retrieved_documents)
        guardrail_details = None

        if self.guardrails_enabled and self.guardrail_service is not None:
            guarded_result = self.guardrail_service.run(
                question=question,
                retrieved_docs=retrieved_documents,
                prompt_strategy=self.prompt_strategy_name,
                scope_decision=scope_only,
            )
            answer = guarded_result.answer
            guardrail_details = {
                "scope_reason": guarded_result.scope_decision.reason if guarded_result.scope_decision else "",
                "scope_in_scope": guarded_result.scope_decision.in_scope if guarded_result.scope_decision else True,
                "grounded_in_context": (
                    guarded_result.validated_output.grounded_in_context
                    if guarded_result.validated_output is not None
                    else None
                ),
                "fallback_reason": (
                    guarded_result.validated_output.fallback_reason
                    if guarded_result.validated_output is not None
                    else ""
                ),
            }
        else:
            answer = self.llm.invoke(prompt)

        return RetrievalResult(
            question=question,
            answer=answer,
            prompt=prompt,
            retrieved_documents=retrieved_documents,
            retrieval_mode=self.retrieval_mode,
            retrieval_kwargs=self._build_retrieval_kwargs(),
            guardrails_enabled=self.guardrails_enabled,
            blocked_by_guardrails=False,
            guardrail_details=guardrail_details,
        )

    def _retrieve_documents(self, question: str) -> list:
        if self.retrieval_mode == "mmr":
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.top_k,
                    "fetch_k": self.mmr_fetch_k,
                    "lambda_mult": self.mmr_lambda_mult,
                },
            )
            return retriever.invoke(question)
        return self.vector_store.similarity_search(question, k=self.top_k)

    def _build_retrieval_kwargs(self) -> dict:
        if self.retrieval_mode == "mmr":
            return {
                "k": self.top_k,
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult,
            }
        return {"k": self.top_k}


@lru_cache(maxsize=4)
def get_retrieval_engine(
    db_path: str | None = None,
    model_name: str | None = None,
    k: int | None = None,
    prompt_strategy: str = "zero-shot",
    retrieval_mode: str = "similarity",
    use_guardrails: bool = False,
) -> RetrievalEngine:
    embedding_model = build_embedding_model()
    vector_store = load_chroma_index(db_path or DB_DIR, embedding_model)
    top_k = k or int(os.getenv("TOP_K", "3"))
    resolved_model_name = model_name or resolve_ollama_model()
    llm = build_local_llm(resolved_model_name)
    guardrail_service = GuardrailService(resolved_model_name) if use_guardrails else None
    return RetrievalEngine(
        vector_store=vector_store,
        llm=llm,
        top_k=top_k,
        prompt_strategy_name=prompt_strategy,
        retrieval_mode=retrieval_mode,
        mmr_fetch_k=20,
        mmr_lambda_mult=1.0,
        guardrails_enabled=use_guardrails,
        guardrail_service=guardrail_service,
    )


def answer_question(
    question: str,
    db_path: str | None = None,
    model_name: str | None = None,
    k: int | None = None,
    prompt_strategy: str = "zero-shot",
    retrieval_mode: str = "similarity",
    use_guardrails: bool = False,
) -> RetrievalResult:
    engine = get_retrieval_engine(
        db_path=db_path,
        model_name=model_name,
        k=k,
        prompt_strategy=prompt_strategy,
        retrieval_mode=retrieval_mode,
        use_guardrails=use_guardrails,
    )
    return engine.answer(question)
