from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from src.embeddings.factory import build_embedding_model
from src.llm.local_model import build_local_llm
from src.llm.prompts import build_rag_prompt
from src.retrieval.vector_store import load_chroma_index
from src.utils.paths import DB_DIR


@dataclass(slots=True)
class RetrievalResult:
    question: str
    answer: str
    prompt: str
    retrieved_documents: list


@dataclass(slots=True)
class RetrievalEngine:
    vector_store: object
    llm: object
    top_k: int

    def answer(self, question: str) -> RetrievalResult:
        retrieved_documents = self.vector_store.similarity_search(question, k=self.top_k)
        prompt = build_rag_prompt(question, retrieved_documents)
        answer = self.llm.invoke(prompt)
        return RetrievalResult(
            question=question,
            answer=answer,
            prompt=prompt,
            retrieved_documents=retrieved_documents,
        )


@lru_cache(maxsize=4)
def get_retrieval_engine(
    db_path: str | None = None,
    model_name: str | None = None,
    k: int | None = None,
) -> RetrievalEngine:
    embedding_model = build_embedding_model()
    vector_store = load_chroma_index(db_path or DB_DIR, embedding_model)
    top_k = k or int(os.getenv("TOP_K", "3"))
    llm = build_local_llm(model_name)
    return RetrievalEngine(vector_store=vector_store, llm=llm, top_k=top_k)


def answer_question(
    question: str,
    db_path: str | None = None,
    model_name: str | None = None,
    k: int | None = None,
) -> RetrievalResult:
    engine = get_retrieval_engine(db_path=db_path, model_name=model_name, k=k)
    return engine.answer(question)
