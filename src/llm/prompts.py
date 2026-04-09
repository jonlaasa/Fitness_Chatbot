from __future__ import annotations

from src.llm.prompt_strategies import build_zero_shot_prompt


def build_rag_prompt(question, retrieved_docs) -> str:
    return build_zero_shot_prompt(question, retrieved_docs)
