from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document


PromptBuilder = Callable[[str, list[Document]], str]


@dataclass(frozen=True, slots=True)
class PromptStrategy:
    name: str
    description: str
    builder: PromptBuilder


def build_zero_shot_prompt(question: str, retrieved_docs: list[Document]) -> str:
    context = _format_context(retrieved_docs)
    return (
        "You are a local academic RAG assistant specialized in sports science, "
        "fitness, training, and basic nutrition.\n"
        "Answer only using the retrieved context.\n"
        "If the context is insufficient, say clearly: 'Not enough information in the retrieved documents.'\n"
        "Do not invent studies, values, exercises, ingredients, or recommendations.\n"
        "Keep the answer concise, factual, and easy to justify in an academic project.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Final answer:"
    )


def build_one_shot_prompt(question: str, retrieved_docs: list[Document]) -> str:
    context = _format_context(retrieved_docs)
    return (
        "You are a local academic RAG assistant specialized in sports science, "
        "fitness, training, and basic nutrition.\n"
        "Use only the retrieved context.\n"
        "If the context is not sufficient, say: 'Not enough information in the retrieved documents.'\n"
        "Do not invent missing details.\n\n"
        "Example:\n"
        "Question: Which exercises target the glutes?\n"
        "Retrieved context: [Document 1] Glute Bridge ... Primary muscles: glutes.\n"
        "Answer: The retrieved context indicates that Glute Bridge targets the glutes.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )


def build_few_shot_prompt(question: str, retrieved_docs: list[Document]) -> str:
    context = _format_context(retrieved_docs)
    return (
        "You are a local academic RAG assistant specialized in sports science, "
        "fitness, training, and basic nutrition.\n"
        "Answer only from the retrieved context. If the context is insufficient, say: "
        "'Not enough information in the retrieved documents.'\n"
        "Do not invent missing facts.\n\n"
        "Example 1:\n"
        "Question: Which exercises target the chest?\n"
        "Retrieved context: [Document 1] Chest Press - Machine ... Primary muscles: chest.\n"
        "Answer: The retrieved context indicates that Chest Press - Machine targets the chest.\n\n"
        "Example 2:\n"
        "Question: Which dish has chicken and how much protein does it have?\n"
        "Retrieved context: [Document 1] dish_123 ... Ingredients: chicken ... Protein (g): 21.7.\n"
        "Answer: The retrieved context indicates that dish_123 contains chicken and has 21.7 g of protein.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )


def build_chain_of_thought_prompt(question: str, retrieved_docs: list[Document]) -> str:
    context = _format_context(retrieved_docs)
    return (
        "You are a local academic RAG assistant specialized in sports science, "
        "fitness, training, and basic nutrition.\n"
        "Use only the retrieved context.\n"
        "Reason carefully about which retrieved facts support the answer, but do not expose hidden reasoning.\n"
        "Instead, provide a short evidence-based answer followed by a brief justification.\n"
        "If the context is insufficient, say: 'Not enough information in the retrieved documents.'\n"
        "Do not invent missing details.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Output format:\n"
        "Answer: <final answer>\n"
        "Justification: <1-2 sentences grounded in the retrieved context>\n"
    )


PROMPT_STRATEGIES: dict[str, PromptStrategy] = {
    "zero-shot": PromptStrategy(
        name="zero-shot",
        description="Direct instruction with no examples.",
        builder=build_zero_shot_prompt,
    ),
    "one-shot": PromptStrategy(
        name="one-shot",
        description="One worked example before the real question.",
        builder=build_one_shot_prompt,
    ),
    "few-shot": PromptStrategy(
        name="few-shot",
        description="Two short worked examples before the real question.",
        builder=build_few_shot_prompt,
    ),
    "chain-of-thought": PromptStrategy(
        name="chain-of-thought",
        description="Internal step-by-step reasoning with concise answer and short justification.",
        builder=build_chain_of_thought_prompt,
    ),
}


def get_prompt_strategy(name: str) -> PromptStrategy:
    try:
        return PROMPT_STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(PROMPT_STRATEGIES)
        raise ValueError(f"Unknown prompt strategy '{name}'. Available: {available}") from exc


def list_prompt_strategies() -> list[PromptStrategy]:
    return list(PROMPT_STRATEGIES.values())


def _format_context(retrieved_docs: list[Document]) -> str:
    if not retrieved_docs:
        return "No retrieved documents."
    blocks: list[str] = []
    for index, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("id") or f"document_{index}"
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Document {index}] {title} | source={source}\n{doc.page_content}")
    return "\n\n".join(blocks)
