from __future__ import annotations

import numexpr
from langchain_core.tools import tool

from src.embeddings.factory import build_embedding_model
from src.retrieval.vector_store import load_chroma_index
from src.utils.paths import DB_DIR


def _get_vector_store():
    embedding_model = build_embedding_model()
    return load_chroma_index(DB_DIR, embedding_model)


@tool
def search_fitness_knowledge(query: str) -> str:
    """Search the local fitness and nutrition knowledge base.

    Use this tool to retrieve exercises, training information, and nutrition
    metadata from the local Chroma vector database.
    """

    vector_store = _get_vector_store()
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents were found in the local knowledge base."

    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("id") or f"document_{index}"
        source = doc.metadata.get("source", "unknown")
        metadata = {
            key: value
            for key, value in doc.metadata.items()
            if value not in ("", None)
        }
        preview = " ".join(doc.page_content.split())[:220].strip()
        blocks.append(
            f"[Document {index}] {title} | source={source}\n"
            f"Metadata: {metadata}\n"
            f"Preview: {preview} ..."
        )
    return "\n\n".join(blocks)


@tool
def fitness_calculator(expression: str) -> str:
    """Evaluate simple fitness-related calculations.

    Use this for formulas such as BMI, calorie arithmetic, macro totals,
    unit conversions, repetitions, or training volume calculations.
    Example input: '(80 / (1.78**2))'
    """

    try:
        result = numexpr.evaluate(expression)
        return str(float(result))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def get_agent_tools():
    return [search_fitness_knowledge, fitness_calculator]
