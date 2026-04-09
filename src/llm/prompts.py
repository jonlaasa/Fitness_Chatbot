from __future__ import annotations

from langchain_core.documents import Document


def build_rag_prompt(question: str, retrieved_docs: list[Document]) -> str:
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


def _format_context(retrieved_docs: list[Document]) -> str:
    if not retrieved_docs:
        return "No retrieved documents."
    blocks: list[str] = []
    for index, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("id") or f"document_{index}"
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Document {index}] {title} | source={source}\n{doc.page_content}")
    return "\n\n".join(blocks)
