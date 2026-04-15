from __future__ import annotations

from langchain_core.documents import Document

from src.processing.schemas import NormalizedRecord


CHUNK_SIZE = 450
CHUNK_OVERLAP = 80


def build_documents(records: list[NormalizedRecord]) -> list[Document]:
    """Convert normalized records into chunked retrieval documents."""

    chunks: list[Document] = []
    for record in records:
        chunks.extend(build_document_chunks(record))
    return chunks


def build_document_chunks(record: NormalizedRecord) -> list[Document]:
    content = _render_semantic_content(record)
    chunk_texts = _split_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunk_texts:
        chunk_texts = [content]

    base_metadata = _build_metadata(record)
    documents: list[Document] = []
    chunk_count = len(chunk_texts)
    for index, chunk_text in enumerate(chunk_texts, start=1):
        chunk_metadata = {
            **base_metadata,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "chunk_id": f"{record.id}__chunk_{index}",
            "parent_id": record.id,
            "parent_title": record.title,
        }
        documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    return documents


def _build_metadata(record: NormalizedRecord) -> dict:
    return {
        "id": record.id,
        "source": record.source,
        "record_type": record.record_type,
        "title": record.title,
        "category": record.category,
        "exercise_name": record.exercise_name,
        "dish_name": record.dish_name,
        "primary_muscles": ", ".join(record.primary_muscles),
        "secondary_muscles": ", ".join(record.secondary_muscles),
        "tags": ", ".join(record.tags),
        "ingredients": ", ".join(record.ingredients),
        "calories": record.calories,
        "protein": record.protein,
        "carbs": record.carbs,
        "fat": record.fat,
    }


def _render_semantic_content(record: NormalizedRecord) -> str:
    if record.record_type == "exercise":
        return _render_exercise_content(record)
    if record.record_type == "dish":
        return _render_dish_content(record)
    if record.record_type == "diet_pdf":
        return _render_diet_pdf_content(record)
    return _render_generic_content(record)


def _render_exercise_content(record: NormalizedRecord) -> str:
    parts = [
        f"Exercise name: {record.exercise_name or record.title}",
        f"Category: {record.category or 'unknown'}",
        f"Primary muscles: {_join_or_default(record.primary_muscles)}",
        f"Secondary muscles: {_join_or_default(record.secondary_muscles)}",
        f"Instructions: {_join_sentences(record.instructions)}",
        f"Notes: {_join_sentences(record.notes)}",
        f"Tags: {_join_or_default(record.tags)}",
    ]
    return "\n".join(parts)


def _render_dish_content(record: NormalizedRecord) -> str:
    parts = [
        f"Dish identifier: {record.dish_name or record.title}",
        f"Category: {record.category or 'nutrition_metadata'}",
        f"Ingredients: {_join_or_default(record.ingredients)}",
        f"Calories: {_number_or_default(record.calories, 'unknown')}",
        f"Protein (g): {_number_or_default(record.protein, 'unknown')}",
        f"Carbs (g): {_number_or_default(record.carbs, 'unknown')}",
        f"Fat (g): {_number_or_default(record.fat, 'unknown')}",
        f"Notes: {_join_sentences(record.notes)}",
    ]
    return "\n".join(parts)


def _render_diet_pdf_content(record: NormalizedRecord) -> str:
    return record.document_text or "No PDF content available."


def _render_generic_content(record: NormalizedRecord) -> str:
    parts = [
        f"Title: {record.title}",
        f"Category: {record.category or 'unknown'}",
        f"Notes: {_join_sentences(record.notes)}",
    ]
    return "\n".join(parts)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            split_point = normalized.rfind(" ", start, end)
            if split_point > start + 40:
                end = split_point
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _join_or_default(values: list[str], default: str = "not provided") -> str:
    cleaned = [value for value in values if value]
    return ", ".join(cleaned) if cleaned else default


def _join_sentences(values: list[str], default: str = "not provided") -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    return " ".join(cleaned) if cleaned else default


def _number_or_default(value: float | None, default: str) -> str:
    if value is None:
        return default
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"
