from __future__ import annotations

from langchain_core.documents import Document

from src.processing.schemas import NormalizedRecord


def build_documents(records: list[NormalizedRecord]) -> list[Document]:
    """Convert normalized records into semantically complete RAG documents."""

    return [build_document(record) for record in records]


def build_document(record: NormalizedRecord) -> Document:
    if record.record_type == "exercise":
        page_content = _render_exercise_document(record)
    elif record.record_type == "dish":
        page_content = _render_dish_document(record)
    else:
        page_content = _render_generic_document(record)

    metadata = {
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
    return Document(page_content=page_content, metadata=metadata)


def _render_exercise_document(record: NormalizedRecord) -> str:
    parts = [
        f"Document type: exercise",
        f"Exercise name: {record.exercise_name or record.title}",
        f"Category: {record.category or 'unknown'}",
        f"Primary muscles: {_join_or_default(record.primary_muscles)}",
        f"Secondary muscles: {_join_or_default(record.secondary_muscles)}",
        f"Instructions: {_join_sentences(record.instructions)}",
        f"Notes: {_join_sentences(record.notes)}",
        f"Tags: {_join_or_default(record.tags)}",
        f"Source: {record.source}",
    ]
    return "\n".join(parts)


def _render_dish_document(record: NormalizedRecord) -> str:
    parts = [
        "Document type: dish",
        f"Dish identifier: {record.dish_name or record.title}",
        f"Category: {record.category or 'nutrition_metadata'}",
        f"Ingredients: {_join_or_default(record.ingredients)}",
        f"Calories: {_number_or_default(record.calories, 'unknown')}",
        f"Protein (g): {_number_or_default(record.protein, 'unknown')}",
        f"Carbs (g): {_number_or_default(record.carbs, 'unknown')}",
        f"Fat (g): {_number_or_default(record.fat, 'unknown')}",
        f"Notes: {_join_sentences(record.notes)}",
        f"Source: {record.source}",
    ]
    return "\n".join(parts)


def _render_generic_document(record: NormalizedRecord) -> str:
    parts = [
        f"Document type: {record.record_type}",
        f"Title: {record.title}",
        f"Category: {record.category or 'unknown'}",
        f"Notes: {_join_sentences(record.notes)}",
        f"Source: {record.source}",
    ]
    return "\n".join(parts)


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
