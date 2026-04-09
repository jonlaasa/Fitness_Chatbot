from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.processing.schemas import NormalizedRecord


class ExerciseIngestionError(Exception):
    """Raised when exercise files cannot be loaded or normalized."""


def load_exercise_records(dataset_dir: str | Path) -> list[NormalizedRecord]:
    """Load and normalize the longhaul-fitness exercises dataset.

    Expected usage:
    - the repository or extracted JSON files live inside `dataset_dir`
    - every exercise becomes exactly one normalized record
    - no text chunking is applied here
    """

    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise ExerciseIngestionError(f"Dataset path does not exist: {base_path}")

    json_files = sorted(base_path.rglob("*.json"))
    if not json_files:
        raise ExerciseIngestionError(f"No JSON files found under: {base_path}")

    normalized_records: list[NormalizedRecord] = []
    for json_file in json_files:
        payload = _read_json_file(json_file)
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_records.append(
                normalize_exercise_record(item=item, source_path=json_file)
            )
    return normalized_records


def normalize_exercise_record(item: dict[str, Any], source_path: Path) -> NormalizedRecord:
    """Map a raw exercise payload into the common schema."""

    exercise_name = _first_non_empty(
        item,
        ["exercise_name", "name", "title", "exercise"],
        default=source_path.stem,
    )
    exercise_id = _first_non_empty(
        item,
        ["id", "exercise_id", "slug", "uuid"],
        default=f"{source_path.stem}:{exercise_name}".lower().replace(" ", "_"),
    )
    category = _first_non_empty(
        item,
        ["category", "bodyPart", "type", "exercise_type"],
        default=source_path.parent.name,
    )

    primary_muscles = _ensure_list(
        _first_existing(item, ["primary_muscles", "primaryMuscles", "muscles", "target"])
    )
    secondary_muscles = _ensure_list(
        _first_existing(item, ["secondary_muscles", "secondaryMuscles", "synergists"])
    )
    instructions = _ensure_list(
        _first_existing(item, ["instructions", "steps", "description"])
    )
    notes = _ensure_list(
        _first_existing(item, ["notes", "tips", "comments", "equipment"])
    )
    tags = _collect_tags(item)

    return NormalizedRecord(
        id=str(exercise_id),
        source="longhaul-fitness/exercises",
        record_type="exercise",
        title=str(exercise_name),
        category=str(category),
        exercise_name=str(exercise_name),
        primary_muscles=primary_muscles,
        secondary_muscles=secondary_muscles,
        instructions=instructions,
        notes=notes,
        tags=tags,
        raw_payload=item,
    )


def records_to_jsonl(records: Iterable[NormalizedRecord], output_path: str | Path) -> Path:
    """Persist normalized records to JSONL for transparent inspection."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output


def _read_json_file(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_existing(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, "", []):
            return item[key]
    return None


def _first_non_empty(item: dict[str, Any], keys: list[str], default: str = "") -> str:
    value = _first_existing(item, keys)
    if value is None:
        return default
    if isinstance(value, list):
        joined = ", ".join(str(part).strip() for part in value if str(part).strip())
        return joined or default
    return str(value).strip() or default


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if "|" in cleaned:
            parts = cleaned.split("|")
        elif ";" in cleaned:
            parts = cleaned.split(";")
        else:
            parts = [cleaned]
        return [part.strip() for part in parts if part.strip()]
    return [str(value).strip()]


def _collect_tags(item: dict[str, Any]) -> list[str]:
    tag_keys = ["tags", "tag", "equipment", "level", "mechanic", "force"]
    collected: list[str] = []
    for key in tag_keys:
        collected.extend(_ensure_list(item.get(key)))
    return sorted(set(collected))
