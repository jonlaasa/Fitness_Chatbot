from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NormalizedRecord:
    """Common schema shared by all knowledge sources."""

    id: str
    source: str
    record_type: str
    title: str
    category: str = ""
    exercise_name: str = ""
    primary_muscles: list[str] = field(default_factory=list)
    secondary_muscles: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dish_name: str = ""
    ingredients: list[str] = field(default_factory=list)
    calories: float | None = None
    protein: float | None = None
    carbs: float | None = None
    fat: float | None = None
    document_text: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
