from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EvalSample:
    question: str
    reference: str


def load_eval_samples(csv_path: str | Path, max_rows: int | None = None) -> list[EvalSample]:
    path = Path(csv_path)
    rows: list[EvalSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            question = row["Pregunta"].strip()
            reference = row["Respuesta (Ground Truth)"].strip()
            overflow_parts = row.get(None, []) or []

            if overflow_parts:
                if question.lower().startswith("which muscles are targeted by") and reference.lower().startswith(" and "):
                    question = f"{question}{reference}"
                    reference = overflow_parts[0].strip()
                    overflow_parts = overflow_parts[1:]

                if overflow_parts:
                    joined_overflow = ", ".join(part.strip() for part in overflow_parts if part.strip())
                    if joined_overflow:
                        reference = f"{reference}, {joined_overflow}" if reference else joined_overflow

            rows.append(
                EvalSample(
                    question=question,
                    reference=reference,
                )
            )
    return rows
