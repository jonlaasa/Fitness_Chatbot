from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from src.processing.schemas import NormalizedRecord


class NutritionIngestionError(Exception):
    """Raised when Nutrition5k metadata cannot be loaded or normalized."""


def load_nutrition_records(dataset_dir: str | Path) -> list[NormalizedRecord]:
    """Load and normalize Nutrition5k metadata CSV files.

    The public metadata files are lightweight compared with the full dataset.
    We intentionally use only the dish-level nutrition metadata and ingredient names.
    """

    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise NutritionIngestionError(f"Dataset path does not exist: {base_path}")

    csv_files = _find_metadata_csv_files(base_path)
    if not csv_files:
        raise NutritionIngestionError(f"No Nutrition5k metadata CSV files found under: {base_path}")

    normalized_records: list[NormalizedRecord] = []
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                normalized_records.append(
                    normalize_nutrition_row(row=row, source_path=csv_file)
                )
    return normalized_records


def normalize_nutrition_row(row: list[str], source_path: Path) -> NormalizedRecord:
    """Map a raw Nutrition5k metadata row into the common schema."""

    if len(row) < 6:
        raise NutritionIngestionError(
            f"Nutrition row in {source_path} has fewer than 6 base columns."
        )

    dish_id = row[0].strip()
    calories = _to_float(row[1])
    total_mass = _to_float(row[2])
    fat = _to_float(row[3])
    carbs = _to_float(row[4])
    protein = _to_float(row[5])

    ingredient_groups = _parse_ingredient_groups(row[6:])
    ingredient_names = [group["name"] for group in ingredient_groups if group["name"]]

    notes: list[str] = [f"ingredient_count: {len(ingredient_groups)}"]
    if total_mass is not None:
        notes.append(f"total_mass_g: {total_mass}")

    return NormalizedRecord(
        id=dish_id,
        source="google-research-datasets/Nutrition5k",
        record_type="dish",
        title=dish_id,
        category="nutrition_metadata",
        dish_name=dish_id,
        ingredients=ingredient_names,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        notes=notes,
        tags=["nutrition", "dish", source_path.stem],
        raw_payload={
            "dish_id": dish_id,
            "total_calories": calories,
            "total_mass": total_mass,
            "total_fat": fat,
            "total_carb": carbs,
            "total_protein": protein,
            "ingredients": ingredient_groups,
        },
    )


def records_to_jsonl(records: Iterable[NormalizedRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output


def _find_metadata_csv_files(base_path: Path) -> list[Path]:
    preferred = sorted(base_path.rglob("dish_metadata*.csv"))
    if preferred:
        return preferred
    return sorted(
        path for path in base_path.rglob("*.csv") if "metadata" in str(path).lower()
    )


def _parse_ingredient_groups(values: list[str]) -> list[dict[str, float | str | None]]:
    groups: list[dict[str, float | str | None]] = []
    group_size = 7
    for index in range(0, len(values), group_size):
        group = values[index : index + group_size]
        if len(group) < group_size:
            continue
        ingredient_id = group[0].strip()
        ingredient_name = group[1].strip()
        if not ingredient_id and not ingredient_name:
            continue
        groups.append(
            {
                "ingredient_id": ingredient_id,
                "name": ingredient_name,
                "grams": _to_float(group[2]),
                "calories": _to_float(group[3]),
                "fat": _to_float(group[4]),
                "carbs": _to_float(group[5]),
                "protein": _to_float(group[6]),
            }
        )
    return groups


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
