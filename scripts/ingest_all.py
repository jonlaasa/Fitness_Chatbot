from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.exercises import (
    ExerciseIngestionError,
    load_exercise_records,
    records_to_jsonl as exercises_to_jsonl,
)
from src.ingestion.nutrition import (
    NutritionIngestionError,
    load_nutrition_records,
    records_to_jsonl as nutrition_to_jsonl,
)
from src.utils.paths import EXERCISES_RAW_DIR, NUTRITION_RAW_DIR, PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both lightweight ingestion pipelines for exercises and Nutrition5k metadata."
    )
    parser.add_argument(
        "--exercises-dir",
        default=str(EXERCISES_RAW_DIR),
        help="Directory containing exercise JSON files.",
    )
    parser.add_argument(
        "--nutrition-dir",
        default=str(NUTRITION_RAW_DIR),
        help="Directory containing Nutrition5k metadata CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        exercise_records = load_exercise_records(args.exercises_dir)
    except ExerciseIngestionError as exc:
        print(f"Exercise ingestion error: {exc}")
        exercise_records = []

    try:
        nutrition_records = load_nutrition_records(args.nutrition_dir)
    except NutritionIngestionError as exc:
        print(f"Nutrition ingestion error: {exc}")
        nutrition_records = []

    if exercise_records:
        exercises_output = exercises_to_jsonl(
            exercise_records, PROCESSED_DATA_DIR / "exercises_normalized.jsonl"
        )
        print(f"Normalized {len(exercise_records)} exercise records -> {exercises_output}")

    if nutrition_records:
        nutrition_output = nutrition_to_jsonl(
            nutrition_records, PROCESSED_DATA_DIR / "nutrition_normalized.jsonl"
        )
        print(f"Normalized {len(nutrition_records)} nutrition records -> {nutrition_output}")

    if not exercise_records and not nutrition_records:
        print("No datasets were processed. Add raw files and run the script again.")


if __name__ == "__main__":
    main()
