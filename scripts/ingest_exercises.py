from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.exercises import load_exercise_records, records_to_jsonl
from src.utils.paths import EXERCISES_RAW_DIR, PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the exercise dataset into the project common schema."
    )
    parser.add_argument(
        "--input-dir",
        default=str(EXERCISES_RAW_DIR),
        help="Directory containing the exercise JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(PROCESSED_DATA_DIR / "exercises_normalized.jsonl"),
        help="Output JSONL path for normalized exercise records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_exercise_records(args.input_dir)
    output_path = records_to_jsonl(records, args.output_path)
    print(f"Normalized {len(records)} exercise records.")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
