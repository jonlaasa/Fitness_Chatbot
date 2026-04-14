from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.diets import load_diet_pdf_records, records_to_jsonl
from src.utils.paths import DIETS_RAW_DIR, PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize diet PDF files into the common project schema."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DIETS_RAW_DIR),
        help="Directory containing diet PDF files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(PROCESSED_DATA_DIR / "diets_normalized.jsonl"),
        help="Output JSONL path for normalized diet PDF records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_diet_pdf_records(args.input_dir)
    output_path = records_to_jsonl(records, args.output_path)
    print(f"Normalized {len(records)} diet PDF records.")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
