from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.factory import build_embedding_model
from src.processing.documents import build_documents
from src.processing.jsonl import load_normalized_records
from src.retrieval.vector_store import build_chroma_index
from src.utils.paths import DB_DIR, PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local ChromaDB index from normalized exercise and nutrition files."
    )
    parser.add_argument(
        "--exercises-path",
        default=str(PROCESSED_DATA_DIR / "exercises_normalized.jsonl"),
        help="Path to normalized exercise JSONL.",
    )
    parser.add_argument(
        "--nutrition-path",
        default=str(PROCESSED_DATA_DIR / "nutrition_normalized.jsonl"),
        help="Path to normalized nutrition JSONL.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_DIR),
        help="Target Chroma persistence directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []

    for source_path in (args.exercises_path, args.nutrition_path):
        path = Path(source_path)
        if path.exists():
            records.extend(load_normalized_records(path))

    if not records:
        raise FileNotFoundError(
            "No normalized JSONL files were found. Run the ingestion scripts first."
        )

    documents = build_documents(records)
    embedding_model = build_embedding_model()
    build_chroma_index(documents, embedding_model, args.db_path, reset=True)

    print(f"Indexed {len(records)} normalized records.")
    print(f"Created {len(documents)} semantic documents.")
    print(f"Chroma database saved to: {args.db_path}")


if __name__ == "__main__":
    main()
