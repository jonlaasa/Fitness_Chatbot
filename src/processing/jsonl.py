from __future__ import annotations

import json
from pathlib import Path

from src.processing.schemas import NormalizedRecord


def load_normalized_records(jsonl_path: str | Path) -> list[NormalizedRecord]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized data file not found: {path}")

    records: list[NormalizedRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(NormalizedRecord(**payload))
    return records
