from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _resolve_from_project(relative_or_absolute: str, default_relative: str) -> Path:
    raw_value = os.getenv(relative_or_absolute, default_relative)
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXERCISES_RAW_DIR = _resolve_from_project("EXERCISES_DATA_DIR", "data/raw/exercises")
NUTRITION_RAW_DIR = _resolve_from_project("NUTRITION_DATA_DIR", "data/raw/nutrition5k")
DIETS_RAW_DIR = _resolve_from_project("DIETS_DATA_DIR", "data/raw/diets")
DB_DIR = _resolve_from_project("CHROMA_DB_PATH", "db")
