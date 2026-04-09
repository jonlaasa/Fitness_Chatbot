from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import EXERCISES_RAW_DIR, NUTRITION_RAW_DIR


EXERCISE_FILES = {
    "strength.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/strength.json",
    "cardio.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/cardio.json",
    "flexibility.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/flexibility.json",
}

NUTRITION_FILES = {
    "metadata/dish_metadata_cafe1.csv": "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv": "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv",
}


def download_files(target_root: Path, files: dict[str, str]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for relative_path, url in files.items():
        destination = target_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urlretrieve(url, destination)
        print(f"Saved to {destination}")


def main() -> None:
    download_files(EXERCISES_RAW_DIR, EXERCISE_FILES)
    download_files(NUTRITION_RAW_DIR, NUTRITION_FILES)
    print("Finished downloading the lightweight source files.")


if __name__ == "__main__":
    main()
