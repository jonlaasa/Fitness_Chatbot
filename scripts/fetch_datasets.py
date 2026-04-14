from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DIETS_RAW_DIR, EXERCISES_RAW_DIR, NUTRITION_RAW_DIR


EXERCISE_FILES = {
    "strength.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/strength.json",
    "cardio.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/cardio.json",
    "flexibility.json": "https://raw.githubusercontent.com/longhaul-fitness/exercises/main/flexibility.json",
}

NUTRITION_FILES = {
    "metadata/dish_metadata_cafe1.csv": "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv": "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv",
}

DEFAULT_DIETS_DRIVE_URL = "https://drive.google.com/drive/folders/1TlcrD1lR4xJvCuCJQma3YMc-q2ri6OOA?usp=sharing"


def download_files(target_root: Path, files: dict[str, str]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for relative_path, url in files.items():
        destination = target_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urlretrieve(url, destination)
        print(f"Saved to {destination}")


def download_drive_folder(folder_url: str, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Google Drive PDFs from {folder_url}")
    gdown.download_folder(
        url=folder_url,
        output=str(target_root),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    print(f"Saved Google Drive PDFs to {target_root}")


def main() -> None:
    download_files(EXERCISES_RAW_DIR, EXERCISE_FILES)
    download_files(NUTRITION_RAW_DIR, NUTRITION_FILES)
    diets_drive_url = os.getenv("DIETS_DRIVE_FOLDER_URL", DEFAULT_DIETS_DRIVE_URL)
    download_drive_folder(diets_drive_url, DIETS_RAW_DIR)
    print("Finished downloading the lightweight source files and the diet PDFs.")


if __name__ == "__main__":
    main()
