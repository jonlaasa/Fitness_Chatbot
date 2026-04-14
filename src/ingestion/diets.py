from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.processing.schemas import NormalizedRecord


class DietPdfIngestionError(Exception):
    """Raised when diet PDFs cannot be loaded or normalized."""


def load_diet_pdf_records(dataset_dir: str | Path) -> list[NormalizedRecord]:
    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise DietPdfIngestionError(f"Dataset path does not exist: {base_path}")

    pdf_files = sorted(base_path.rglob("*.pdf"))
    if not pdf_files:
        raise DietPdfIngestionError(f"No PDF files found under: {base_path}")

    return [normalize_diet_pdf(pdf_file) for pdf_file in pdf_files]


def normalize_diet_pdf(pdf_path: Path) -> NormalizedRecord:
    text = _extract_pdf_text(pdf_path)
    cleaned_title = _clean_pdf_title(pdf_path.stem)
    normalized_id = _slugify(cleaned_title)
    summary_text = _truncate_text(text, max_chars=2000)

    return NormalizedRecord(
        id=normalized_id,
        source="google_drive/diet_pdfs",
        record_type="diet_pdf",
        title=cleaned_title,
        category="diet_pdf",
        notes=[
            f"file_name: {pdf_path.name}",
            f"page_count: {_get_page_count(pdf_path)}",
        ],
        tags=["diet", "pdf", "supplement", "nutrition"],
        document_text=summary_text,
        raw_payload={
            "file_name": pdf_path.name,
            "clean_title": cleaned_title,
        },
    )


def records_to_jsonl(records: Iterable[NormalizedRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(pages)
    return re.sub(r"\s+", " ", text).strip()


def _get_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def _clean_pdf_title(stem: str) -> str:
    title = re.sub(r"^copia de\s*", "", stem, flags=re.IGNORECASE).strip()
    title = re.sub(r"^\d+\.\s*", "", title).strip()
    title = title.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    if title and len(title) > 1:
        title = title[0].upper() + title[1:]
    return title


def _truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars].strip()


def _slugify(value: str) -> str:
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")
