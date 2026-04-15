from __future__ import annotations


def extract_chunk_preview(page_content: str, max_chars: int = 440) -> str:
    normalized = " ".join(page_content.split()).strip()
    return normalized[:max_chars].strip()


def extract_prompt_excerpt(page_content: str, max_chars: int = 700) -> str:
    normalized = " ".join(page_content.split()).strip()
    return normalized[:max_chars].strip()
