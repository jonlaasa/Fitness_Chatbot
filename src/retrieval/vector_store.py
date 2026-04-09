from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_chroma_index(
    documents: list[Document],
    embedding_function,
    persist_directory: str | Path,
    reset: bool = True,
) -> Chroma:
    db_path = Path(persist_directory)
    if reset and db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_chroma_index(persist_directory: str | Path, embedding_function) -> Chroma:
    db_path = Path(persist_directory)
    if not db_path.exists():
        raise FileNotFoundError(f"Chroma database path not found: {db_path}")
    return Chroma(
        persist_directory=str(db_path),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
