from __future__ import annotations

import os
from functools import lru_cache

from huggingface_hub.utils import logging as hf_logging
from langchain_huggingface import HuggingFaceEmbeddings
from transformers.utils import logging as transformers_logging


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _configure_embedding_runtime() -> None:
    # Keep the CLI clean during repeated local queries.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()


@lru_cache(maxsize=2)
def build_embedding_model(model_name: str | None = None) -> HuggingFaceEmbeddings:
    _configure_embedding_runtime()
    resolved_model = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=resolved_model)
