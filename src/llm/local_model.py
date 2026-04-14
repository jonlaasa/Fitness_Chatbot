from __future__ import annotations

import os
import subprocess

from langchain_ollama import OllamaLLM


DEFAULT_MODEL = "phi3:mini"


def resolve_ollama_model() -> str:
    model_from_env = os.getenv("OLLAMA_MODEL")
    if model_from_env:
        return model_from_env

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DEFAULT_MODEL

    model_names = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            model_names.append(parts[0])

    for preferred_prefix in ("phi3", "qwen", "mistral", "llama3"):
        for model_name in model_names:
            if model_name.startswith(f"{preferred_prefix}:"):
                return model_name

    return model_names[0] if model_names else DEFAULT_MODEL


def build_local_llm(model_name: str | None = None) -> OllamaLLM:
    return OllamaLLM(
        model=model_name or resolve_ollama_model(),
        temperature=0.1,
        num_predict=256,
    )
