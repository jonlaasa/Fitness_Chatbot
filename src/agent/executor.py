from __future__ import annotations

import subprocess

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.agent.tools import get_agent_tools
from src.llm.local_model import resolve_ollama_model


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "Answer the user's question as best you can using the available tools. "
    "Always use the local tools when they are useful. "
    "Stay within the domains of exercises, training, nutrition, and healthy habits. "
    "Do not invent tool results."
)


def build_chat_model(model_name: str | None = None) -> ChatOllama:
    resolved_model = model_name or resolve_agent_model()
    return ChatOllama(
        base_url="http://localhost:11434",
        model=resolved_model,
        num_ctx=2048,
        seed=42,
    )


def resolve_agent_model() -> str:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        model_names = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                model_names.append(parts[0])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return resolve_ollama_model()

    for preferred_prefix in ("qwen", "llama3", "mistral"):
        for model_name in model_names:
            if model_name.startswith(preferred_prefix):
                return model_name

    return resolve_ollama_model()


def build_agent_executor(
    model_name: str | None = None,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
):
    llm = build_chat_model(model_name)
    tools = get_agent_tools()
    return create_agent(llm, tools, system_prompt=system_prompt)
