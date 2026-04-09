# LLM-Fit Model Selection Notes

This notebook-style document is not part of the RAG runtime pipeline.

Its purpose is to document how LLM-Fit was used during the model selection phase of the project. It acts as a support artifact for:

- explaining hardware constraints
- justifying the choice of the local LLM
- showing that model selection was guided by feasibility

## Why this file exists

The final runtime pipeline uses:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB for vector retrieval
- Ollama with Phi-3 Mini for answer generation

LLM-Fit was used before that stage, only as a decision-support tool.

It was useful to answer a practical question:

- which local LLM is realistic for our laptops with 16 GB RAM and no dedicated GPU

## Hardware context

### Device 1

- Model: Asus Zenbook 14 OLED
- RAM: 16 GB
- CPU: Intel Core Ultra
- GPU: Intel Arc integrated graphics

### Device 2

- Model: HP 250 G9
- RAM: 16 GB
- CPU: Intel Core i5 12th Gen
- GPU: integrated graphics

## Suggested workflow with LLM-Fit

1. Open LLM-Fit on the target laptop.
2. Run the hardware analysis.
3. Review the candidate models suggested by the tool.
4. Compare them by local feasibility, expected responsiveness, and ease of deployment.
5. Discard options that are too heavy or too unstable for a reproducible local setup.
6. Keep the model that best matches the academic goal of a lightweight first iteration.

## How we interpret the results

LLM-Fit is not treated as a production dependency. It is treated as a preliminary evaluation tool.

The main criteria we care about are:

- whether the model can run comfortably within the available RAM
- whether it is realistic to use it without dedicated GPU
- whether installation is manageable on both machines
- whether the model is suitable for conversational interaction

## Project decision

For this project, Phi-3 Mini is the final choice because:

- it is lightweight enough for local use
- it integrates well with Ollama
- it is suitable for chat-based interaction
- it aligns with the scope of a first local RAG prototype

## How to cite this in the report

Suggested sentence:

> LLM-Fit was used as an auxiliary tool during the model selection phase in order to estimate the viability of different local LLMs on the available hardware. It was not integrated into the final RAG pipeline, but it supported the decision to use Phi-3 Mini as the most appropriate model for this first local iteration.

## Notes

- This file is intentionally descriptive rather than executable.
- It exists for documentation, reproducibility, and academic explanation.
