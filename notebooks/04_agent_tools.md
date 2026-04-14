# Agent Tools for the Fitness Project

This note documents the agent-oriented part of the project inspired by the class example.

## Why add an agent layer

The main RAG pipeline already answers questions using retrieval plus generation. However, the class example introduces an additional idea: a local agent that can decide when to call tools.

That is useful for this project because it allows us to show a second interaction style:

- the standard RAG pipeline
- a simple agent with tools

## Agent design in this project

The agent is intentionally small and simple. It uses:

- `ChatOllama` as the local chat model
- `create_agent` from LangChain
- two domain-relevant tools

## Tools included

### 1. `search_fitness_knowledge`

This tool queries the local Chroma database and retrieves relevant exercise or nutrition documents from the same knowledge base used by the RAG system.

Use cases:

- find exercises for a muscle group
- retrieve dish information
- inspect local training or nutrition knowledge

### 2. `fitness_calculator`

This tool evaluates simple calculations with `numexpr`.

Use cases:

- BMI
- calorie arithmetic
- macro totals
- training volume calculations
- unit conversions

## Why these tools make sense

They are both directly connected to the project domain:

- one tool retrieves domain knowledge
- one tool performs useful domain calculations

This makes them more meaningful than adding a generic web search for a local first iteration.

## Example command

```powershell
.\.venv\Scripts\python scripts\run_agent.py --question "Find exercises for glutes and calculate the BMI for 80 kg and 1.78 m"
```

## Academic interpretation

This agent does not replace the main RAG pipeline. It complements it by showing how tool use can be added on top of a local LLM workflow in a controlled and explainable way.
