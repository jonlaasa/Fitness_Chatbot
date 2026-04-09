# Prompt Engineering Experiments

This document is an academic support note for the prompt engineering part of the project.

It is not a replacement for the main RAG pipeline. Instead, it sits on top of the existing retrieval system and allows controlled comparison of several prompting techniques.

## Goal

The goal is to compare how different prompt formulations affect the final answer when the retrieved context is kept fixed.

The techniques included in this repository are:

- zero-shot
- one-shot
- few-shot
- chain-of-thought

## Why this is useful

This experiment helps answer an important project question:

- if retrieval stays the same, does prompt style change the final answer quality, structure, or groundedness?

That makes it a good section for academic analysis because it isolates the prompting variable from the retrieval variable.

## How it is integrated

The main query script supports a strategy flag:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --strategy zero-shot
```

Available strategies:

- `zero-shot`
- `one-shot`
- `few-shot`
- `chain-of-thought`

There is also a comparison script:

```powershell
.\.venv\Scripts\python scripts\compare_prompting.py --question "Which exercises target the glutes?"
```

That script runs the same question through every strategy and prints the outputs one after another.

## Strategy overview

### Zero-shot

The model receives direct instructions and the retrieved context, but no examples.

### One-shot

The model receives one miniature example before the real question.

### Few-shot

The model receives two short examples before the real question.

### Chain-of-thought

The model is encouraged to reason more carefully before answering, but the prompt asks for a concise final answer and a short justification instead of exposing long hidden reasoning.

## Recommended evaluation criteria

When comparing outputs, useful criteria are:

- factual grounding in retrieved context
- clarity
- conciseness
- tendency to hallucinate
- consistency across repeated runs

## Suggested write-up idea

You can describe this module as:

> A prompt engineering layer was added on top of the local RAG pipeline in order to compare different prompting strategies while keeping retrieval fixed. This allowed us to study the influence of zero-shot, one-shot, few-shot, and chain-of-thought style prompting on the quality and grounding of the generated answers.
