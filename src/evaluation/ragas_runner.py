from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.embeddings.factory import build_embedding_model


@dataclass(slots=True)
class RagasRunOutput:
    summary: dict
    detailed_results: pd.DataFrame


def list_eval_metrics() -> list[str]:
    return [
        "answer_relevancy",
        "faithfulness",
        "context_recall",
        "factual_correctness",
    ]


def evaluate_predictions_with_ragas(
    eval_dataset_rows: list[dict],
    evaluator_model: str,
    embeddings_model: str | None = None,
    metric_names: list[str] | None = None,
) -> RagasRunOutput:
    prepared_rows = [_prepare_eval_row(row) for row in eval_dataset_rows]
    base_df = pd.DataFrame(
        {
            "user_input": [row["user_input"] for row in prepared_rows],
            "retrieved_contexts_text": [" ".join(row["retrieved_contexts"]) for row in prepared_rows],
            "response": [row["response"] for row in prepared_rows],
            "reference": [row["reference"] for row in prepared_rows],
        }
    )

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=evaluator_model,
            base_url="http://localhost:11434",
            temperature=0,
            num_ctx=1536,
            num_predict=96,
        )
    )
    if embeddings_model:
        local_embeddings_model = OllamaEmbeddings(
            model=embeddings_model,
            base_url="http://localhost:11434",
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            local_embeddings_model
        )
    else:
        local_embeddings_model = build_embedding_model()
        evaluator_embeddings = LangchainEmbeddingsWrapper(local_embeddings_model)

    metric_builders = {
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        "faithfulness": lambda: Faithfulness(llm=evaluator_llm),
        "context_recall": lambda: ContextRecall(llm=evaluator_llm),
        "factual_correctness": lambda: FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
            atomicity="low",
            coverage="low",
        ),
    }
    selected_metric_names = metric_names or list_eval_metrics()

    summary: dict[str, float | None] = {}
    detailed_df = base_df.copy()

    for metric_name in selected_metric_names:
        builder = metric_builders[metric_name]
        metric_df = _evaluate_single_metric(
            prepared_rows=prepared_rows,
            metric_name=metric_name,
            metric=builder(),
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )
        detailed_df = detailed_df.merge(metric_df, on="user_input", how="left")
        score_column = _find_metric_column(metric_df, metric_name)
        if score_column:
            source_column = f"{metric_name}_source"
            if source_column not in detailed_df.columns:
                detailed_df[source_column] = "ragas"
            _fill_metric_with_fallback(
                df=detailed_df,
                score_column=score_column,
                source_column=source_column,
                metric_name=metric_name,
                embeddings_model=local_embeddings_model,
            )
        summary[metric_name] = _safe_mean(detailed_df, score_column) if score_column else None

    return RagasRunOutput(summary=summary, detailed_results=detailed_df)


def _evaluate_single_metric(
    prepared_rows: list[dict],
    metric_name: str,
    metric,
    evaluator_llm,
    evaluator_embeddings,
) -> pd.DataFrame:
    dataset = EvaluationDataset.from_list(prepared_rows)
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(timeout=45, max_workers=1, max_retries=1),
            batch_size=1,
            raise_exceptions=False,
            show_progress=False,
        )
        metric_df = result.to_pandas()
        selected_columns = ["user_input"]
        score_column = _find_metric_column(metric_df, metric_name)
        if score_column:
            selected_columns.append(score_column)
            metric_df[f"{metric_name}_source"] = "ragas"
            selected_columns.append(f"{metric_name}_source")
        return metric_df[selected_columns]
    except Exception:
        return pd.DataFrame(
            {
                "user_input": [row["user_input"] for row in prepared_rows],
                metric_name: [None] * len(prepared_rows),
                f"{metric_name}_source": ["fallback"] * len(prepared_rows),
            }
        )


def _prepare_eval_row(row: dict) -> dict:
    truncated_contexts = [
        _truncate_text(context, max_chars=280)
        for context in row["retrieved_contexts"][:3]
    ]
    return {
        "user_input": _truncate_text(row["user_input"], max_chars=260),
        "retrieved_contexts": truncated_contexts,
        "response": _truncate_text(row["response"], max_chars=420),
        "reference": _truncate_text(row["reference"], max_chars=420),
    }


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _find_metric_column(df: pd.DataFrame, metric_name: str) -> str | None:
    if metric_name in df.columns:
        return metric_name
    for column in df.columns:
        if column.startswith(metric_name):
            return column
    return None


def _safe_mean(df: pd.DataFrame, column: str | None) -> float | None:
    if not column or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return float(series.mean())


def _fill_metric_with_fallback(
    df: pd.DataFrame,
    score_column: str,
    source_column: str,
    metric_name: str,
    embeddings_model,
) -> None:
    scores = pd.to_numeric(df[score_column], errors="coerce") if score_column in df.columns else pd.Series(dtype=float)
    if score_column not in df.columns:
        df[score_column] = None
        scores = pd.to_numeric(df[score_column], errors="coerce")
    if source_column not in df.columns:
        df[source_column] = "ragas"

    missing_indexes = scores[scores.isna()].index.tolist()
    if not missing_indexes:
        return

    for index in missing_indexes:
        row = df.loc[index]
        fallback_score = _fallback_metric_score(row, metric_name, embeddings_model)
        df.at[index, score_column] = fallback_score
        df.at[index, source_column] = "fallback"


def _fallback_metric_score(row: pd.Series, metric_name: str, embeddings_model) -> float:
    question = str(row.get("user_input", ""))
    response = str(row.get("response", ""))
    reference = str(row.get("reference", ""))

    if metric_name == "answer_relevancy":
        return _cosine_similarity_from_texts(question, response, embeddings_model)
    if metric_name == "faithfulness":
        contexts = str(row.get("retrieved_contexts_text", ""))
        return _cosine_similarity_from_texts(response, contexts, embeddings_model)
    if metric_name == "context_recall":
        contexts = str(row.get("retrieved_contexts_text", ""))
        return _keyword_overlap(reference, contexts)
    if metric_name == "factual_correctness":
        semantic = _cosine_similarity_from_texts(reference, response, embeddings_model)
        overlap = _keyword_overlap(reference, response)
        return round((semantic + overlap) / 2, 4)
    return 0.0


def _cosine_similarity_from_texts(text_a: str, text_b: str, embeddings_model) -> float:
    try:
        vector_a = np.array(embeddings_model.embed_query(text_a), dtype=float)
        vector_b = np.array(embeddings_model.embed_query(text_b), dtype=float)
        denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        if denominator == 0.0:
            return 0.0
        similarity = float(np.dot(vector_a, vector_b) / denominator)
        similarity = max(0.0, min(1.0, similarity))
        return round(similarity, 4)
    except Exception:
        return 0.0


def _keyword_overlap(reference: str, response: str) -> float:
    reference_tokens = {token.strip(".,;:!?()[]{}\"'").lower() for token in reference.split() if token.strip()}
    response_tokens = {token.strip(".,;:!?()[]{}\"'").lower() for token in response.split() if token.strip()}
    reference_tokens = {token for token in reference_tokens if len(token) > 2}
    response_tokens = {token for token in response_tokens if len(token) > 2}
    if not reference_tokens:
        return 0.0
    overlap = len(reference_tokens & response_tokens) / len(reference_tokens)
    return round(max(0.0, min(1.0, overlap)), 4)
