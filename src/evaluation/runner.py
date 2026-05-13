from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.dataset import load_eval_samples
from src.evaluation.ragas_runner import evaluate_predictions_with_ragas, list_eval_metrics
from src.evaluation.variants import list_eval_variants, run_variant_prediction


def run_evaluation(
    csv_path: str | Path,
    output_dir: str | Path,
    variants: list[str] | None = None,
    metrics: list[str] | None = None,
    max_rows: int | None = None,
    model_name: str | None = None,
    evaluator_model: str = "qwen3.5:4b",
    embeddings_model: str = "nomic-embed-text:latest",
    resume_run_dir: str | Path | None = None,
) -> Path:
    # Orquestador principal de la evaluación:
    # recorre variantes, genera respuestas, calcula métricas y guarda tablas.
    selected_variants = variants or list_eval_variants()
    selected_metrics = metrics or list_eval_metrics()
    samples = load_eval_samples(csv_path, max_rows=max_rows)

    if resume_run_dir:
        run_dir = Path(resume_run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"ragas_eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "ragas_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=["variant", *list_eval_metrics()])

    for variant_name in selected_variants:
        # Primero generamos las predicciones de la variante concreta.
        predictions = []
        for sample in samples:
            prediction = run_variant_prediction(
                variant_name=variant_name,
                question=sample.question,
                reference=sample.reference,
                model_name=model_name,
            )
            predictions.append(prediction)

        prediction_rows = [
            {
                "user_input": prediction.user_input,
                "retrieved_contexts": prediction.retrieved_contexts,
                "response": prediction.response,
                "reference": prediction.reference,
            }
            for prediction in predictions
        ]

        ragas_output = evaluate_predictions_with_ragas(
            eval_dataset_rows=prediction_rows,
            evaluator_model=evaluator_model,
            embeddings_model=embeddings_model,
            metric_names=selected_metrics,
        )

        variant_dir = run_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Guardamos tanto la respuesta final como el contexto recuperado para
        # poder inspeccionar después qué ocurrió en cada experimento.
        (variant_dir / "predictions.json").write_text(
            json.dumps([asdict(pred) for pred in predictions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        detailed_df = ragas_output.detailed_results.copy()
        detailed_df.insert(0, "variant", variant_name)
        detailed_path = variant_dir / "ragas_detailed.csv"
        if detailed_path.exists():
            existing_detailed_df = pd.read_csv(detailed_path)
            detailed_df = _merge_variant_details(existing_detailed_df, detailed_df)
        detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")

        # La tabla resumen se va completando de forma incremental.
        summary_df = _upsert_summary_row(
            summary_df=summary_df,
            variant_name=variant_name,
            metric_values=ragas_output.summary,
        )

    summary_df = _normalize_summary_columns(summary_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    (run_dir / "ragas_summary.md").write_text(
        _to_markdown_table(summary_df),
        encoding="utf-8",
    )

    config = {
        "csv_path": str(csv_path),
        "variants": selected_variants,
        "metrics": selected_metrics,
        "max_rows": max_rows,
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "embeddings_model": embeddings_model,
        "sample_count": len(samples),
        "resume_run_dir": str(run_dir) if resume_run_dir else None,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir


def _to_markdown_table(df: pd.DataFrame) -> str:
    # Evitamos depender de librerías extra para generar una tabla simple en Markdown.
    if df.empty:
        return "| variant | answer_relevancy | faithfulness | context_recall | factual_correctness |\n|---|---|---|---|---|\n"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body_lines = []
    for _, row in df.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row]
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body_lines]) + "\n"


def _upsert_summary_row(
    summary_df: pd.DataFrame,
    variant_name: str,
    metric_values: dict,
) -> pd.DataFrame:
    # Inserta una variante nueva o actualiza una ya existente si estamos
    # completando resultados en varias ejecuciones.
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=["variant", *list_eval_metrics()])

    if "variant" not in summary_df.columns:
        summary_df.insert(0, "variant", [])

    row_mask = summary_df["variant"] == variant_name
    if not row_mask.any():
        new_row = {"variant": variant_name}
        for metric_name in list_eval_metrics():
            new_row[metric_name] = metric_values.get(metric_name)
        return pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    row_index = summary_df.index[row_mask][0]
    for metric_name, metric_value in metric_values.items():
        if metric_name not in summary_df.columns:
            summary_df[metric_name] = None
        if metric_value is not None:
            summary_df.at[row_index, metric_name] = metric_value
    return summary_df


def _merge_variant_details(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Cuando reanudamos una evaluación, unimos las columnas nuevas con las
    # ya existentes sin perder resultados anteriores.
    key_columns = ["variant", "user_input"]
    for column in key_columns:
        if column not in existing_df.columns:
            existing_df[column] = None
        if column not in new_df.columns:
            new_df[column] = None

    merged_df = existing_df.merge(
        new_df,
        on=key_columns,
        how="outer",
        suffixes=("_old", ""),
    )

    for column in list(merged_df.columns):
        if column.endswith("_old"):
            base_column = column[:-4]
            if base_column in merged_df.columns:
                merged_df[base_column] = merged_df[base_column].combine_first(merged_df[column])
                merged_df = merged_df.drop(columns=[column])
            else:
                merged_df = merged_df.rename(columns={column: base_column})
    return merged_df


def _normalize_summary_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    # Mantiene el orden de columnas de la tabla final estable entre ejecuciones.
    desired_columns = ["variant", *list_eval_metrics()]
    for column in desired_columns:
        if column not in summary_df.columns:
            summary_df[column] = None
    return summary_df[desired_columns]
