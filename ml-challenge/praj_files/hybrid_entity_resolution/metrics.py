"""Retrieval and entity resolution metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def truth_map(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for row in ground_truth.itertuples(index=False):
        matched_ids = row.matched_entity_ids
        if matched_ids is None:
            targets: set[str] = set()
        else:
            targets = {str(target_id) for target_id in matched_ids if target_id is not None}
        mapping[str(row.source1_entity_id)] = targets
    return mapping


def candidate_recall(
    master_pool_path: str,
    ground_truth: pd.DataFrame,
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    """Compute positive-pair recall from a potentially large Parquet cache."""
    truth = truth_map(ground_truth)
    total = sum(len(targets) for targets in truth.values())
    captured = 0
    parquet_file = pq.ParquetFile(master_pool_path)
    for batch in parquet_file.iter_batches(
        columns=["query_id", "target_id"], batch_size=batch_size
    ):
        query_ids = batch.column(0).to_pylist()
        target_ids = batch.column(1).to_pylist()
        captured += sum(
            target_id in truth.get(query_id, set())
            for query_id, target_id in zip(query_ids, target_ids)
        )
    return {
        "captured_positive_pairs": int(captured),
        "total_positive_pairs": int(total),
        "candidate_recall": float(captured / total) if total else 1.0,
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    query_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute pair precision/recall and challenge-style macro F0.5 per query."""
    truth = truth_map(ground_truth)
    predicted = dict(
        zip(predictions["query_id"].astype(str), predictions["prediction"].astype(str))
    )
    selected_ids = list(query_ids) if query_ids is not None else list(truth)
    beta_squared = 0.25
    f_values: list[float] = []
    total_true_positive = 0
    total_predicted = 0
    total_truth = 0

    for query_id in selected_ids:
        true_targets = truth.get(str(query_id), set())
        prediction = predicted.get(str(query_id), "NO_MATCH")
        predicted_targets = set() if prediction == "NO_MATCH" else {prediction}
        overlap = len(true_targets & predicted_targets)
        false_positive = len(predicted_targets) - overlap
        false_negative = len(true_targets) - overlap
        denominator = (1.0 + beta_squared) * overlap + false_positive + beta_squared * false_negative
        f_beta = (
            (1.0 + beta_squared) * overlap / denominator
            if denominator
            else 1.0
        )
        f_values.append(f_beta)
        total_true_positive += overlap
        total_predicted += len(predicted_targets)
        total_truth += len(true_targets)

    return {
        "queries": len(selected_ids),
        "precision": total_true_positive / total_predicted if total_predicted else 0.0,
        "recall": total_true_positive / total_truth if total_truth else 1.0,
        "macro_f0_5": float(np.mean(f_values)) if selected_ids else 0.0,
        "true_positive_pairs": total_true_positive,
        "predicted_pairs": total_predicted,
        "ground_truth_pairs": total_truth,
    }
