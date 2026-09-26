"""Audit-only plots, explanations, errors, and subgroup metrics."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from config import Config
from prepare import atomic_json, cache_dir
from train import macro_at_threshold


LOGGER = logging.getLogger(__name__)


def _save(path, title: str, xlabel: str = "", ylabel: str = "") -> None:
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _audit_rows(config: Config) -> list[dict]:
    rows = []
    for part in sorted((cache_dir(config) / "parts").glob("part_*.parquet")):
        table = pq.read_table(part, columns=[
            "source1_entity_id", "candidate_entity_id", "query_index", "split", "label",
            "raw_name_t1", "raw_name_t2", "raw_address_t1", "raw_address_t2",
            "country_t1", "source_t2", "address_missing_right", "script_target_nonlatin",
        ])
        rows.extend(row for row in table.to_pylist() if row["split"] == "audit")
    return rows


def create_diagnostics(config: Config, booster, arrays: dict, query_rows: list[dict], predictions: dict, metrics: dict, importance: list[dict], history: dict, curve: list[dict]) -> None:
    run_dir = config.output / "run"
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    threshold = metrics["selected_threshold"]
    probability = predictions["audit"]
    _, labels, groups = arrays["audit"]
    predicted = probability >= threshold

    plt.figure(figsize=(8, 5))
    for label, name in ((0, "Nonmatch"), (1, "Match")):
        plt.hist(probability[labels == label], bins=80, density=True, alpha=0.55, label=name)
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.3f}")
    plt.yscale("log")
    plt.legend()
    _save(plots / "score_distributions.png", "Audit score distributions", "Predicted probability", "Density (log)")

    plt.figure(figsize=(8, 5))
    recalls, precisions = None, None
    precisions, recalls, _ = precision_recall_curve(labels, probability)
    plt.plot(recalls, precisions)
    _save(plots / "precision_recall.png", "Audit pair precision–recall", "Recall", "Precision")

    plt.figure(figsize=(8, 5))
    fpr, tpr, _ = roc_curve(labels, probability)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    _save(plots / "roc.png", "Audit pair ROC", "False positive rate", "True positive rate")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(labels, predicted, labels=[0, 1], ax=ax, colorbar=False)
    _save(plots / "confusion_matrix.png", "Audit pair confusion matrix")

    plt.figure(figsize=(8, 5))
    plt.plot([row["threshold"] for row in curve], [row["macro_f0.5"] for row in curve], label="Macro F0.5")
    plt.plot([row["threshold"] for row in curve], [row["pair_precision"] for row in curve], label="Pair precision", alpha=0.7)
    plt.plot([row["threshold"] for row in curve], [row["pair_recall"] for row in curve], label="Pair recall", alpha=0.7)
    plt.axvline(threshold, color="black", linestyle="--")
    plt.legend()
    _save(plots / "threshold_curve.png", "Threshold selection on calibration split", "Threshold", "Score")

    plt.figure(figsize=(8, 5))
    for split in ("fit", "early"):
        plt.plot(history[split]["aucpr"], label=f"{split} AUPRC")
    plt.legend()
    _save(plots / "learning_curve.png", "XGBoost learning curve", "Boosting iteration", "AUPRC")

    top = importance[:30][::-1]
    plt.figure(figsize=(9, 9))
    plt.barh([row["feature"] for row in top], [row["gain"] for row in top])
    _save(plots / "feature_importance.png", "XGBoost gain importance", "Gain")

    # Native XGBoost contributions are TreeSHAP values; no separate shap package needed.
    matrix = arrays["audit"][0]
    if len(matrix):
        rng = np.random.default_rng(config.seed)
        sample_idx = rng.choice(len(matrix), min(2_000, len(matrix)), replace=False)
        feature_names = json.loads((cache_dir(config) / "prepare_manifest.json").read_text())["feature_names"]
        sample = xgb.DMatrix(matrix[sample_idx], feature_names=feature_names)
        contributions = booster.predict(sample, pred_contribs=True, iteration_range=(0, metrics["best_iteration"] + 1))
        mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-25:]
        plt.figure(figsize=(9, 8))
        plt.barh([feature_names[idx] for idx in top_idx], mean_abs[top_idx])
        _save(plots / "shap_importance.png", "Mean absolute TreeSHAP contribution", "Mean |log-odds contribution|")
        atomic_json(run_dir / "shap_summary.json", {"sample_size": len(sample_idx), "mean_absolute_contributions": {name: float(value) for name, value in zip(feature_names, mean_abs)}})

    audit_rows = _audit_rows(config)
    if len(audit_rows) != len(probability):
        raise ValueError("Audit rows do not align with cached predictions")
    for row, score, selection in zip(audit_rows, probability, predicted):
        row["score"] = float(score)
        row["predicted"] = bool(selection)
    with (run_dir / "audit_predictions.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source1_entity_id", "candidate_entity_id", "label", "score", "predicted"], delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)
    error_rows = sorted((row for row in audit_rows if row["label"] != row["predicted"]), key=lambda row: (-row["score"] if not row["label"] else row["score"]))[:500]
    with (run_dir / "error_analysis.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["source1_entity_id", "candidate_entity_id", "label", "score", "raw_name_t1", "raw_name_t2", "raw_address_t1", "raw_address_t2", "country_t1", "source_t2"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(error_rows)

    truth_count = np.array([row["truth_count"] for row in query_rows], dtype=np.int32)
    audit_query_ids = np.array([row["query_index"] for row in query_rows if row["split"] == "audit"], dtype=np.int32)
    query_meta = {row["query_index"]: row for row in query_rows}
    subgroup_specs = {
        "source2_pairs": np.array([row["source_t2"] == "S2" for row in audit_rows]),
        "source3_pairs": np.array([row["source_t2"] == "S3" for row in audit_rows]),
        "target_address_missing": np.array([bool(row["address_missing_right"]) for row in audit_rows]),
        "target_nonlatin": np.array([bool(row["script_target_nonlatin"]) for row in audit_rows]),
    }
    subgroups = {}
    for name, mask in subgroup_specs.items():
        if mask.any():
            selected = predicted[mask]
            truth = labels[mask].astype(bool)
            tp, fp, fn = int((selected & truth).sum()), int((selected & ~truth).sum()), int((~selected & truth).sum())
            subgroups[name] = {"pairs": int(mask.sum()), "tp": tp, "fp": fp, "fn_among_candidates": fn, "precision": tp / max(1, tp + fp), "recall_among_candidates": tp / max(1, tp + fn)}
    for name, predicate in (
        ("US", lambda row: row["country"] == "US"),
        ("India", lambda row: row["country"] == "India"),
        ("singleton", lambda row: row["truth_count"] == 0),
        ("multi_match", lambda row: row["truth_count"] > 1),
    ):
        selected_queries = np.array([idx for idx in audit_query_ids if predicate(query_meta[idx])], dtype=np.int32)
        if len(selected_queries):
            subgroups[name] = macro_at_threshold(probability, labels, groups, truth_count, selected_queries, threshold)
            subgroups[name]["queries"] = len(selected_queries)
    atomic_json(run_dir / "subgroups.json", subgroups)
    LOGGER.info("Saved audit plots, explanations, predictions, errors, and subgroup metrics")
