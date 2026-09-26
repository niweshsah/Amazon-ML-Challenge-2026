from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

from .io_utils import atomic_json, ensure_layout, load_config, output_root


def _parse_ids(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def _f05(truth: set[str], prediction: set[str]) -> float:
    if not truth:
        return 1.0 if not prediction else 0.0
    if not prediction:
        return 0.0
    tp = len(truth & prediction)
    precision = tp / len(prediction)
    recall = tp / len(truth)
    beta2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall) if tp else 0.0


def _predictions(rows: list[dict[str, Any]], k: int, threshold: float | None, margin: float | None) -> set[str]:
    eligible = [r for r in rows if r["rank"] <= k]
    if threshold is None:
        return {r["candidate_entity_id"] for r in eligible}
    best = max((r["cosine_similarity"] for r in eligible), default=-np.inf)
    return {
        r["candidate_entity_id"] for r in eligible
        if r["cosine_similarity"] >= threshold and (margin is None or r["cosine_similarity"] >= best - margin)
    }


def evaluate(config_path: str | None, split: str = "train", margin: float | None = None) -> dict[str, Any]:
    if split != "train":
        raise ValueError("Ground-truth evaluation is only permitted for split=train")
    config = load_config(config_path)
    ensure_layout(config)
    raw_dir = output_root(config) / "retrieval" / "train" / "raw_top50"
    summary = json.loads((raw_dir / "summary.json").read_text())
    raw = ds.dataset(summary["parts"], format="parquet").to_table().to_pylist()
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_query[row["source1_entity_id"]].append(row)
    truth_path = Path(config["paths"]["data_dir"]) / "train_ground_truth.parquet"
    truth: dict[str, set[str]] = {}
    wanted = set(by_query)
    for batch in ds.dataset(str(truth_path), format="parquet").scanner(batch_size=131072).to_batches():
        data = batch.to_pydict()
        for source1_id, matched in zip(data["source1_entity_id"], data["matched_entity_ids"]):
            if source1_id in wanted:
                truth[source1_id] = _parse_ids(matched)
    # Keep query IDs even if a malformed/missing GT row exists, treating it as empty and reporting coverage.
    matched_ground_truth_rows = len(truth)
    for query_id in wanted:
        truth.setdefault(query_id, set())
    ks = [1, 5, 10, 25, 50]
    recall_at_k = {}
    total_pairs = sum(len(value) for value in truth.values())
    for k in ks:
        hits = 0
        at_least = 0
        all_hits = 0
        single_hits = single_total = multi_hits = multi_total = 0
        for query_id, expected in truth.items():
            predicted = _predictions(by_query[query_id], k, None, None)
            found = len(expected & predicted)
            hits += found
            at_least += bool(found) if expected else 0
            all_hits += bool(expected) and found == len(expected)
            if len(expected) == 1:
                single_hits += found; single_total += 1
            elif len(expected) > 1:
                multi_hits += found; multi_total += len(expected)
        nonempty = sum(bool(x) for x in truth.values())
        recall_at_k[str(k)] = {
            "pair_recall": hits / total_pairs if total_pairs else 0.0,
            "entities_at_least_one_recalled": at_least / nonempty if nonempty else 0.0,
            "entities_all_matches_recalled": all_hits / nonempty if nonempty else 0.0,
            "single_match_pair_recall": single_hits / single_total if single_total else 0.0,
            "multi_match_pair_recall": multi_hits / multi_total if multi_total else 0.0,
        }
    scores = np.asarray([row["cosine_similarity"] for row in raw], dtype=np.float32)
    base_thresholds = config["retrieval"]["thresholds"]
    if len(scores):
        quantiles = np.quantile(scores, [0.5, 0.75, 0.9, 0.95, 0.975, 0.99])
        fine = [round(float(q + offset), 4) for q in quantiles for offset in (-0.02, -0.01, 0, 0.01, 0.02)]
    else:
        fine = []
    thresholds = sorted({float(x) for x in [*base_thresholds, *fine] if -1 <= float(x) <= 1})
    comparisons = []
    k = max(ks)
    for threshold in thresholds:
        tp = predicted_pairs = 0
        f_scores = []
        singleton_correct = singleton_total = 0
        counts = []
        for query_id, expected in truth.items():
            predicted = _predictions(by_query[query_id], k, threshold, margin)
            tp += len(expected & predicted)
            predicted_pairs += len(predicted)
            f_scores.append(_f05(expected, predicted))
            counts.append(len(predicted))
            if not expected:
                singleton_total += 1
                singleton_correct += not predicted
        comparisons.append({
            "threshold": threshold, "pair_precision": tp / predicted_pairs if predicted_pairs else 1.0,
            "pair_recall": tp / total_pairs if total_pairs else 0.0,
            "macro_f0_5": statistics.fmean(f_scores) if f_scores else 0.0,
            "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
            "predicted_matches_per_query": {
                "mean": statistics.fmean(counts) if counts else 0.0, "median": statistics.median(counts) if counts else 0.0,
                "p95": float(np.quantile(counts, .95)) if counts else 0.0, "max": max(counts, default=0),
            },
        })
    best = max(comparisons, key=lambda x: x["macro_f0_5"]) if comparisons else None
    report = {
        "generated_at_epoch": time.time(), "debug_only": bool(summary.get("debug_only")),
        "queries": len(truth), "ground_truth_coverage": matched_ground_truth_rows / max(len(wanted), 1),
        "true_pairs": total_pairs, "recall_at_k": recall_at_k, "score_distribution": {
            "min": float(scores.min()) if len(scores) else None, "median": float(np.median(scores)) if len(scores) else None,
            "p95": float(np.quantile(scores, .95)) if len(scores) else None, "max": float(scores.max()) if len(scores) else None,
        }, "margin": margin, "threshold_comparison": comparisons, "best_threshold": best,
    }
    reports = output_root(config) / "reports"
    atomic_json(reports / "train_retrieval_metrics.json", report)
    if not report["debug_only"] and best:
        atomic_json(reports / "selected_config.json", {
            "frozen_from": "train_retrieval_metrics.json", "model": config["model"], "text": config["text"],
            "top_k": k, "threshold": best["threshold"], "margin": margin,
        })
    lines = ["# Train retrieval metrics", ""]
    if report["debug_only"]:
        lines += ["> **DEBUG ONLY:** target sampling was enabled; do not use these metrics or threshold for test.", ""]
    lines += [f"Queries: {len(truth):,}; true pairs: {total_pairs:,}", "", "## Recall", "", "| K | Pair recall | Any true match | All true matches |", "|---:|---:|---:|---:|"]
    for key, value in recall_at_k.items():
        lines.append(f"| {key} | {value['pair_recall']:.6f} | {value['entities_at_least_one_recalled']:.6f} | {value['entities_all_matches_recalled']:.6f} |")
    lines += ["", "## Thresholds", "", "| Threshold | Precision | Recall | Macro F0.5 | Singleton accuracy | Mean predictions |", "|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        singleton = "n/a" if row["singleton_accuracy"] is None else f"{row['singleton_accuracy']:.6f}"
        lines.append(f"| {row['threshold']:.4f} | {row['pair_precision']:.6f} | {row['pair_recall']:.6f} | {row['macro_f0_5']:.6f} | {singleton} | {row['predicted_matches_per_query']['mean']:.3f} |")
    (reports / "train_retrieval_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
