"""CUDA XGBoost fitting, macro-F0.5 threshold tuning, and audit scoring."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from config import Config
from prepare import atomic_json, cache_dir


LOGGER = logging.getLogger(__name__)
SPLITS = ("fit", "early", "threshold", "audit")


def verify_cuda() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this container")
    free, total = torch.cuda.mem_get_info()
    probe = xgb.DMatrix(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), label=np.array([0, 0, 1, 1]))
    booster = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_depth": 2}, probe, num_boost_round=1)
    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost fell back to {device}")
    return {"gpu": torch.cuda.get_device_name(), "gpu_free_bytes": free, "gpu_total_bytes": total, "xgboost_device": device}


def load_arrays(config: Config, feature_names: list[str]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict]]:
    root = cache_dir(config)
    query_rows = pq.read_table(root / "queries.parquet").to_pylist()
    columns = ["split", "query_index", "label", *feature_names]
    vectors: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    labels: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    groups: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    parts = sorted((root / "parts").glob("part_*.parquet"))
    if not parts:
        raise ValueError("No cached feature parts exist")
    for number, part in enumerate(parts, 1):
        table = pq.read_table(part, columns=columns)
        frame = table.to_pandas()
        matrix = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite feature in {part}")
        split_values = frame["split"].to_numpy()
        for split in SPLITS:
            mask = split_values == split
            if mask.any():
                vectors[split].append(matrix[mask])
                labels[split].append(frame.loc[mask, "label"].to_numpy(dtype=np.uint8))
                groups[split].append(frame.loc[mask, "query_index"].to_numpy(dtype=np.int32))
        if number % 10 == 0 or number == len(parts):
            LOGGER.info("Loaded %s/%s cached feature parts", number, len(parts))
    arrays = {}
    for split in SPLITS:
        arrays[split] = (
            np.concatenate(vectors[split]) if vectors[split] else np.empty((0, len(feature_names)), dtype=np.float32),
            np.concatenate(labels[split]) if labels[split] else np.empty(0, dtype=np.uint8),
            np.concatenate(groups[split]) if groups[split] else np.empty(0, dtype=np.int32),
        )
        LOGGER.info("%s: %s pairs, %s positives", split, len(arrays[split][1]), int(arrays[split][1].sum()))
    return arrays, query_rows


def macro_at_threshold(probability: np.ndarray, label: np.ndarray, query_index: np.ndarray, truth_count: np.ndarray, selected_queries: np.ndarray, threshold: float) -> dict:
    selected = probability >= threshold
    size = len(truth_count)
    tp = np.bincount(query_index, weights=(selected & (label == 1)).astype(np.uint8), minlength=size)
    fp = np.bincount(query_index, weights=(selected & (label == 0)).astype(np.uint8), minlength=size)
    fn = truth_count - tp
    denominator = 1.25 * tp + 0.25 * fn + fp
    score = np.divide(1.25 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    score[(truth_count == 0) & (fp == 0)] = 1.0
    selected_scores = score[selected_queries]
    pair_tp = int(tp[selected_queries].sum())
    pair_fp = int(fp[selected_queries].sum())
    pair_fn = int(fn[selected_queries].sum())
    singleton = selected_queries[truth_count[selected_queries] == 0]
    return {
        "threshold": float(threshold),
        "macro_f0.5": float(selected_scores.mean()),
        "pair_precision": pair_tp / max(1, pair_tp + pair_fp),
        "pair_recall": pair_tp / max(1, pair_tp + pair_fn),
        "singleton_accuracy": float(np.mean(fp[singleton] == 0)) if len(singleton) else None,
        "tp": pair_tp, "fp": pair_fp, "fn": pair_fn,
    }


def threshold_grid(probability: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(probability, np.linspace(0.1, 0.9999, 180))
    fixed = np.linspace(0.05, 0.995, 100)
    return np.unique(np.r_[0.0, quantiles, fixed, 0.999, 0.9999, 1.0])


def score_split(probability: np.ndarray, label: np.ndarray, query_index: np.ndarray, query_rows: list[dict], split: str, threshold: float) -> dict:
    truth_count = np.array([row["truth_count"] for row in query_rows], dtype=np.int32)
    selected_queries = np.array([row["query_index"] for row in query_rows if row["split"] == split], dtype=np.int32)
    result = macro_at_threshold(probability, label, query_index, truth_count, selected_queries, threshold)
    prediction = probability >= threshold
    result["confusion_matrix"] = confusion_matrix(label, prediction, labels=[0, 1]).tolist()
    result["pr_auc"] = float(average_precision_score(label, probability)) if len(np.unique(label)) == 2 else None
    result["roc_auc"] = float(roc_auc_score(label, probability)) if len(np.unique(label)) == 2 else None
    result["query_count"] = len(selected_queries)
    result["retrieved_pair_count"] = len(label)
    result["retrieved_positive_count"] = int(label.sum())
    result["total_truth_count"] = int(truth_count[selected_queries].sum())
    result["candidate_recall"] = result["retrieved_positive_count"] / max(1, result["total_truth_count"])
    return result


def train(config: Config) -> dict:
    started = time.monotonic()
    gpu = verify_cuda()
    LOGGER.info("CUDA verified: %s; %.2f GiB free", gpu["gpu"], gpu["gpu_free_bytes"] / 2**30)
    root = cache_dir(config)
    manifest_path = root / "prepare_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare before train")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Feature cache is incomplete")
    features = manifest["feature_names"]
    arrays, query_rows = load_arrays(config, features)
    if any(len(np.unique(arrays[name][1])) < 2 for name in SPLITS):
        raise ValueError("Each split needs both classes; increase --max-queries")
    run_dir = config.output / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    matrices = {
        "fit": xgb.QuantileDMatrix(arrays["fit"][0], label=arrays["fit"][1], feature_names=features),
    }
    matrices["early"] = xgb.QuantileDMatrix(
        arrays["early"][0], label=arrays["early"][1], feature_names=features,
        ref=matrices["fit"],
    )
    params = {
        "objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist", "device": "cuda", "max_depth": config.max_depth,
        "eta": config.learning_rate, "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight, "lambda": config.reg_lambda,
        "max_bin": 256, "seed": config.seed, "nthread": config.nthread,
    }
    history: dict = {}
    booster = xgb.train(
        params, matrices["fit"], num_boost_round=config.rounds,
        evals=[(matrices["fit"], "fit"), (matrices["early"], "early")],
        early_stopping_rounds=config.early_stopping_rounds,
        maximize=True, evals_result=history,
        verbose_eval=25,
    )
    # XGBoost's early-stop prediction uses the best iteration explicitly.
    best_iteration = int(booster.best_iteration)
    LOGGER.info("Best iteration: %s, early AUPRC: %s", best_iteration, booster.best_score)
    model_path = run_dir / "model.json"
    temporary = run_dir / "model.tmp.json"
    booster.save_model(temporary)
    os.replace(temporary, model_path)
    predictions: dict[str, np.ndarray] = {}
    for name in ("threshold", "audit", "early"):
        matrix, _, _ = arrays[name]
        predictions[name] = booster.predict(xgb.DMatrix(matrix, feature_names=features), iteration_range=(0, best_iteration + 1))
        np.save(run_dir / f"{name}_probabilities.npy", predictions[name])
    truth_count = np.array([row["truth_count"] for row in query_rows], dtype=np.int32)
    threshold_queries = np.array([row["query_index"] for row in query_rows if row["split"] == "threshold"], dtype=np.int32)
    _, threshold_labels, threshold_groups = arrays["threshold"]
    curve = [macro_at_threshold(predictions["threshold"], threshold_labels, threshold_groups, truth_count, threshold_queries, value) for value in threshold_grid(predictions["threshold"])]
    best = max(curve, key=lambda row: (row["macro_f0.5"], row["pair_precision"], row["threshold"]))
    LOGGER.info("Selected threshold %.5f; calibration macro F0.5 %.5f", best["threshold"], best["macro_f0.5"])
    results = {name: score_split(predictions[name], arrays[name][1], arrays[name][2], query_rows, name, best["threshold"]) for name in ("threshold", "audit", "early")}
    LOGGER.info("Untouched audit macro F0.5 %.5f, precision %.5f, recall %.5f", results["audit"]["macro_f0.5"], results["audit"]["pair_precision"], results["audit"]["pair_recall"])
    gain = booster.get_score(importance_type="gain")
    importance = sorted(({"feature": feature, "gain": gain.get(feature, 0.0)} for feature in features), key=lambda row: row["gain"], reverse=True)
    atomic_json(run_dir / "feature_importance.json", {"importance": importance})
    atomic_json(run_dir / "threshold_curve.json", {"rows": curve})
    atomic_json(run_dir / "learning_curve.json", history)
    environment = {
        "python": platform.python_version(), "xgboost": xgb.__version__,
        "numpy": np.__version__, "gpu": gpu, "device_config": json.loads(booster.save_config())["learner"]["generic_param"]["device"],
    }
    results_document = {
        "config": config.serializable(), "cache_signature": manifest["signature"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "best_iteration": best_iteration, "selected_threshold": best["threshold"],
        "results": results, "environment": environment,
        "training_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "metrics.json", results_document)
    from diagnostics import create_diagnostics
    create_diagnostics(config, booster, arrays, query_rows, predictions, results_document, importance, history, curve)
    LOGGER.info("Training and diagnostics complete: %s", run_dir)
    return results_document
