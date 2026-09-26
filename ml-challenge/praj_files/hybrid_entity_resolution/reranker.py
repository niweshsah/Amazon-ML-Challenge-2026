"""XGBoost pair reranking, thresholded inference, and evaluation."""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .metrics import evaluate_predictions


LOGGER = logging.getLogger(__name__)
MODEL_FEATURES = [
    "dense_score",
    "sparse_score",
    "levenshtein_ratio",
    "token_set_ratio",
    "token_sort_ratio",
    "len_diff_abs",
    "digits_exact_match",
]


def _is_validation_query(query_id: str, validation_fraction: float) -> bool:
    bucket = zlib.crc32(query_id.encode("utf-8")) % 10_000
    return bucket < round(validation_fraction * 10_000)


@dataclass
class EntityReranker:
    """Train and apply an XGBoost binary pair classifier."""

    threshold: float = 0.50
    validation_fraction: float = 0.20
    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.08
    device: str = "cuda"
    random_state: int = 42

    def _new_model(self, labels: np.ndarray) -> XGBClassifier:
        positives = int(labels.sum())
        negatives = len(labels) - positives
        scale_pos_weight = negatives / positives if positives else 1.0
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=self.device,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    @staticmethod
    def _add_labels(features: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
        positives = ground_truth[["source1_entity_id", "matched_entity_ids"]].explode(
            "matched_entity_ids"
        )
        positives = positives.dropna(subset=["matched_entity_ids"])
        positive_pairs = set(
            zip(
                positives["source1_entity_id"].astype(str),
                positives["matched_entity_ids"].astype(str),
            )
        )
        output = features.copy()
        output["label"] = np.fromiter(
            (
                (str(query_id), str(target_id)) in positive_pairs
                for query_id, target_id in zip(output["query_id"], output["target_id"])
            ),
            dtype=np.int8,
            count=len(output),
        )
        return output

    def _predictions(self, pairs: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
        scored = pairs[["query_id", "target_id"]].copy()
        scored["match_probability"] = probabilities.astype(np.float32, copy=False)
        top_indices = scored.groupby("query_id", sort=False)["match_probability"].idxmax()
        top = scored.loc[top_indices].reset_index(drop=True)
        top["prediction"] = np.where(
            top["match_probability"] >= self.threshold,
            top["target_id"],
            "NO_MATCH",
        )
        return top[["query_id", "prediction", "target_id", "match_probability"]]

    def train_evaluate(
        self,
        feature_path: Path,
        ground_truth_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Evaluate by query split, fit the final model, and write all predictions."""
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_parquet(feature_path)
        ground_truth = pd.read_parquet(ground_truth_path)
        labeled = self._add_labels(features, ground_truth)
        labeled_path = output_dir / "labeled_candidate_features.parquet"
        labeled.to_parquet(labeled_path, index=False)
        if int(labeled["label"].sum()) == 0:
            raise ValueError("No positive candidate pairs remain after dynamic slicing")

        validation_query_mask = labeled["query_id"].astype(str).map(
            lambda value: _is_validation_query(value, self.validation_fraction)
        )
        validation_query_ids = set(labeled.loc[validation_query_mask, "query_id"].astype(str))
        train_mask = ~validation_query_mask
        if not validation_query_ids or not train_mask.any():
            raise ValueError("validation_fraction must leave nonempty train and validation queries")

        validation_model = self._new_model(labeled.loc[train_mask, "label"].to_numpy())
        validation_model.fit(
            labeled.loc[train_mask, MODEL_FEATURES],
            labeled.loc[train_mask, "label"],
        )
        validation_pairs = labeled.loc[validation_query_mask]
        validation_probabilities = validation_model.predict_proba(
            validation_pairs[MODEL_FEATURES]
        )[:, 1]
        validation_predictions = self._predictions(
            validation_pairs, validation_probabilities
        )
        validation_predictions.to_parquet(
            output_dir / "validation_predictions.parquet", index=False
        )
        validation_metrics = evaluate_predictions(
            validation_predictions,
            ground_truth,
            query_ids=sorted(validation_query_ids),
        )
        LOGGER.info("Validation metrics: %s", validation_metrics)

        final_model = self._new_model(labeled["label"].to_numpy())
        final_model.fit(labeled[MODEL_FEATURES], labeled["label"])
        model_path = output_dir / "entity_reranker.ubj"
        final_model.save_model(model_path)
        probabilities = final_model.predict_proba(labeled[MODEL_FEATURES])[:, 1]
        predictions = self._predictions(labeled, probabilities)
        predictions_path = output_dir / "predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)

        return {
            "validation": validation_metrics,
            "threshold": self.threshold,
            "training_pairs": int(len(labeled)),
            "positive_training_pairs": int(labeled["label"].sum()),
            "labeled_features_path": str(labeled_path),
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "reranker_seconds": time.perf_counter() - started,
        }
