"""Dynamic candidate slicing and pair feature extraction."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .metrics import candidate_recall


LOGGER = logging.getLogger(__name__)
_NUMBER_PATTERN = re.compile(r"\d+")
FEATURE_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("target_id", pa.string()),
        ("dense_score", pa.float32()),
        ("sparse_score", pa.float32()),
        ("combined_score", pa.float32()),
        ("levenshtein_ratio", pa.float32()),
        ("token_set_ratio", pa.float32()),
        ("token_sort_ratio", pa.float32()),
        ("len_diff_abs", pa.int32()),
        ("digits_exact_match", pa.float32()),
    ]
)


@dataclass
class PairFeatureExtractor:
    """Select fused Top-K pairs and calculate string similarity features."""

    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    read_batch_size: int = 500_000

    def _select_top_k(self, frame: pd.DataFrame, max_k: int) -> pd.DataFrame:
        frame = frame.copy()
        frame["combined_score"] = (
            self.dense_weight * frame["dense_score"]
            + self.sparse_weight * frame["sparse_score"]
        )
        return (
            frame.sort_values(
                ["query_id", "combined_score"],
                ascending=[True, False],
                kind="stable",
            )
            .groupby("query_id", sort=False, as_index=False)
            .head(max_k)
        )

    @staticmethod
    def _features_for_pairs(
        pairs: pd.DataFrame,
        query_text: dict[str, str],
        target_text: dict[str, str],
    ) -> pd.DataFrame:
        query_values = [query_text[str(value)] for value in pairs["query_id"]]
        target_values = [target_text[str(value)] for value in pairs["target_id"]]
        pairs = pairs.copy()
        pairs["levenshtein_ratio"] = np.asarray(
            [
                Levenshtein.normalized_similarity(query, target)
                for query, target in zip(query_values, target_values)
            ],
            dtype=np.float32,
        )
        pairs["token_set_ratio"] = np.asarray(
            [
                fuzz.token_set_ratio(query, target) / 100.0
                for query, target in zip(query_values, target_values)
            ],
            dtype=np.float32,
        )
        pairs["token_sort_ratio"] = np.asarray(
            [
                fuzz.token_sort_ratio(query, target) / 100.0
                for query, target in zip(query_values, target_values)
            ],
            dtype=np.float32,
        )
        pairs["len_diff_abs"] = np.asarray(
            [abs(len(query) - len(target)) for query, target in zip(query_values, target_values)],
            dtype=np.int32,
        )
        pairs["digits_exact_match"] = np.asarray(
            [
                float(set(_NUMBER_PATTERN.findall(query)) == set(_NUMBER_PATTERN.findall(target)))
                for query, target in zip(query_values, target_values)
            ],
            dtype=np.float32,
        )
        return pairs[list(FEATURE_SCHEMA.names)]

    def extract(
        self,
        master_pool_path: Path,
        normalized_t1_path: Path,
        normalized_t2_path: Path,
        ground_truth_path: Path,
        output_path: Path,
        max_k: int = 50,
    ) -> dict[str, Any]:
        """Stream the cache, slice each query group, and persist pair features."""
        if max_k <= 0 or max_k > 300:
            raise ValueError("max_k must be between 1 and 300")
        started = time.perf_counter()
        query_frame = pd.read_parquet(
            normalized_t1_path, columns=["entity_id", "normalized_text"]
        )
        target_frame = pd.read_parquet(
            normalized_t2_path, columns=["entity_id", "normalized_text"]
        )
        query_text = dict(
            zip(query_frame["entity_id"].astype(str), query_frame["normalized_text"])
        )
        target_text = dict(
            zip(target_frame["entity_id"].astype(str), target_frame["normalized_text"])
        )
        del query_frame, target_frame

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary_path, FEATURE_SCHEMA, compression="zstd")
        parquet_file = pq.ParquetFile(master_pool_path)
        pending = pd.DataFrame()
        written = 0
        try:
            for batch in parquet_file.iter_batches(batch_size=self.read_batch_size):
                current = batch.to_pandas()
                if not pending.empty:
                    current = pd.concat((pending, current), ignore_index=True)
                final_query_id = current["query_id"].iloc[-1]
                complete = current[current["query_id"] != final_query_id]
                pending = current[current["query_id"] == final_query_id].copy()
                if complete.empty:
                    continue
                selected = self._select_top_k(complete, max_k)
                featured = self._features_for_pairs(selected, query_text, target_text)
                writer.write_table(pa.Table.from_pandas(featured, schema=FEATURE_SCHEMA, preserve_index=False))
                written += len(featured)
                LOGGER.info("Extracted %s pair feature rows", f"{written:,}")

            if not pending.empty:
                selected = self._select_top_k(pending, max_k)
                featured = self._features_for_pairs(selected, query_text, target_text)
                writer.write_table(pa.Table.from_pandas(featured, schema=FEATURE_SCHEMA, preserve_index=False))
                written += len(featured)
        finally:
            writer.close()
        os.replace(temporary_path, output_path)

        ground_truth = pd.read_parquet(ground_truth_path)
        metrics = candidate_recall(str(output_path), ground_truth)
        metrics.update(
            {
                "max_k": max_k,
                "feature_rows": written,
                "feature_path": str(output_path),
                "feature_extraction_seconds": time.perf_counter() - started,
            }
        )
        LOGGER.info(
            "Top-%d feature-pool recall: %.6f; extraction time: %.2fs",
            max_k,
            metrics["candidate_recall"],
            metrics["feature_extraction_seconds"],
        )
        return metrics
