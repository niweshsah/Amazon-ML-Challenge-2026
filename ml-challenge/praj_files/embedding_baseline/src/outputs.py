from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .io_utils import ensure_layout, load_config, output_root


def write_outputs(
    config_path: str | None, split: str, threshold: float, top_k: int = 50,
    margin: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_layout(config)
    raw_dir = output_root(config) / "retrieval" / split / "raw_top50"
    out_dir = output_root(config) / "outputs"
    prediction_path = out_dir / f"{split}_baseline_predictions.tsv"
    candidate_path = out_dir / "candidate_pairs.tsv" if split == "test" else out_dir / f"{split}_candidate_pairs.tsv"

    pred_fd, pred_tmp = tempfile.mkstemp(prefix=f".{prediction_path.name}.", suffix=".tmp", dir=out_dir)
    cand_fd, cand_tmp = tempfile.mkstemp(prefix=f".{candidate_path.name}.", suffix=".tmp", dir=out_dir)
    query_count = 0
    try:
        with os.fdopen(pred_fd, "w", newline="", encoding="utf-8") as pred_handle, os.fdopen(cand_fd, "w", newline="", encoding="utf-8") as cand_handle:
            pred_writer = csv.writer(pred_handle, delimiter="\t", lineterminator="\n")
            cand_writer = csv.writer(cand_handle, delimiter="\t", lineterminator="\n")
            pred_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            summary = json.loads((raw_dir / "summary.json").read_text())
            for part_text in summary["parts"]:
                part_path = Path(part_text)
                by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
                parquet_file = pq.ParquetFile(part_path)
                for batch in parquet_file.iter_batches(batch_size=131072):
                    for row in batch.to_pylist():
                        by_query[row["source1_entity_id"]].append(row)
                for query_id, query_rows in by_query.items():
                    eligible = [row for row in query_rows if row["rank"] <= top_k]
                    eligible.sort(key=lambda row: (-row["cosine_similarity"], row["candidate_entity_id"]))
                    candidate_ids = list(dict.fromkeys(row["candidate_entity_id"] for row in eligible))
                    best = max((row["cosine_similarity"] for row in eligible), default=-2.0)
                    predicted_ids = list(dict.fromkeys(
                        row["candidate_entity_id"] for row in eligible
                        if row["cosine_similarity"] >= threshold
                        and (margin is None or row["cosine_similarity"] >= best - margin)
                    ))
                    pred_writer.writerow((query_id, ",".join(predicted_ids)))
                    cand_writer.writerow((query_id, ",".join(candidate_ids)))
                    query_count += 1
            pred_handle.flush(); os.fsync(pred_handle.fileno())
            cand_handle.flush(); os.fsync(cand_handle.fileno())
        os.replace(pred_tmp, prediction_path); os.replace(cand_tmp, candidate_path)
    finally:
        if os.path.exists(pred_tmp): os.unlink(pred_tmp)
        if os.path.exists(cand_tmp): os.unlink(cand_tmp)
    return {"queries": query_count, "predictions": str(prediction_path), "candidates": str(candidate_path)}
