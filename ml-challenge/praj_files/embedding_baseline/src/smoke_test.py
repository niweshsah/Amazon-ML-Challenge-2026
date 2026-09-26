from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .embed import create_embedder, encode_with_retry
from .gpu_utils import gpu_memory
from .io_utils import atomic_json, ensure_layout, load_config, output_root, parquet_dataset
from .text_builder import build_texts


def _ids(value: str | None) -> set[str]:
    return {x.strip() for x in (value or "").split(",") if x.strip()}


def _unique_rows(table: pa.Table) -> pa.Table:
    rows = []
    seen = set()
    for row in table.to_pylist():
        if row["entity_id"] not in seen:
            seen.add(row["entity_id"]); rows.append(row)
    return pa.Table.from_pylist(rows, schema=table.schema)


def smoke_ground_truth(config_path: str | None, queries: int = 32, distractors: int = 256) -> dict[str, Any]:
    """Quality smoke test with all true targets injected; explicitly not full-pool Recall@K."""
    config = load_config(config_path)
    ensure_layout(config)
    data = Path(config["paths"]["data_dir"])
    columns = ["entity_id", "business_name", "business_address", "country"]
    query_table = parquet_dataset(data / "train_source1.parquet").head(queries, columns=columns)
    query_ids = query_table["entity_id"].to_pylist()
    gt_table = parquet_dataset(data / "train_ground_truth.parquet").to_table(
        filter=ds.field("source1_entity_id").isin(query_ids)
    )
    truth = {row["source1_entity_id"]: _ids(row["matched_entity_ids"]) for row in gt_table.to_pylist()}
    true_target_ids = set().union(*(truth.get(query_id, set()) for query_id in query_ids))
    target_tables = {}
    for source in ("source2", "source3"):
        source_prefix = "S2-" if source == "source2" else "S3-"
        ids_for_source = [value for value in true_target_ids if value.startswith(source_prefix)]
        dataset = parquet_dataset(data / f"train_{source}.parquet")
        matched = dataset.to_table(columns=columns, filter=ds.field("entity_id").isin(ids_for_source))
        target_tables[source] = _unique_rows(pa.concat_tables([matched, dataset.head(distractors, columns=columns)]))
    embedder = create_embedder(config)
    batch = int(config["embedding"]["initial_batch_size"])
    minimum = int(config["embedding"]["min_batch_size"])
    log_path = output_root(config) / "logs" / "smoke_failure.json"

    def encode(table: pa.Table) -> np.ndarray:
        vectors, _, _ = encode_with_retry(
            embedder, build_texts(table, config["text"]["representation"]), batch, minimum, log_path
        )
        return vectors

    started = time.perf_counter()
    query_vectors = encode(query_table)
    retrieved: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
    score_stats = []
    for source, table in target_tables.items():
        target_vectors = encode(table)
        target_ids = np.asarray(table["entity_id"].to_pylist(), dtype=object)
        similarities = query_vectors @ target_vectors.T
        top_k = min(50, similarities.shape[1])
        indexes = np.argsort(-similarities, axis=1)[:, :top_k]
        for row, query_id in enumerate(query_ids):
            retrieved[query_id].extend(target_ids[indexes[row]].tolist())
            score_stats.extend(similarities[row, indexes[row]].tolist())
    metrics = {}
    total_pairs = sum(len(truth.get(query_id, set())) for query_id in query_ids)
    for k in (1, 5, 10, 25, 50):
        hits = 0
        entities_any = entities_all = nonempty = 0
        for query_id in query_ids:
            expected = truth.get(query_id, set())
            # Results were appended as S2 top-50 then S3 top-50; apply K independently to each source.
            ranked = retrieved[query_id]
            predicted = set(ranked[:k] + ranked[50:50 + k])
            found = len(expected & predicted)
            hits += found
            if expected:
                nonempty += 1; entities_any += bool(found); entities_all += found == len(expected)
        metrics[str(k)] = {
            "pair_recall": hits / total_pairs if total_pairs else 0.0,
            "entities_any": entities_any / nonempty if nonempty else 0.0,
            "entities_all": entities_all / nonempty if nonempty else 0.0,
        }
    report = {
        "debug_only": True,
        "warning": "All true targets were injected into a small distractor pool; these are not realistic full-pool metrics.",
        "queries": len(query_ids), "true_pairs": total_pairs,
        "target_rows": {key: len(value) for key, value in target_tables.items()},
        "recall_at_k": metrics,
        "score_min": min(score_stats, default=None), "score_max": max(score_stats, default=None),
        "elapsed_seconds": time.perf_counter() - started, "gpu": gpu_memory(),
    }
    atomic_json(output_root(config) / "reports" / "smoke_ground_truth_metrics.json", report)
    return report

