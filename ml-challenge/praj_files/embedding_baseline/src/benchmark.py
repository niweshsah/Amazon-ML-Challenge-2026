from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .embed import create_embedder, encode_with_retry
from .gpu_utils import environment_report, gpu_memory
from .io_utils import atomic_json, count_rows, dataset_path, ensure_layout, iter_table_shards, load_config, output_root
from .text_builder import build_texts, deduplicate_texts


def benchmark(config_path: str | None, sample_records: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_layout(config)
    sample_records = sample_records or int(config["benchmark"]["sample_records"])
    source_path = dataset_path(config, "train", "source2")
    _, _, table = next(iter_table_shards(
        source_path, ["entity_id", "business_name", "business_address", "country"], sample_records, sample_records
    ))
    configurations = [
        ("name_address_country", 512, 64), ("name_address_country", 256, 64),
        ("name_only", 512, 64), ("name_only", 256, 64),
    ]
    # Optional lengths remain a cheap CLI/config extension; default benchmark stays bounded.
    model_config = copy.deepcopy(config["model"])
    model_config["output_dim"] = max(x[1] for x in configurations)
    model_config["max_length"] = max(x[2] for x in configurations)
    local_config = copy.deepcopy(config); local_config["model"] = model_config
    embedder = create_embedder(local_config)
    warmup_texts = build_texts(table.slice(0, min(2048, len(table))), "name_address_country")
    warmup_unique, _ = deduplicate_texts(warmup_texts)
    batch_trials = []
    for trial_batch in (128, 256, 512, 1024):
        started = time.perf_counter()
        try:
            trial_vectors, _, used_batch = encode_with_retry(
                embedder, warmup_unique, trial_batch, int(config["embedding"]["min_batch_size"]),
                output_root(config) / "logs" / "benchmark_failure.json",
            )
            trial_seconds = time.perf_counter() - started
            batch_trials.append({
                "requested_batch_size": trial_batch, "used_batch_size": used_batch,
                "rows_per_second": len(warmup_texts) / trial_seconds, "success": used_batch == trial_batch,
            })
            del trial_vectors
        except (RuntimeError, MemoryError) as exc:
            batch_trials.append({"requested_batch_size": trial_batch, "success": False, "error": str(exc)})
    successful_batches = [x for x in batch_trials if x["success"]]
    selected_batch = max(successful_batches, key=lambda x: x["rows_per_second"])["used_batch_size"] if successful_batches else int(config["embedding"]["initial_batch_size"])
    results = []
    total_target_rows = count_rows(dataset_path(config, "train", "source2")) + count_rows(dataset_path(config, "train", "source3"))
    for representation, dimension, max_length in configurations:
        print(f"[benchmark] start text={representation} dim={dimension} max_length={max_length} batch={selected_batch}", flush=True)
        embedder.config["output_dim"] = dimension
        embedder.config["max_length"] = max_length
        texts = build_texts(table, representation)
        unique, _ = deduplicate_texts(texts)
        started = time.perf_counter()
        vectors, timing, batch_size = encode_with_retry(
            embedder, unique, selected_batch,
            int(config["embedding"]["min_batch_size"]), output_root(config) / "logs" / "benchmark_failure.json",
        )
        elapsed = time.perf_counter() - started
        norms = np.linalg.norm(vectors, axis=1)
        speed = len(texts) / elapsed
        results.append({
            "engine": "transformers", "dtype": embedder.dtype, "quantization": None,
            "text_representation": representation, "embedding_dim": dimension, "max_length": max_length,
            "records": len(texts), "unique_texts": len(unique), "elapsed_seconds": elapsed,
            "rows_per_second": speed, "unique_texts_per_second": len(unique) / elapsed,
            "tokens_per_second": timing["tokens"] / max(timing["embedding_seconds"] + timing["tokenization_seconds"], 1e-9),
            "estimated_full_target_seconds": total_target_rows / max(speed, 1e-9),
            "batch_size": batch_size, "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "finite_fraction": float(np.isfinite(vectors).mean()), "gpu": gpu_memory(),
            "host_rss_bytes": psutil.Process().memory_info().rss,
        })
        print(f"[benchmark] done text={representation} dim={dimension}: {speed:.1f} rows/s, {elapsed:.1f}s", flush=True)
    # Quality selection is finalized by train evaluation; before then prefer the requested 512d representation.
    recommended = max(
        (row for row in results if row["text_representation"] == config["text"]["representation"] and row["embedding_dim"] == config["model"]["output_dim"]),
        key=lambda row: row["rows_per_second"], default=max(results, key=lambda row: row["rows_per_second"]),
    )
    report = {
        "environment": environment_report(), "sample_records": sample_records, "batch_size_trials": batch_trials,
        "selected_batch_size": selected_batch, "results": results,
        "quantization": {
            "tested": False,
            "reason": "No local verified quantized checkpoint/runtime was available; no conversion or large extra download attempted.",
        },
        "recommended_before_retrieval_quality_check": recommended,
        "selection_note": "Do not promote 256d or another representation until sampled Recall@K is compared.",
    }
    reports = output_root(config) / "reports"
    atomic_json(reports / "embedding_benchmark.json", report)
    lines = ["# Embedding benchmark", "", f"Sample records: {sample_records:,}", "", "| Text | Dim | Length | rows/s | tokens/s | Full target ETA (h) |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['text_representation']} | {row['embedding_dim']} | {row['max_length']} | {row['rows_per_second']:.1f} | {row['tokens_per_second']:.1f} | {row['estimated_full_target_seconds']/3600:.2f} |")
    lines += ["", "Quantization was not tested because no verified local quantized model/runtime was available.", ""]
    (reports / "embedding_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return report
