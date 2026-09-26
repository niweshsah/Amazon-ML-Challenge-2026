from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyarrow.compute as pc

from .io_utils import atomic_json, ensure_layout, load_config, output_root, parquet_dataset


DATASETS = [
    "train_source1.parquet", "train_source2.parquet", "train_source3.parquet",
    "train_ground_truth.parquet", "test_source1.parquet", "test_source2.parquet", "test_source3.parquet",
]


def _profile_dataset(path: Path, examples: int = 5) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = parquet_dataset(path)
    row_count = dataset.count_rows()
    null_counts = {field.name: 0 for field in dataset.schema}
    empty_counts = {field.name: 0 for field in dataset.schema}
    for batch in dataset.scanner(batch_size=131072).to_batches():
        for name in null_counts:
            array = batch.column(batch.schema.get_field_index(name))
            null_counts[name] += array.null_count
            if str(array.type) in {"string", "large_string"}:
                empty_counts[name] += int(pc.sum(pc.equal(pc.fill_null(array, ""), "")).as_py() or 0)
    sample = dataset.head(examples).to_pylist()
    fields = [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in dataset.schema]
    return {
        "path": str(path), "row_count": row_count, "columns": fields,
        "null_counts": null_counts,
        "null_percent": {k: round(100 * v / row_count, 6) for k, v in null_counts.items()},
        "empty_or_null_counts": empty_counts,
        "empty_or_null_percent": {k: round(100 * v / row_count, 6) for k, v in empty_counts.items()},
        "examples": sample, "elapsed_seconds": time.perf_counter() - started,
    }


def inspect_all(config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_layout(config)
    data_dir = Path(config["paths"]["data_dir"])
    report = {"generated_at_epoch": time.time(), "datasets": {}}
    for name in DATASETS:
        print(f"[inspect] {name}", flush=True)
        report["datasets"][name] = _profile_dataset(data_dir / name)
    gt = report["datasets"]["train_ground_truth.parquet"]
    gt["verified_columns"] = [column["name"] for column in gt["columns"]]
    path = output_root(config) / "reports" / "dataset_profile.json"
    atomic_json(path, report)
    print(f"[inspect] wrote {path}")
    return report

