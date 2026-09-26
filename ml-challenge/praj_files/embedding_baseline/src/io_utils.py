from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def dataset_path(config: dict[str, Any], split: str, source: str) -> Path:
    name = f"{split}_{source}.parquet"
    path = Path(config["paths"]["data_dir"]) / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_dir"])


def ensure_layout(config: dict[str, Any]) -> None:
    root = output_root(config)
    for name in ("embeddings", "indexes", "checkpoints", "retrieval", "reports", "logs", "outputs"):
        (root / name).mkdir(parents=True, exist_ok=True)


def parquet_dataset(path: str | Path) -> ds.Dataset:
    return ds.dataset(str(path), format="parquet")


def count_rows(path: str | Path) -> int:
    return parquet_dataset(path).count_rows()


def iter_table_shards(
    path: str | Path,
    columns: list[str] | None,
    shard_rows: int,
    max_rows: int | None = None,
) -> Iterator[tuple[int, int, pa.Table]]:
    """Stream deterministic contiguous row shards from a file or dataset directory."""
    scanner = parquet_dataset(path).scanner(columns=columns, batch_size=min(shard_rows, 65536), use_threads=True)
    chunks: list[pa.RecordBatch] = []
    buffered = 0
    emitted = 0
    shard_id = 0
    for batch in scanner.to_batches():
        if max_rows is not None and emitted + buffered >= max_rows:
            break
        if max_rows is not None and emitted + buffered + len(batch) > max_rows:
            batch = batch.slice(0, max_rows - emitted - buffered)
        chunks.append(batch)
        buffered += len(batch)
        while buffered >= shard_rows:
            table = pa.Table.from_batches(chunks).combine_chunks()
            yield shard_id, emitted, table.slice(0, shard_rows)
            remainder = table.slice(shard_rows)
            chunks = remainder.to_batches() if len(remainder) else []
            buffered -= shard_rows
            emitted += shard_rows
            shard_id += 1
    if buffered:
        table = pa.Table.from_batches(chunks).combine_chunks()
        yield shard_id, emitted, table


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: str | Path, table: pa.Table, compression: str = "zstd") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression=compression)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_numpy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy.tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

