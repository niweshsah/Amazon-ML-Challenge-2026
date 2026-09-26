from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import config, record_text, save_json, source_path
from train import encode_numpy, load_model


def embed(cfg: dict, split: str, source: int, batch_size: int, shard_rows: int,
          max_rows: int | None = None) -> dict:
    model_path = Path(cfg["output_dir"]) / "final_adapter"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained adapter not found: {model_path}")
    tokenizer, model = load_model(cfg, model_path)
    root = Path(cfg["output_dir"]) / "embeddings" / f"{split}_source{source}"
    root.mkdir(parents=True, exist_ok=True)
    # Keep one shard in host memory and only one encoder batch in GPU memory.
    import pyarrow.dataset as ds
    stream = ds.dataset(source_path(cfg, split, source), format="parquet").to_batches(
        columns=["entity_id", "business_name", "business_address", "country"],
        batch_size=min(shard_rows, 32768))
    row_buffer = []
    shard_id = 0
    processed = 0

    def flush(rows: list[dict], index: int) -> None:
        vector_path = root / f"part_{index:05d}.npy"
        ids_path = root / f"part_{index:05d}.ids.parquet"
        if vector_path.exists() and ids_path.exists():
            if len(np.load(vector_path, mmap_mode="r")) == len(rows) and pq.read_metadata(ids_path).num_rows == len(rows):
                return
            raise RuntimeError(f"Incomplete existing shard {index}; inspect before resuming")
        texts = [record_text(row) for row in rows]
        vectors = encode_numpy(model, tokenizer, texts, cfg["max_length"], batch_size)
        temp_vectors = vector_path.with_suffix(".tmp")
        with temp_vectors.open("wb") as handle:
            np.save(handle, vectors.astype(np.float16), allow_pickle=False)
        temp_vectors.replace(vector_path)
        temp_ids = ids_path.with_suffix(".tmp")
        pq.write_table(pa.table({"entity_id": [row["entity_id"] for row in rows]}), temp_ids)
        temp_ids.replace(ids_path)

    for batch in stream:
        row_buffer.extend(batch.to_pylist())
        while len(row_buffer) >= shard_rows:
            flush(row_buffer[:shard_rows], shard_id)
            row_buffer = row_buffer[shard_rows:]
            processed += shard_rows
            shard_id += 1
            print(f"{split} source{source}: {processed} rows", flush=True)
            if max_rows and processed >= max_rows:
                break
        if max_rows and processed >= max_rows:
            break
    if row_buffer and (not max_rows or processed < max_rows):
        remaining = row_buffer[:max_rows - processed] if max_rows else row_buffer
        flush(remaining, shard_id)
        processed += len(remaining)
    result = {"split": split, "source": source, "rows": processed,
              "dimension": 1024, "dtype": "float16", "output": str(root)}
    save_json(root / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--source", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-rows", type=int, default=50000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--config")
    args = parser.parse_args()
    print(json.dumps(embed(config(args.config), args.split, args.source, args.batch_size,
                           args.shard_rows, args.max_rows), indent=2))
