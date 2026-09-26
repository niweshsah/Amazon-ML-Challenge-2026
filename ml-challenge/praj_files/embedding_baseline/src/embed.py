from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from .checkpoint import valid_embedding_manifest
from .gpu_utils import cuda_cleanup, gpu_memory, has_module
from .io_utils import (
    atomic_json, atomic_numpy, atomic_parquet, count_rows, dataset_path, ensure_layout,
    format_duration, iter_table_shards, load_config, output_root, sha256_file,
)
from .text_builder import build_texts, deduplicate_texts


class TransformersEmbedder:
    def __init__(self, model_config: dict[str, Any]):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.config = model_config
        dtype_name = model_config.get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"], revision=model_config.get("revision"),
            trust_remote_code=model_config.get("trust_remote_code", True), padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            model_config["name"], revision=model_config.get("revision"), dtype=dtype,
            trust_remote_code=model_config.get("trust_remote_code", True),
        ).eval().to("cuda")
        self.dtype = str(dtype).replace("torch.", "")
        self.revision = getattr(self.model.config, "_commit_hash", None) or model_config.get("revision")

    def encode(self, texts: list[str], batch_size: int) -> tuple[np.ndarray, dict[str, float]]:
        torch = self.torch
        outputs: list[np.ndarray] = []
        token_count = 0
        token_time = 0.0
        inference_time = 0.0
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            t0 = time.perf_counter()
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.config["max_length"], return_tensors="pt"
            )
            token_time += time.perf_counter() - t0
            token_count += int(encoded["attention_mask"].sum())
            encoded = {key: value.to("cuda", non_blocking=True) for key, value in encoded.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                # Left padding makes the final token the final non-padding token for every sequence.
                pooled = hidden[:, -1, : self.config["output_dim"]]
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            torch.cuda.synchronize()
            inference_time += time.perf_counter() - t0
            outputs.append(pooled.cpu().numpy())
            del encoded, hidden, pooled
        vectors = np.concatenate(outputs) if outputs else np.empty((0, self.config["output_dim"]), np.float32)
        return vectors, {"tokenization_seconds": token_time, "embedding_seconds": inference_time, "tokens": token_count}


class VLLMEmbedder:
    """Thin adapter around the vLLM pooling API; instantiated only when vLLM is installed."""
    def __init__(self, model_config: dict[str, Any]):
        from vllm import LLM
        self.config = model_config
        kwargs: dict[str, Any] = {
            "model": model_config["name"], "task": "embed", "dtype": model_config.get("dtype", "bfloat16"),
            "max_model_len": model_config["max_length"], "trust_remote_code": model_config.get("trust_remote_code", True),
        }
        if model_config.get("revision"):
            kwargs["revision"] = model_config["revision"]
        if model_config.get("quantization") not in (None, "auto"):
            kwargs["quantization"] = model_config["quantization"]
        self.model = LLM(**kwargs)
        self.dtype = model_config.get("dtype", "auto")
        self.revision = model_config.get("revision")

    def encode(self, texts: list[str], batch_size: int) -> tuple[np.ndarray, dict[str, float]]:
        vectors = []
        embedding_seconds = 0.0
        for start in range(0, len(texts), batch_size):
            t0 = time.perf_counter()
            outputs = self.model.embed(texts[start:start + batch_size], use_tqdm=False)
            embedding_seconds += time.perf_counter() - t0
            vectors.extend(output.outputs.embedding for output in outputs)
        array = np.asarray(vectors, dtype=np.float32)[:, : self.config["output_dim"]]
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array /= np.maximum(norms, 1e-12)
        return array, {"tokenization_seconds": 0.0, "embedding_seconds": embedding_seconds, "tokens": 0}


def resolve_engine(config: dict[str, Any]) -> str:
    requested = config["model"].get("engine", "auto")
    if requested == "vllm" and not has_module("vllm"):
        raise RuntimeError("engine=vllm requested, but vLLM is not installed")
    # vLLM support varies by release; Transformers is the verified fallback and remains explicit in manifests.
    if requested == "auto":
        return "vllm" if has_module("vllm") else "transformers"
    return requested


def create_embedder(config: dict[str, Any]):
    engine = resolve_engine(config)
    if engine == "vllm":
        return VLLMEmbedder(config["model"])
    return TransformersEmbedder(config["model"])


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def encode_with_retry(embedder, texts: list[str], initial_batch: int, min_batch: int, log_path: Path):
    batch = initial_batch
    while True:
        try:
            vectors, timing = embedder.encode(texts, batch)
            return vectors, timing, batch
        except (RuntimeError, MemoryError) as exc:
            if not _is_oom(exc) or batch <= min_batch:
                atomic_json(log_path, {"success": False, "batch_size": batch, "error": repr(exc), "gpu": gpu_memory()})
                raise
            new_batch = max(min_batch, batch // 2)
            print(f"[embed] OOM at batch={batch}; retrying same shard with batch={new_batch}", flush=True)
            atomic_json(log_path, {"success": False, "oom": True, "old_batch": batch, "new_batch": new_batch, "error": str(exc)})
            batch = new_batch
            cuda_cleanup()


def embed_source(
    config_path: str | None, split: str, source: str, resume: bool = True,
    max_rows: int | None = None, shard_rows: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_layout(config)
    source_path = dataset_path(config, split, source)
    total_rows = min(count_rows(source_path), max_rows or count_rows(source_path))
    shard_rows = shard_rows or int(config["embedding"].get(
        "query_shard_rows" if source == "source1" else "shard_rows",
        config["embedding"]["shard_rows"],
    ))
    out_dir = output_root(config) / "embeddings" / f"{split}_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "model": config["model"]["name"], "embedding_dim": config["model"]["output_dim"],
        "text_representation": config["text"]["representation"], "max_length": config["model"]["max_length"],
    }
    embedder = None
    job_started = time.perf_counter()
    rows_done = 0
    summaries = []
    active_manifest_paths: list[str] = []
    last_stage_end = time.perf_counter()
    columns = ["entity_id", "business_name", "business_address", "country"]
    for shard_id, row_start, table in iter_table_shards(source_path, columns, shard_rows, max_rows):
        parquet_read_seconds = time.perf_counter() - last_stage_end
        prefix = out_dir / f"part_{shard_id:06d}"
        manifest_path = prefix.with_suffix(".manifest.json")
        if resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if valid_embedding_manifest(manifest, {**expected, "row_start": row_start, "rows": len(table)}):
                rows_done += len(table)
                summaries.append(manifest)
                active_manifest_paths.append(str(manifest_path))
                print(f"[embed {split}_{source}] skip completed shard {shard_id}", flush=True)
                last_stage_end = time.perf_counter()
                continue
        if embedder is None:
            embedder = create_embedder(config)
        shard_started = time.perf_counter()
        t0 = time.perf_counter()
        texts = build_texts(table, config["text"]["representation"])
        unique, inverse = deduplicate_texts(texts) if config["text"].get("deduplicate", True) else (texts, list(range(len(texts))))
        text_seconds = time.perf_counter() - t0
        failure_path = prefix.with_suffix(".failure.json")
        vectors, timing, used_batch = encode_with_retry(
            embedder, unique, int(config["embedding"]["initial_batch_size"]),
            int(config["embedding"]["min_batch_size"]), failure_path,
        )
        if failure_path.exists():
            failure_path.unlink()
        dtype = np.float16 if config["embedding"]["save_dtype"] == "float16" else np.float32
        vectors = vectors.astype(dtype)
        vector_path = prefix.with_suffix(".vectors.npy")
        mapping_path = prefix.with_suffix(".mapping.parquet")
        t0 = time.perf_counter()
        atomic_numpy(vector_path, vectors)
        mapping = pa.table({
            "entity_id": table["entity_id"],
            "text_index": pa.array(inverse, type=pa.int32()),
        })
        atomic_parquet(mapping_path, mapping)
        write_seconds = time.perf_counter() - t0
        elapsed = time.perf_counter() - shard_started
        rows_done += len(table)
        speed = len(table) / elapsed
        eta = (total_rows - rows_done) / max(speed, 1e-9)
        manifest = {
            **expected, "success": True, "source_file": str(source_path), "shard_id": shard_id,
            "row_start": row_start, "row_end": row_start + len(table), "rows": len(table),
            "unique_texts": len(unique), "duplicate_percent": 100 * (1 - len(unique) / max(len(table), 1)),
            "embedding_work_avoided": len(table) - len(unique), "model_revision": embedder.revision,
            "engine": resolve_engine(config), "quantization": config["model"].get("quantization"),
            "dtype": embedder.dtype, "save_dtype": str(dtype), "batch_size": used_batch,
            "timing": {"parquet_read_seconds": parquet_read_seconds, "text_build_seconds": text_seconds, **timing, "disk_write_seconds": write_seconds, "total_seconds": elapsed},
            "records_per_second": speed, "gpu": gpu_memory(),
            "files": [
                {"path": str(vector_path), "bytes": vector_path.stat().st_size, "sha256": sha256_file(vector_path)},
                {"path": str(mapping_path), "bytes": mapping_path.stat().st_size, "sha256": sha256_file(mapping_path)},
            ],
        }
        atomic_json(manifest_path, manifest)
        summaries.append(manifest)
        active_manifest_paths.append(str(manifest_path))
        print(
            f"[embed {split}_{source}] shard {shard_id + 1} rows {rows_done:,}/{total_rows:,} "
            f"speed {speed:,.1f} rows/s elapsed {format_duration(time.perf_counter()-job_started)} "
            f"ETA {format_duration(eta)} batch {used_batch} output {vector_path}", flush=True,
        )
        benchmark_path = output_root(config) / "reports" / "embedding_benchmark.json"
        if benchmark_path.exists():
            benchmark_report = json.loads(benchmark_path.read_text())
            benchmark_speed = benchmark_report.get("recommended_before_retrieval_quality_check", {}).get("rows_per_second")
            if benchmark_speed and speed < benchmark_speed / float(config["execution"].get("bad_speed_multiplier", 3.0)):
                print(
                    f"[embed] WARNING: {speed:.1f} rows/s is more than the configured slowdown from "
                    f"benchmark {benchmark_speed:.1f} rows/s; checkpoint preserved. Stop and diagnose before continuing.", flush=True,
                )
        last_stage_end = time.perf_counter()
    summary = {
        "success": rows_done == total_rows, "split": split, "source": source, "rows": rows_done,
        "total_seconds": time.perf_counter() - job_started,
        "records_per_second": rows_done / max(time.perf_counter() - job_started, 1e-9), "shards": len(summaries),
        "manifests": active_manifest_paths,
    }
    atomic_json(out_dir / "summary.json", summary)
    return summary
