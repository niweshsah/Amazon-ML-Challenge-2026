from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .gpu_utils import cuda_cleanup, gpu_memory
from .io_utils import atomic_json, atomic_parquet, ensure_layout, format_duration, load_config, output_root


def _merge_topk(
    best_scores: np.ndarray, best_ids: np.ndarray, new_scores: np.ndarray,
    new_ids: np.ndarray, top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate((best_scores, new_scores), axis=1)
    ids = np.concatenate((best_ids, new_ids), axis=1)
    keep = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    rows = np.arange(len(scores))[:, None]
    selected_scores = scores[rows, keep]
    selected_ids = ids[rows, keep]
    order = np.argsort(-selected_scores, axis=1)
    return selected_scores[rows, order], selected_ids[rows, order]


def _search_source(
    query_vectors: np.ndarray, target_shards: list[dict[str, Any]], top_k: int,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    started = time.perf_counter()
    all_scores = np.full((len(query_vectors), top_k), -np.inf, dtype=np.float32)
    all_ids = np.full((len(query_vectors), top_k), "", dtype=object)
    for target_num, shard in enumerate(target_shards):
        unique_vectors = np.load(shard["vectors"], mmap_mode="r")
        mapping = pq.read_table(shard["mapping"], columns=["entity_id", "text_index"])
        text_indexes = mapping["text_index"].to_numpy(zero_copy_only=False)
        entity_ids = np.asarray(mapping["entity_id"].to_pylist(), dtype=object)
        t = torch.from_numpy(np.asarray(unique_vectors[text_indexes], dtype=np.float16)).to("cuda")
        batch = query_batch_size
        start = 0
        while start < len(query_vectors):
            stop = min(len(query_vectors), start + batch)
            try:
                q = torch.from_numpy(query_vectors[start:stop].astype(np.float16, copy=False)).to("cuda")
                # Expand only this target shard. FP16 storage stays compact on disk/host and GPU.
                scores = (q @ t.T).float()
                local_k = min(top_k, scores.shape[1])
                values, indices = torch.topk(scores, local_k, dim=1)
                values_np = values.cpu().numpy()
                ids_np = entity_ids[indices.cpu().numpy()]
                if local_k < top_k:
                    values_np = np.pad(values_np, ((0, 0), (0, top_k-local_k)), constant_values=-np.inf)
                    ids_np = np.pad(ids_np, ((0, 0), (0, top_k-local_k)), constant_values="")
                merged = _merge_topk(all_scores[start:stop], all_ids[start:stop], values_np, ids_np, top_k)
                all_scores[start:stop], all_ids[start:stop] = merged
                del q, scores, values, indices
                start = stop
            except (RuntimeError, MemoryError) as exc:
                if "memory" not in str(exc).lower() or batch <= 1:
                    raise
                batch = max(1, batch // 2)
                print(f"[retrieve] OOM; retrying query block with batch={batch}", flush=True)
                cuda_cleanup()
        del t
        print(f"[retrieve] searched target shard {target_num+1}/{len(target_shards)}", flush=True)
    return all_scores, all_ids, time.perf_counter() - started


def retrieve(
    config_path: str | None, split: str, top_k: int | None = None,
    sample_fraction: float | None = None, max_source1: int | None = None,
    resume: bool = True, tiny_debug: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_layout(config)
    top_k = top_k or int(config["retrieval"]["top_k"])
    index_path = output_root(config) / "indexes" / f"{split}_index.json"
    catalog = json.loads(index_path.read_text())
    index_checksum = catalog["index_signature"]
    query_dir = output_root(config) / "embeddings" / f"{split}_source1"
    query_manifests = sorted(query_dir.glob("part_*.manifest.json"))
    if not query_manifests:
        raise FileNotFoundError(f"No query embeddings found in {query_dir}")
    available = sum(json.loads(p.read_text())["rows"] for p in query_manifests)
    requested = max_source1
    if requested is None and sample_fraction is not None:
        requested = max(1, math.ceil(available * sample_fraction))
    requested = min(requested or available, available)
    if tiny_debug:
        print("[retrieve] WARNING: tiny-debug target sampling produces DEBUG ONLY metrics", flush=True)
        for source in catalog["sources"].values():
            source["shards"] = source["shards"][:1]
    out_dir = output_root(config) / "retrieval" / split / "raw_top50"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_started = time.perf_counter()
    completed = 0
    output_parts = []
    for query_num, manifest_path in enumerate(query_manifests):
        if completed >= requested:
            break
        manifest = json.loads(manifest_path.read_text())
        part_path = out_dir / f"part_{query_num:06d}.parquet"
        done_path = out_dir / f"part_{query_num:06d}.manifest.json"
        rows_this = min(manifest["rows"], requested - completed)
        if resume and done_path.exists() and part_path.exists():
            done = json.loads(done_path.read_text())
            query_checksum = manifest["files"][0]["sha256"]
            if (done.get("success") and done.get("rows") == rows_this and done.get("top_k") == top_k
                    and done.get("index_checksum") == index_checksum and done.get("query_checksum") == query_checksum):
                completed += rows_this
                output_parts.append(str(part_path))
                continue
        vectors = np.load(manifest["files"][0]["path"], mmap_mode="r")
        mapping = pq.read_table(manifest["files"][1]["path"])
        inverse = mapping["text_index"].to_numpy(zero_copy_only=False)[:rows_this]
        query_vectors = np.asarray(vectors[inverse], dtype=np.float32)
        query_ids = mapping["entity_id"].to_pylist()[:rows_this]
        result_tables = []
        search_seconds = 0.0
        for source in ("source2", "source3"):
            scores, ids, seconds = _search_source(
                query_vectors, catalog["sources"][source]["shards"], top_k,
                int(config["retrieval"]["query_batch_size"]),
            )
            search_seconds += seconds
            flat_ids = ids.reshape(-1)
            flat_scores = scores.reshape(-1)
            valid = flat_ids != ""
            result_tables.append(pa.table({
                "source1_entity_id": pa.array(np.repeat(np.asarray(query_ids, dtype=object), top_k)[valid]),
                "candidate_entity_id": pa.array(flat_ids[valid].tolist()),
                "candidate_source": pa.array(np.repeat(source, int(valid.sum()))),
                "rank": pa.array(np.tile(np.arange(1, top_k + 1, dtype=np.int16), len(query_ids))[valid]),
                "cosine_similarity": pa.array(flat_scores[valid], pa.float32()),
            }))
        t0 = time.perf_counter()
        table = pa.concat_tables(result_tables)
        atomic_parquet(part_path, table)
        write_seconds = time.perf_counter() - t0
        completed += rows_this
        elapsed = time.perf_counter() - job_started
        speed = completed / max(elapsed, 1e-9)
        atomic_json(done_path, {
            "success": True, "rows": rows_this, "top_k": top_k, "search_seconds": search_seconds,
            "result_write_seconds": write_seconds, "gpu": gpu_memory(), "debug_only": tiny_debug,
            "index_checksum": index_checksum, "query_checksum": manifest["files"][0]["sha256"],
        })
        output_parts.append(str(part_path))
        print(
            f"[retrieve {split}] queries {completed:,}/{requested:,} speed {speed:,.1f}/s "
            f"elapsed {format_duration(elapsed)} ETA {format_duration((requested-completed)/max(speed,1e-9))}", flush=True,
        )
    summary = {
        "success": completed == requested, "split": split, "queries": completed, "top_k_per_source": top_k,
        "total_seconds": time.perf_counter() - job_started, "debug_only": tiny_debug, "parts": output_parts,
    }
    atomic_json(out_dir / "summary.json", summary)
    return summary
