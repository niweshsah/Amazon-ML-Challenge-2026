"""Embedding index, sampled-pool retrieval, calibration, and TSV output."""
from __future__ import annotations

import csv
import resource
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from core import metrics, save_json
from encoder import embed_rows, load_encoder


def build_index(root: Path, logger, adapter: Path | None = None,
                max_length: int = 96, batch_size: int = 128,
                index_type: str = "ivf", nlist: int = 2048) -> dict:
    targets = pq.read_table(root / "eval_targets.parquet").to_pylist()
    tokenizer, model = load_encoder(adapter)
    stats = embed_rows(model, tokenizer, targets, root / "target_vectors.npy", max_length, batch_size, logger)
    del model
    torch.cuda.empty_cache()
    vectors = np.load(root / "target_vectors.npy", mmap_mode="r")
    dim = vectors.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, max(1, len(targets) // 40)), faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(42)
        sample = rng.choice(len(targets), min(100000, len(targets)), replace=False)
        index.train(np.asarray(vectors[sample]))
    for start in range(0, len(targets), 10000):
        index.add(np.asarray(vectors[start:start + 10000]))
        if start % 100000 == 0:
            logger("index_progress", rows_done=min(len(targets), start + 10000), rows_total=len(targets))
    faiss.write_index(index, str(root / "target.index"))
    report = {"label": "sampled-pool", "pool_size": len(targets), "index_type": index_type,
              "nlist": getattr(index, "nlist", None), "embedding": stats,
              "adapter": str(adapter) if adapter else None, "max_length": max_length,
              "peak_process_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    save_json(root / "index_manifest.json", report)
    return report


def search(root: Path, logger, adapter: Path | None = None, max_length: int = 96,
           batch_size: int = 128, top_k: int = 20, nprobe: int = 64) -> dict:
    import json
    manifest = json.loads((root / "index_manifest.json").read_text())
    if manifest["adapter"] != (str(adapter) if adapter else None) or manifest["max_length"] != max_length:
        raise ValueError("Index model configuration differs from query encoder")
    queries = pq.read_table(root / "eval_queries.parquet").to_pylist()
    targets = pq.read_table(root / "eval_targets.parquet", columns=["entity_id"]).to_pylist()
    tokenizer, model = load_encoder(adapter)
    embedding = embed_rows(model, tokenizer, queries, root / "query_vectors.npy", max_length, batch_size, logger)
    del model
    torch.cuda.empty_cache()
    index = faiss.read_index(str(root / "target.index"))
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    vectors = np.load(root / "query_vectors.npy", mmap_mode="r")
    scores = np.empty((len(queries), top_k), dtype=np.float32)
    positions = np.empty((len(queries), top_k), dtype=np.int64)
    started = time.monotonic()
    for start in range(0, len(queries), 512):
        chunk = np.asarray(vectors[start:start + 512])
        scores[start:start + len(chunk)], positions[start:start + len(chunk)] = index.search(chunk, top_k)
        if start % 10000 == 0:
            logger("search_progress", queries_done=start + len(chunk), queries_total=len(queries),
                   queries_per_second=(start + len(chunk)) / max(1e-6, time.monotonic() - started))
    np.save(root / "retrieval_scores.npy", scores)
    np.save(root / "retrieval_positions.npy", positions)
    report = {"label": "sampled-pool", "pool_size": len(targets), "queries": len(queries),
              "top_k": top_k, "nprobe": nprobe, "embedding": embedding,
              "search_seconds": time.monotonic() - started,
              "search_queries_per_second": len(queries) / max(1e-6, time.monotonic() - started),
              "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
              "peak_process_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    save_json(root / "inference_throughput.json", report)
    return report


def score(root: Path, logger, top_k: int = 20, calibration_count: int = 20000) -> dict:
    queries = pq.read_table(root / "eval_queries.parquet").to_pylist()
    target_ids = pq.read_table(root / "eval_targets.parquet", columns=["entity_id"]).column(0).to_pylist()
    positions = np.load(root / "retrieval_positions.npy")
    scores = np.load(root / "retrieval_scores.npy")
    if len(queries) != len(positions) or len(queries) < calibration_count + 1:
        raise ValueError("Query count does not support separate calibration and score splits")
    if top_k > positions.shape[1] or top_k < 20:
        raise ValueError("Top-K must be at least 20 and within cached retrieval width")
    ranked = [[target_ids[j] for j in row[:top_k] if j >= 0] for row in positions]
    calibration = queries[:calibration_count]
    thresholds = np.arange(0.20, 0.951, 0.01)
    truth_sets = [set(filter(None, q["truth"].split(","))) for q in calibration]
    truth_counts = np.array([len(x) for x in truth_sets], dtype=np.float32)
    hit_matrix = np.array([[tid in truth_sets[i] for tid in ids] +
                           [False] * (top_k - len(ids)) for i, ids in enumerate(ranked[:calibration_count])],
                          dtype=np.bool_)
    def calibration_score(threshold: float) -> float:
        selected = scores[:calibration_count, :top_k] >= threshold
        predicted = selected.sum(axis=1)
        hits = (selected & hit_matrix).sum(axis=1)
        values = np.where(truth_counts == 0, predicted == 0,
                          1.25 * hits / np.maximum(1e-9, .25 * truth_counts + predicted))
        return float(values.mean())
    threshold_curve = [{"threshold": round(float(x), 3),
                        "macro_f0.5": calibration_score(float(x))} for x in thresholds]
    chosen = max((point["macro_f0.5"], -point["threshold"], point["threshold"])
                 for point in threshold_curve)
    threshold = chosen[2]
    pool_size = len(target_ids)
    cal_report = metrics(calibration, ranked[:calibration_count], scores[:calibration_count], threshold, top_k)
    score_report = metrics(queries[calibration_count:], ranked[calibration_count:],
                           scores[calibration_count:], threshold, top_k)
    full_report = metrics(queries, ranked, scores, threshold, top_k)
    for item in (cal_report, score_report, full_report):
        item.update({"label": "sampled-pool", "pool_size": pool_size})
    report = {"label": "sampled-pool", "pool_size": pool_size,
              "threshold_selected_on": calibration_count, "threshold": threshold,
              "threshold_curve": threshold_curve,
              "calibration": cal_report, "heldout_score": score_report,
              "full_100k_diagnostics": full_report}
    save_json(root / "metrics.json", report)
    with (root / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as cf, \
         (root / "matching_results.tsv").open("w", encoding="utf-8", newline="") as mf:
        cw, mw = csv.writer(cf, delimiter="\t"), csv.writer(mf, delimiter="\t")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        for query, ids, row in zip(queries, ranked, scores):
            cw.writerow([query["entity_id"], ",".join(ids)])
            mw.writerow([query["entity_id"], ",".join(tid for tid, score in zip(ids, row) if score >= threshold)])
    # Validator expects the evaluated Source 1 roster in a test-style directory.
    validation_dir = root / "validation_dataset"
    validation_dir.mkdir(exist_ok=True)
    with (validation_dir / "test_source1.tsv").open("w", encoding="utf-8", newline="") as out:
        out.write("entity_id\n")
        out.writelines(q["entity_id"] + "\n" for q in queries)
    for source in (2, 3):
        with (validation_dir / f"test_source{source}.tsv").open("w", encoding="utf-8", newline="") as out:
            out.write("entity_id\n")
            out.writelines(tid + "\n" for tid in target_ids if tid.startswith(f"S{source}-"))
    logger("score_complete", threshold=threshold, heldout_macro_f05=score_report["macro_f0.5"],
           pool_size=pool_size)
    return report
