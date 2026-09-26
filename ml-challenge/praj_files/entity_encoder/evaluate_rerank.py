"""Evaluate an encoder on full-source BGE candidates with held-out threshold tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import config, record_text, rows, save_json, source_path
from evaluate_bge_100k import digest, sample_truth
from train import encode_numpy, load_model


def prepare_candidates(cfg: dict, query_count: int) -> dict:
    root = Path(cfg["output_dir"]) / f"rerank_{query_count}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "prepare_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    truth = sample_truth(cfg, query_count)
    query_ids = set(truth)
    candidates: dict[str, list[str]] = {x: [] for x in query_ids}
    candidate_root = Path(__file__).resolve().parents[1] / "output"
    for source in (2, 3):
        path = candidate_root / f"candidate_pairs_bge_m3_s{source}.tsv"
        found = set()
        with path.open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                entity_id, _, values = line.rstrip("\n\r").partition("\t")
                if entity_id in query_ids:
                    candidates[entity_id].extend(filter(None, values.split(",")))
                    found.add(entity_id)
        if found != query_ids:
            raise RuntimeError(f"Missing {len(query_ids - found)} queries in source {source} candidates")
    target_ids = {x for ids in candidates.values() for x in ids}
    queries = {}
    targets = {}
    cols = ["entity_id", "business_name", "business_address", "country"]
    for row in rows(source_path(cfg, "train", 1), cols):
        if row["entity_id"] in query_ids:
            queries[row["entity_id"]] = row
    for source in (2, 3):
        for row in rows(source_path(cfg, "train", source), cols):
            if row["entity_id"] in target_ids:
                targets[row["entity_id"]] = row
    if len(queries) != query_count or len(targets) != len(target_ids):
        raise RuntimeError(f"Missing query/target records: {len(queries)}/{query_count}, {len(targets)}/{len(target_ids)}")
    query_order = sorted(query_ids)
    target_order = sorted(target_ids)
    pq.write_table(pa.table({"entity_id": query_order,
                             "text": [record_text(queries[x]) for x in query_order],
                             "country": [queries[x]["country"] or "" for x in query_order],
                             "truth": [",".join(sorted(truth[x])) for x in query_order],
                             "candidates": [",".join(candidates[x]) for x in query_order]}),
                   root / "queries.parquet", compression="zstd")
    pq.write_table(pa.table({"entity_id": target_order,
                             "text": [record_text(targets[x]) for x in target_order]}),
                   root / "targets.parquet", compression="zstd")
    result = {"query_count": query_count, "unique_candidate_targets": len(target_order),
              "candidate_source": "full 10.32M T2/T3 BGE-M3 IVF-PQ top100 per source",
              "split": "first 25% by salted T1 hash for threshold tuning; remaining 75% for held-out scoring"}
    save_json(manifest, result)
    return result


def vectors(cfg: dict, root: Path, model_label: str, adapter: Path | None,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    query_rows = pq.read_table(root / "queries.parquet", columns=["text"]).column(0).to_pylist()
    target_rows = pq.read_table(root / "targets.parquet", columns=["text"]).column(0).to_pylist()
    tokenizer, model = load_model(cfg, adapter)
    q_vec = encode_numpy(model, tokenizer, query_rows, cfg["max_length"], batch_size)
    t_vec = encode_numpy(model, tokenizer, target_rows, cfg["max_length"], batch_size)
    return q_vec, t_vec


def f05(truth: set[str], selected: set[str]) -> float:
    if not truth:
        return float(not selected)
    if not selected:
        return 0.0
    hit = len(truth & selected)
    if not hit:
        return 0.0
    p, r = hit / len(selected), hit / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def score(cfg: dict, query_count: int, model_label: str, adapter: Path | None,
          batch_size: int) -> dict:
    root = Path(cfg["output_dir"]) / f"rerank_{query_count}"
    queries = pq.read_table(root / "queries.parquet").to_pylist()
    targets = pq.read_table(root / "targets.parquet", columns=["entity_id"]).column(0).to_pylist()
    index = {entity_id: i for i, entity_id in enumerate(targets)}
    q_vec, t_vec = vectors(cfg, root, model_label, adapter, batch_size)
    similarities = []
    for i, row in enumerate(queries):
        ids = row["candidates"].split(",")
        scores = q_vec[i] @ t_vec[[index[x] for x in ids]].T
        similarities.append((ids, scores))
    # Threshold selection and final reporting use disjoint T1 IDs.
    calibration = [i for i, row in enumerate(queries)
                   if digest(row["entity_id"], b"threshold") % 4 == 0]
    test = [i for i in range(len(queries)) if i not in set(calibration)]
    thresholds = np.arange(0.30, 0.951, 0.01)
    def macro(indices: list[int], threshold: float) -> float:
        return float(np.mean([
            f05(set(filter(None, queries[i]["truth"].split(","))),
                {x for x, s in zip(*similarities[i]) if s >= threshold})
            for i in indices]))
    tuning = [(float(t), macro(calibration, float(t))) for t in thresholds]
    threshold, calibration_f05 = max(tuning, key=lambda x: x[1])
    test_f05 = macro(test, threshold)
    pair_total = pair_hit = 0
    rank_hits = {k: 0 for k in (1, 5, 10, 50, 200)}
    mrr_sum = 0.0
    nonempty = 0
    group = {}
    for country in sorted({queries[i]["country"] for i in test}):
        subset = [i for i in test if queries[i]["country"] == country]
        group[country] = {"queries": len(subset), "macro_f0.5": macro(subset, threshold)}
    for i in test:
        true_ids = set(filter(None, queries[i]["truth"].split(",")))
        if not true_ids:
            continue
        nonempty += 1
        pair_total += len(true_ids)
        ids, scores = similarities[i]
        order = np.argsort(-scores)
        ranked = [ids[j] for j in order]
        pair_hit += len(true_ids & set(ids))
        mrr_sum += next((1 / (j + 1) for j, x in enumerate(ranked) if x in true_ids), 0)
        for k in rank_hits:
            rank_hits[k] += len(true_ids & set(ranked[:k]))
    result = {"model": cfg["model"], "adapter": str(adapter) if adapter else None,
              "queries": len(queries), "calibration_queries": len(calibration),
              "test_queries": len(test), "test_nonempty_queries": nonempty,
              "target_pool_full_source_rows": 5034616 + 5285603,
              "candidate_rows_per_query": 200,
              "candidate_pair_recall_ceiling": pair_hit / max(1, pair_total),
              "threshold": threshold, "calibration_macro_f0.5": calibration_f05,
              "test_macro_f0.5": test_f05,
              "test_pair_recall_at_k": {str(k): rank_hits[k] / max(1, pair_total) for k in rank_hits},
              "test_mrr": mrr_sum / max(1, nonempty), "country_groups": group,
              "limitation": "BGE full-source IVF-PQ candidate recall is a fixed ceiling; this evaluates reranking and thresholding, not new candidate generation by this encoder."}
    save_json(root / f"{model_label}_evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "baseline", "fine_tuned"])
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "prepare":
        result = prepare_candidates(cfg, args.queries)
    else:
        adapter = Path(cfg["output_dir"]) / "final_adapter" if args.command == "fine_tuned" else None
        result = score(cfg, args.queries, args.command, adapter, args.batch_size)
    print(json.dumps(result, indent=2))
