"""Resumable sampled-pool retrieval recall analysis for K=1..50.

Reads the cached query vectors and FAISS target index, searches once at K=50,
then reports pair- and query-level retrieval coverage for every K. Search rows
are checkpointed to a memmap so an interrupted run resumes without repeating
completed queries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id = path, run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()

    def emit(self, event: str, **fields: object) -> None:
        row = {"timestamp_utc": utc_now(), "run_id": self.run_id,
               "event": event, "elapsed_seconds": round(time.monotonic() - self.start, 3), **fields}
        line = json.dumps(row, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)


def atomic_json(path: Path, obj: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def truth_ids(query_rows: list[dict]) -> list[set[str]]:
    return [set(x for x in (row.get("truth") or "").split(",") if x) for row in query_rows]


def make_table(queries: list[dict], targets: list[dict], positions: np.ndarray) -> list[dict]:
    truths = truth_ids(queries)
    total_pairs = sum(map(len, truths))
    nonempty = sum(bool(x) for x in truths)
    singletons = sum(len(x) == 1 for x in truths)
    multi = sum(len(x) > 1 for x in truths)
    rows: list[dict] = []
    cumulative_hits = np.zeros(len(queries), dtype=np.int32)
    cumulative_retrieved = np.zeros(len(queries), dtype=np.int32)
    for k in range(1, positions.shape[1] + 1):
        col = positions[:, k - 1]
        valid = col >= 0
        cumulative_retrieved += valid.astype(np.int32)
        for i in np.flatnonzero(valid):
            if targets[int(col[i])]["entity_id"] in truths[i]:
                cumulative_hits[i] += 1
        hits = cumulative_hits
        found_pairs = int(hits.sum())
        any_hit = int(((hits > 0) & np.fromiter((bool(x) for x in truths), bool, len(truths))).sum())
        all_hit = int(sum(bool(truths[i]) and hits[i] == len(truths[i]) for i in range(len(queries))))
        singleton_hit = int(sum(len(truths[i]) == 1 and hits[i] == 1 for i in range(len(queries))))
        multi_all = int(sum(len(truths[i]) > 1 and hits[i] == len(truths[i]) for i in range(len(queries))))
        retrieved = int(cumulative_retrieved.sum())
        rows.append({
            "k": k,
            "total_ground_truth_pairs": total_pairs,
            "ground_truth_pairs_found": found_pairs,
            "pair_recall_pct": 100.0 * found_pairs / total_pairs if total_pairs else 0.0,
            "queries_with_truth": nonempty,
            "queries_with_any_match": any_hit,
            "any_match_recall_pct": 100.0 * any_hit / nonempty if nonempty else 0.0,
            "queries_with_all_matches": all_hit,
            "all_match_recall_pct": 100.0 * all_hit / nonempty if nonempty else 0.0,
            "singleton_queries": singletons,
            "singleton_correct": singleton_hit,
            "singleton_accuracy_pct": 100.0 * singleton_hit / singletons if singletons else 0.0,
            "multi_match_queries": multi,
            "multi_match_all_found": multi_all,
            "multi_match_complete_pct": 100.0 * multi_all / multi if multi else 0.0,
            "retrieved_candidates": retrieved,
            "candidate_precision_pct": 100.0 * found_pairs / retrieved if retrieved else 0.0,
            "mean_true_pairs_found_per_query": found_pairs / len(queries) if queries else 0.0,
        })
    return rows


def write_outputs(root: Path, run_id: str, queries: list[dict], targets: list[dict],
                  positions: np.ndarray, logger: EventLog) -> None:
    rows = make_table(queries, targets, positions)
    csv_path = root / "recall_at_k_1_to_50.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = root / "recall_at_k_1_to_50.md"
    columns = ["k", "ground_truth_pairs_found", "total_ground_truth_pairs", "pair_recall_pct",
               "queries_with_any_match", "any_match_recall_pct", "queries_with_all_matches",
               "all_match_recall_pct", "singleton_correct", "singleton_accuracy_pct",
               "multi_match_all_found", "multi_match_complete_pct", "candidate_precision_pct"]
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# Sampled-pool retrieval coverage at K=1..50\n\n")
        stream.write(f"Queries: **{len(queries):,}**; targets in pool: **{len(targets):,}**. Pair recall counts all ground-truth pairs, including multiple true targets for one query.\n\n")
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("|" + "|".join(["---:"] * len(columns)) + "|\n")
        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            stream.write("| " + " | ".join(values) + " |\n")
    report = {
        "label": "sampled-pool",
        "run_id": run_id,
        "query_count": len(queries),
        "target_pool_size": len(targets),
        "total_ground_truth_pairs": rows[0]["total_ground_truth_pairs"],
        "queries_with_truth": rows[0]["queries_with_truth"],
        "singleton_queries": rows[0]["singleton_queries"],
        "multi_match_queries": rows[0]["multi_match_queries"],
        "metrics_definition": {
            "pair_recall_pct": "retrieved true pairs / all ground-truth pairs",
            "any_match_recall_pct": "queries with >=1 true target in top K / queries with >=1 true target",
            "all_match_recall_pct": "queries with every true target in top K / queries with >=1 true target",
            "candidate_precision_pct": "retrieved true pairs / all candidate slots retrieved",
            "singleton_accuracy_pct": "single-target queries whose target is in top K / single-target queries",
        },
        "rows": rows,
    }
    atomic_json(root / "recall_at_k_1_to_50.json", report)
    logger.emit("analysis_complete", csv=str(csv_path), markdown=str(md_path), json=str(root / "recall_at_k_1_to_50.json"),
                metrics_at_k50=rows[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("praj_files/dense_encoder_v2/artifacts"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--flush-rows", type=int, default=5120)
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root / "logs"
    logger = EventLog(logs / f"recall_at_k_{args.run_id}.jsonl", args.run_id)
    logger.emit("start", argv=sys.argv, root=str(root), k=args.k, batch_size=args.batch_size, nprobe=args.nprobe)
    try:
        q_path, t_path, v_path, idx_path = (root / "eval_queries.parquet", root / "eval_targets.parquet",
                                            root / "query_vectors.npy", root / "target.index")
        for path in (q_path, t_path, v_path, idx_path):
            if not path.exists():
                raise FileNotFoundError(path)
        queries = pq.read_table(q_path).to_pylist()
        targets = pq.read_table(t_path, columns=["entity_id"]).to_pylist()
        vectors = np.load(v_path, mmap_mode="r")
        if len(queries) != len(vectors):
            raise ValueError(f"query/vector count mismatch: {len(queries)} != {len(vectors)}")
        if not 1 <= args.k <= 100 or args.batch_size <= 0 or args.flush_rows <= 0:
            raise ValueError("k, batch-size, and flush-rows must be positive (k <= 100)")
        partial = root / f"retrieval_positions_top{args.k}.partial.npy"
        final = root / f"retrieval_positions_top{args.k}.npy"
        state_path = root / f"retrieval_positions_top{args.k}.progress.json"
        if final.exists():
            positions = np.load(final, mmap_mode="r")
            if positions.shape != (len(queries), args.k):
                raise ValueError(f"existing result has unexpected shape {positions.shape}")
            logger.emit("search_cache_found", result=str(final), rows=len(queries))
        else:
            if partial.exists():
                positions = np.load(partial, mmap_mode="r+")
                if positions.shape != (len(queries), args.k):
                    raise ValueError("partial result shape differs from current evaluation data")
            else:
                positions = np.lib.format.open_memmap(partial, mode="w+", dtype=np.int64,
                                                       shape=(len(queries), args.k))
                positions[:] = -1
                positions.flush()
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            start_at = int(state.get("next_query", 0))
            # A row written before the last progress record is harmless to repeat; protect against gaps.
            while start_at < len(queries) and np.all(positions[start_at] >= 0):
                start_at += 1
            index = faiss.read_index(str(idx_path))
            if hasattr(index, "nprobe"):
                index.nprobe = args.nprobe
            started = time.monotonic()
            logger.emit("search_started", queries=len(queries), targets=len(targets), resume_from=start_at,
                        index_type=type(index).__name__)
            cursor = start_at
            while cursor < len(queries):
                end = min(len(queries), cursor + args.batch_size)
                _, batch_pos = index.search(np.asarray(vectors[cursor:end]), args.k)
                positions[cursor:end] = batch_pos
                cursor = end
                if cursor % args.flush_rows < args.batch_size or cursor == len(queries):
                    positions.flush()
                    atomic_json(state_path, {"run_id": args.run_id, "next_query": cursor,
                                             "query_count": len(queries), "k": args.k,
                                             "updated_utc": utc_now()})
                    elapsed = time.monotonic() - started
                    logger.emit("search_progress", queries_done=cursor, queries_total=len(queries),
                                queries_per_second=round((cursor - start_at) / max(elapsed, 1e-6), 2),
                                estimated_remaining_seconds=round((len(queries) - cursor) /
                                    max((cursor - start_at) / max(elapsed, 1e-6), 1e-6), 1))
            positions.flush()
            del positions
            os.replace(partial, final)
            positions = np.load(final, mmap_mode="r")
            atomic_json(state_path, {"run_id": args.run_id, "next_query": len(queries),
                                     "query_count": len(queries), "k": args.k,
                                     "complete": True, "updated_utc": utc_now()})
            logger.emit("search_complete", result=str(final), seconds=round(time.monotonic() - started, 2))
        write_outputs(root, args.run_id, queries, targets, positions, logger)
    except Exception as exc:
        logger.emit("failed", error=repr(exc), traceback=traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
