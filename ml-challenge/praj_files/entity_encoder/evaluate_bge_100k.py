"""Evaluate existing full-pool zero-shot BGE-M3 top-100 retrieval on 100k held-out T1s."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

from common import config, rows, save_json, source_path


K_VALUES = (1, 5, 10, 50, 100)


def digest(text: str, salt: bytes = b"") -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), key=salt, digest_size=8).digest(), "big")


def sample_truth(cfg: dict, count: int) -> dict[str, set[str]]:
    # Exclude both training anchors and the 400-query model-selection slice.
    heap: list[tuple[int, str]] = []
    path = Path(cfg["data_dir"]) / "train_ground_truth.parquet"
    for row in rows(path, ["source1_entity_id", "matched_entity_ids"]):
        entity_id = row["source1_entity_id"]
        if digest(entity_id) / 2**64 < cfg["anchor_fraction"] + cfg["validation_fraction"]:
            continue
        score = digest(entity_id, b"bge100k")
        item = (-score, entity_id)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    ids = {entity_id for _, entity_id in heap}
    if len(ids) != count:
        raise RuntimeError(f"Selected {len(ids)} IDs, expected {count}")
    truth = {}
    for row in rows(path, ["source1_entity_id", "matched_entity_ids"]):
        entity_id = row["source1_entity_id"]
        if entity_id in ids:
            truth[entity_id] = set(filter(None, (row["matched_entity_ids"] or "").split(",")))
    return truth


def evaluate(cfg: dict, count: int, output: Path) -> dict:
    truth = sample_truth(cfg, count)
    countries = {}
    for row in rows(source_path(cfg, "train", 1), ["entity_id", "country"]):
        if row["entity_id"] in truth:
            countries[row["entity_id"]] = row["country"] or ""
    per_query = {entity_id: {"hits": set(), "best_rank": None,
                             "hits_at_k": Counter(), "candidate_count": 0}
                 for entity_id in truth}
    root = Path(__file__).resolve().parents[1] / "output"
    source_reports = {}
    for source in (2, 3):
        file_path = root / f"candidate_pairs_bge_m3_s{source}.tsv"
        metadata = json.loads((root / f"faiss_index_bge_m3_train_source{source}.index.json").read_text())
        if metadata["model"] != "BAAI/bge-m3" or metadata["index_type"] != "IndexIVFPQ":
            raise RuntimeError(f"Unexpected index metadata for source {source}")
        if Path(metadata["source_data"]["path"]).name != source_path(cfg, "train", source).name:
            raise RuntimeError(f"Index input differs from configured data for source {source}")
        actual_fragments = sorted(x.stat().st_size for x in source_path(cfg, "train", source).rglob("*.parquet"))
        indexed_fragments = sorted(x["size"] for x in metadata["source_data"]["fragments"])
        if actual_fragments != indexed_fragments:
            raise RuntimeError(f"Index input fragment sizes differ for source {source}")
        seen = set()
        total_rows = 0
        source_hits = Counter()
        source_truth = 0
        with file_path.open(encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\n\r")
            if header != "source1_entity_id\tcandidate_entity_ids":
                raise RuntimeError(f"Bad header in {file_path}")
            for line in handle:
                total_rows += 1
                entity_id, _, candidates = line.rstrip("\n\r").partition("\t")
                if entity_id not in truth:
                    continue
                if entity_id in seen:
                    raise RuntimeError(f"Duplicate sampled ID {entity_id}")
                seen.add(entity_id)
                ids = candidates.split(",") if candidates else []
                expected = {x for x in truth[entity_id] if x.startswith(f"S{source}-")}
                source_truth += len(expected)
                state = per_query[entity_id]
                state["candidate_count"] += len(ids)
                for rank, candidate in enumerate(ids, 1):
                    if candidate in expected:
                        state["hits"].add(candidate)
                        previous = state["best_rank"]
                        state["best_rank"] = rank if previous is None else min(previous, rank)
                        for k in K_VALUES:
                            if rank <= k:
                                source_hits[k] += 1
                                state["hits_at_k"][k] += 1
        if seen != truth.keys():
            raise RuntimeError(f"Missing {len(truth.keys() - seen)} sampled IDs in {file_path}")
        source_reports[f"source{source}"] = {
            "index_model": metadata["model"], "index_type": metadata["index_type"],
            "full_target_rows": 5034616 if source == 2 else 5285603,
            "candidate_file_rows": total_rows, "true_pairs": source_truth,
            "pair_recall_at_k": {str(k): source_hits[k] / max(1, source_truth) for k in K_VALUES}}
    pair_count = sum(len(v) for v in truth.values())
    total_hits = sum(len(state["hits"]) for state in per_query.values())
    total_candidates = sum(state["candidate_count"] for state in per_query.values())
    nonempty = [entity_id for entity_id, ids in truth.items() if ids]
    macro_ceiling = []
    for entity_id, ids in truth.items():
        if not ids:
            macro_ceiling.append(1.0)
            continue
        r = len(per_query[entity_id]["hits"]) / len(ids)
        macro_ceiling.append(1.25 * r / (0.25 + r) if r else 0.0)
    groups = {}
    for country in sorted(set(countries.values())):
        subset = [entity_id for entity_id in nonempty if countries[entity_id] == country]
        groups[country] = {"nonempty_queries": len(subset),
                           "any_match_at_100": sum(bool(per_query[x]["hits"]) for x in subset) / max(1, len(subset)),
                           "all_matches_at_100": sum(per_query[x]["hits"] == truth[x] for x in subset) / max(1, len(subset)),
                           "pair_recall_at_100": sum(len(per_query[x]["hits"]) for x in subset) /
                           max(1, sum(len(truth[x]) for x in subset))}
    result = {
        "protocol": "100000 deterministic held-out training T1 IDs; full train T2/T3 IVF-PQ top-100 candidate files",
        "model": "BAAI/bge-m3", "queries": len(truth),
        "nonempty_queries": len(nonempty), "singletons": len(truth) - len(nonempty),
        "total_true_pairs": pair_count, "total_retrieved_true_pairs": total_hits,
        "pair_recall_at_100": total_hits / max(1, pair_count),
        "any_match_at_100": sum(bool(per_query[x]["hits"]) for x in nonempty) / max(1, len(nonempty)),
        "all_matches_at_100": sum(per_query[x]["hits"] == truth[x] for x in nonempty) / max(1, len(nonempty)),
        "best_per_source_rank_mrr": sum(1 / per_query[x]["best_rank"] for x in nonempty
                                        if per_query[x]["best_rank"]) / max(1, len(nonempty)),
        "candidate_precision_at_100_each_source": total_hits / max(1, total_candidates),
        "oracle_macro_f0.5_ceiling_at_100": sum(macro_ceiling) / len(macro_ceiling),
        "country_groups": groups, "sources": source_reports,
        "caveat": "FAISS IVF-PQ is approximate; candidate TSVs have ranks but no scores. Oracle F0.5 assumes perfect filtering of retrieved candidates and is an upper bound, not model performance."
    }
    save_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--queries", type=int, default=100000)
    args = parser.parse_args()
    cfg = config(args.config)
    output = Path(cfg["output_dir"]) / f"zero_shot_bge_full_pool_{args.queries}.json"
    print(json.dumps(evaluate(cfg, args.queries, output), indent=2))
