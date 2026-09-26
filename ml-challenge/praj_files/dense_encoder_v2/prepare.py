"""Deterministic grouped splits and fixed sampled target pool."""
from __future__ import annotations

import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from core import DATA, digest, save_json, serialize, tags


def tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def smallest(heap: list, limit: int, score: int, item: object) -> None:
    entry = (-score, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def prepare(output: Path, data: Path = DATA, eval_count: int = 100000,
            dev_count: int = 2000, pair_count: int = 200000,
            distractor_count: int = 500000, logger=None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if pair_count < 200000 and eval_count == 100000:
        raise ValueError("Production preparation requires at least 200,000 pairs")
    truth_path = data / "train_ground_truth.tsv"
    group_heap: list = []
    group_count = 0
    for row in tsv(truth_path):
        group_count += 1
        smallest(group_heap, eval_count + dev_count, digest(row["source1_entity_id"]), row["source1_entity_id"])
        if logger and group_count % 500000 == 0:
            logger("split_scan_progress", groups_done=group_count)
    selected = sorted((-score, sid) for score, sid in group_heap)
    eval_ids = {sid for _, sid in selected[:eval_count]}
    dev_ids = {sid for _, sid in selected[eval_count:]}
    if len(eval_ids) != eval_count or len(dev_ids) != dev_count or eval_ids & dev_ids:
        raise ValueError("Split count or disjointness failure")
    if logger:
        logger("split_selected", total_groups=group_count, eval_groups=len(eval_ids), dev_groups=len(dev_ids))
    truth = {}
    train_heap: list = []
    candidate_limit = min(max(pair_count * 2, pair_count + 20000), 500000)
    positive_count = 0
    singleton_count = 0
    for row in tsv(truth_path):
        sid = row["source1_entity_id"]
        targets = [x for x in row["matched_entity_ids"].split(",") if x]
        if sid in eval_ids or sid in dev_ids:
            truth[sid] = targets
            singleton_count += not bool(targets)
        else:
            for tid in targets:
                positive_count += 1
                smallest(train_heap, candidate_limit, digest(sid + ":" + tid, "pair"), (sid, tid))
    if len(train_heap) < pair_count:
        raise ValueError("Too few training positives")
    if logger:
        logger("truth_scan_complete", training_positives=positive_count,
               sampled_pair_candidates=len(train_heap), heldout_singletons=singleton_count)
    train_pairs = [item for _, item in train_heap]
    train_query_ids = {sid for sid, _ in train_pairs}
    selected_query_ids = set(truth) | train_query_ids
    query_rows = {}
    for row in tsv(data / "train_source1.tsv"):
        if row["entity_id"] in selected_query_ids:
            query_rows[row["entity_id"]] = row
    if set(query_rows) != selected_query_ids:
        raise ValueError("Missing selected Source 1 records")
    if logger:
        logger("query_records_loaded", records=len(query_rows))
    eval_true_targets = {tid for sid, ids in truth.items() if sid in eval_ids for tid in ids}
    dev_true_targets = {tid for sid, ids in truth.items() if sid in dev_ids for tid in ids}
    mandatory = eval_true_targets | dev_true_targets
    train_targets = {tid for _, tid in train_pairs}
    if mandatory & train_targets:
        raise ValueError("Target appears in both train and held-out groups")
    required = mandatory | train_targets
    distractor_heap: list = []
    target_rows = {}
    for source in (2, 3):
        for row in tsv(data / f"train_source{source}.tsv"):
            tid = row["entity_id"]
            if tid in required:
                target_rows[tid] = row
            elif tid not in required:
                smallest(distractor_heap, distractor_count, digest(tid, "distractor"), tid)
    distractor_ids = {tid for _, tid in distractor_heap}
    if len(distractor_ids) != distractor_count:
        raise ValueError("Too few distractors")
    for source in (2, 3):
        for row in tsv(data / f"train_source{source}.tsv"):
            if row["entity_id"] in distractor_ids:
                target_rows[row["entity_id"]] = row
    if set(target_rows) != required | distractor_ids:
        raise ValueError(f"Missing {len((required | distractor_ids) - set(target_rows))} targets")
    if logger:
        logger("target_records_loaded", records=len(target_rows), distractors=len(distractor_ids))
    scored = []
    for sid, tid in train_pairs:
        pair_tags = tags(query_rows[sid], target_rows[tid])
        # Selection is deterministic. Reserve coverage, then fill by pair hash.
        scored.append((digest(sid + ":" + tid, "final_pair"), sid, tid, pair_tags))
    priority_tags = ["cross_script", "mixed_script", "bengali", "tamil", "telugu",
                     "gujarati", "kannada", "malayalam", "gurmukhi", "oriya",
                     "devanagari", "missing_field", "short_text", "noisy_text", "near_duplicate"]
    selected_pairs = {}
    quota = max(1, pair_count // 50)
    for tag in priority_tags:
        matches = sorted((x for x in scored if tag in x[3]), key=lambda x: x[0])
        for item in matches[:quota]:
            if len(selected_pairs) >= pair_count:
                break
            selected_pairs[(item[1], item[2])] = item
    for item in sorted(scored, key=lambda x: x[0]):
        if len(selected_pairs) >= pair_count:
            break
        selected_pairs[(item[1], item[2])] = item
    if len(selected_pairs) != pair_count:
        raise ValueError("Pair selection count failure")
    if logger:
        logger("training_pairs_selected", pairs=len(selected_pairs))
    pairs = []
    for _, sid, tid, pair_tags in sorted(selected_pairs.values()):
        pairs.append({"anchor_id": sid, "positive_id": tid,
                      "anchor": serialize(query_rows[sid]), "positive": serialize(target_rows[tid]),
                      "tags": ",".join(pair_tags),
                      "country": query_rows[sid].get("country", ""),
                      "name_key": query_rows[sid].get("business_name", "").casefold().strip()})
    # A lexical negative is safe only when it belongs to another Source 1 group.
    by_name = defaultdict(list)
    by_country = defaultdict(list)
    for pair in pairs:
        by_name[(pair["country"], pair["name_key"])].append(pair)
        by_country[pair["country"]].append(pair)
    for pair in pairs:
        sid = pair["anchor_id"]
        options = by_name[(pair["country"], pair["name_key"])][:256] + by_country[pair["country"]][:256]
        negative = next((x for x in options if x["anchor_id"] != sid), None)
        pair["negative_id"] = negative["positive_id"] if negative else ""
        pair["negative_owner_id"] = negative["anchor_id"] if negative else ""
        pair["negative"] = negative["positive"] if negative else ""
        pair.pop("country")
        pair.pop("name_key")
    eval_rows = []
    dev_rows = []
    for sid, ids in truth.items():
        row = query_rows[sid]
        query_tags = set(tags(row))
        for tid in ids:
            query_tags.update(tags(row, target_rows[tid]))
        entry = {"entity_id": sid, "text": serialize(row), "truth": ",".join(ids),
                 "tags": ",".join(sorted(query_tags | ({"singleton"} if not ids else set())))}
        (eval_rows if sid in eval_ids else dev_rows).append(entry)
    eval_rows.sort(key=lambda x: digest(x["entity_id"], "calibration"))
    dev_rows.sort(key=lambda x: x["entity_id"])
    eval_targets = sorted(eval_true_targets | distractor_ids)
    dev_pool = sorted(dev_true_targets | set(eval_targets[:20000]))
    write_table(output / "train_pairs.parquet", pairs)
    write_table(output / "eval_queries.parquet", eval_rows)
    write_table(output / "dev_queries.parquet", dev_rows)
    write_table(output / "eval_targets.parquet", [{"entity_id": tid, "text": serialize(target_rows[tid])} for tid in eval_targets])
    write_table(output / "dev_targets.parquet", [{"entity_id": tid, "text": serialize(target_rows[tid])} for tid in dev_pool])
    manifest = {"seed": 42, "split_method": "smallest seeded blake2b Source 1 ID hashes",
                "calibration_method": "first 20,000 evaluation rows by independent seeded hash",
                "train_groups": group_count - eval_count - dev_count,
                "dev_groups": dev_count, "eval_groups": eval_count, "calibration_groups": min(20000, eval_count),
                "score_groups": max(0, eval_count - 20000), "heldout_singletons": singleton_count,
                "available_train_positive_pairs": positive_count, "train_pairs": len(pairs),
                "train_pairs_with_hard_negative": sum(bool(x["negative_id"]) for x in pairs),
                "eval_true_targets": len(eval_true_targets), "dev_true_targets": len(dev_true_targets),
                "distractors": len(distractor_ids),
                "eval_pool_size": len(eval_targets), "dev_pool_size": len(dev_pool),
                "pair_tags": dict(Counter(tag for row in pairs for tag in row["tags"].split(",") if tag)),
                "paths": {name: str(output / f"{name}.parquet") for name in
                          ("train_pairs", "eval_queries", "dev_queries", "eval_targets", "dev_targets")}}
    save_json(output / "split_manifest.json", manifest)
    return manifest


def verify_prepared(output: Path) -> dict:
    train = pq.read_table(output / "train_pairs.parquet",
                          columns=["anchor_id", "positive_id", "negative_id"]).to_pylist()
    dev = pq.read_table(output / "dev_queries.parquet", columns=["entity_id", "truth"]).to_pylist()
    evaluation = pq.read_table(output / "eval_queries.parquet", columns=["entity_id", "truth"]).to_pylist()
    pool = pq.read_table(output / "eval_targets.parquet", columns=["entity_id"]).column(0).to_pylist()
    train_ids = {x["anchor_id"] for x in train}
    dev_ids = {x["entity_id"] for x in dev}
    eval_ids = {x["entity_id"] for x in evaluation}
    if train_ids & dev_ids or train_ids & eval_ids or dev_ids & eval_ids:
        raise ValueError("Source 1 split overlap")
    if len(eval_ids) != len(evaluation) or len(dev_ids) != len(dev):
        raise ValueError("Duplicate held-out Source 1 row")
    if len(pool) != len(set(pool)):
        raise ValueError("Duplicate target in evaluation pool")
    pool_set = set(pool)
    eval_true = {tid for row in evaluation for tid in row["truth"].split(",") if tid}
    if not eval_true <= pool_set:
        raise ValueError(f"Missing {len(eval_true - pool_set)} evaluation positives")
    train_positive = {x["positive_id"] for x in train}
    if train_positive & eval_true:
        raise ValueError("Target leakage between train and evaluation")
    if len({(x["anchor_id"], x["positive_id"]) for x in train}) != len(train):
        raise ValueError("Duplicate training pair")
    if any(x["negative_id"] == x["positive_id"] for x in train):
        raise ValueError("Positive used as its own hard negative")
    report = {"passed": True, "train_pairs": len(train), "train_groups_in_pairs": len(train_ids),
              "dev_queries": len(dev), "evaluation_queries": len(evaluation),
              "evaluation_positives": len(eval_true), "pool_size": len(pool),
              "missing_evaluation_positives": 0, "split_overlap": 0}
    save_json(output / "data_verification.json", report)
    return report
