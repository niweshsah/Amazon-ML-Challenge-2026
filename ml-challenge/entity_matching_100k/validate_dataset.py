"""Validate the canonical sample against its Parquet files and original TSVs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from build_dataset import FIELDS, SAMPLE_SIZE, field_stats, rows, sha256


def read(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(source_dir: Path, output_dir: Path, rebuild_check: bool = False) -> dict:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_files"].items():
        require(sha256(output_dir / name) == expected, f"Output checksum differs: {name}")
    for name, expected in manifest["source_files"].items():
        require(sha256(source_dir / name) == expected, f"Source checksum differs: {name}")

    t1 = read(output_dir / "t1.parquet")
    t2 = read(output_dir / "t2.parquet")
    ground_truth = read(output_dir / "ground_truth.parquet")
    positives = read(output_dir / "positives.parquet")
    negatives = read(output_dir / "negatives.parquet")
    selection = read(output_dir / "selection.parquet")
    t1_by_id = {row["entity_id"]: row for row in t1}
    t2_by_id = {row["entity_id"]: row for row in t2}
    truth_by_id = {row["source1_entity_id"]: row["matched_entity_ids"] for row in ground_truth}
    require(len(t1) == SAMPLE_SIZE == len(t1_by_id), "T1 must contain 100,000 unique records")
    require(len(ground_truth) == SAMPLE_SIZE == len(truth_by_id), "Ground truth must contain one row per T1")
    require(set(truth_by_id) == set(t1_by_id), "Ground-truth T1 IDs differ from T1 table")
    require(len(selection) == SAMPLE_SIZE and {row["source1_entity_id"] for row in selection} == set(t1_by_id), "Selection metadata does not cover T1 exactly")
    require(len(t2) == len(t2_by_id), "T2 contains duplicate IDs")
    require(all(row["entity_id"].startswith("S1-") for row in t1), "Invalid T1 ID prefix")
    require(all(row["entity_id"].startswith(("S2-", "S3-")) for row in t2), "Invalid T2 ID prefix")
    require(all(row["source"] == row["entity_id"][:2] for row in t2), "T2 source column disagrees with ID")

    expected_positive = {(sid, tid, position) for sid, targets in truth_by_id.items() for position, tid in enumerate(targets)}
    observed_positive = {(row["source1_entity_id"], row["matched_entity_id"], row["ground_truth_position"]) for row in positives}
    require(len(positives) == len(expected_positive) and observed_positive == expected_positive, "Positive pairs do not exactly expand ground truth")
    positive_ids = {tid for _, tid, _ in expected_positive}
    require(positive_ids <= set(t2_by_id), "A positive T2 ID is absent from T2 table")
    require(all(len(targets) == len(set(targets)) for targets in truth_by_id.values()), "A T1 has duplicate positive IDs")
    require(len(positive_ids) == len(positives), "A T2 positive is owned by multiple T1 records")
    require(all(t2_by_id[tid]["pool_role"] == "positive" for tid in positive_ids), "Positive pool role is incorrect")
    distractors = {tid for tid, row in t2_by_id.items() if row["pool_role"] == "distractor"}
    require(len(distractors) == 500_000, "Expected 500,000 distractors")
    require(len(t2) == len(positive_ids) + len(distractors), "T2 has an invalid pool role")
    require(Counter(tid[:2] for tid in distractors) == {"S2": 250_000, "S3": 250_000}, "Distractor source balance is wrong")

    negative_pairs = [(row["source1_entity_id"], row["negative_entity_id"]) for row in negatives]
    require(len(negative_pairs) == len(set(negative_pairs)) == SAMPLE_SIZE * 6, "Expected six unique negatives per T1")
    neg_by_t1: dict[str, list[dict]] = defaultdict(list)
    for row in negatives:
        sid, tid = row["source1_entity_id"], row["negative_entity_id"]
        require(sid in t1_by_id and tid in distractors, f"Invalid negative mapping: {sid}, {tid}")
        require(tid not in truth_by_id[sid], f"Negative overlaps a positive: {sid}, {tid}")
        require(row["source"] == tid[:2], f"Negative source disagrees with ID: {tid}")
        if row["negative_kind"] == "same_country":
            require(t1_by_id[sid]["country"] == t2_by_id[tid]["country"], f"Same-country negative differs: {sid}, {tid}")
        else:
            require(row["negative_kind"] == "global", f"Unknown negative kind: {row['negative_kind']}")
        neg_by_t1[sid].append(row)
    require(set(neg_by_t1) == set(t1_by_id), "Some T1 records lack negatives")
    for sid, negs in neg_by_t1.items():
        require(Counter((row["source"], row["negative_kind"]) for row in negs) == {
            ("S2", "same_country"): 2, ("S2", "global"): 1,
            ("S3", "same_country"): 2, ("S3", "global"): 1,
        }, f"Negative allocation differs for {sid}")

    seen_t1: set[str] = set()
    for row in rows(source_dir / "train_source1.tsv"):
        sid = row["entity_id"]
        if sid in t1_by_id:
            require(row == t1_by_id[sid], f"T1 raw fields changed: {sid}")
            seen_t1.add(sid)
    require(seen_t1 == set(t1_by_id), "Some T1 IDs are absent from the original source")
    seen_truth: set[str] = set()
    for row in rows(source_dir / "train_ground_truth.tsv"):
        sid = row["source1_entity_id"]
        if sid in truth_by_id:
            original = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            require(original == truth_by_id[sid], f"Ground truth changed: {sid}")
            seen_truth.add(sid)
    require(seen_truth == set(truth_by_id), "Some T1 truth rows are absent from original ground truth")
    seen_t2: set[str] = set()
    for source in (2, 3):
        for row in rows(source_dir / f"train_source{source}.tsv"):
            tid = row["entity_id"]
            if tid in t2_by_id:
                require(all(row[field] == t2_by_id[tid][field] for field in FIELDS), f"T2 raw fields changed: {tid}")
                seen_t2.add(tid)
    require(seen_t2 == set(t2_by_id), "Some T2 IDs are absent from the original sources")
    require(field_stats(t1) == stats["t1"], "T1 statistics differ")
    require(field_stats(t2) == stats["t2"], "T2 statistics differ")
    require(sum(not targets for targets in truth_by_id.values()) == stats["singletons"], "Singleton count differs")
    require(len(positives) == stats["positive_pairs"] and len(negatives) == stats["negative_pairs"], "Pair counts differ")

    if rebuild_check:
        with tempfile.TemporaryDirectory(prefix="entity_matching_100k_rebuild_") as temporary:
            command = [sys.executable, str(Path(__file__).resolve().with_name("build_dataset.py")),
                       "--source-dir", str(source_dir), "--output-dir", temporary]
            subprocess.run(command, check=True)
            for name, expected in manifest["output_files"].items():
                require(sha256(Path(temporary) / name) == expected, f"Rebuild differs: {name}")
    return {"status": "PASS", "t1": len(t1), "t2": len(t2), "positives": len(positives),
            "negatives": len(negatives), "singletons": stats["singletons"],
            "raw_round_trip": True, "rebuild_identical": rebuild_check}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "student_resource/dataset/train")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--rebuild-check", action="store_true", help="Rebuild in a temporary directory and compare checksums")
    args = parser.parse_args()
    print(json.dumps(validate(args.source_dir.resolve(), args.output_dir.resolve(), args.rebuild_check), indent=2))


if __name__ == "__main__":
    main()
