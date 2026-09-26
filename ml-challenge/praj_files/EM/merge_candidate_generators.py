"""Interleave and deduplicate per-source candidates from two retrieval runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PRAJ_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PRAJ_DIR / "output"


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        yield from reader


def merge_pair(first: Path, second: Path, output: Path, top_k: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(str(output) + ".tmp")
    with temp_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        first_rows = iter(read_rows(first))
        second_rows = iter(read_rows(second))
        row_number = 0
        while True:
            left = next(first_rows, None)
            right = next(second_rows, None)
            if left is None or right is None:
                if left is not None or right is not None:
                    raise ValueError(f"Candidate files have different row counts: {first} vs {second}")
                break
            row_number += 1
            query_id = left["source1_entity_id"]
            if query_id != right["source1_entity_id"]:
                raise ValueError(f"Query order mismatch at row {row_number}: {query_id} vs {right['source1_entity_id']}")
            lists = [
                [item for item in (left.get("candidate_entity_ids", "") or "").split(",") if item],
                [item for item in (right.get("candidate_entity_ids", "") or "").split(",") if item],
            ]
            merged: list[str] = []
            seen: set[str] = set()
            for rank in range(max(map(len, lists), default=0)):
                for candidates in lists:
                    if rank < len(candidates):
                        candidate = candidates[rank]
                        if candidate not in seen:
                            seen.add(candidate)
                            merged.append(candidate)
                            if len(merged) == top_k:
                                break
                if len(merged) == top_k:
                    break
            writer.writerow([query_id, ",".join(merged)])
    os.replace(temp_output, output)
    print(f"[saved] {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a fixed-size union by alternating ranks from two candidate generators."
    )
    parser.add_argument("--top-k", type=int, default=100, help="Maximum candidates per source/query in the union")
    parser.add_argument("--first-prefix", type=Path, default=OUTPUT_DIR / "candidate_pairs_faiss")
    parser.add_argument("--second-prefix", type=Path, default=OUTPUT_DIR / "candidate_pairs_bge_m3")
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_DIR / "candidate_pairs_union")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    for source in (2, 3):
        merge_pair(
            Path(f"{args.first_prefix}_s{source}.tsv"),
            Path(f"{args.second_prefix}_s{source}.tsv"),
            Path(f"{args.output_prefix}_s{source}.tsv"),
            args.top_k,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
