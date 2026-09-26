"""Benchmark ranked candidate files on the first million Source 1 training rows.

Reports pair recall, all-match retention, and the perfect-matcher macro F0.5
ceiling. A fair round-robin union uses the same 100-per-source budget as each
individual retriever. The full union is reported separately for comparison.
"""

from __future__ import annotations

import argparse
import csv
import resource
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from paper_blocking_probe import ROOT, TRAIN
from multilingual_candidate_probe import name_script


def read_first_ids(path: Path, limit: int) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        result = []
        for row in reader:
            result.append(row["entity_id"])
            if len(result) == limit:
                break
    return result


def read_truth(path: Path, query_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id = row["source1_entity_id"]
            if query_id in query_ids:
                truth[query_id] = set(filter(None, row["matched_entity_ids"].split(",")))
    if len(truth) != len(query_ids):
        raise ValueError(f"Ground truth has {len(truth):,} of {len(query_ids):,} selected queries")
    return truth


def ordered_union(first: list[str], second: list[str], limit: int) -> set[str]:
    result: set[str] = set()
    for rank in range(max(len(first), len(second))):
        if rank < len(first):
            result.add(first[rank])
            if len(result) == limit:
                break
        if rank < len(second):
            result.add(second[rank])
            if len(result) == limit:
                break
    return result


class Score:
    def __init__(self) -> None:
        self.hits = 0
        self.pairs = 0
        self.full_queries = 0
        self.positive_queries = 0
        self.oracle_total = 0.0
        self.candidates = 0

    def add(self, truth: set[str], selected: set[str]) -> set[str]:
        found = truth & selected
        self.hits += len(found)
        self.pairs += len(truth)
        self.candidates += len(selected)
        if not truth:
            self.oracle_total += 1.0
        else:
            self.positive_queries += 1
            self.full_queries += len(found) == len(truth)
            recall = len(found) / len(truth)
            if recall:
                self.oracle_total += 1.25 * recall / (0.25 + recall)
        return truth - found


def script_report(
    true_target_counts: Counter[str], missed_target_counts: dict[str, Counter[str]],
    output_path: Path,
) -> None:
    script_counts: dict[str, Counter[str]] = defaultdict(Counter)
    remaining = set(true_target_counts)
    for source in (2, 3):
        path = TRAIN / f"train_source{source}.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                target_id = row["entity_id"]
                if target_id not in remaining:
                    continue
                remaining.remove(target_id)
                script = name_script(row["business_name"])
                result = script_counts[script]
                result["true_pairs"] += true_target_counts[target_id]
                for method, missed in missed_target_counts.items():
                    result[f"{method}_missed"] += missed[target_id]
        print(f"[script] scanned {path.name}; remaining truth IDs={len(remaining):,}", flush=True)
    if remaining:
        raise ValueError(f"Could not find {len(remaining):,} ground-truth target IDs")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        methods = list(missed_target_counts)
        writer.writerow(["script", "true_pairs"] + [f"{method}_recall" for method in methods])
        for script in ("Latin", "Devanagari", "Tamil", "Other non-Latin"):
            counts = script_counts[script]
            total = counts["true_pairs"]
            if total:
                writer.writerow([script, total] + [f"{1 - counts[f'{method}_missed'] / total:.6f}" for method in methods])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--bge-s2", type=Path, default=ROOT / "praj_files/output/candidate_pairs_bge_m3_s2.tsv")
    parser.add_argument("--bge-s3", type=Path, default=ROOT / "praj_files/output/candidate_pairs_bge_m3_s3.tsv")
    parser.add_argument("--faiss-s2", type=Path, default=ROOT / "praj_files/output/candidate_pairs_faiss_s2.tsv")
    parser.add_argument("--faiss-s3", type=Path, default=ROOT / "praj_files/output/candidate_pairs_faiss_s3.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "praj_files/EM/reports/first_million_retrieval.tsv")
    parser.add_argument("--script-output", type=Path, default=ROOT / "praj_files/EM/reports/first_million_script.tsv")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    started = time.monotonic()
    query_ids = read_first_ids(TRAIN / "train_source1.tsv", args.limit)
    truth = read_truth(TRAIN / "train_ground_truth.tsv", set(query_ids))
    print(f"Loaded {len(query_ids):,} queries and their truth in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    methods = ["bge_10", "bge_50", "bge_100", "faiss_10", "faiss_50", "faiss_100", "round_robin_100", "full_union_200"]
    scores = {method: Score() for method in methods}
    true_target_counts: Counter[str] = Counter()
    missed_target_counts = {method: Counter() for method in ("bge_100", "faiss_100", "round_robin_100", "full_union_200")}
    paths = (args.bge_s2, args.bge_s3, args.faiss_s2, args.faiss_s3)
    with ExitStack() as stack:
        readers = [csv.reader(stack.enter_context(path.open(encoding="utf-8", newline="")), delimiter="\t") for path in paths]
        for reader, path in zip(readers, paths):
            if next(reader, None) != ["source1_entity_id", "candidate_entity_ids"]:
                raise ValueError(f"Unexpected candidate file header: {path}")
        for number, query_id in enumerate(query_ids, 1):
            rows = [next(reader, None) for reader in readers]
            if any(row is None or row[0] != query_id for row in rows):
                raise ValueError(f"Candidate row mismatch at row {number:,} ({query_id})")
            bge_s2, bge_s3, faiss_s2, faiss_s3 = [row[1].split(",") if row[1] else [] for row in rows]
            expected = truth.pop(query_id)
            true_target_counts.update(expected)
            for k in (10, 50, 100):
                bge = set(bge_s2[:k] + bge_s3[:k])
                faiss = set(faiss_s2[:k] + faiss_s3[:k])
                bge_missing = scores[f"bge_{k}"].add(expected, bge)
                faiss_missing = scores[f"faiss_{k}"].add(expected, faiss)
                if k == 100:
                    missed_target_counts["bge_100"].update(bge_missing)
                    missed_target_counts["faiss_100"].update(faiss_missing)
            round_robin = ordered_union(bge_s2, faiss_s2, 100) | ordered_union(bge_s3, faiss_s3, 100)
            full_union = set(bge_s2 + bge_s3 + faiss_s2 + faiss_s3)
            missed_target_counts["round_robin_100"].update(scores["round_robin_100"].add(expected, round_robin))
            missed_target_counts["full_union_200"].update(scores["full_union_200"].add(expected, full_union))
            if number % 100_000 == 0:
                print(f"Scored {number:,}/{len(query_ids):,} queries in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["method", "queries", "true_pairs", "hits", "pair_recall", "all_match_retention", "oracle_macro_f05", "candidate_pairs", "mean_candidates_per_query"])
        for method in methods:
            score = scores[method]
            writer.writerow([method, len(query_ids), score.pairs, score.hits, f"{score.hits / score.pairs:.6f}",
                             f"{score.full_queries / score.positive_queries:.6f}",
                             f"{score.oracle_total / len(query_ids):.6f}", score.candidates,
                             f"{score.candidates / len(query_ids):.3f}"])
    print(args.output.read_text(encoding="utf-8"), flush=True)
    script_report(true_target_counts, missed_target_counts, args.script_output)
    print(args.script_output.read_text(encoding="utf-8"), flush=True)
    print(f"Finished in {(time.monotonic() - started) / 60:.1f} min; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2:.2f} GiB")


if __name__ == "__main__":
    main()
