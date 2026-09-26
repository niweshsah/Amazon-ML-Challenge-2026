"""Probe paper-inspired token blocking against the complete training corpus.

This is a bounded, sampled retrieval experiment, not a full submission generator.
It uses standard token blocks, optional name q-gram blocks, block-frequency
weighting, and per-query top-k pruning. No training labels affect retrieval.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "student_resource" / "dataset" / "train"
WORD = re.compile(r"[^a-z0-9]+")
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "pvt", "private"}


def words(value: str) -> list[str]:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return [token for token in WORD.split(value) if token]


def signatures(name: str, address: str, qgrams: bool) -> set[str]:
    name_tokens = [token for token in words(name) if token not in LEGAL]
    address_tokens = words(address)
    result = {"n:" + token for token in name_tokens if len(token) >= 3}
    result.update("a:" + token for token in address_tokens if len(token) >= 4)
    result.update("d:" + token for token in address_tokens if token.isdigit() and len(token) >= 2)
    if name_tokens:
        result.add("e:" + " ".join(name_tokens))
    if qgrams:
        for token in name_tokens:
            if len(token) >= 5:
                result.update("g:" + token[i:i + 4] for i in range(len(token) - 3))
    return result


def sample_queries(path: Path, size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if position < size:
                sample.append(row)
            else:
                replacement = rng.randrange(position + 1)
                if replacement < size:
                    sample[replacement] = row
    return sample


def build_query_index(
    queries: list[dict[str, str]], max_query_frequency: int, qgrams: bool
) -> tuple[dict[str, list[int]], dict[str, float]]:
    all_keys = [signatures(row["business_name"], row["business_address"], qgrams) for row in queries]
    frequency = Counter(key for keys in all_keys for key in keys)
    postings: dict[str, list[int]] = defaultdict(list)
    weights: dict[str, float] = {}
    for query_number, keys in enumerate(all_keys):
        for key in keys:
            if frequency[key] > max_query_frequency and not key.startswith("e:"):
                continue
            postings[key].append(query_number)
            weight = math.log1p(len(queries) / frequency[key])
            weights[key] = weight * (4.0 if key.startswith("e:") else 1.0 if key.startswith("n:") else 0.55)
    return postings, weights


def retrieve_source(
    path: Path,
    postings: dict[str, list[int]],
    weights: dict[str, float],
    query_count: int,
    top_k: int,
    qgrams: bool,
) -> tuple[list[list[str]], int]:
    heaps: list[list[tuple[float, str]]] = [[] for _ in range(query_count)]
    comparisons = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle, delimiter="\t"), 1):
            scores: dict[int, float] = defaultdict(float)
            for key in signatures(row["business_name"], row["business_address"], qgrams):
                for query_number in postings.get(key, ()):
                    scores[query_number] += weights[key]
            comparisons += len(scores)
            candidate_id = row["entity_id"]
            for query_number, score in scores.items():
                heap = heaps[query_number]
                item = (score, candidate_id)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if row_number % 1_000_000 == 0:
                print(f"[{path.name}] scanned {row_number:,} records", flush=True)
    return [[candidate for _, candidate in sorted(heap, reverse=True)] for heap in heaps], comparisons


def read_truth(path: Path, query_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id = row["source1_entity_id"]
            if query_id in query_ids:
                truth[query_id] = set(filter(None, row["matched_entity_ids"].split(",")))
    return truth


def read_baseline(path: Path, query_ids: set[str]) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id = row["source1_entity_id"]
            if query_id in query_ids:
                selected[query_id] = set(filter(None, row["candidate_entity_ids"].split(",")))
    return selected


def report(name: str, truth: dict[str, set[str]], candidate_sets: dict[str, set[str]]) -> None:
    positives = sum(map(len, truth.values()))
    hits = sum(len(expected & candidate_sets.get(query_id, set())) for query_id, expected in truth.items())
    complete = sum(bool(expected) and expected <= candidate_sets.get(query_id, set()) for query_id, expected in truth.items())
    non_singletons = sum(bool(expected) for expected in truth.values())
    count = sum(len(items) for items in candidate_sets.values())
    oracle_total = 0.0
    for query_id, expected in truth.items():
        if not expected:
            oracle_total += 1.0
            continue
        recall = len(expected & candidate_sets.get(query_id, set())) / len(expected)
        if recall:
            oracle_total += 1.25 * recall / (0.25 + recall)
    print(f"{name}: pair_recall={hits / positives:.6f} ({hits:,}/{positives:,}), "
          f"all_match_retention={complete / non_singletons:.6f}, "
          f"oracle_macro_f0.5={oracle_total / len(truth):.6f}, candidates={count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100, help="Candidates per target source")
    parser.add_argument("--max-query-frequency", type=int, default=20)
    parser.add_argument("--qgrams", action="store_true", help="Add name character 4-gram blocks")
    parser.add_argument("--baseline-s2", type=Path)
    parser.add_argument("--baseline-s3", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "praj_files" / "paper_blocking_probe.tsv")
    args = parser.parse_args()
    if args.sample_size < 1 or args.top_k < 1 or args.max_query_frequency < 1:
        parser.error("sample-size, top-k, and max-query-frequency must be positive")
    if bool(args.baseline_s2) != bool(args.baseline_s3):
        parser.error("provide both baseline paths together")

    queries = sample_queries(TRAIN / "train_source1.tsv", args.sample_size, args.seed)
    query_ids = {row["entity_id"] for row in queries}
    postings, weights = build_query_index(queries, args.max_query_frequency, args.qgrams)
    print(f"Indexed {len(queries):,} queries into {len(postings):,} blocks", flush=True)
    source_results = []
    for source in (2, 3):
        results, comparisons = retrieve_source(
            TRAIN / f"train_source{source}.tsv", postings, weights,
            len(queries), args.top_k, args.qgrams,
        )
        print(f"Source {source}: {comparisons:,} distinct block comparisons", flush=True)
        source_results.append(results)
    candidate_sets = {
        row["entity_id"]: set(source_results[0][index] + source_results[1][index])
        for index, row in enumerate(queries)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        for row in queries:
            query_id = row["entity_id"]
            writer.writerow((query_id, ",".join(sorted(candidate_sets[query_id]))))
    truth = read_truth(TRAIN / "train_ground_truth.tsv", query_ids)
    report("paper_blocking", truth, candidate_sets)
    if args.baseline_s2:
        baseline_s2 = read_baseline(args.baseline_s2, query_ids)
        baseline_s3 = read_baseline(args.baseline_s3, query_ids)
        baseline = {query_id: baseline_s2.get(query_id, set()) | baseline_s3.get(query_id, set()) for query_id in query_ids}
        report("baseline", truth, baseline)
        union = {query_id: baseline[query_id] | candidate_sets[query_id] for query_id in query_ids}
        report("union", truth, union)
    print(f"Saved sampled candidates to {args.output}")


if __name__ == "__main__":
    main()
