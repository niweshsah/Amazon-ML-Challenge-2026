"""Measure and augment BGE-M3 recall for cross-script training matches.

On a sampled Source 1 query set, this script scans all target records, uses ICU
romanization for records containing non-Latin text, and retrieves candidates
through normalized name token/character blocks plus address numbers. It reports
candidate recall by target name script without using labels during retrieval.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from paper_blocking_probe import ROOT, TRAIN, read_baseline, read_truth, report, sample_queries


WORD = re.compile(r"[^a-z0-9]+")
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "pvt", "private"}


def name_script(value: str) -> str:
    if any("\u0900" <= char <= "\u097f" for char in value):
        return "Devanagari"
    if any("\u0b80" <= char <= "\u0bff" for char in value):
        return "Tamil"
    if any(char.isalpha() and ord(char) > 127 and not unicodedata.name(char, "").startswith("LATIN") for char in value):
        return "Other non-Latin"
    return "Latin"


def has_non_latin(value: str) -> bool:
    return name_script(value) != "Latin"


def keys(name: str, address: str) -> set[str]:
    name_tokens = [token for token in WORD.split(name.lower()) if len(token) >= 3 and token not in LEGAL]
    address_tokens = [token for token in WORD.split(address.lower()) if token]
    result = {"n:" + token for token in name_tokens}
    if name_tokens:
        result.add("e:" + " ".join(name_tokens))
    for token in name_tokens:
        if len(token) >= 5:
            result.update("g:" + token[index:index + 4] for index in range(len(token) - 3))
    result.update("d:" + token for token in address_tokens if token.isdigit() and len(token) >= 2)
    return result


def query_index(queries: list[dict[str, str]], max_frequency: int) -> tuple[dict[str, list[int]], dict[str, float]]:
    all_keys = [keys(row["business_name"], row["business_address"]) for row in queries]
    frequency = Counter(key for item in all_keys for key in item)
    index: dict[str, list[int]] = defaultdict(list)
    weight: dict[str, float] = {}
    for query_number, item in enumerate(all_keys):
        for key in item:
            if frequency[key] > max_frequency and not key.startswith("e:"):
                continue
            index[key].append(query_number)
            multiplier = 5.0 if key.startswith("e:") else 2.0 if key.startswith("n:") else 1.0
            weight[key] = multiplier * math.log1p(len(queries) / frequency[key])
    return index, weight


def scan_target(
    path: Path,
    truth_target_ids: set[str],
    index: dict[str, list[int]],
    weight: dict[str, float],
    query_count: int,
    top_k: int,
) -> tuple[list[list[str]], dict[str, str], int, int]:
    heaps: list[list[tuple[float, str]]] = [[] for _ in range(query_count)]
    truth_scripts: dict[str, str] = {}
    comparisons = 0
    non_latin_records = 0
    with path.open("rb") as icu_input, path.open(encoding="utf-8", newline="") as raw_input:
        process = subprocess.Popen(
            ["uconv", "-x", "Any-Latin; Latin-ASCII"],
            stdin=icu_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        assert process.stdout is not None
        raw_rows = csv.DictReader(raw_input, delimiter="\t")
        romanized_rows = csv.DictReader(process.stdout, delimiter="\t")
        row_number = 0
        for row_number, (raw, romanized) in enumerate(zip(raw_rows, romanized_rows, strict=True), 1):
            candidate_id = raw["entity_id"]
            if romanized["entity_id"] != candidate_id:
                raise ValueError(f"ICU output lost row alignment at {path}:{row_number}")
            raw_name = raw["business_name"]
            raw_address = raw["business_address"]
            script = name_script(raw_name)
            if candidate_id in truth_target_ids:
                truth_scripts[candidate_id] = script
            if script == "Latin" and not has_non_latin(raw_address):
                continue
            non_latin_records += 1
            scores: dict[int, float] = defaultdict(float)
            for key in keys(romanized["business_name"], romanized["business_address"]):
                for query_number in index.get(key, ()):
                    scores[query_number] += weight[key]
            comparisons += len(scores)
            for query_number, score in scores.items():
                heap = heaps[query_number]
                pair = (score, candidate_id)
                if len(heap) < top_k:
                    heapq.heappush(heap, pair)
                elif pair > heap[0]:
                    heapq.heapreplace(heap, pair)
            if row_number % 1_000_000 == 0:
                print(f"[{path.name}] scanned {row_number:,} rows", flush=True)
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.wait() != 0:
            raise RuntimeError(f"ICU transliteration failed for {path}: {stderr}")
    print(f"[{path.name}] rows={row_number:,}, non_Latin={non_latin_records:,}, comparisons={comparisons:,}")
    return (
        [[candidate for _, candidate in sorted(heap, reverse=True)] for heap in heaps],
        truth_scripts,
        non_latin_records,
        comparisons,
    )


def print_script_metrics(
    truth: dict[str, set[str]],
    target_scripts: dict[str, str],
    baseline: dict[str, set[str]],
    added: dict[str, set[str]],
) -> None:
    print("script\ttrue_pairs\tbaseline_hits\tbaseline_recall\tunion_hits\tunion_recall\tnew_hits")
    for script in ("Latin", "Devanagari", "Tamil", "Other non-Latin"):
        pairs = [(query_id, target_id) for query_id, expected in truth.items() for target_id in expected
                 if target_scripts.get(target_id) == script]
        if not pairs:
            continue
        baseline_hits = sum(target_id in baseline[query_id] for query_id, target_id in pairs)
        union_hits = sum(target_id in baseline[query_id] or target_id in added[query_id]
                         for query_id, target_id in pairs)
        print(f"{script}\t{len(pairs)}\t{baseline_hits}\t{baseline_hits / len(pairs):.6f}\t"
              f"{union_hits}\t{union_hits / len(pairs):.6f}\t{union_hits - baseline_hits}")
    missing = sum(target_id not in target_scripts for expected in truth.values() for target_id in expected)
    if missing:
        raise ValueError(f"Failed to locate {missing} ground-truth target IDs in the source files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100, help="Added candidates per target source")
    parser.add_argument("--max-query-frequency", type=int, default=30)
    parser.add_argument("--baseline-s2", type=Path, default=ROOT / "praj_files/output/candidate_pairs_bge_m3_s2.tsv")
    parser.add_argument("--baseline-s3", type=Path, default=ROOT / "praj_files/output/candidate_pairs_bge_m3_s3.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "praj_files/EM/reports/multilingual_probe_5000.tsv")
    args = parser.parse_args()
    if min(args.sample_size, args.top_k, args.max_query_frequency) < 1:
        parser.error("sample-size, top-k, and max-query-frequency must be positive")

    queries = sample_queries(TRAIN / "train_source1.tsv", args.sample_size, args.seed)
    query_ids = {row["entity_id"] for row in queries}
    truth = read_truth(TRAIN / "train_ground_truth.tsv", query_ids)
    truth_target_ids = set().union(*truth.values())
    index, weight = query_index(queries, args.max_query_frequency)
    print(f"Indexed {len(queries):,} queries into {len(index):,} blocks", flush=True)
    source_results = []
    target_scripts: dict[str, str] = {}
    for source in (2, 3):
        results, scripts, _, _ = scan_target(
            TRAIN / f"train_source{source}.tsv", truth_target_ids,
            index, weight, len(queries), args.top_k,
        )
        source_results.append(results)
        target_scripts.update(scripts)
    added = {
        row["entity_id"]: set(source_results[0][number] + source_results[1][number])
        for number, row in enumerate(queries)
    }
    baseline_s2 = read_baseline(args.baseline_s2, query_ids)
    baseline_s3 = read_baseline(args.baseline_s3, query_ids)
    baseline = {query_id: baseline_s2.get(query_id, set()) | baseline_s3.get(query_id, set()) for query_id in query_ids}
    union = {query_id: baseline[query_id] | added[query_id] for query_id in query_ids}
    report("baseline", truth, baseline)
    report("transliteration", truth, added)
    report("union", truth, union)
    print_script_metrics(truth, target_scripts, baseline, added)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        for row in queries:
            query_id = row["entity_id"]
            writer.writerow((query_id, ",".join(sorted(added[query_id]))))
    print(f"Saved transliteration candidates to {args.output}")


if __name__ == "__main__":
    main()
