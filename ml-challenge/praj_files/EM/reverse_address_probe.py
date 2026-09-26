"""Reverse address anchored retrieval for non-Latin target names.

The first N Source 1 records form an address-block index. Each non-Latin
Source 2/3 target probes the index and retains its best Source 1 candidates.
This avoids losing all but the first hundred target records for one query.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.dataset as ds

from benchmark_first_million import read_first_ids, read_truth
from multilingual_candidate_probe import name_script
from paper_blocking_probe import ROOT, TRAIN


TOKEN = re.compile(r"[a-z0-9]+")


def parts(value: str) -> tuple[set[str], set[str]]:
    tokens = set(TOKEN.findall(value.lower()))
    numbers = {token for token in tokens if any(char.isdigit() for char in token) and len(token) <= 12}
    words = {token for token in tokens if len(token) >= 4 and token.isalpha()}
    return numbers, words


def address_keys(numbers: set[str], words: set[str], df: Counter[str], max_word_df: int) -> set[int]:
    useful = sorted((word for word in words if df[word] <= max_word_df), key=lambda word: (df[word], word))[:10]
    keys = {hash(("nw", number, word)) for number in sorted(numbers)[:4] for word in useful}
    for index, first in enumerate(useful[:4]):
        for second in useful[index + 1:4]:
            keys.add(hash(("ww", first, second)))
    return keys


def chargrams(value: str) -> set[str]:
    text = " ".join(TOKEN.findall(value.lower()))
    return {text[index:index + 3] for index in range(max(0, len(text) - 2))}


def name_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def build_index(limit: int, max_word_df: int) -> tuple[list[str], list[str], dict[int, list[int]], Counter[str]]:
    ids: list[str] = []
    names: list[str] = []
    addresses: list[tuple[set[str], set[str]]] = []
    df: Counter[str] = Counter()
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ids.append(row["entity_id"])
            names.append(row["business_name"])
            number, words = parts(row["business_address"])
            addresses.append((number, words))
            df.update(words)
            if len(ids) == limit:
                break
    index: dict[int, list[int]] = defaultdict(list)
    for query_number, (numbers, words) in enumerate(addresses):
        for key in address_keys(numbers, words, df, max_word_df):
            index[key].append(query_number)
    return ids, names, index, df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--max-word-df", type=int, default=100_000)
    parser.add_argument("--max-posting", type=int, default=300)
    parser.add_argument("--queries-per-target", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "praj_files/EM/reports/first_million_reverse_address.tsv")
    args = parser.parse_args()
    if min(args.limit, args.max_word_df, args.max_posting, args.queries_per_target) < 1:
        parser.error("all numeric options must be positive")
    start = time.monotonic()
    ids, names, index, df = build_index(args.limit, args.max_word_df)
    print(f"Indexed {len(ids):,} queries into {len(index):,} address blocks in {(time.monotonic() - start) / 60:.1f} min", flush=True)
    truth = read_truth(TRAIN / "train_ground_truth.tsv", set(ids))
    output: dict[int, list[str]] = defaultdict(list)
    comparisons = 0
    total_targets = 0
    @lru_cache(maxsize=200_000)
    def query_grams(query_number: int) -> set[str]:
        return chargrams(names[query_number])

    for source in (2, 3):
        path = ROOT / f"praj_files/datasets/preprocessed_latin/train_source{source}.parquet"
        dataset = ds.dataset(path, format="parquet")
        columns = ["entity_id", "business_name", "business_name_latin", "business_address_latin"]
        for batch in dataset.to_batches(columns=columns, batch_size=20_000):
            values = batch.to_pydict()
            for target_id, raw_name, latin_name, address in zip(*(values[column] for column in columns)):
                if name_script(raw_name or "") == "Latin":
                    continue
                total_targets += 1
                numbers, words = parts(address or "")
                scores: dict[int, int] = defaultdict(int)
                for key in address_keys(numbers, words, df, args.max_word_df):
                    postings = index.get(key, ())
                    if len(postings) > args.max_posting:
                        continue
                    for query_number in postings:
                        scores[query_number] += 1
                comparisons += len(scores)
                if not scores:
                    continue
                likely = heapq.nlargest(100, scores, key=scores.get)
                target_grams = chargrams(latin_name or "")
                best = heapq.nlargest(
                    args.queries_per_target, likely,
                    key=lambda query_number: (
                        name_score(query_grams(query_number), target_grams)
                        + 0.10 * min(scores[query_number], 4),
                        scores[query_number],
                    ),
                )
                for query_number in best:
                    output[query_number].append(target_id)
        print(f"[source {source}] non-Latin targets={total_targets:,}; compared pairs={comparisons:,}; elapsed={(time.monotonic() - start) / 60:.1f} min", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    total = 0
    all_found = 0
    positive_queries = 0
    candidate_count = 0
    oracle = 0.0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for query_number, query_id in enumerate(ids):
            candidates = set(output.get(query_number, ()))
            candidate_count += len(candidates)
            expected = truth[query_id]
            hits = len(expected & candidates)
            total += len(expected)
            found += hits
            if not expected:
                oracle += 1.0
            else:
                positive_queries += 1
                all_found += hits == len(expected)
                recall = hits / len(expected)
                if recall:
                    oracle += 1.25 * recall / (0.25 + recall)
            writer.writerow([query_id, ",".join(sorted(candidates))])
    print(f"pair_recall={found / total:.6f} ({found:,}/{total:,}); all_match_retention={all_found / positive_queries:.6f}; "
          f"oracle_macro_f05={oracle / len(ids):.6f}; candidates={candidate_count:,}; "
          f"non_Latin_targets={total_targets:,}; comparisons={comparisons:,}; elapsed={(time.monotonic() - start) / 60:.1f} min", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
