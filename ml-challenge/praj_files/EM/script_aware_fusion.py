"""Compare fixed-budget BGE/FAISS candidate fusion on the first million rows."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from benchmark_first_million import Score, ordered_union, read_first_ids, read_truth
from multilingual_candidate_probe import name_script
from paper_blocking_probe import ROOT, TRAIN


def load_non_latin_names() -> dict[str, str]:
    result: dict[str, str] = {}
    for source in (2, 3):
        path = TRAIN / f"train_source{source}.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                script = name_script(row["business_name"])
                if script != "Latin":
                    result[row["entity_id"]] = script
        print(f"[script] loaded {path.name}; non-Latin target count={len(result):,}", flush=True)
    return result


def fuse(
    bge: list[str], faiss: list[str], non_latin: dict[str, str], reserve: int, limit: int = 100,
) -> set[str]:
    selected: set[str] = set(bge[:limit - reserve])
    for target_id in faiss:
        if target_id in non_latin:
            selected.add(target_id)
            if len(selected) == limit:
                return selected
    for target_id in bge[limit - reserve:]:
        selected.add(target_id)
        if len(selected) == limit:
            return selected
    for target_id in faiss:
        selected.add(target_id)
        if len(selected) == limit:
            return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "praj_files/EM/reports/first_million_fusion.tsv")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    started = time.monotonic()
    non_latin = load_non_latin_names()
    ids = read_first_ids(TRAIN / "train_source1.tsv", args.limit)
    truth = read_truth(TRAIN / "train_ground_truth.tsv", set(ids))
    quotas = (10, 20, 30, 40, 50)
    methods = ["bge_100", "round_robin_100"] + [f"script_reserve_{quota}" for quota in quotas]
    scores = {method: Score() for method in methods}
    script_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    paths = [ROOT / f"praj_files/output/candidate_pairs_{kind}_s{source}.tsv"
             for kind in ("bge_m3", "faiss") for source in (2, 3)]
    with ExitStack() as stack:
        readers = [csv.reader(stack.enter_context(path.open(encoding="utf-8", newline="")), delimiter="\t") for path in paths]
        for path, reader in zip(paths, readers):
            if next(reader, None) != ["source1_entity_id", "candidate_entity_ids"]:
                raise ValueError(f"Unexpected candidate header: {path}")
        for number, query_id in enumerate(ids, 1):
            rows = [next(reader, None) for reader in readers]
            if any(row is None or row[0] != query_id for row in rows):
                raise ValueError(f"Candidate row mismatch at {number:,}: {query_id}")
            b2, b3, f2, f3 = [row[1].split(",") if row[1] else [] for row in rows]
            expected = truth.pop(query_id)
            selected: dict[str, set[str]] = {
                "bge_100": set(b2 + b3),
                "round_robin_100": ordered_union(b2, f2, 100) | ordered_union(b3, f3, 100),
            }
            for quota in quotas:
                selected[f"script_reserve_{quota}"] = (
                    fuse(b2, f2, non_latin, quota) | fuse(b3, f3, non_latin, quota)
                )
            for method, candidates in selected.items():
                scores[method].add(expected, candidates)
                result = script_counts[method]
                for target_id in expected:
                    script = non_latin.get(target_id, "Latin")
                    result[script]["pairs"] += 1
                    result[script]["hits"] += target_id in candidates
            if number % 100_000 == 0:
                print(f"[fusion] {number:,}/{len(ids):,} queries; {(time.monotonic() - started) / 60:.1f} min", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["method", "pair_recall", "all_match_retention", "oracle_macro_f05", "mean_candidates_per_query",
                         "latin_recall", "devanagari_recall", "tamil_recall", "other_non_latin_recall"])
        for method in methods:
            score = scores[method]
            slices = script_counts[method]
            writer.writerow([
                method, f"{score.hits / score.pairs:.6f}",
                f"{score.full_queries / score.positive_queries:.6f}",
                f"{score.oracle_total / len(ids):.6f}",
                f"{score.candidates / len(ids):.3f}",
                *[f"{slices[script]['hits'] / slices[script]['pairs']:.6f}"
                  for script in ("Latin", "Devanagari", "Tamil", "Other non-Latin")],
            ])
    print(args.output.read_text(encoding="utf-8"), flush=True)
    print(f"Finished in {(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
