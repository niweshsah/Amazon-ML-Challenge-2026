"""Measure how often FAISS top-k candidates retain training ground-truth matches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP_KS = [5, 10, 20, 30, 50, 75, 100]


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """Return Source 1 ID -> all ground-truth candidate IDs."""
    frame = pd.read_parquet(path, columns=["source1_entity_id", "matched_entity_ids"])
    truth: dict[str, set[str]] = {}
    for query_id, matched_ids in zip(
        frame["source1_entity_id"], frame["matched_entity_ids"].fillna("")
    ):
        truth[str(query_id)] = {
            value.strip() for value in str(matched_ids).split(",") if value.strip()
        }
    return truth


def candidate_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as candidate_file:
        reader = csv.DictReader(candidate_file, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            yield row["source1_entity_id"], [
                value.strip()
                for value in (row.get("candidate_entity_ids", "") or "").split(",")
                if value.strip()
            ]


def empty_metrics() -> dict[str, Any]:
    return {
        "queries": 0,
        "truth_pairs": 0,
        "hits": 0,
        "any_hit_queries": 0,
        "all_hit_queries": 0,
        "candidate_pairs": 0,
    }


def add_result(
    metrics: dict[str, dict[int, dict[str, Any]]],
    top_ks: list[int],
    source_key: str,
    expected: set[str],
    ranked_candidates: list[str],
    second_source_candidates: list[str] | None = None,
) -> None:
    """Accumulate metrics; queries without truth for this scope are excluded."""
    if not expected:
        return
    for top_k in top_ks:
        selected = set(ranked_candidates[:top_k])
        if second_source_candidates is not None:
            selected.update(second_source_candidates[:top_k])
        hits = expected & selected
        result = metrics[source_key][top_k]
        result["queries"] += 1
        result["truth_pairs"] += len(expected)
        result["hits"] += len(hits)
        result["any_hit_queries"] += bool(hits)
        result["all_hit_queries"] += len(hits) == len(expected)
        result["candidate_pairs"] += len(selected)


def evaluate(
    data_dir: Path,
    truth: dict[str, set[str]],
    top_ks: list[int],
    s2_path: Path,
    s3_path: Path,
) -> list[dict[str, Any]]:
    metrics = {
        scope: {top_k: empty_metrics() for top_k in top_ks}
        for scope in ("S2", "S3", "combined")
    }
    s2_rows = iter(candidate_rows(s2_path))
    s3_rows = iter(candidate_rows(s3_path))
    source1_path = data_dir / "train_source1.parquet"
    source1_ids = pd.read_parquet(source1_path, columns=["entity_id"])["entity_id"].astype(str)

    for row_number, query_id in enumerate(source1_ids, start=1):
        try:
            s2_query_id, s2_candidates = next(s2_rows)
            s3_query_id, s3_candidates = next(s3_rows)
        except StopIteration as exc:
            raise ValueError(
                f"Candidate files ended before Source 1 row {row_number:,} ({query_id}). "
                "Regenerate complete per-source candidate files."
            ) from exc
        if s2_query_id != query_id or s3_query_id != query_id:
            raise ValueError(
                f"Candidate row order/IDs do not match train_source1.parquet at row "
                f"{row_number:,}: expected {query_id}, got S2={s2_query_id}, S3={s3_query_id}."
            )

        expected_all = truth.get(query_id, set())
        expected_s2 = {entity_id for entity_id in expected_all if entity_id.startswith("S2-")}
        expected_s3 = {entity_id for entity_id in expected_all if entity_id.startswith("S3-")}
        add_result(metrics, top_ks, "S2", expected_s2, s2_candidates)
        add_result(metrics, top_ks, "S3", expected_s3, s3_candidates)
        add_result(
            metrics,
            top_ks,
            "combined",
            expected_all,
            s2_candidates,
            s3_candidates,
        )
        if row_number % 500_000 == 0:
            print(f"[read] evaluated {row_number:,} Source 1 queries", flush=True)

    # Catch extra rows too; otherwise misaligned candidate output can go unnoticed.
    if next(s2_rows, None) is not None or next(s3_rows, None) is not None:
        raise ValueError("Candidate TSV contains more rows than train_source1.parquet")

    rows: list[dict[str, Any]] = []
    for scope in ("S2", "S3", "combined"):
        for top_k in top_ks:
            result = metrics[scope][top_k]
            queries = result["queries"]
            truth_pairs = result["truth_pairs"]
            candidate_pairs = result["candidate_pairs"]
            rows.append(
                {
                    "scope": scope,
                    "top_k_per_source": top_k,
                    "queries_with_ground_truth": queries,
                    "ground_truth_pairs": truth_pairs,
                    "retrieved_true_pairs": result["hits"],
                    "pair_recall": result["hits"] / truth_pairs if truth_pairs else 0.0,
                    "queries_with_any_match": result["any_hit_queries"],
                    "any_match_retention": result["any_hit_queries"] / queries if queries else 0.0,
                    "queries_with_all_matches": result["all_hit_queries"],
                    "full_query_retention": result["all_hit_queries"] / queries if queries else 0.0,
                    "candidate_pairs": candidate_pairs,
                    "candidate_precision": result["hits"] / candidate_pairs if candidate_pairs else 0.0,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure pair recall and ground-truth retention in FAISS top-k candidates."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="praj_files directory")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Preprocessed dataset directory; defaults to datasets/preprocessed_libpostal.",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=DEFAULT_TOP_KS,
        help="Candidate cutoffs to evaluate (maximum 100), e.g. --top-ks 10 20 50 100",
    )
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--s2-candidates", type=Path, default=None)
    parser.add_argument("--s3-candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Metrics TSV output path")
    args = parser.parse_args()

    if not args.top_ks or any(k < 1 or k > 100 for k in args.top_ks):
        parser.error("every --top-ks value must be between 1 and 100")
    args.top_ks = sorted(set(args.top_ks))
    data_dir = args.data_dir or args.root / "datasets" / "preprocessed_libpostal"
    ground_truth_path = args.ground_truth or (
        data_dir / "train_ground_truth.parquet"
    )
    s2_path = args.s2_candidates or args.root / "output" / "candidate_pairs_faiss_s2.tsv"
    s3_path = args.s3_candidates or args.root / "output" / "candidate_pairs_faiss_s3.tsv"
    # Candidate outputs may be mounted read-only or owned by a different
    # container UID. Keep the small metrics report in the writable project dir.
    output_path = args.output or args.root / "faiss_retention_metrics.tsv"
    for path in (ground_truth_path, s2_path, s3_path):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")

    truth = load_ground_truth(ground_truth_path)
    rows = evaluate(data_dir, truth, args.top_ks, s2_path, s3_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False, float_format="%.6f")

    print("scope  k  pair_recall  any_match_retention  full_query_retention  candidate_precision  queries")
    for row in rows:
        print(
            f"{row['scope']:<8} {row['top_k_per_source']:>3} "
            f"{row['pair_recall']:.4f}       {row['any_match_retention']:.4f}              "
            f"{row['full_query_retention']:.4f}               {row['candidate_precision']:.4f}              "
            f"{row['queries_with_ground_truth']:,}"
        )
    print(f"\nSaved detailed metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
