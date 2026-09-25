"""Evaluate blocking candidates and entity-resolution predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import Config


def parse_entity_ids(value: object) -> set[str]:
    """Parse a comma-separated entity-id field into a set."""
    if value is None or pd.isna(value):
        return set()
    return {entity_id.strip() for entity_id in str(value).split(",") if entity_id.strip()}


def load_entity_mapping(path: Path, id_column: str, values_column: str) -> dict[str, set[str]]:
    """Load an entity-to-match-list TSV mapping."""
    frame = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    required_columns = {id_column, values_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    return {
        str(row[id_column]): parse_entity_ids(row[values_column])
        for _, row in frame.iterrows()
    }


def blocking_recall(
    ground_truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
) -> tuple[float, int, int]:
    """Return recall, captured true matches, and total true matches."""
    total_true_matches = sum(len(match_ids) for match_ids in ground_truth.values())
    captured_matches = sum(
        len(true_ids.intersection(candidates.get(source1_id, set())))
        for source1_id, true_ids in ground_truth.items()
    )
    recall = captured_matches / total_true_matches if total_true_matches else 1.0
    return recall, captured_matches, total_true_matches


def f05_score(precision: float, recall: float) -> float:
    """Calculate F0.5, weighting precision twice as strongly as recall."""
    beta_squared = 0.5**2
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 1.0 if precision == recall == 0 else 0.0
    return (1 + beta_squared) * precision * recall / denominator


def macro_f05(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> tuple[float, pd.DataFrame]:
    """Return macro F0.5 and one precision/recall row per Source 1 entity."""
    rows = []
    entity_ids = set(ground_truth).union(predictions)
    for source1_id in sorted(entity_ids):
        true_ids = ground_truth.get(source1_id, set())
        predicted_ids = predictions.get(source1_id, set())
        true_positive = len(true_ids.intersection(predicted_ids))
        false_positive = len(predicted_ids - true_ids)
        false_negative = len(true_ids - predicted_ids)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        rows.append(
            {
                "source1_entity_id": source1_id,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f0.5": f05_score(precision, recall),
            }
        )

    per_entity = pd.DataFrame(rows)
    score = float(per_entity["f0.5"].mean()) if not per_entity.empty else 0.0
    return score, per_entity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--matching-results", type=Path)
    args = parser.parse_args()

    cfg = Config()
    ground_truth_path = args.ground_truth or cfg.DATA_DIR / "train_ground_truth.tsv"
    candidates_path = args.candidates or cfg.OUTPUT_DIR / "candidate_pairs.tsv"
    matching_path = args.matching_results or cfg.OUTPUT_DIR / "matching_results.tsv"

    ground_truth = load_entity_mapping(
        ground_truth_path, "source1_entity_id", "matched_entity_ids"
    )
    candidates = (
        load_entity_mapping(candidates_path, "source1_entity_id", "candidate_entity_ids")
        if candidates_path.exists() and candidates_path.stat().st_size > 0
        else {}
    )
    recall, captured, total = blocking_recall(ground_truth, candidates)

    print("Evaluation results")
    print(f"Blocking recall: {recall:.4%} ({captured:,}/{total:,} true matches)")

    if not matching_path.exists() or matching_path.stat().st_size == 0:
        print(f"Macro F0.5: unavailable (no predictions found at {matching_path})")
        return

    predictions = load_entity_mapping(
        matching_path, "source1_entity_id", "matched_entity_ids"
    )
    score, per_entity = macro_f05(ground_truth, predictions)
    print(f"Macro F0.5: {score:.4f} ({len(per_entity):,} Source 1 entities)")


if __name__ == "__main__":
    main()