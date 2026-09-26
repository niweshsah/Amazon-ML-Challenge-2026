import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def is_validation_query(source1_id, validation_fraction, seed):
    digest = hashlib.blake2b(
        f"{seed}:{source1_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 < validation_fraction


def evaluate_source(
    source_name,
    source1_ids,
    target_ids,
    truth,
    chunk_dir,
    thresholds,
    top_ks,
    validation_fraction,
    seed,
):
    chunk_paths = sorted(
        Path(chunk_dir).glob("chunk_*.npz"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No score checkpoints found in {chunk_dir}")

    metrics = {
        (threshold, top_k): {"queries": 0, "true_pairs": 0, "hits": 0, "any": 0, "all": 0, "predicted": 0}
        for threshold in thresholds
        for top_k in top_ks
    }

    for chunk_path in chunk_paths:
        chunk_idx = int(chunk_path.stem.split("_")[1])
        start = chunk_idx * 25000
        end = min(start + 25000, len(source1_ids))
        checkpoint = np.load(chunk_path)
        indices = checkpoint["indices"]
        scores = checkpoint["scores"]

        for source1_id, row_indices, row_scores in zip(
            source1_ids[start:end], indices, scores
        ):
            if is_validation_query(source1_id, validation_fraction, seed):
                split_match = True
            else:
                split_match = False
            expected = {
                value
                for value in truth.get(source1_id, "").split(",")
                if value.startswith(f"{source_name}-")
            }
            if not split_match or not expected:
                continue

            for threshold in thresholds:
                for top_k in top_ks:
                    key = (threshold, top_k)
                    selected = row_scores[:top_k] >= threshold
                    predicted = set(target_ids[row_indices[:top_k][selected]])
                    hits = expected & predicted
                    result = metrics[key]
                    result["queries"] += 1
                    result["true_pairs"] += len(expected)
                    result["hits"] += len(hits)
                    result["any"] += bool(hits)
                    result["all"] += len(hits) == len(expected)
                    result["predicted"] += len(predicted)

    for (threshold, top_k), result in metrics.items():
        queries = result["queries"]
        true_pairs = result["true_pairs"]
        predicted = result["predicted"]
        result.update(
            threshold=threshold,
            top_k=top_k,
            pair_recall=result["hits"] / true_pairs if true_pairs else 0.0,
            pair_precision=result["hits"] / predicted if predicted else 1.0,
            any_match_recall=result["any"] / queries if queries else 0.0,
            all_match_recall=result["all"] / queries if queries else 0.0,
        )
    return source_name, list(metrics.values())


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS threshold and top-k recall.")
    parser.add_argument("--root", default="/workspace/ml-challenge/praj_files")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 5, 10, 20])
    args = parser.parse_args()

    root = Path(args.root)
    source1_ids = pd.read_parquet(
        root / "datasets/preprocessed/train_source1.parquet", columns=["entity_id"]
    )["entity_id"].to_numpy()
    truth_frame = pd.read_parquet(root / "datasets/preprocessed/train_ground_truth.parquet")
    truth = dict(zip(truth_frame["source1_entity_id"], truth_frame["matched_entity_ids"].fillna("")))

    for source_name in ("S2", "S3"):
        target_ids = pd.read_parquet(
            root / f"datasets/preprocessed/train_source{source_name[-1]}.parquet",
            columns=["entity_id"],
        )["entity_id"].to_numpy()
        preferred_chunk_dir = root / f"output/faiss_{source_name.lower()}_chunks_top{max(args.top_ks)}"
        legacy_chunk_dir = root / f"output/faiss_{source_name.lower()}_chunks"
        chunk_dir = preferred_chunk_dir if preferred_chunk_dir.exists() else legacy_chunk_dir
        try:
            _, results = evaluate_source(
                source_name,
                source1_ids,
                target_ids,
                truth,
                chunk_dir,
                args.thresholds,
                args.top_ks,
                args.validation_fraction,
                args.seed,
            )
        except FileNotFoundError:
            print(
                f"\n{source_name}: skipped; no .npz score checkpoints found in {chunk_dir}. "
                "Run 5_faiss_gpu_search.py first with the updated score-checkpoint code."
            )
            continue
        print(f"\n{source_name} validation results")
        print("threshold top_k true_positive predicted false_positive precision recall any_match_recall all_match_recall queries")
        for result in results:
            print(
                f"{result['threshold']:.2f} {result['top_k']:5d} "
                f"{result['hits']} {result['predicted']} "
                f"{result['predicted'] - result['hits']} "
                f"{result['pair_precision']:.4f} {result['pair_recall']:.4f} "
                f"{result['any_match_recall']:.4f} {result['all_match_recall']:.4f} "
                f"{result['queries']}"
            )


if __name__ == "__main__":
    main()
