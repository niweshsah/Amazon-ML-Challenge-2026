"""Command-line entry point for the multilingual entity resolution pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .features import PairFeatureExtractor
from .metrics import candidate_recall
from .reranker import EntityReranker
from .retrieval import HybridGPURetriever


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "entity_matching_100k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid GPU multilingual entity resolution for Parquet inputs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser("retrieve", help="Build the reusable Top-150 pool")
    _common_parser(retrieve)
    retrieve.add_argument("--model", default="BAAI/bge-m3")
    retrieve.add_argument("--device", default="cuda:0")
    retrieve.add_argument("--embedding-batch-size", type=int, default=128)
    retrieve.add_argument("--dense-query-batch-size", type=int, default=2_048)
    retrieve.add_argument("--sparse-query-batch-size", type=int, default=8)
    retrieve.add_argument("--max-seq-length", type=int, default=512)
    retrieve.add_argument("--max-tfidf-features", type=int)
    retrieve.add_argument("--min-candidate-recall", type=float, default=0.99)
    retrieve.add_argument("--transliterator", choices=["sanscript", "indicxlit"], default="sanscript")
    retrieve.add_argument("--indicxlit-model-dir", type=Path)

    features = subparsers.add_parser(
        "features", help="Slice the cached pool and calculate pair features"
    )
    _common_parser(features)
    features.add_argument("--max-k", type=int, default=50)
    features.add_argument("--read-batch-size", type=int, default=500_000)

    evaluate_retrieval = subparsers.add_parser(
        "evaluate-retrieval",
        help="Measure recall from an existing retrieval_master_pool.parquet cache",
    )
    _common_parser(evaluate_retrieval)

    rerank = subparsers.add_parser(
        "train-evaluate", help="Train XGBoost, infer, and evaluate"
    )
    _common_parser(rerank)
    rerank.add_argument("--threshold", type=float, default=0.50)
    rerank.add_argument("--validation-fraction", type=float, default=0.20)
    rerank.add_argument("--n-estimators", type=int, default=300)
    rerank.add_argument("--max-depth", type=int, default=8)
    rerank.add_argument("--learning-rate", type=float, default=0.08)
    rerank.add_argument("--xgb-device", default="cuda")

    all_stages = subparsers.add_parser("all", help="Run retrieval, features, and reranking")
    _common_parser(all_stages)
    all_stages.add_argument("--model", default="BAAI/bge-m3")
    all_stages.add_argument("--device", default="cuda:0")
    all_stages.add_argument("--embedding-batch-size", type=int, default=128)
    all_stages.add_argument("--dense-query-batch-size", type=int, default=2_048)
    all_stages.add_argument("--sparse-query-batch-size", type=int, default=8)
    all_stages.add_argument("--max-seq-length", type=int, default=512)
    all_stages.add_argument("--max-tfidf-features", type=int)
    all_stages.add_argument("--min-candidate-recall", type=float, default=0.99)
    all_stages.add_argument("--max-k", type=int, default=50)
    all_stages.add_argument("--read-batch-size", type=int, default=500_000)
    all_stages.add_argument("--threshold", type=float, default=0.50)
    all_stages.add_argument("--validation-fraction", type=float, default=0.20)
    all_stages.add_argument("--n-estimators", type=int, default=300)
    all_stages.add_argument("--max-depth", type=int, default=8)
    all_stages.add_argument("--learning-rate", type=float, default=0.08)
    all_stages.add_argument("--xgb-device", default="cuda")
    all_stages.add_argument("--transliterator", choices=["sanscript", "indicxlit"], default="sanscript")
    all_stages.add_argument("--indicxlit-model-dir", type=Path)
    return parser


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "t1": args.data_dir / "t1.parquet",
        "t2": args.data_dir / "t2.parquet",
        "ground_truth": args.data_dir / "ground_truth.parquet",
        "master": args.output_dir / "retrieval_master_pool.parquet",
        "normalized_t1": args.output_dir / "normalized_t1.parquet",
        "normalized_t2": args.output_dir / "normalized_t2.parquet",
        "features": args.output_dir / "candidate_features.parquet",
    }


def _run_retrieval(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    retriever = HybridGPURetriever(
        model_name=args.model,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        dense_query_batch_size=args.dense_query_batch_size,
        sparse_query_batch_size=args.sparse_query_batch_size,
        max_seq_length=args.max_seq_length,
        max_tfidf_features=args.max_tfidf_features,
        transliterator=args.transliterator,
        indicxlit_model_dir=args.indicxlit_model_dir,
    )
    return retriever.retrieve(
        paths["t1"],
        paths["t2"],
        paths["ground_truth"],
        args.output_dir,
        min_candidate_recall=args.min_candidate_recall,
    )


def _run_features(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    extractor = PairFeatureExtractor(read_batch_size=args.read_batch_size)
    return extractor.extract(
        paths["master"],
        paths["normalized_t1"],
        paths["normalized_t2"],
        paths["ground_truth"],
        paths["features"],
        max_k=args.max_k,
    )


def _run_reranker(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    reranker = EntityReranker(
        threshold=args.threshold,
        validation_fraction=args.validation_fraction,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        device=args.xgb_device,
    )
    return reranker.train_evaluate(
        paths["features"], paths["ground_truth"], args.output_dir
    )


def _evaluate_cached_retrieval(paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    result = candidate_recall(
        str(paths["master"]),
        pd.read_parquet(paths["ground_truth"]),
    )
    result["master_pool_path"] = str(paths["master"])
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(args)
    metrics: dict[str, Any] = {}
    if args.command in {"retrieve", "all"}:
        metrics["retrieval"] = _run_retrieval(args, paths)
    if args.command in {"features", "all"}:
        metrics["features"] = _run_features(args, paths)
    if args.command in {"train-evaluate", "all"}:
        metrics["reranker"] = _run_reranker(args, paths)
    if args.command == "evaluate-retrieval":
        metrics["retrieval"] = _evaluate_cached_retrieval(paths, args.output_dir)

    metrics_path = args.output_dir / f"{args.command}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
