"""Command-line entry point for the sampled GPU tree baseline."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import Config


def configure_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "pipeline.log", encoding="utf-8")],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "run", "evaluate", "smoke"])
    parser.add_argument("--dataset", type=Path, default=Config.dataset)
    parser.add_argument("--output", type=Path, default=Config.output)
    parser.add_argument("--max-queries", type=int, default=Config.max_queries)
    parser.add_argument("--max-candidates-per-source", type=int, default=Config.max_candidates_per_source)
    parser.add_argument("--feature-batch-size", type=int, default=Config.feature_batch_size)
    parser.add_argument("--rounds", type=int, default=Config.rounds)
    args = parser.parse_args()
    if args.max_queries < 1 or args.max_candidates_per_source < 1 or args.feature_batch_size < 1 or args.rounds < 1:
        parser.error("Query, candidate, batch, and round counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = Config(dataset=args.dataset, output=args.output, max_queries=args.max_queries,
                    max_candidates_per_source=args.max_candidates_per_source,
                    feature_batch_size=args.feature_batch_size, rounds=args.rounds)
    if args.command == "smoke":
        config = replace(config, output=args.output / "smoke", max_queries=min(250, args.max_queries),
                         feature_batch_size=50, rounds=min(8, args.rounds), early_stopping_rounds=4,
                         retrieval_progress_every=50)
    configure_logging(config.output)
    logging.info("Started %s at %s", args.command, datetime.now(timezone.utc).isoformat())
    logging.info("Configuration: %s", config.serializable())
    if args.command in {"prepare", "run", "smoke"}:
        from prepare import prepare
        prepare(config)
    if args.command in {"train", "run", "smoke"}:
        from train import train
        train(config)
    if args.command == "evaluate":
        import json
        path = config.output / "run" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError("Train a model before evaluating")
        from diagnostics import create_diagnostics
        from prepare import cache_dir
        from train import load_arrays
        import numpy as np
        import xgboost as xgb
        metrics = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((cache_dir(config) / "prepare_manifest.json").read_text(encoding="utf-8"))
        arrays, query_rows = load_arrays(config, manifest["feature_names"])
        booster = xgb.Booster()
        booster.load_model(config.output / "run" / "model.json")
        predictions = {name: np.load(config.output / "run" / f"{name}_probabilities.npy") for name in ("audit", "early", "threshold")}
        run_dir = config.output / "run"
        importance = json.loads((run_dir / "feature_importance.json").read_text(encoding="utf-8"))["importance"]
        history = json.loads((run_dir / "learning_curve.json").read_text(encoding="utf-8"))
        curve = json.loads((run_dir / "threshold_curve.json").read_text(encoding="utf-8"))["rows"]
        create_diagnostics(config, booster, arrays, query_rows, predictions, metrics, importance, history, curve)
        print(json.dumps(metrics["results"], indent=2))
    logging.info("Finished %s", args.command)


if __name__ == "__main__":
    main()
