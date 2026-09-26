#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.benchmark import benchmark
from src.build_index import build_index
from src.embed import embed_source
from src.evaluate import evaluate
from src.inspect_data import inspect_all
from src.io_utils import count_rows, dataset_path, load_config
from src.outputs import write_outputs
from src.retrieve import retrieve
from src.smoke_test import smoke_ground_truth


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None)
    root = argparse.ArgumentParser(description="Resumable GPU embedding baseline")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", parents=[common])
    smoke = sub.add_parser("smoke-test", parents=[common]); smoke.add_argument("--queries", type=int, default=32); smoke.add_argument("--distractors", type=int, default=256)
    bench = sub.add_parser("benchmark", parents=[common]); bench.add_argument("--sample-records", type=int, default=None)
    emb = sub.add_parser("embed", parents=[common]); emb.add_argument("--split", choices=["train", "test"], required=True); emb.add_argument("--source", choices=["source1", "source2", "source3"], required=True); emb.add_argument("--resume", action="store_true"); emb.add_argument("--max-rows", type=int); emb.add_argument("--shard-rows", type=int)
    idx = sub.add_parser("build-index", parents=[common]); idx.add_argument("--split", choices=["train", "test"], required=True); idx.add_argument("--resume", action="store_true")
    ret = sub.add_parser("retrieve", parents=[common]); ret.add_argument("--split", choices=["train", "test"], required=True); ret.add_argument("--top-k", type=int); ret.add_argument("--sample-fraction", type=float); ret.add_argument("--max-source1", type=int); ret.add_argument("--resume", action="store_true"); ret.add_argument("--tiny-debug", action="store_true")
    ev = sub.add_parser("evaluate", parents=[common]); ev.add_argument("--split", default="train", choices=["train"]); ev.add_argument("--margin", type=float)
    out = sub.add_parser("write-outputs", parents=[common]); out.add_argument("--split", choices=["train", "test"], required=True); out.add_argument("--threshold", type=float, required=True); out.add_argument("--top-k", type=int, default=50); out.add_argument("--margin", type=float)
    full = sub.add_parser("full-train-eval", parents=[common]); full.add_argument("--sample-fraction", type=float, default=.01); full.add_argument("--max-source1", type=int); full.add_argument("--resume", action="store_true")
    test = sub.add_parser("full-test", parents=[common]); test.add_argument("--resume", action="store_true"); test.add_argument("--threshold", type=float)
    return root


def main() -> None:
    args = parser().parse_args()
    cfg = args.config
    if args.command == "inspect": result = inspect_all(cfg)
    elif args.command == "smoke-test": result = smoke_ground_truth(cfg, args.queries, args.distractors)
    elif args.command == "benchmark": result = benchmark(cfg, args.sample_records)
    elif args.command == "embed": result = embed_source(cfg, args.split, args.source, args.resume, args.max_rows, args.shard_rows)
    elif args.command == "build-index": result = build_index(cfg, args.split, args.resume)
    elif args.command == "retrieve": result = retrieve(cfg, args.split, args.top_k, args.sample_fraction, args.max_source1, args.resume, args.tiny_debug)
    elif args.command == "evaluate": result = evaluate(cfg, args.split, args.margin)
    elif args.command == "write-outputs": result = write_outputs(cfg, args.split, args.threshold, args.top_k, args.margin)
    elif args.command == "full-train-eval":
        config = load_config(cfg)
        total = count_rows(dataset_path(config, "train", "source1"))
        query_rows = args.max_source1 or max(1, math.ceil(total * args.sample_fraction))
        embed_source(cfg, "train", "source2", args.resume)
        embed_source(cfg, "train", "source3", args.resume)
        embed_source(cfg, "train", "source1", args.resume, query_rows)
        build_index(cfg, "train", args.resume)
        retrieve(cfg, "train", max_source1=query_rows, resume=args.resume)
        result = evaluate(cfg, "train")
        if result.get("best_threshold"):
            result["outputs"] = write_outputs(cfg, "train", result["best_threshold"]["threshold"])
    elif args.command == "full-test":
        config = load_config(cfg)
        selected_path = Path(config["paths"]["output_dir"]) / "reports" / "selected_config.json"
        if not selected_path.exists():
            raise RuntimeError("full-test requires reports/selected_config.json from a non-debug train evaluation")
        selected = json.loads(selected_path.read_text())
        for section in ("model", "text"):
            if selected[section] != config[section]:
                raise RuntimeError(f"Current {section} configuration differs from the frozen train selection")
        threshold = selected["threshold"] if args.threshold is None else args.threshold
        if abs(threshold - selected["threshold"]) > 1e-12:
            raise RuntimeError("Explicit test threshold differs from the frozen train-selected threshold")
        for source in ("source2", "source3", "source1"):
            embed_source(cfg, "test", source, args.resume)
        build_index(cfg, "test", args.resume)
        retrieve(cfg, "test", resume=args.resume)
        result = write_outputs(cfg, "test", threshold, selected["top_k"], selected.get("margin"))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
