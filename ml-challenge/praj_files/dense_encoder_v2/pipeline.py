"""Command line entrypoint with timestamped terminal and file logs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from core import DATA, MODEL, MODEL_REVISION


class Logger:
    def __init__(self, root: Path, command: str):
        self.root = root
        self.command = command
        self.path = root / "logs" / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.run_id = os.environ.get("DENSE_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def __call__(self, event: str, **fields) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "run_id": self.run_id, "command": self.command, "event": event,
                  "elapsed_seconds": round(time.monotonic() - self.started, 3), **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(line + "\n")
        print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["analyze", "prepare", "verify-data", "train", "resume", "dev-eval", "build-index", "infer", "score", "validate"])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--eval-count", type=int, default=100000)
    parser.add_argument("--dev-count", type=int, default=2000)
    parser.add_argument("--pairs", type=int, default=200000)
    parser.add_argument("--distractors", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-minutes", type=int, default=80)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--index-type", choices=["ivf", "flat"], default="ivf")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logger = Logger(root, args.command)
    adapter = None if args.baseline else root / "best_adapter"
    logger("start", argv=sys.argv, model=MODEL, model_revision=MODEL_REVISION, python=sys.version,
           pid=os.getpid(), data_dir=str(args.data.resolve()), output_dir=str(root))
    try:
        if args.command in ("analyze", "prepare"):
            from prepare import prepare
            result = prepare(root, args.data, args.eval_count, args.dev_count,
                             args.pairs, args.distractors, logger)
            logger("analysis_complete", **result)
        elif args.command == "verify-data":
            from prepare import verify_prepared
            result = verify_prepared(root)
        elif args.command in ("train", "resume"):
            from train import train
            result = train(root, logger, resume=args.command == "resume", batch_size=args.batch_size,
                           max_length=args.max_length, max_minutes=args.max_minutes,
                           max_pairs=args.pairs, checkpoint_steps=args.checkpoint_steps)
        elif args.command == "dev-eval":
            from encoder import load_encoder
            from train import dev_score
            tokenizer, model = load_encoder(adapter)
            result = {"dev_macro_f0.5": dev_score(root, model, tokenizer, args.max_length,
                                                   args.batch_size, logger)}
        elif args.command == "build-index":
            from evaluate import build_index
            result = build_index(root, logger, adapter, args.max_length, args.batch_size, args.index_type)
        elif args.command == "infer":
            from evaluate import search
            result = search(root, logger, adapter, args.max_length, args.batch_size,
                            args.top_k, args.nprobe)
        elif args.command == "score":
            from evaluate import score
            result = score(root, logger, args.top_k, min(20000, args.eval_count // 5))
        else:
            validator = Path(__file__).resolve().parents[2] / "student_resource/utils/validate_submission.py"
            command = [sys.executable, str(validator), "--matching", str(root / "matching_results.tsv"),
                       "--candidate", str(root / "candidate_pairs.tsv"),
                       "--test-dir", str(root / "validation_dataset"), "--check-ids"]
            process = subprocess.run(command, text=True, capture_output=True)
            logger("validator_output", returncode=process.returncode,
                   stdout=process.stdout, stderr=process.stderr)
            if process.returncode:
                raise RuntimeError("Submission validator failed")
            result = {"passed": True}
        logger("complete", result=result)
    except BaseException as error:
        logger("failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
