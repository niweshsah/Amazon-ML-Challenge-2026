#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/ml-challenge
python praj_files/embedding_baseline/run.py full-train-eval --sample-fraction "${SAMPLE_FRACTION:-0.01}" --resume

