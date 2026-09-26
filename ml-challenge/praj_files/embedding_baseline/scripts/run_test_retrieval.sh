#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/ml-challenge
if [[ -n "${BASELINE_THRESHOLD:-}" ]]; then
  python praj_files/embedding_baseline/run.py full-test --threshold "${BASELINE_THRESHOLD}" --resume
else
  python praj_files/embedding_baseline/run.py full-test --resume
fi
