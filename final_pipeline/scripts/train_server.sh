#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ER_DATA_DIR:?Set ER_DATA_DIR to the directory containing train_source*.tsv}"
entity-resolution all --mode train --data-dir "$ER_DATA_DIR" \
  --output-dir "${ER_OUTPUT_DIR:-outputs/train}" --model-dir "${ER_MODEL_DIR:-models}" \
  --train-percent "${ER_TRAIN_PERCENT:-1.0}" "$@"
