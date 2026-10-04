#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ER_DATA_DIR:?Set ER_DATA_DIR to the directory containing test_source*.tsv}"
entity-resolution all --mode test --data-dir "$ER_DATA_DIR" \
  --output-dir "${ER_OUTPUT_DIR:-outputs/test}" --model-dir "${ER_MODEL_DIR:-models}" "$@"
