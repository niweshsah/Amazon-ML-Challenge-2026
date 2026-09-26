#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/artifacts"
mkdir -p "${ROOT}/logs"
export DENSE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${ROOT}/logs/terminal_${DENSE_RUN_ID}.log"
trap 'printf "%s FAILED stage=%s exit=%s\n" "$(date -u +%FT%TZ)" "${CURRENT_STAGE:-unknown}" "$?" | tee -a "${LOG}"' ERR

run_stage() {
    CURRENT_STAGE="$1"
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG}"
    python3 -u "${SCRIPT_DIR}/pipeline.py" "$@" --output "${ROOT}" 2>&1 | tee -a "${LOG}"
    printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG}"
}

run_stage prepare
run_stage verify-data
run_stage train
run_stage build-index
run_stage infer
run_stage score
run_stage validate
