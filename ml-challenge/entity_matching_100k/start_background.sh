#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="pytorch-qwen"

if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
    echo "The ${CONTAINER} container is not running. Start it with ./run_qwen.sh first." >&2
    exit 1
fi

if [[ -f "${ROOT}/job_status.json" ]] && grep -Eq '"state": "(building|validating)"' "${ROOT}/job_status.json"; then
    echo "A dataset job is already marked as running: ${ROOT}/job_status.json" >&2
    exit 1
fi

docker exec --detach --user "$(id -u):$(id -g)" "${CONTAINER}" \
    python3 -u /workspace/ml-challenge/entity_matching_100k/run_job.py

echo "Started dataset build in ${CONTAINER}."
echo "Status: ${ROOT}/job_status.json"
echo "Log:    ${ROOT}/job.log"
