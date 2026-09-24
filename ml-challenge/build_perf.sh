#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="pytorch-llm-perf:latest"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.perf"

echo "[build_perf] Dockerfile: ${DOCKERFILE}"
echo "[build_perf] Image:      ${IMAGE_NAME}"

if ! command -v docker >/dev/null 2>&1; then
    echo "[build_perf] ERROR: Docker is not installed."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[build_perf] ERROR: Docker daemon is unavailable."
    exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[build_perf] ERROR: ${DOCKERFILE} not found."
    exit 1
fi

echo
echo "[build_perf] Building performance image..."
echo

docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo
echo "[build_perf] Build complete."
echo
docker image inspect "${IMAGE_NAME}" \
    --format 'Image: {{.RepoTags}}'