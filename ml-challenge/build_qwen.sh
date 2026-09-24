#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="pytorch-ml:latest"
QWEN_IMAGE="pytorch-qwen:latest"

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.qwen"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

echo "[build_qwen] Build context: ${SCRIPT_DIR}"
echo "[build_qwen] Base image:     ${BASE_IMAGE}"
echo "[build_qwen] Qwen image:     ${QWEN_IMAGE}"

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    echo "[build_qwen] ERROR: Docker is not installed or not in PATH."
    exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[build_qwen] ERROR: Missing ${DOCKERFILE}"
    exit 1
fi

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "[build_qwen] ERROR: Missing ${REQUIREMENTS}"
    exit 1
fi

if [[ ! -s "${REQUIREMENTS}" ]]; then
    echo "[build_qwen] ERROR: ${REQUIREMENTS} is empty."
    echo "[build_qwen] Add the required Python packages first."
    exit 1
fi

# ------------------------------------------------------------
# Check base image
# ------------------------------------------------------------

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo
    echo "[build_qwen] ERROR: Base image '${BASE_IMAGE}' does not exist."
    echo
    echo "Build your PyTorch base environment first."
    echo
    exit 1
fi

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

echo
echo "[build_qwen] Building Qwen image..."
echo

docker build \
    --file "${DOCKERFILE}" \
    --tag "${QWEN_IMAGE}" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --build-arg USER_NAME="developer" \
    "${SCRIPT_DIR}"
    
echo
echo "[build_qwen] Qwen image built successfully."
echo
echo "[build_qwen] Image:"
docker image inspect "${QWEN_IMAGE}" \
    --format '  {{.RepoTags}}'
echo