#!/usr/bin/env bash

set -Eeuo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

HOST_ML_DIR="${SCRIPT_DIR}"
CONTAINER_ML_DIR="/workspace/ml-challenge"
HOST_DATASET_DIR="${SCRIPT_DIR}/../challenge-dataset"
CONTAINER_DATASET_DIR="${CONTAINER_ML_DIR}/challenge-dataset"

# --- CACHE DIRECTORY CONFIGURATION ---
# Change these paths if you want to store caches in a specific local folder 
# (e.g., HOST_HF_CACHE="${SCRIPT_DIR}/.cache/huggingface")
HOST_HF_CACHE="${HOME}/.cache/huggingface"
CONTAINER_HF_CACHE="/home/developer/.cache/huggingface"

# Added Triton cache so Liger Kernel doesn't recompile every time
HOST_TRITON_CACHE="${HOME}/.triton/cache"
CONTAINER_TRITON_CACHE="/home/developer/.triton/cache"

IMAGE_NAME="pytorch-qwen:latest"
CONTAINER_NAME="pytorch-qwen"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

error() {
    echo "[run_qwen] ERROR: $*" >&2
    exit 1
}

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    error "Docker is not installed or not in PATH."
fi

if ! docker info >/dev/null 2>&1; then
    error "Docker daemon is not running or current user cannot access Docker."
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[run_qwen] Image '${IMAGE_NAME}' does not exist."
    echo
    echo "[run_qwen] Build it first with:"
    echo
    echo "    ${SCRIPT_DIR}/build_qwen.sh"
    echo
    exit 1
fi

if [[ ! -d "${HOST_DATASET_DIR}" ]]; then
    error "Dataset directory not found: ${HOST_DATASET_DIR}"
fi

# ------------------------------------------------------------
# Host directories
# ------------------------------------------------------------

# Ensure the cache directories exist on the host before mounting
mkdir -p "${HOST_HF_CACHE}"
mkdir -p "${HOST_TRITON_CACHE}"

# ------------------------------------------------------------
# Already running?
# ------------------------------------------------------------

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then

    echo "[run_qwen] Container '${CONTAINER_NAME}' is already running."
    echo "[run_qwen] Attaching..."

    exec docker exec \
        --interactive \
        --tty \
        "${CONTAINER_NAME}" \
        bash
fi

# ------------------------------------------------------------
# Existing stopped container?
# ------------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then

    echo "[run_qwen] Found stopped container '${CONTAINER_NAME}'."
    echo "[run_qwen] Removing it before creating a new one..."

    docker rm "${CONTAINER_NAME}" >/dev/null
fi

# ------------------------------------------------------------
# Start new container
# ------------------------------------------------------------

echo
echo "[run_qwen] Starting Qwen container..."
echo
echo "[run_qwen] Image:"
echo "    ${IMAGE_NAME}"
echo
echo "[run_qwen] Project:"
echo "    ${HOST_ML_DIR}"
echo "    -> ${CONTAINER_ML_DIR}"
echo
echo "[run_qwen] Hugging Face cache:"
echo "    ${HOST_HF_CACHE}"
echo "    -> ${CONTAINER_HF_CACHE}"
echo
echo "[run_qwen] Triton / Liger cache:"
echo "    ${HOST_TRITON_CACHE}"
echo "    -> ${CONTAINER_TRITON_CACHE}"
echo

exec docker run \
    --interactive \
    --tty \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env HF_HOME="${CONTAINER_HF_CACHE}" \
    --env TRITON_CACHE_DIR="${CONTAINER_TRITON_CACHE}" \
    --mount "type=bind,source=${HOST_ML_DIR},target=${CONTAINER_ML_DIR}" \
    --mount "type=bind,source=${HOST_DATASET_DIR},target=${CONTAINER_DATASET_DIR},readonly" \
    --mount "type=bind,source=${HOST_HF_CACHE},target=${CONTAINER_HF_CACHE}" \
    --mount "type=bind,source=${HOST_TRITON_CACHE},target=${CONTAINER_TRITON_CACHE}" \
    --workdir "${CONTAINER_ML_DIR}" \
    "${IMAGE_NAME}" \
    bash