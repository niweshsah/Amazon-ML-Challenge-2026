#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE_NAME="pytorch-ml:latest"
readonly CONTAINER_NAME="pytorch-ml-dev"
readonly HOST_ML_DIR="${HOME}/workspace/niwesh/ml-challenge"
readonly CONTAINER_ML_DIR="/workspace/ml-challenge"

readonly PYTHON_VERSION="3.10"
readonly PYTORCH_VERSION="2.12.1"
readonly PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"
readonly CUDA_BASE_IMAGE="nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04"
readonly CUDA_TEST_IMAGE="nvidia/cuda:13.0.3-base-ubuntu22.04"
readonly MIN_DRIVER_VERSION="580.65.06"

log() { printf '[run_pytorch] %s\n' "$*"; }
fail() { printf '[run_pytorch][ERROR] %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [[ -n "${BUILD_DIR:-}" && -d "${BUILD_DIR}" ]]; then
        rm -rf "${BUILD_DIR}"
    fi
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 \
    || fail "Docker is not installed or not in PATH."

docker info >/dev/null 2>&1 \
    || fail "Docker is not running, or the current user cannot access the Docker daemon."

if ! command -v nvidia-ctk >/dev/null 2>&1 && \
   ! command -v nvidia-container-cli >/dev/null 2>&1; then
    fail "NVIDIA Container Toolkit is not available."
fi

command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi is not installed or not in PATH."

HOST_DRIVER_VERSION="$(
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
    head -n1 | tr -d '[:space:]'
)" || fail "Unable to query the NVIDIA driver."

[[ -n "${HOST_DRIVER_VERSION}" ]] \
    || fail "nvidia-smi did not report a driver version."

if command -v dpkg >/dev/null 2>&1 &&
   dpkg --compare-versions "${HOST_DRIVER_VERSION}" lt "${MIN_DRIVER_VERSION}"; then
    fail "NVIDIA driver ${HOST_DRIVER_VERSION} is too old. Required: ${MIN_DRIVER_VERSION}+."
fi

[[ -d "${HOST_ML_DIR}" ]] \
    || fail "Required host directory does not exist: ${HOST_ML_DIR}"

log "Host NVIDIA driver: ${HOST_DRIVER_VERSION}"
log "Host GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
    sed 's/^/  - /' || true

log "Checking NVIDIA GPU passthrough through Docker..."
if ! docker run --rm --gpus all "${CUDA_TEST_IMAGE}" nvidia-smi -L; then
    fail "NVIDIA GPU is not accessible from Docker with --gpus all."
fi

if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    log "Docker image ${IMAGE_NAME} already exists; reusing it without rebuilding."
else
    log "Docker image ${IMAGE_NAME} not found; building it now..."

    BUILD_DIR="$(mktemp -d -t run-pytorch-build.XXXXXX)"

    cat > "${BUILD_DIR}/Dockerfile" <<DOCKERFILE
FROM ${CUDA_BASE_IMAGE}

ARG PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTORCH_VERSION=${PYTORCH_VERSION}
ARG PYTORCH_CUDA_INDEX=${PYTORCH_CUDA_INDEX}

ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_NAME=developer

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:\$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        build-essential \
        git \
        curl \
        ca-certificates \
        pkg-config \
    && python3.10 -m venv "\${VIRTUAL_ENV}" \
    && "\${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        --index-url "\${PYTORCH_CUDA_INDEX}" \
        --extra-index-url https://pypi.org/simple \
        "torch==\${PYTORCH_VERSION}+cu130" \
    && pip install \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        matplotlib \
        tqdm \
        pyyaml

# Create a real user matching the host UID/GID. This fixes:
#   "groups: cannot find name for group ID ..."
#   "I have no name!"
# and keeps files created in the bind mount owned by the host user.
RUN groupadd --gid "\${USER_GID}" "\${USER_NAME}" \
    && useradd --uid "\${USER_UID}" \
        --gid "\${USER_GID}" \
        --create-home \
        --shell /bin/bash \
        "\${USER_NAME}" \
    && mkdir -p /workspace/ml-challenge \
    && chown -R "\${USER_UID}:\${USER_GID}" /workspace

USER \${USER_NAME}

WORKDIR /workspace/ml-challenge

CMD ["bash"]
DOCKERFILE

    if ! docker build \
        --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
        --build-arg "PYTORCH_VERSION=${PYTORCH_VERSION}" \
        --build-arg "PYTORCH_CUDA_INDEX=${PYTORCH_CUDA_INDEX}" \
        --build-arg "USER_UID=$(id -u)" \
        --build-arg "USER_GID=$(id -g)" \
        --tag "${IMAGE_NAME}" \
        "${BUILD_DIR}"; then
        fail "Docker image build failed."
    fi

    log "Docker image ${IMAGE_NAME} built successfully."
fi

log "Verifying PyTorch CUDA access inside ${IMAGE_NAME}..."
if ! docker run --rm --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${IMAGE_NAME}" \
    python3 -c '
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
raise SystemExit(0 if torch.cuda.is_available() else 1)
'; then
    fail "PyTorch inside the Docker image cannot access the NVIDIA GPU."
fi

log "Starting interactive PyTorch container."
log "Mounted: ${HOST_ML_DIR} -> ${CONTAINER_ML_DIR}"
log "Working directory: ${CONTAINER_ML_DIR}"
log "Type 'exit' to leave the container."

HOST_HF_CACHE="${HOME}/.cache/huggingface"

mkdir -p "${HOST_HF_CACHE}"

# If the container is already running, attach to it.
if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "[run_pytorch] Container '${CONTAINER_NAME}' is already running."
    echo "[run_pytorch] Attaching to existing container..."

    exec docker exec \
        --interactive \
        --tty \
        "${CONTAINER_NAME}" \
        bash
fi

# If a stopped container with the same name exists, remove it.
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "[run_pytorch] Removing stopped container '${CONTAINER_NAME}'..."
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "[run_pytorch] Starting new PyTorch container."
echo "[run_pytorch] Mounted project: ${HOST_ML_DIR} -> ${CONTAINER_ML_DIR}"
echo "[run_pytorch] Mounted Hugging Face cache: ${HOST_HF_CACHE} -> /home/developer/.cache/huggingface"

exec docker run \
    --interactive \
    --tty \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --mount "type=bind,source=${HOST_ML_DIR},target=${CONTAINER_ML_DIR}" \
    --mount "type=bind,source=${HOST_HF_CACHE},target=/home/developer/.cache/huggingface" \
    --workdir "${CONTAINER_ML_DIR}" \
    "${IMAGE_NAME}" \
    bash