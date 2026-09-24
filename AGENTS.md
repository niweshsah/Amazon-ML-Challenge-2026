# AGENTS.md — niwesh workspace

Not a git repo / monorepo. Two independent areas + scratch files at root:

- `ml-challenge/` — active work: QLoRA fine-tune of `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4). See below.
- `isaacsim/` — NVIDIA Isaac Sim clone (v6.1.0, own git repo). Has its own `isaacsim/AGENTS.md` + `skills/` routing; follow that file when working inside it. Do not apply ml-challenge Docker flow there.
- `code.ipynb`, `run_issac.sh`, `run_pytorch.sh` — root-level scratch/launchers.

No root build, lint, tests, or CI. No `opencode.json`.

## ml-challenge — container flow (order matters)

Images are layered; build/run in this order. All scripts are idempotent (reuse image, re-attach to running container, remove stopped same-name container):

1. `./run_pytorch.sh` (root) — builds `pytorch-ml:latest` if missing (CUDA 13.0.3, torch 2.12.1+cu130, python 3.10, venv `/opt/venv`), verifies GPU, drops into interactive shell. Requires: docker, NVIDIA Container Toolkit, `nvidia-smi`, host driver >= 580.65.06, host dir `~/workspace/niwesh/ml-challenge`.
2. `ml-challenge/build_qwen.sh` — builds `pytorch-qwen:latest` FROM `pytorch-ml:latest` + `ml-challenge/requirements.txt`. Fails if base image or non-empty `requirements.txt` missing.
3. `ml-challenge/run_qwen.sh` — daily driver for Qwen work. Starts/attaches `pytorch-qwen` container; mounts `ml-challenge/ -> /workspace/ml-challenge` and `~/.cache/huggingface` (reused for model weights).

`ml-challenge/build_perf.sh` / `Dockerfile.perf` is separate: `pytorch-llm-perf:latest` FROM `vllm/vllm-openai:latest` (+ `liger-kernel`). Inference/perf only, not training.

## ml-challenge — training

Run inside the `pytorch-qwen` container (`WORKDIR /workspace/ml-challenge`):

- Smoke test (no data needed): `python3 qlora_train.py --dummy`
- Real run: `python3 qlora_train.py --data data/train.jsonl` — JSONL, one object per line: `{"prompt": "...", "response": "..."}` (both required; blank lines skipped).
- Key defaults: output `./checkpoints/qwen2.5-vl-qlora`, NF4 double-quant bf16 frozen base, LoRA r16/alpha32/dropout0.05 on `q/k/v/o/gate/up/down_proj`, bf16, `paged_adamw_8bit`, batch 1, grad-accum 8, save/eval every 50 steps. Resume: `--resume <checkpoint-dir>`.
- `quantised_model_load.py` — minimal 4-bit load check (not training).

## Gotchas

- Do NOT use host conda env for torch work: `code.ipynb` shows the local `ml-challenge` conda env is broken (`libcusparseLt.so.0` ImportError on `import torch`). Use the Docker containers above.
- `code.ipynb` targets 24 GB VRAM; its `force_unload()` / `reload_model()` helpers are the established pattern for purging/reloading the 7B 4-bit model in-notebook.
- `run_issac.sh` (note typo: "issac") launches `nvcr.io/nvidia/isaac-sim:6.1.0` as container `isaac-sim` (`--gpus all --network=host -u 1234:1234`, volumes under `~/docker/isaac-sim/`). `docker exec` if already running.
- Container usernames differ: `developer` (pytorch-ml / pytorch-qwen) vs root + `/workspace/ml-challenge` workdir in perf image. HF cache mount path is user-dependent — check the script before assuming.
