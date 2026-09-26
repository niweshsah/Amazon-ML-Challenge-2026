# Repository Guidelines

## Project Structure & Module Organization

This repository explores business entity resolution across three data sources.

- `business_entity_resolution/src/` contains configuration, text preprocessing, candidate blocking, and metrics utilities. `run_blocking.py` generates candidates; `run_pipeline.py` is currently empty.
- `business_entity_resolution/challenge-dataset/` holds datasets and generated outputs.
- `basic_python_scripts/` contains Qwen LoRA/QLoRA training, inference, and dataset preparation experiments.
- `praj_files/` contains additional preprocessing and blocking experiments; `dataset-explore/` contains visualization code.
- `student_resource/` provides challenge rules, submission documentation, and the validator. `checkpoints/` and `qwen-lora-checkpoint/` contain model artifacts.

## Build, Test, and Development Commands

Run commands from this directory unless stated otherwise:

- `bash build_qwen.sh`: build `pytorch-qwen:latest`; requires Docker and an existing `pytorch-ml:latest` base image.
- `bash run_qwen.sh`: open the GPU development container; requires NVIDIA GPU support in Docker.
- `python3 business_entity_resolution/run_blocking.py`: generate `candidate_pairs.tsv` using settings in `src/config.py`. The default mode is `train`; dependencies include NumPy, pandas, and scikit-learn.
- `python3 basic_python_scripts/test_lora.py`: run inference examples using the saved LoRA adapter; requires model dependencies and suitable GPU resources.

The container mounts `challenge-dataset/` read-only. Configure a writable output directory before running blocking there. `requirements.txt` supplements the base environment and includes GPU-specific packages.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase configuration constants. Follow surrounding code, add type hints to new interfaces, and keep preprocessing and blocking logic in their respective modules. No formatter or linter configuration is currently present.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Validate algorithm changes on a held-out training split and report precision, recall, and F0.5. Check generated submissions from `student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Adjust output paths as needed. This checks formatting, not model quality; add `--check-ids` for memory-intensive ID validation. Preserve tab separators, one row per Source 1 entity, and support unseen countries.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `Update code and scripts`; no formal commit convention is established. Prefer more specific subjects describing the affected component. PRs should explain the change, list commands run, link relevant issues, and report evaluation results and runtime or memory impact for algorithm changes. Document dataset splits and configuration needed to reproduce results.
