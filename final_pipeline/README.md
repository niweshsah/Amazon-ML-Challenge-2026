# Final entity-resolution pipeline

Raw records → Unicode preprocessing → trained Indic BGE-M3 CLS retrieval and character-trigram TF-IDF → candidate union → pair features → XGBoost reject/accept/route → supervised BGE reranker → independent pair matches.

Run commands from this directory. The checkout contains no challenge data or neural weights. Historical logs, adapter README files, the legacy TF-IDF vectorizer, PDFs, and reports are preserved under `artifacts/legacy/` and `docs/`; `docs/migration_inventory.json` records original paths and SHA-256 checksums. Historical evidence retains its original paths and is not runnable tooling.

## Install and local engineering verification

Python 3.11 or 3.12 is required. Use `uv sync --frozen --extra test` for the locked CPU environment, or `python -m pip install -e '.[test]'`. On macOS, actual XGBoost needs `brew install libomp`. The fixture uses deterministic substitutes for both neural models and the tree scorer, and real TF-IDF, FAISS, calibration, routing, metrics, and output code.

```bash
uv run --extra test pytest tests -q
uv run python scripts/make_fixture.py .fixture/data
uv run entity-resolution all --fixture --train-percent 100 \
  --data-dir .fixture/data --output-dir .fixture/train \
  --set top_k_dense=8 --set top_k_tfidf=8 --set threshold_grid_size=5
uv run entity-resolution all --fixture --mode test \
  --data-dir .fixture/data --output-dir .fixture/test \
  --model-dir .fixture/train/models \
  --set top_k_dense=8 --set top_k_tfidf=8 --set threshold_grid_size=5
```

These results validate engineering behavior, **not trained-model quality**. Training on challenge data and full inference require the server's assets and GPU. Official challenge data and documentation template were not present; the methodology document includes the required topics from `docs/PS.pdf`.

## GPU smoke, training, and prediction

Use `uv sync --frozen --extra gpu --extra test` or install `requirements-gpu.txt` followed by `pip install --no-deps -e .`. CUDA 12.4 is supplied by the Docker image. GPU models use BF16; ensure the GPU supports it. Default model revisions are pinned and remote downloads are disabled. Prefetch the two configured Hugging Face model snapshots, or explicitly use `--set local_files_only=false` during the initial training/smoke run. This fetches pretrained models, not external business data.

```bash
uv run --extra gpu python scripts/gpu_smoke.py --download
ER_DATA_DIR=/data/dataset/train ER_MODEL_DIR=/models \
  bash scripts/train_server.sh
ER_DATA_DIR=/data/dataset/test ER_MODEL_DIR=/models \
  bash scripts/predict_server.sh
```

The training default is **1.0 percent**, not a fraction of 1.0. Change it with `--train-percent 5` or `ER_TRAIN_PERCENT=5`; valid values are `(0, 100]`. `--target-distractor-percent` independently controls the percentage of remaining records sampled from each target source; its default follows `train-percent`. Every selected positive is retained. Test mode ignores sampling and processes every Source 1 entity against complete Source 2/3 pools.

Training `all` prepares data, trains missing or stale managed models, embeds, retrieves, builds features, trains XGBoost and the cross-encoder, calibrates, predicts, evaluates audit entities, and checks outputs. To load an existing compatible adapter, pass `--biencoder-checkpoint /models/adapter` and/or `--crossencoder-checkpoint /models/reranker`. Checkpoints must contain tokenizer assets, PEFT adapter files, and the pipeline `metadata.json` that records base/revision, serialization, pooling, and maximum length. Legacy adapters without these declarations fail explicitly; inspect their provenance before adding compatibility metadata. An external missing checkpoint is an error, not permission to substitute an untrained model.

```bash
entity-resolution prepare --data-dir /data/train --output-dir outputs/train
entity-resolution train-biencoder --data-dir /data/train --output-dir outputs/train
entity-resolution embed --data-dir /data/train --output-dir outputs/train
entity-resolution retrieve --data-dir /data/train --output-dir outputs/train
entity-resolution features --data-dir /data/train --output-dir outputs/train
entity-resolution train-xgboost --data-dir /data/train --output-dir outputs/train
entity-resolution train-crossencoder --data-dir /data/train --output-dir outputs/train
entity-resolution calibrate --data-dir /data/train --output-dir outputs/train
entity-resolution predict --data-dir /data/train --output-dir outputs/train
entity-resolution evaluate --data-dir /data/train --output-dir outputs/train
```

Use the same configuration/overrides and model directory across stages. All settings in `configs/default.yaml` accept `--set KEY=YAML_VALUE`; paths, mode, sampling, and checkpoints have dedicated flags. `--config path.yaml` merges a YAML file over defaults; relative paths resolve from the working directory. `--policy-dir` allows an independently stored calibration policy. Probability calibration requires both classes and at least two calibration groups; increase sampling if this is not satisfied.

## Outputs, validation, and package

State is disk-backed SQLite. Embeddings are atomic NPY shards with row-ID maps and checksums. Small target pools use exact FAISS inner-product search; larger pools use IVF-PQ. TF-IDF products are bounded by query and target batch settings. Retrieval preserves separate scores/ranks and channel provenance in `pairs.sqlite`; `candidate_pairs.tsv` is its exact union, with no label injection. `matching_results.tsv` contains accepted members only, including empty and multiple-match lists.

```bash
entity-resolution validate --mode test --data-dir /data/test \
  --output-dir outputs/test --model-dir /models
python scripts/validate_submission.py \
  --matching outputs/test/matching_results.tsv \
  --candidate outputs/test/candidate_pairs.tsv --test-dir /data/test
entity-resolution package --mode test --data-dir /data/test \
  --output-dir outputs/test --model-dir /models --archive team_submission.zip
```

Packaging rejects fixture results and includes outputs, runnable source, locked dependencies, methodology, adapters, calibration, and cached base-model snapshots. The zip follows `PS.pdf`: `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`. It can be large because it contains model weights. After unpacking, place test TSVs in `code/business_entity_resolution/data/`, install the environment there, and run `entity-resolution all --config configs/submission.yaml`. The official supplied validator was absent; the included validator implements the PDF rules with complete ID and candidate-containment checks.

`evaluation.json` reports sampled-pool audit macro F0.5 including singletons, pair precision/recall, candidate pair recall, routing volume, subgroup and source results, stage runtime and peak process RSS. Full-pool quality, GPU throughput, disk capacity, and routing cost must be measured on the server. Sparse search scans target shards for each query batch; it bounds memory but needs throughput benchmarking at challenge scale. IVF-PQ is approximate and may reduce retrieval recall. RAM includes one compressed FAISS index, its ID map, a bounded sparse shard, and TF-IDF vocabulary; tune retrieval/index limits to server memory. Atomic commits and content fingerprints reject stale or damaged caches; resume reuses intact embedding and lexical shards.

## Docker

```bash
ER_DATA_DIR=/data/dataset/train ER_MODEL_DIR=/srv/models \
  ER_OUTPUT_DIR=/srv/run docker compose run --rm pipeline \
  entity-resolution all --data-dir /data --output-dir /outputs --model-dir /models
```

Mounts use the consolidated directory, read-only input data, writable outputs/models, and a persistent Hugging Face cache. Training and prediction scripts accept additional CLI flags.
