# GPU embedding baseline

This isolated pipeline embeds business records with `Qwen/Qwen3-Embedding-0.6B`, performs global Source-2 and Source-3 cosine search, preserves raw top-50 candidates, and tunes multi-match thresholding only on training truth. Every artifact it creates stays below this directory.

## Environment and data

From the repository host directory, enter the existing image (do not rebuild it):

```bash
./run_qwen.sh
```

Run all remaining commands inside that shell from `/workspace/ml-challenge`. Input is read from `/workspace/ml-challenge/praj_files/datasets/preprocessed`. Each apparent `.parquet` input is a Parquet dataset directory.

The environment probe found Transformers and GPU FAISS but not vLLM or cuVS. Accordingly, `engine: auto` currently selects Transformers BF16 on the GPU. The index is a checkpointed catalog over FP16 vector shards, searched exactly with GPU inner products; this avoids a second 20+ GB float32 index. If vLLM is later installed, do not use it blindly: verify its pooling API before changing the engine. No quantized configuration is promoted unless it loads, is faster, produces finite unit-normal vectors, and retains sampled retrieval quality.

## Commands

Inspect and benchmark first:

```bash
python praj_files/embedding_baseline/run.py inspect
python praj_files/embedding_baseline/run.py benchmark --sample-records 100000
python praj_files/embedding_baseline/run.py smoke-test --queries 32 --distractors 256
```

A small smoke run can use separate directories via a copied config, or explicit row limits:

```bash
python praj_files/embedding_baseline/run.py embed --split train --source source2 --max-rows 1000 --shard-rows 500 --resume
python praj_files/embedding_baseline/run.py embed --split train --source source3 --max-rows 1000 --shard-rows 500 --resume
python praj_files/embedding_baseline/run.py embed --split train --source source1 --max-rows 100 --shard-rows 100 --resume
python praj_files/embedding_baseline/run.py build-index --split train --resume
python praj_files/embedding_baseline/run.py retrieve --split train --max-source1 100 --top-k 50 --tiny-debug --resume
python praj_files/embedding_baseline/run.py evaluate --split train
```

`--tiny-debug` searches only the first target shard and its metrics are explicitly DEBUG ONLY. A realistic sampled validation embeds all targets and searches sampled Source-1 queries against the complete target pool:

The separate `smoke-test` injects each sampled query's true targets into a small distractor pool. It verifies that real ground-truth matches can be embedded and retrieved, but its report is also labeled DEBUG ONLY and must never be presented as realistic full-pool quality.

```bash
python praj_files/embedding_baseline/run.py full-train-eval --sample-fraction 0.01 --resume
```

The equivalent explicit full-target commands are:

```bash
python praj_files/embedding_baseline/run.py embed --split train --source source2 --resume
python praj_files/embedding_baseline/run.py embed --split train --source source3 --resume
python praj_files/embedding_baseline/run.py embed --split train --source source1 --max-rows 22069 --resume
python praj_files/embedding_baseline/run.py build-index --split train --resume
python praj_files/embedding_baseline/run.py retrieve --split train --max-source1 22069 --top-k 50 --resume
python praj_files/embedding_baseline/run.py evaluate --split train
```

After reading `reports/train_retrieval_metrics.json`, freeze its selected threshold and create validation outputs:

```bash
python praj_files/embedding_baseline/run.py write-outputs --split train --threshold 0.84 --top-k 50
```

Replace `0.84` with the measured best threshold. Only then run test:

```bash
bash praj_files/embedding_baseline/scripts/run_test_retrieval.sh
```

This creates `outputs/test_baseline_predictions.tsv` and `outputs/candidate_pairs.tsv`. A non-debug training evaluation freezes the model, text settings, K, margin, and best threshold in `reports/selected_config.json`; full test refuses to run without it or if settings drift. Candidate pairs contain deduplicated raw top-K Source-2/3 IDs immediately before thresholding. Every retrieved Source-1 query gets exactly one output row; an empty prediction is written as an empty second field.

## Resume, safety, and outputs

Embedding shards contain `part_NNNNNN.vectors.npy`, `part_NNNNNN.mapping.parquet`, and a manifest with row range, model/revision, engine, quantization setting, dimension, representation, length, dtypes, timings, throughput, GPU memory, SHA-256 hashes, and completion state. Writes go to temporary files and atomically rename. Resume validates sizes and hashes before skipping shards. CUDA OOM retries the same unfinished shard with a halved batch size down to the configured minimum; completed shards remain intact.

Raw search results are Parquet shards under `retrieval/<split>/raw_top50` with query ID, candidate ID/source, per-source rank, and cosine similarity. Threshold and K grids are therefore evaluated without repeating GPU work. Reports contain pair Recall@1/5/10/25/50, any/all-match entity recall, single/multi-match recall, pair precision/recall, empty-truth-aware macro F0.5, singleton accuracy, prediction-count distributions, score distributions, and a refined threshold grid.

Important paths:

- `reports/dataset_profile.json`: schemas, counts, examples, and missingness
- `reports/embedding_benchmark.{json,md}`: speed, memory, numerical checks, and ETA
- `reports/train_retrieval_metrics.{json,md}`: retrieval and threshold evaluation
- `embeddings/`, `indexes/`, `checkpoints/`, `retrieval/`: resumable intermediate state
- `outputs/`: final TSV files

Configuration lives in `configs/default.yaml`. Keep the default 512 dimensions until sampled train retrieval demonstrates that 256 dimensions is within the configured quality tolerance. Missing values are removed, never rendered as `nan`; text is whitespace-normalized and deduplicated inside every shard before embedding.

The measured 100k benchmark selected batch size 256. On the current RTX PRO 4000 Blackwell, BF16 512-d `name_address_country` processed 542.6 rows/s and projected about 5.28 hours for all 10.32M train targets. `name_only` reached 565.6 rows/s; 256-d truncation did not speed up transformer inference. Because no realistic full-pool retrieval quality run has yet compared those alternatives, the safe selected configuration remains BF16, 512 dimensions, length 64, and `name_address_country`.
