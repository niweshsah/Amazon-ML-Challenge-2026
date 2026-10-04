# Sample-pool GPU tree entity-matching baseline

This folder is independent of the existing implementations and does not modify the source Parquet data. It retrieves candidate T2 records for each sampled T1, computes cached pair features, trains CUDA XGBoost, and scores complete T1 match lists with the challenge's macro F0.5 (including singletons). It is a **sample-pool benchmark**: `entity_matching_100k/t2.parquet` includes all selected positive T2 records and 500,000 distractors. Its candidate recall and F0.5 do not estimate performance against the full challenge's roughly 10 million T2 records.

The source PDF defines the task and scoring. The prior 1% analysis and `business_entity_resolution/train_gbdt.py` informed the feature choices. In this sample, T1 names are Latin script; many target names contain accents or Indian scripts. Source 3 addresses have weaker overlap with T1, and roughly 4.8% of true target addresses are empty. The code preserves all raw text, adds Unicode-safe normalized and accent-folded forms, and treats country as an open string. `pool_role`, selection reasons, IDs, and labels are never model features. Target IDs only identify output pairs.

## Environment

From the repository root, enter the existing container with `./run_qwen.sh`. The container sees the repository at `/workspace/ml-challenge`. It has been checked for CUDA-visible XGBoost, PyArrow, RapidFuzz, matplotlib, and scikit-learn. The host Python environment is not the training environment. Commands below run inside the container from `/workspace/ml-challenge`.

The new folder's `artifacts/` is writable. Preprocessing is CPU heavy because Unicode normalization, bounded inverted postings, and RapidFuzz operate on variable-length strings; moving these operations to cuDF/cuML would add transfers without a measured benefit. XGBoost fitting and its CUDA smoke check use the GPU. The retriever indexes exact names, informative name/address tokens, and name character 4-grams, including transliterated Indic forms, ranking and retaining at most 32 candidates per target source. Candidate generation does not read labels. Retrieved nonmatches are the hard negatives. It reports recall against *all* true links, including missed candidates.

## Commands

```bash
# Small end-to-end startup check; writes only under this folder's artifacts/smoke.
python3 entity_matching_gpu_tree/main.py smoke

# Reusable processed records, candidates, and pair-feature Parquet cache.
python3 entity_matching_gpu_tree/main.py prepare

# Fit, threshold selection, untouched audit evaluation, and all plots.
python3 entity_matching_gpu_tree/main.py train

# Regenerate all plots and error tables from the saved model and cached scores,
# then print saved evaluation metrics.
python3 entity_matching_gpu_tree/main.py evaluate
```

The requested detached final run starts preprocessing and training in one process. Run this **once** after the smoke check:

```bash
cd /workspace/ml-challenge
nohup python3 -u entity_matching_gpu_tree/main.py run > entity_matching_gpu_tree/artifacts/final_nohup.log 2>&1 < /dev/null &
echo $! > entity_matching_gpu_tree/artifacts/final_nohup.pid
```

The pipeline also writes timestamped lines to `artifacts/pipeline.log`. Check the job later only when desired:

```bash
cat entity_matching_gpu_tree/artifacts/final_nohup.pid
ps -p "$(cat entity_matching_gpu_tree/artifacts/final_nohup.pid)" -o pid,etime,stat,cmd
tail -n 80 entity_matching_gpu_tree/artifacts/final_nohup.log
```

Stop it with `kill -TERM "$(cat entity_matching_gpu_tree/artifacts/final_nohup.pid)"`. After a complete `prepare`, restarting `train` reuses the feature cache. An interrupted `prepare` restarts candidate generation from the beginning, writing deterministic parts in the same cache directory; a complete cache is reused automatically. An interrupted XGBoost fit restarts fitting from the cached features. To rerun the full pipeline after completion, choose a new output folder with `--output /workspace/ml-challenge/entity_matching_gpu_tree/artifacts/<run_name>` so old results remain intact; the output folder must be under a writable project path.

## Splits and artifacts

Stable hashes of T1 IDs assign 70% to fitting, 10% to early stopping, 10% to threshold selection, and 10% to a final audit. Every T1's pairs stay together. Target IDs are not features. All unsupervised token frequencies are measured from the retrieval corpus available at inference; no labels enter those statistics. This T2 pool is label-enriched by construction, which is a documented limitation of the dataset itself.

`artifacts/cache/<signature>/` contains processed T1/T2 records with raw and normalized text, partitioned pair-feature Parquet, all T1 truth counts and IDs, and a manifest with the feature list, input hash, and candidate recall. `artifacts/run/` contains the model, selected threshold and metrics, predictions, threshold and learning curves, gain importance, sampled native TreeSHAP values, false-positive/false-negative examples, subgroup metrics, and 300 dpi plots. Pair PR-AUC and ROC-AUC apply only to retrieved pairs; macro F0.5 and candidate recall account for omitted true pairs. The audit partition is scored once after threshold selection.

If a future full challenge submission is needed, it requires retrieving against the complete train/test sources and writing/validating both exact-format TSVs. This sampled baseline does not claim to produce those submission files.
