# Multilingual hybrid entity resolution

This package matches the canonical `entity_matching_100k` Parquet query set to
its target pool. It supports two independent normalization variants:
rule-based sanscript (default) and AI4Bharat IndicXlit neural Indic-to-English
transliteration. Both use exact BGE-M3 inner-product search on FAISS GPU, GPU
character trigram TF-IDF search, dynamic candidate slicing, and an XGBoost
reranker.

The two retrieval channels always collect Top-150 independently. Their union,
including cross-filled scores from both channels, is saved as
`retrieval_master_pool.parquet`. Changing `--max-k` only reruns inexpensive
feature extraction and reranking.

## Install

Use a CUDA PyTorch environment with a GPU-enabled FAISS build, then install:

```bash
python3 -m pip install -r praj_files/hybrid_entity_resolution/requirements.txt
```

## Run the full pipeline

From the repository root:

```bash
python3 -m praj_files.hybrid_entity_resolution.pipeline all \
  --data-dir entity_matching_100k \
  --output-dir praj_files/hybrid_entity_resolution/artifacts \
  --max-k 50
```

The retrieval command raises an error when master-pool candidate recall is
below `0.99`, while preserving the cache for analysis.

## Run a separate IndicXlit comparison

Keep the existing sanscript process and artifacts untouched. In the same
container, install IndicXlit inference dependencies and download its native to
Roman checkpoint:

```bash
python3 -m pip install -r praj_files/hybrid_entity_resolution/requirements-indicxlit.txt
bash praj_files/hybrid_entity_resolution/setup_indicxlit_model.sh
```

Then start a separate run with a separate output directory and log:

```bash
mkdir -p praj_files/hybrid_entity_resolution/logs
nohup python3 -m praj_files.hybrid_entity_resolution.pipeline all \
  --data-dir entity_matching_100k \
  --output-dir praj_files/hybrid_entity_resolution/artifacts_indicxlit \
  --transliterator indicxlit \
  --indicxlit-model-dir praj_files/hybrid_entity_resolution/indicxlit_model \
  --max-k 50 \
  > praj_files/hybrid_entity_resolution/logs/indicxlit_pipeline.log 2>&1 &
```

After both runs finish, compare retrieval recall, Top-K recall, and validation
metrics:

```bash
python3 -m praj_files.hybrid_entity_resolution.compare_runs \
  praj_files/hybrid_entity_resolution/artifacts \
  praj_files/hybrid_entity_resolution/artifacts_indicxlit
```

IndicXlit's upstream inference expects its Indic-to-English release archive
(`corpus-bin/`, `transformer/indicxlit.pt`, and `lang_list.txt`) and the
`fairseq-interactive` command. Its official instructions describe word-level
inputs; the separate normalizer batches Indic-script runs by detected Unicode
script and language code. Devanagari is routed as Hindi because the script does
not identify Hindi, Marathi, or Nepali on its own.

## Reuse retrieval for another K

```bash
python3 -m praj_files.hybrid_entity_resolution.pipeline features \
  --data-dir entity_matching_100k \
  --output-dir praj_files/hybrid_entity_resolution/artifacts \
  --max-k 30

python3 -m praj_files.hybrid_entity_resolution.pipeline train-evaluate \
  --data-dir entity_matching_100k \
  --output-dir praj_files/hybrid_entity_resolution/artifacts
```

The output directory contains normalized text caches, the retrieval master
pool, candidate features, labels, a validation prediction file, the final
XGBoost model, full predictions, and JSON metrics. Validation is split by query
ID so candidate rows for one entity never cross the train/validation boundary.
Reported precision and recall are pair-level totals; F0.5 is calculated per T1
and macro-averaged, including singleton queries.

Useful memory controls are `--embedding-batch-size`,
`--dense-query-batch-size`, and `--sparse-query-batch-size`. Sparse search
materializes one query batch by all targets on the GPU, so lower the last value
first if CUDA memory is tight.
