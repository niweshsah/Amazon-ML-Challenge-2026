# Codex Implementation Plan: Resumable GPU Embedding Baseline for Entity Resolution

## 1. Goal

Implement a fast, restartable embedding-based baseline using the already-preprocessed Parquet datasets under:

```text
praj_files/datasets/preprocessed/
├── train_source1.parquet
├── train_source2.parquet
├── train_source3.parquet
├── train_ground_truth.parquet
├── test_source1.parquet
├── test_source2.parquet
└── test_source3.parquet
```

The first milestone is not a final learned matcher. It is a strong candidate-retrieval / similarity baseline that:

1. embeds Source 2 and Source 3 records once;
2. builds GPU ANN indexes over their embeddings;
3. embeds Source 1 records in resumable shards;
4. retrieves TOP-K similar records from both Source 2 and Source 3;
5. keeps every candidate whose similarity is above a tuned threshold, allowing zero, one, or many returned entities per Source 1 record;
6. saves raw TOP-K search results so thresholds can be changed later without rerunning embeddings or ANN search;
7. evaluates thresholds and K values on training ground truth;
8. runs the frozen configuration on test;
9. writes challenge-compatible candidate output and a baseline matching output.

The implementation must prioritize restartability, bounded GPU/host memory, measurable runtime, and graceful OOM recovery.

---

## 2. Challenge Rules the Implementation Must Preserve

- Source 1 is the reference source.
- A Source 1 entity may match zero, one, or many records from Source 2 and Source 3.
- Search both Source 2 and Source 3 for every Source 1 entity.
- Do not enforce one-to-one matching.
- Country is an open string field; do not hard-code only US/India.
- Every test Source 1 entity must appear exactly once in the final TSV outputs.
- Returned IDs must come only from test Source 2 or Source 3.
- No external business lookup, geocoding, APIs, or external enrichment.
- The candidate file must contain the final candidate set considered by a later matcher; for this baseline, the thresholded embedding candidates can serve as that set.

---

## 3. Baseline Model Strategy

### 3.1 Embedding model

Start with:

```text
Qwen/Qwen3-Embedding-0.6B
```

Preferred serving path:

```text
vLLM BF16 embedding inference
```

Do not begin with quantization. First benchmark BF16. Add quantized inference only as an optional experiment after the BF16 pipeline works and only keep it if:

- throughput improves materially;
- validation retrieval quality does not degrade materially;
- implementation complexity remains low.

### 3.2 Text representation

Implement two configurable views.

**Baseline view:**

```text
{normalized_business_name} | {normalized_business_address} | {country}
```

**Fast fallback view:**

```text
{normalized_business_name} | {country}
```

Use the name+address view first if the average token count is reasonable. If encoding throughput is poor, benchmark the name-only view.

Do not destroy Unicode text. Preserve accents and non-Latin scripts. Normalize whitespace and nulls conservatively.

### 3.3 Sequence length

Start with:

```yaml
max_length: 96
```

Also benchmark 64 tokens. Avoid long sequences unless validation proves they are necessary.

### 3.4 Embedding dimension

Start with:

```yaml
embedding_dim: 512
```

If supported by the chosen Qwen embedding wrapper, use its dimensionality-reduction / MRL mechanism rather than arbitrary post-hoc PCA.

Also benchmark 256 dimensions if ANN memory or disk becomes a bottleneck.

### 3.5 Similarity

Use cosine similarity.

L2-normalize vectors before indexing/search so inner product equals cosine similarity.

---

## 4. Retrieval Policy: TOP-K + Similarity Threshold

The baseline must not force TOP-1.

For each Source 1 record:

1. retrieve TOP-K from Source 2;
2. retrieve TOP-K from Source 3;
3. merge both result lists;
4. sort globally by cosine similarity;
5. preserve source ID and similarity;
6. keep every candidate satisfying the configured similarity policy.

Initial search values:

```yaml
search_k_per_source: 50
max_candidates_after_merge: 100
```

The ANN search should always save the raw TOP-K results, even if the final baseline uses fewer candidates.

### 4.1 Thresholded multiple-match policy

For each S1 entity:

```text
accepted = all merged candidates with similarity >= threshold
```

Then:

```text
0 above threshold  -> empty prediction
1 above threshold  -> one prediction
N above threshold  -> N predictions
```

This directly supports the challenge requirement that one Source 1 record can have zero, one, or many true matches.

### 4.2 Optional relative-score guard

Implement but disable by default:

```text
candidate_score >= absolute_threshold
AND
candidate_score >= best_score - max_score_gap
```

This can prevent a long tail of weak candidates from being returned when one candidate is clearly stronger.

Suggested initial grid:

```yaml
absolute_thresholds:
  - 0.70
  - 0.75
  - 0.80
  - 0.82
  - 0.84
  - 0.86
  - 0.88
  - 0.90
  - 0.92
  - 0.94

score_gaps:
  - null
  - 0.02
  - 0.04
  - 0.06
```

Do not assume these thresholds are correct. Tune them from the training ground truth.

### 4.3 Candidate list vs baseline prediction

Persist both concepts separately:

- `retrieved_topk`: raw ANN search output;
- `candidate_pairs`: the candidates surviving the selected candidate threshold/policy;
- `baseline_matches`: the subset returned by the current embedding-only baseline prediction policy.

For the first embedding-only baseline, `candidate_pairs` and `baseline_matches` may use the same threshold policy. Keep the files logically separate so a later matcher can consume a broader candidate set.

---

## 5. Required Project Structure

Create:

```text
business_entity_resolution/
├── configs/
│   └── embedding_baseline.yaml
├── src/
│   ├── config.py
│   ├── io.py
│   ├── normalize.py
│   ├── text_builder.py
│   ├── embedder.py
│   ├── embedding_store.py
│   ├── ann_index.py
│   ├── search.py
│   ├── threshold_tuning.py
│   ├── metrics.py
│   ├── checkpoints.py
│   ├── monitoring.py
│   └── output.py
├── scripts/
│   ├── inspect_data.py
│   ├── benchmark_embedding.py
│   ├── embed_targets.py
│   ├── build_indexes.py
│   ├── search_train.py
│   ├── tune_threshold.py
│   ├── search_test.py
│   └── build_submission.py
└── run_embedding_baseline.py
```

Use one orchestrator so Codex can run stages independently and resume:

```bash
python3 business_entity_resolution/run_embedding_baseline.py \
  --config business_entity_resolution/configs/embedding_baseline.yaml \
  --stage benchmark
```

Supported stages:

```text
inspect
benchmark
embed_targets_train
index_train
search_train
tune_threshold
embed_targets_test
index_test
search_test
build_outputs
validate
all
```

---

## 6. Configuration

Create a single YAML config containing at least:

```yaml
paths:
  data_dir: praj_files/datasets/preprocessed
  artifacts_dir: artifacts/embedding_baseline
  train_s1: praj_files/datasets/preprocessed/train_source1.parquet
  train_s2: praj_files/datasets/preprocessed/train_source2.parquet
  train_s3: praj_files/datasets/preprocessed/train_source3.parquet
  train_gt: praj_files/datasets/preprocessed/train_ground_truth.parquet
  test_s1: praj_files/datasets/preprocessed/test_source1.parquet
  test_s2: praj_files/datasets/preprocessed/test_source2.parquet
  test_s3: praj_files/datasets/preprocessed/test_source3.parquet

model:
  name: Qwen/Qwen3-Embedding-0.6B
  backend: vllm
  dtype: bfloat16
  max_length: 96
  embedding_dim: 512
  normalize_embeddings: true

text:
  view: name_address_country

embedding:
  initial_batch_size: 256
  min_batch_size: 8
  shard_rows: 100000
  save_dtype: float16

search:
  k_per_source: 50
  max_merged_candidates: 100
  index_backend: cuvs_or_faiss_gpu
  similarity: cosine

thresholding:
  candidate_threshold: null
  prediction_threshold: null
  max_score_gap: null

resources:
  gpu_free_reserve_gb: 2
  host_memory_limit_gb: 10
  disk_reserve_gb: 20

runtime:
  resume: true
  log_every_seconds: 15
  slow_shard_multiplier: 2.0
```

---

## 7. Stage 0: Inspect Parquet Schema Before Coding Assumptions

Codex must first inspect each Parquet schema and a small sample.

Verify columns such as:

```text
entity_id
business_name
business_address
country
```

Do not silently assume exact column names if the preprocessed files differ.

Report:

- schema;
- row counts;
- null counts;
- average/max text lengths;
- average estimated tokenizer length on a sample;
- unique entity IDs;
- duplicate IDs;
- countries present;
- disk size per file.

Write:

```text
artifacts/embedding_baseline/inspect/dataset_profile.json
```

If schema validation fails, stop before expensive work.

---

## 8. Stage 1: 100k-Record Benchmark Before Full Embedding

Never launch embeddings for millions of records without a benchmark.

Sample 100,000 representative target rows and benchmark:

```text
vLLM BF16
max_length = 64
max_length = 96
embedding_dim = 512
```

Measure:

```text
records/sec
tokens/sec
seconds/100k
peak GPU memory
peak host memory
output bytes/record
average tokens/record
p95 tokens/record
```

Forecast:

```text
ETA_train_targets
ETA_train_queries
ETA_test_targets
ETA_test_queries
embedding_disk_usage
```

Save:

```text
artifacts/embedding_baseline/benchmarks/benchmark.json
```

If a single 100k benchmark is unexpectedly slow, try:

1. shorter max length;
2. larger/smaller vLLM batch settings;
3. name-only view;
4. 256-d embeddings;
5. only then quantized vLLM.

Do not proceed blindly when ETA is unreasonable.

---

## 9. Stage 2: Target Embedding Generation

Targets are Source 2 and Source 3.

Process train and test separately.

For each split/source:

```text
train_source2
train_source3
test_source2
test_source3
```

Read Parquet in bounded row groups / batches. Do not load the full file into Python objects.

For each shard:

1. read rows;
2. construct normalized input text;
3. optionally deduplicate identical input texts inside the shard;
4. embed using vLLM;
5. normalize vectors;
6. restore duplicated rows if text deduplication was used;
7. save IDs + embeddings;
8. write a completed shard manifest;
9. free tensors;
10. clear unnecessary GPU references;
11. continue.

Persist approximately every 100,000 records:

```text
artifacts/embedding_baseline/embeddings/test/source2/part_000000.parquet
artifacts/embedding_baseline/embeddings/test/source2/part_000001.parquet
...
```

Each shard must contain at least:

```text
row_id
entity_id
embedding
```

For efficiency, binary NumPy/memmap/Arrow vector storage is acceptable if Parquet vector columns become a bottleneck, but maintain a manifest mapping rows to entity IDs.

Never keep all 10M target embeddings as a Python list.

---

## 10. Checkpoint / Resume Contract

Every expensive stage must maintain a manifest, for example:

```json
{
  "stage": "embed_targets_test_source2",
  "input_file": "...",
  "input_fingerprint": "...",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "max_length": 96,
  "embedding_dim": 512,
  "completed_shards": [0, 1, 2],
  "rows_completed": 300000,
  "last_batch_size": 256
}
```

On restart:

- validate the config hash;
- validate already-saved shard files;
- skip valid completed shards;
- resume from the first incomplete shard.

Use temporary filenames:

```text
part_000042.tmp
```

Then atomic rename only after the shard is fully written and validated.

Never mark a partially written shard complete.

---

## 11. OOM Recovery

Wrap GPU embedding and ANN operations with explicit CUDA OOM handling.

If embedding OOM occurs:

```text
batch_size = max(min_batch_size, batch_size // 2)
```

Then:

1. release failed tensors;
2. run Python GC;
3. empty unused CUDA cache when appropriate;
4. retry only the current unfinished batch/shard;
5. log the OOM event and new batch size.

If batch size reaches the configured minimum and still fails, stop with a clear actionable error. Do not restart prior completed shards.

For ANN index OOM:

- reduce index shard size;
- build/search one target partition at a time;
- merge per-partition TOP-K results globally;
- do not lower K silently.

---

## 12. Stage 3: ANN Index Construction

Build separate indexes for Source 2 and Source 3 for each split.

Logical indexes:

```text
train_s2.index
train_s3.index
test_s2.index
test_s3.index
```

Preferred GPU backend:

```text
cuVS if already compatible with the RAPIDS environment
otherwise FAISS-GPU
```

Choose the simplest reliable cosine/IP index first.

A flat exact GPU index is acceptable for a small benchmark. For the full ~5M-row source indexes, use a memory-safe approximate or partitioned index if exact search is not feasible.

Persist:

- index metadata;
- embedding shard fingerprints;
- vector dimension;
- metric;
- index parameters;
- entity row mapping.

If the index itself is not safely serializable/reloadable, persist enough partition metadata so rebuilding only the missing partition is possible.

---

## 13. Stage 4: Source 1 Query Embeddings + Search

Do not precompute all Source 1 embeddings unless it materially helps. Streaming query embedding + immediate search is preferred.

For each Source 1 shard:

1. load 50k-100k S1 rows;
2. build input text;
3. embed;
4. normalize;
5. search Source 2 index for TOP-50;
6. search Source 3 index for TOP-50;
7. merge to global TOP-100;
8. save raw search results immediately;
9. update checkpoint;
10. release query embeddings.

Save raw results with one row per pair:

```text
source1_entity_id
candidate_entity_id
candidate_source
similarity
source_rank
global_rank
```

Path:

```text
artifacts/embedding_baseline/search/train/part_000000.parquet
artifacts/embedding_baseline/search/test/part_000000.parquet
```

This raw TOP-K cache is critical. Threshold tuning must operate from these files without rerunning ANN search.

---

## 14. Stage 5: Threshold Tuning on Training Ground Truth

Use `train_ground_truth.parquet` only for evaluation/tuning.

Do not inject true matches into the retrieved candidate set.

From cached train TOP-K results, evaluate:

```text
K per source: 10, 25, 50
absolute similarity thresholds: configured grid
optional best-score gap: configured grid
```

Report at minimum:

### Retrieval metrics

```text
pair Recall@K
entity hit-rate@K
all-true-matches-covered@K
average candidate count
p50/p95 candidate count
zero-candidate rate
```

### Baseline prediction metrics

For each threshold policy:

```text
macro F0.5
pair precision
pair recall
singleton accuracy
false-positive rate on singletons
average predicted matches/entity
```

Because F0.5 is precision-heavy, do not select a threshold using recall alone.

Save all results:

```text
artifacts/embedding_baseline/evaluation/threshold_grid.parquet
artifacts/embedding_baseline/evaluation/best_policy.json
```

The selected policy must record:

```text
k_per_source
candidate_threshold
prediction_threshold
max_score_gap
validation metrics
```

---

## 15. Recommended Two-Threshold Design

To support a later model, use two thresholds instead of one when useful:

### Candidate threshold

Lower threshold, optimized for high recall:

```text
candidate_similarity >= candidate_threshold
```

This produces `candidate_pairs.tsv`.

### Baseline prediction threshold

Higher threshold, optimized for macro F0.5:

```text
similarity >= prediction_threshold
```

This produces `matching_results.tsv` for the embedding-only baseline.

This separation is preferred because the candidate stage should preserve plausible matches, while final predictions should be more precision-oriented.

Example only:

```text
candidate_threshold = 0.78
prediction_threshold = 0.88
```

Do not hard-code these example numbers; learn them from train validation.

---

## 16. Train/Validation Split for Threshold Selection

Do not tune thresholds on every training S1 and then report the same data as validation.

Create a deterministic S1-level split, for example:

```text
80% threshold-development
20% held-out validation
seed = 42
```

Use the development portion to choose threshold/K.

Report the final selected policy once on the held-out portion.

Do not split individual positive pairs from the same Source 1 entity across dev/validation.

---

## 17. Stage 6: Test Search

After threshold selection, freeze:

```text
model
text representation
max_length
embedding dimension
ANN config
K
candidate threshold
prediction threshold
score-gap rule
```

Then run test S1 against test S2 and test S3.

Do not use test labels because none exist.

Save raw TOP-K test search results before thresholding.

---

## 18. Output Generation

Generate two TSV files.

### candidate_pairs.tsv

For each test S1:

```text
all S2/S3 IDs surviving the selected candidate threshold
```

### matching_results.tsv

For each test S1:

```text
all S2/S3 IDs surviving the selected prediction threshold
```

If none survive, write an empty second field.

Required characteristics:

- literal tab separator;
- exactly one row per test S1;
- deterministic ID ordering, preferably descending similarity then stable entity ID;
- no duplicate candidate IDs;
- S2/S3 IDs only;
- every predicted ID must also exist in the corresponding candidate list.

Also write an audit file:

```text
artifacts/embedding_baseline/output/test_scored_pairs.parquet
```

containing:

```text
source1_entity_id
candidate_entity_id
similarity
global_rank
passed_candidate_threshold
passed_prediction_threshold
```

---

## 19. Runtime Monitoring

Every expensive loop must report periodically:

```text
stage
source/split
shard number / total shards
rows completed / total rows
rows/sec
tokens/sec when embedding
elapsed time
ETA
GPU allocated memory
GPU reserved memory
GPU free memory
host RSS / available memory
disk free space
current batch size
output checkpoint path
```

Example:

```text
[embed test_s2]
shard=18/49
rows=1,800,000/4,887,273
speed=1,742 rows/s
elapsed=00:17:13
eta=00:29:32
gpu_alloc=6.8GB
gpu_free=12.1GB
batch=256
saved=.../part_000017.parquet
```

---

## 20. Detecting a Process That Is Taking Too Long

Do not use arbitrary hard time limits initially.

After the first 2-3 completed shards, establish a rolling median shard duration.

Warn when:

```text
current_shard_time > 2.0 × rolling_median_shard_time
```

Also warn on sustained throughput degradation greater than 30%.

When triggered, log diagnostics:

- token length distribution for current shard;
- GPU utilization/memory;
- host memory;
- disk write speed if observable;
- batch size;
- duplicate-text ratio;
- records/sec compared with benchmark.

Do not kill the process automatically solely because one shard is slow.

---

## 21. Fast Smoke Test Before Full Run

Codex must implement a smoke test over a tiny slice:

```text
1,000 S1
10,000 S2
10,000 S3
```

Validate:

- embedding shapes;
- normalization;
- ANN search ordering;
- source ID mapping;
- merged ranking;
- multiple candidates above threshold;
- empty results at high threshold;
- checkpoint/resume;
- OOM batch-size fallback using a mocked/reduced memory condition if practical.

Then run the 100k benchmark.

Only after both pass should full target embedding begin.

---

## 22. Implementation Order for Codex

Implement in this exact order:

1. inspect Parquet schemas and dataset counts;
2. config loader;
3. checkpoint/manifest helpers;
4. runtime/ETA/GPU-memory logger;
5. text builder;
6. Qwen vLLM embedder;
7. 1k smoke test;
8. 100k embedding benchmark;
9. resumable target embedding writer;
10. ANN index abstraction;
11. small exact-search correctness test;
12. resumable S1 search with TOP-K cache;
13. training ground-truth parser;
14. threshold/K evaluation;
15. policy freezing;
16. full train retrieval/evaluation;
17. full test target embeddings/index;
18. full test S1 search;
19. TSV generation;
20. validation and summary report.

Do not implement cross-encoders, XGBoost, or fine-tuning before this baseline is complete and measured.

---

## 23. Acceptance Criteria

The baseline is complete only when all of the following are true:

- all seven input Parquet files are schema-validated;
- target embeddings are produced in resumable shards;
- rerunning skips completed valid shards;
- a failed shard can resume without recomputing earlier shards;
- CUDA OOM reduces batch size and retries only unfinished work;
- train S1 retrieves TOP-K from both train S2 and train S3;
- raw TOP-K results are persisted;
- threshold changes require no re-embedding or ANN rerun;
- K and thresholds are evaluated against `train_ground_truth.parquet`;
- policy supports zero, one, or multiple returned candidates;
- runtime/ETA/memory are logged throughout;
- full test S1 coverage is preserved;
- no S1 IDs appear as candidate IDs;
- no unknown S2/S3 IDs are emitted;
- every baseline match is a subset of its candidate list;
- final output files are deterministic and restartable.

---

## 24. First Configuration to Try

Use this as the first serious baseline, then tune from measurements:

```yaml
model: Qwen/Qwen3-Embedding-0.6B
backend: vllm
dtype: bfloat16
text_view: name_address_country
max_length: 96
embedding_dim: 512
embedding_shard_rows: 100000
initial_batch_size: 256
search_k_per_source: 50
similarity: cosine
candidate_threshold: tuned_on_train
prediction_threshold: tuned_on_train
```

The critical principle is:

```text
embed once -> save everything -> search once -> save TOP-K scores -> tune thresholds cheaply many times
```

Do not repeatedly re-embed or re-search merely to change the match threshold.
