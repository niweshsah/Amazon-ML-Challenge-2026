# Multilingual entity encoder

This pipeline uses the raw Unicode name, address, and country fields. The old
`*_clean` columns are intentionally excluded because their regex splits Indic
combining characters. It trains a 1024-dimensional CLS pooled multilingual
encoder, initialized from the MIT-licensed Indic-tuned BGE-M3 checkpoint, with
BF16 LoRA and a contrastive loss over true T1 to T2/T3 links,
in-batch competitors, and explicit hard negatives.

Run inside `./run_qwen.sh` from `/workspace/ml-challenge` (or use `docker exec -w
/workspace/ml-challenge pytorch-qwen` for a running container):

```bash
python3 praj_files/entity_encoder/prepare.py
python3 praj_files/entity_encoder/train.py baseline
python3 praj_files/entity_encoder/evaluate_bge_100k.py --queries 100000
python3 praj_files/entity_encoder/evaluate_rerank.py prepare --queries 2000
python3 praj_files/entity_encoder/evaluate_rerank.py baseline --queries 2000
python3 praj_files/entity_encoder/evaluate_rerank.py fine_tuned --queries 2000
python3 praj_files/entity_encoder/train.py smoke --smoke-steps 3
python3 praj_files/entity_encoder/train.py train
python3 praj_files/entity_encoder/train.py evaluate
python3 praj_files/entity_encoder/export_encoder.py
python3 praj_files/entity_encoder/embed_all.py --split train --source 1
python3 praj_files/entity_encoder/embed_all.py --split train --source 2
python3 praj_files/entity_encoder/embed_all.py --split train --source 3
python3 praj_files/entity_encoder/embed_all.py --split test --source 1
python3 praj_files/entity_encoder/embed_all.py --split test --source 2
python3 praj_files/entity_encoder/embed_all.py --split test --source 3
```

Configuration and seed are in `config.json`. `prepare.py` deterministically
assigns T1 entities to train or validation by ID hash, removes validation
entities whose known target ID also appears in training, and saves the sampled
triples and validation pool as Parquet. The 500k positives are sampled without
replacement with extra weight on missing fields, non-ASCII scripts, and
single-field agreement. Negatives preferentially share the exact name, then
the first name token and country. The validation pool contains all known
matches for held-out T1 queries and about 12k sampled distractors; its metrics
are **sampled-pool metrics**, not full 10M-target retrieval estimates.

The baseline and fine-tuned evaluation files report Recall@K, MRR, cosine
scores, and best macro F0.5 over an identical held-out pool. Threshold tuning
on this small pool is diagnostic and must be repeated on a larger held-out
pool before final challenge matching. `embed_all.py` writes unit normalized
float16 `.npy` vector shards and matching ID Parquet shards, with small GPU
batches and resumable completed shards. It does not generate final challenge
matches; downstream retrieval and thresholding remain necessary.

`evaluate_bge_100k.py` audits existing full-pool zero-shot `BAAI/bge-m3`
FAISS IVF-PQ candidate files in `praj_files/output/`. It selects exactly 100k
T1 records outside the training and model-selection ID partitions, scans both
stored top-100 lists per source, and compares them against ground truth. This
measures blocking/candidate recall against all training T2/T3 rows. The stored
files have ranks but no cosine scores, so the script reports an oracle F0.5
ceiling rather than presenting a thresholded matching score as measured.

`evaluate_rerank.py` makes the next practical comparison: it takes 2,000 T1
IDs from the same untouched holdout, retrieves their previously saved full-
source BGE top-100 candidates from each target source, and scores those actual
candidates with the zero-shot and final Indic encoders. It tunes a cosine
threshold on a salted 25% T1 calibration split and reports F0.5 on the
remaining 75%. This tests a realistic candidate stage plus the Indic matching
encoder without injecting positives; its candidate recall still cannot exceed
the upstream BGE stage. Preparation scans the large input files and inference
must embed each distinct candidate with each model, so allow time and disk
space. These commands are supplied for the user to run; no result is claimed
until they finish.

Training is capped at 115 minutes. If this limit stops an epoch early, the
actual number of pairs processed is recorded in `training_result.json`.
Checkpoint adapters are saved every 500 steps and at the end. Logs record
loss, elapsed time, and peak GPU allocation. Keep the adapter with its base
model and tokenizer for inference.

The provided `final_adapter` and `final_encoder` are the validated step-500
checkpoint. The initial run was stopped at step 800 when the user requested
the code handoff; only step 500 had a measured validation result. Running
`train.py train` starts a fresh full capped run and can replace these outputs,
so copy the provided checkpoint first if you want to retain it.

All six source files together contain roughly 24 million rows, so 1024-d
float16 vectors alone need about 49 GB before IDs and checkpoints. Pass
`--embedding-dir /path/to/large/volume` to `embed_all.py` if the workspace
does not have enough space. Each 50k-row shard is about 100 MB and can be
resumed independently; the script checks the adapter hash before reusing a
shard directory.
