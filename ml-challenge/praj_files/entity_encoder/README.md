# Multilingual entity encoder

This pipeline uses the raw Unicode name, address, and country fields. The old
`*_clean` columns are intentionally excluded because their regex splits Indic
combining characters. It trains a 1024-dimensional CLS pooled multilingual
encoder with BF16 LoRA and a contrastive loss over true T1 to T2/T3 links,
in-batch competitors, and explicit hard negatives.

Run inside `./run_qwen.sh` from `/workspace/ml-challenge` (or use `docker exec -w
/workspace/ml-challenge pytorch-qwen` for a running container):

```bash
python3 praj_files/entity_encoder/prepare.py
python3 praj_files/entity_encoder/train.py baseline
python3 praj_files/entity_encoder/train.py smoke --smoke-steps 3
python3 praj_files/entity_encoder/train.py train
python3 praj_files/entity_encoder/train.py evaluate
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

Training is capped at 115 minutes. If this limit stops an epoch early, the
actual number of pairs processed is recorded in `training_result.json`.
Checkpoint adapters are saved every 500 steps and at the end. Logs record
loss, elapsed time, and peak GPU allocation. Keep the adapter with its base
model and tokenizer for inference.
