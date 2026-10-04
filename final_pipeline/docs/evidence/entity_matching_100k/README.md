# Canonical 100K entity-matching dataset

This folder contains a reusable sample of the **training** data from the Business Entity Resolution Challenge. It preserves original UTF-8 field values and IDs in Parquet. No model, tokenizer, normalized text, embedding, or fixed train/validation/test split is stored here. The original files under `student_resource/` are read only.

## Source and relationship

The source files are `student_resource/dataset/train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`. `PS.pdf` defines Source 1 as the deduplicated reference source. Each Source 1 (T1) entity has zero or more ground-truth matches in Source 2 or Source 3 (together called T2 here). The original ground truth has one row per T1, with a comma-separated list of T2 IDs; an empty list means a singleton. All raw source records have `entity_id`, `business_name`, `business_address`, and `country` fields. The ID prefix (`S1-`, `S2-`, `S3-`) identifies the source.

The full training data has 2,206,821 T1 records. Of these, 123,247 are singletons. A T2 ID appears in at most one training ground-truth list. The challenge evaluates predictions with **macro F0.5 per T1**, including singletons. Its test set has an unseen country, France; the canonical schema keeps `country` as an unrestricted string.

## Files and schemas

| File | Contents |
| --- | --- |
| `t1.parquet` | Exactly 100,000 T1 rows; the four original raw fields. |
| `t2.parquet` | All positive T2 records for selected T1s plus 500,000 sampled distractors; original raw fields plus `source` (`S2`/`S3`) and `pool_role` (`positive`/`distractor`). Each T2 appears once. |
| `ground_truth.parquet` | One row per T1: `source1_entity_id`, `matched_entity_ids` as an ordered `list<string>`. Singletons have an empty list. |
| `positives.parquet` | One row per true link: `source1_entity_id`, `matched_entity_id`, `ground_truth_position`. |
| `negatives.parquet` | Six sampled false links per T1: `source1_entity_id`, `negative_entity_id`, `negative_kind`, `source`. |
| `selection.parquet` | T1 selection reason and diagnostic sampling tags; these are metadata, not model inputs. |
| `statistics.json` | Counts, countries, missing fields, script proxies, match multiplicity, and sampling coverage. |
| `manifest.json` | Seed, build parameters, and SHA-256 hashes of source and generated files. |
| `SCHEMA.md` / `schema.json` | Human-readable and machine-readable schema references for all six Parquet tables. |

The original TSV's empty fields remain empty strings in Parquet. The files do not infer `null` from an empty field, change Unicode, or alter name/address punctuation or casing. The diagnostic script counts are **Unicode script proxies**, not language labels.

## Sampling

The fixed seed is **20260926**. Stable BLAKE2b ranks select a 500,000-T1 candidate set regardless of TSV row order. The first 70,000 selected T1s are a representative hash sample. Ten diversity categories contribute up to 3,000 additional T1s each, without duplicate T1 IDs: singleton, seven-or-more matches, rare Indic script in a linked T2, Devanagari in a linked T2, mixed-script T2 name, missing T2 field, short/noisy text, numeric/punctuation-heavy text, duplicate-like T1 name/address, and low T1/T2 name-token overlap. If a category has fewer available records, the remaining places are filled by hash rank. Sampling tags only guide selection; raw values remain unchanged.

The T2 pool includes every positive for the selected T1s. It also contains the 250,000 lowest-hash distractor IDs from each of Sources 2 and 3, excluding selected positives. Each T1 gets two random same-country distractors and one unrestricted random distractor from each source, for six negatives. The per-T1 choices use the fixed seed and stable hashes, never reuse an ID for that T1, and exclude its **entire** positive set. The original ground truth is treated as exhaustive, as specified by the problem statement. More difficult negatives can later be mined from `t2.parquet` while checking `ground_truth.parquet` before assigning labels.

This is a **sampled retrieval pool**. Retrieval quality or runtime on these T2 records does not estimate full 10-million-T2 performance without an additional scale test. When making future splits, group by T1 ID and keep all its positive and negative links together. Score macro F0.5 on held-out T1s, including singletons.

## Build and validate

Requires Python 3 and `pyarrow` (tested with 24.0.0). Run from the repository root:

```bash
python3 entity_matching_100k/build_dataset.py
python3 entity_matching_100k/validate_dataset.py
```

The builder refuses to replace an existing dataset unless `--force` is supplied. `--source-dir` and `--output-dir` can point to other locations. For a full deterministic rebuild check:

```bash
python3 entity_matching_100k/validate_dataset.py --rebuild-check
```

The validator checks file hashes, mapping completeness, IDs, six negative links per T1, positive/negative separation, country-constrained negative slots, and byte-for-byte-equivalent decoded raw fields against the source TSVs. `--rebuild-check` additionally rebuilds in a temporary directory and compares generated file hashes.

For a detached run inside the existing `pytorch-qwen` container started by `./run_qwen.sh`, run `bash entity_matching_100k/start_background.sh`. The job runs the builder and full validator, writes progress to `job.log`, and records `building`, `validating`, `complete`, or `failed` in `job_status.json`. The container must remain running until the job finishes.

## Load the data

```python
from pathlib import Path
import pyarrow.parquet as pq

root = Path("entity_matching_100k")
t1 = pq.read_table(root / "t1.parquet").to_pandas()
t2 = pq.read_table(root / "t2.parquet").to_pandas()
truth = pq.read_table(root / "ground_truth.parquet").to_pylist()
positives = pq.read_table(root / "positives.parquet").to_pandas()
negatives = pq.read_table(root / "negatives.parquet").to_pandas()

# All records for a singleton have matched_entity_ids == [].
singleton_ids = [row["source1_entity_id"] for row in truth if not row["matched_entity_ids"]]
# Join positives or negatives to raw T1/T2 fields by their original IDs as needed.
```
