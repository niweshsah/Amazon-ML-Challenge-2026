# Parquet schema reference

All string columns use UTF-8 Arrow `string` values. Original source fields are preserved as read from TSV; empty TSV cells are stored as empty strings (`""`), not Parquet nulls. ID prefixes identify source: `S1-` for T1, and `S2-` or `S3-` for T2.

## `t1.parquet`

One row per selected Source 1 entity (100,000 rows).

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `entity_id` | `string` | Original T1 ID; primary key. |
| `business_name` | `string` | Original business name. |
| `business_address` | `string` | Original address; may be empty. |
| `country` | `string` | Original country label. |

## `t2.parquet`

One row per T2 record in the sampled retrieval pool (855,265 rows in this build). This includes every positive T2 referenced by selected T1 rows and 500,000 distractors.

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `entity_id` | `string` | Original T2 ID; primary key. |
| `business_name` | `string` | Original business name. |
| `business_address` | `string` | Original address; may be empty. |
| `country` | `string` | Original country label. |
| `source` | `string` | ID source, either `S2` or `S3`. |
| `pool_role` | `string` | `positive` if referenced by a selected T1 ground-truth mapping; otherwise `distractor`. |

## `ground_truth.parquet`

Exactly one row per T1 (100,000 rows). The ordered list retains the original order of IDs from `train_ground_truth.tsv`; a singleton has an empty list.

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `source1_entity_id` | `string` | T1 ID; primary key and foreign key to `t1.entity_id`. |
| `matched_entity_ids` | `list<string>` | All ground-truth T2 IDs for the T1; may be empty. IDs refer to `t2.entity_id`. |

## `positives.parquet`

One row per positive T1–T2 link (355,265 rows in this build).

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `source1_entity_id` | `string` | T1 ID; foreign key to `t1.entity_id`. |
| `matched_entity_id` | `string` | Positive T2 ID; foreign key to `t2.entity_id`. |
| `ground_truth_position` | `int32` | Zero-based position of the T2 ID in `ground_truth.matched_entity_ids`. |

The pair `(source1_entity_id, matched_entity_id)` is unique. A T1 with no matches has no rows in this table; use `ground_truth.parquet` to identify singletons.

## `negatives.parquet`

Six sampled negative links per T1 (600,000 rows in this build).

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `source1_entity_id` | `string` | T1 ID; foreign key to `t1.entity_id`. |
| `negative_entity_id` | `string` | Sampled T2 distractor ID; foreign key to `t2.entity_id`. |
| `negative_kind` | `string` | `same_country` for country-matched samples, or `global` for unrestricted samples. |
| `source` | `string` | T2 source, either `S2` or `S3`; agrees with the negative ID prefix. |

Each T1 has three negatives from each source: two `same_country` and one `global`. The pair `(source1_entity_id, negative_entity_id)` is unique. Negative T2 IDs are drawn from the distractor pool and do not occur in that T1's ground-truth list.

## `selection.parquet`

One row per selected T1 (100,000 rows). These columns document the sample design and are metadata, not replacement entity text.

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `source1_entity_id` | `string` | T1 ID; primary/foreign key to `t1.entity_id`. |
| `selection_reason` | `string` | First category that selected this T1, or representative sampling. |
| `sampling_tags` | `list<string>` | Every heuristic category detected for the T1 or its positive T2 records; may be empty. |

Current diversity tag values are `singleton`, `many_matches`, `rare_indic`, `devanagari`, `mixed_script`, `missing_t2_field`, `short_noisy`, `numeric_punctuation`, `duplicate_like`, and `low_name_overlap`. `selection_reason` may also be `representative` or `representative_fill`.

## Join examples

- Positive training pairs: join `positives.source1_entity_id` to `t1.entity_id` and `positives.matched_entity_id` to `t2.entity_id`.
- Negative training pairs: join `negatives.source1_entity_id` to `t1.entity_id` and `negatives.negative_entity_id` to `t2.entity_id`.
- All ground truth, including singletons: join `ground_truth.source1_entity_id` to `t1.entity_id`; explode `matched_entity_ids` only when individual positive rows are needed.

The exact table and field types are also recorded in [`schema.json`](schema.json). Per-build row counts and data distributions are in [`statistics.json`](statistics.json).
