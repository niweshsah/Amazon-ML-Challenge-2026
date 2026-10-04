# Business Entity Resolution Dataset Analysis

Dataset root: `/workspace/ml-challenge/student_resource/dataset`. Backend: `gpu`.
GPU mode covers batched string-length calculations only; TSV parsing, Unicode tokenization, joins, duplicate checks, and reporting use CPU/SQLite.

Requested data percentage: 1%. Counts describe only the analyzed records. Training Source 1 IDs are sampled by a stable seeded hash; all their known Source 2/3 matches and sampled background records are included. Test sources are sampled independently. Actual per-file percentages may differ.
Displayed examples are bounded reservoirs; optional similarity and blocking estimates use sampled training pairs. Percentage selection still reads the TSVs and does not estimate full-dataset totals.

## Dataset inventory

| split | source | input_rows | rows | sample_percent_actual | file_size_bytes |
| --- | --- | --- | --- | --- | --- |
| train | source1 | 2206821 | 22037 | 0.999 | 210069713 |
| train | source2 | 5034616 | 86886 | 1.726 | 489301488 |
| train | source3 | 5285603 | 91727 | 1.735 | 503705637 |
| test | source1 | 1732544 | 17194 | 0.992 | 175022086 |
| test | source2 | 4887273 | 48739 | 0.997 | 509456422 |
| test | source3 | 5082316 | 50811 | 1.000 | 506002772 |

Detailed file and column profiles: `dataset_summary.tsv`, `column_summary.tsv`, and `common_values.tsv`.

## Missingness and text

| split | source | column | missing_percent | unique_values |
| --- | --- | --- | --- | --- |
| train | source1 | business_name | 0.000 | 21354 |
| train | source1 | business_address | 0.000 | 22021 |
| train | source1 | country | 0.000 | 2 |
| train | source2 | business_name | 0.000 | 84687 |
| train | source2 | business_address | 3.827 | 79114 |
| train | source2 | country | 0.000 | 2 |
| train | source3 | business_name | 0.000 | 89584 |
| train | source3 | business_address | 3.687 | 84273 |
| train | source3 | country | 0.000 | 2 |
| test | source1 | business_name | 0.000 | 16791 |
| test | source1 | business_address | 0.000 | 17186 |
| test | source1 | country | 0.000 | 3 |

See `missing_values.tsv`, `string_statistics.tsv`, `character_patterns.tsv`, `normalization_patterns.tsv`, and the token tables for source and country detail.

## Tokens and blocking clues

| token | frequency | document_frequency |
| --- | --- | --- |
| limited | 44505 | 44309 |
| private | 41866 | 41595 |
| llc | 31563 | 31402 |
| ltd | 25193 | 25044 |
| inc | 23137 | 23002 |
| pvt | 13891 | 13809 |
| com | 11320 | 11281 |
| center | 11095 | 10928 |
| services | 8990 | 8849 |
| लिमिटेड | 8378 | 8378 |

Source/country-associated name tokens among each group's most frequent candidates (lift is relative document frequency):

| split | source | country | token | lift |
| --- | --- | --- | --- | --- |
| test | source1 | France | parents | 10.693 |
| test | source1 | France | amis | 9.535 |
| test | source1 | France | amicale | 9.295 |
| test | source1 | France | sarl | 9.214 |
| test | source1 | France | sas | 9.206 |
| test | source1 | France | ets | 9.050 |
| test | source1 | France | maison | 8.970 |
| test | source1 | France | du | 8.248 |
| test | source1 | France | sportive | 8.198 |
| test | source2 | France | développement | 8.097 |

Frequent tokens may create large candidate blocks; rare tokens may help recall. Inspect `token_disproportion.tsv`, `rare_tokens.tsv`, and `token_frequency_distribution.tsv` before selecting rules.

## Ground truth and true matches

Singleton share: 5.75% of ground-truth rows.

| measure | count |
| --- | --- |
| source1_rows | 22037 |
| match_pairs | 75824 |
| singletons | 1267 |
| one_match | 1210 |
| multiple_matches | 19560 |
| max_matches | 10 |

| source | field | measure | mean | median |
| --- | --- | --- | --- | --- |
| source2 | business_address | exact_equality | 0.000 | 0.000 |
| source2 | business_address | normalized_equality | 0.126 | 0.000 |
| source2 | business_address | token_jaccard_milli | 0.691 | 0.714 |
| source2 | business_name | exact_equality | 0.047 | 0.000 |
| source2 | business_name | normalized_equality | 0.211 | 0.000 |
| source2 | business_name | token_jaccard_milli | 0.611 | 0.667 |
| source3 | business_address | exact_equality | 0.043 | 0.000 |
| source3 | business_address | normalized_equality | 0.043 | 0.000 |
| source3 | business_address | token_jaccard_milli | 0.520 | 0.500 |
| source3 | business_name | exact_equality | 0.046 | 0.000 |
| source3 | business_name | normalized_equality | 0.227 | 0.000 |
| source3 | business_name | token_jaccard_milli | 0.631 | 0.667 |

Examples are in `true_match_examples.tsv`; inspect low-similarity examples before choosing normalization or blocking rules.

## Duplicates, blocking, and train/test shift

| split | source | type | groups | extra_records |
| --- | --- | --- | --- | --- |
| train | source1 | normalized_name_address | 0 | 0 |
| test | source1 | normalized_name_address | 0 | 0 |

Countries seen only in the analyzed test subset: France. Sampling can create apparent country shifts.
See `duplicate_summary.tsv`, `suspicious_duplicates.tsv`, `train_test_comparison.tsv`, and `data_quality_issues.tsv`.

Pairwise near-duplicate search, edit/TF-IDF similarity, and blocking experiments were skipped. Enable `--expensive-analysis` to run them.

## Data-quality warnings

- `test/source1/strange_character`: 5
- `test/source2/strange_character`: 16
- `test/source3/strange_character`: 11
- `train/source1/strange_character`: 6
- `train/source2/strange_character`: 27
- `train/source3/strange_character`: 21

## Interpretation notes

A normalized duplicate is a review candidate, not proof of identity. A true pair can have different country labels or weak text overlap. Randomly paired records in the optional baseline are illustrative and are not guaranteed negatives. Empty and whitespace-only fields are reported separately; literal 'NA' is preserved as text. No external data was consulted.
