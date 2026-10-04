# Business entity resolution: dataset findings and action plan

This report interprets [PS.pdf](../../../PS.pdf) using the existing analysis in this folder. It is an **engineering plan**, not a measured model score. The analysis was run on a seeded 1% sample of Source 1 and test records; training Source 2/3 include all known matches for sampled Source 1 plus sampled background records. This makes the training Source 2/3 sample match-enriched. Counts labelled *full input rows* came from scanning the full TSV files; most other statistics describe the analyzed subset. No expensive blocking experiment or held-out model evaluation was run.

![Dataset overview](ps_figures/ps_dataset_overview.png)

![Matching diagnostics](ps_figures/ps_matching_diagnostics.png)

## 1. What the challenge requires

For **every test Source 1 ID**, return zero or more IDs from **test Source 2 and Source 3** representing the same business. Source 1 is the deduplicated reference, but the output is not a one-to-one assignment: one Source 1 record can link to several S2 and S3 records. Produce two tab-separated files, `matching_results.tsv` and `candidate_pairs.tsv`, each with exactly one row per test Source 1 ID. The candidate file must contain the **final set actually scored by the model**; every predicted match must occur in that row's candidate list. Empty lists are valid and necessary. The score is **macro F0.5 over Source 1 entities**, including singletons. A singleton scores 1 only if its predicted list is empty.

The challenge bans external business lookups, geocoding APIs, and outside data augmentation. The final model must satisfy the stated MIT/Apache-2.0 license and at-most-8B-parameter rule. The final zip needs runnable code, pinned environment, both outputs, and the methodology document. Run the provided validator before packaging.

## 2. Dataset scale and sampling limits

| Split | S1 full rows | S2 full rows | S3 full rows | Total source rows | Analyzed S1 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 12,527,040 | 22,037 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | 11,702,133 | 17,194 |

There are **24,229,173 source records** across the six files. An unrestricted test S1 × (S2 + S3) join would form about **17.27 trillion pairs**. Blocking, bounded indexes, streaming or chunking, and measured memory use are essential. The sampled training S2/S3 rows are 1.73% of their respective full files, while the sampled test rows are about 1%; never interpret the different sampled counts as a production source ratio. The analysis backend marked `gpu`, but only batched string-length calculations used GPU; the analysis does **not** measure GPU matching throughput.

## 3. Findings and what to handle

| Priority | Observed evidence | Required handling | Acceptance check |
| --- | --- | --- | --- |
| **P0** | 1.73m test S1 and 9.97m test S2/S3 records | Build scalable multi-pass candidate generation; cap huge common-token postings without discarding all alternate routes | Log candidates per S1, pair count, runtime, RAM, and **pair recall** on validation |
| **P0** | 19,560/22,037 sampled training S1 have >1 match; 17,681 have matches in both S2 and S3 | Score S1–S2 and S1–S3 independently and keep every pair above calibrated acceptance criteria | Report macro F0.5 and recall separately for S2 and S3; no forced top-1 or global one-to-one constraint |
| **P0** | 1,267/22,037 sampled training S1 are singletons (5.75%) | Allow an empty predicted list; tune a conservative decision threshold for macro F0.5 | Report singleton accuracy and false-positive rate as well as precision/recall/F0.5 |
| **P0** | France occurs in test only; 2,603/17,194 sampled test S1 are French (15.1%) | Treat country as an open string; support Unicode and French address/name patterns without relying on a France training label | Every France S1 gets an output row; inspect candidate counts for France; leave unknown labels processable |
| **P1** | S2/S3 address missing in 3.83%/3.69% of sampled training rows and 2.79%/2.73% of sampled test rows; 3,305 true pairs have missing target address | Keep a **name-only candidate and scoring path** and explicit missingness features; never treat two empty addresses as evidence of identity | Pair recall for true matches with missing address; no forced nonmatch merely because address is empty |
| **P1** | Among true pairs, normalized name equality is only 21.1% (S2) and 22.7% (S3); normalized address equality is 12.6% (S2) and 4.3% (S3) | Use fuzzy name/address features and multiple blocking keys; exact normalized equality can be a strong feature but not the only retrieval rule | Pair recall by low-similarity strata and by source |
| **P1** | Normalized same-name/different-address cases occur; S1 itself has repeated names but no sampled normalized name+address duplicates | Keep address and other context when deciding identity; do not merge on name alone | Inspect hard negatives with shared name and different address; measure false merges |
| **P2** | 86 sampled rows were flagged for unusual characters (54 train, 32 test); examples include Devanagari and French accents | Normalize Unicode carefully while preserving original text and record IDs; use char n-grams/transliteration only when helpful | Round-trip IDs and test retrieval on accent/script and typo cases |

The 5.75% singleton rate is from sampled **training S1**. It is not an estimate of the hidden test singleton rate. Likewise, the 100% country agreement among **75,824 joined sampled true pairs** supports a same-country blocking pass, but it does not prove all future true pairs agree; measure the effect of a strict country filter on a proper held-out split. The existing generic analysis note says true pairs *can* disagree on country; none did in the sampled labels.

### Text and address normalization

Use one deterministic, reversible pipeline for train and test: Unicode NFKC/case folding, whitespace and punctuation handling, tokenization that retains accented and Indian-script letters, and original strings alongside normalized versions. Normalize common variants such as `Rd/Road`, `St/Street`, `Ave/Avenue`, `&/and`, and legal suffixes (`Pvt/Private`, `Ltd/Limited`, `Corp/Corporation`) as **additional features or alternate search forms**. Do not erase legal suffixes or house numbers from every representation: they may distinguish otherwise similar businesses. Preserve numeric components, postal codes, units, locality, and state; compare components separately when present. Partial addresses and reordered components make token, character n-gram, and edit-distance signals more useful than raw equality.

There is source-specific noise. In true pairs, Source 2 address median token Jaccard is **0.714**, but Source 3 is **0.500**; address exact equality is **0%** for S2 and **4.3%** for S3. Names have median token Jaccard **0.667** for both target sources. Some real pairs have zero token overlap in the name at the 5th percentile, so a single shared-name-token rule creates a recall ceiling. Source 3 addresses particularly need robust fuzzy retrieval and partial-address comparison. These similarities are **positive-pair diagnostics**; without negative-pair measurements they do not define a safe threshold.

Frequent name tokens in the analyzed sample include `limited`, `private`, `llc`, `ltd`, `inc`, and `pvt`. Common legal/company words form large, ambiguous blocks. Weight tokens by document frequency or omit them from *individual blocking keys*, while keeping their presence as matching features. Rare tokens, address numbers, locality, and postal fragments can narrow blocks. Use several candidate paths (for example uncommon name tokens, character n-grams/TF-IDF nearest neighbors, address number plus location, and a fallback for missing addresses), then union and deduplicate their candidate IDs. Each path must be evaluated for **incremental pair recall** and candidate cost; these are proposed experiments, not results of this analysis.

### Country shift and unseen vocabulary

Sampled train S1 is 60.6% US and 39.4% India; sampled test S1 is 38.4% US, 46.4% India, and 15.1% France. France has no training labels. Across all sampled source records, **57.5% of distinct test name tokens** and **54.6% of distinct test address tokens** were unseen in the sampled train vocabulary. These are *unique-token shares*, not the share of test rows or token occurrences affected; sample size and France inflate them. Avoid a fixed US/India category encoder or a dictionary that rejects unseen strings. Favor character and token similarity that transfers to new languages, with train-only learned statistics and a fallback when country-specific parsing fails. French legal terms such as `SARL` and `SAS`, and `rue`-style addresses, should survive preprocessing even without labeled French examples.

### Duplicates, links, and negatives

The sample contains no duplicate entity IDs or exact duplicate rows in the six source files. Source 1 has no sampled duplicate normalized name+address pairs, but **722 extra training S1 rows share a normalized name** with another S1 row. Thus same-name candidates can be distinct businesses. In sampled train S2 and S3 there are respectively **609** and **485** extra records sharing normalized name+address with another row in that source. They may each be legitimate IDs to output; do not collapse them unless an explicit mapping preserves every original ID. The sampled test S2/S3 duplicate counts are much smaller, which may reflect different sampling methods and should not be treated as a true distribution shift without a full-data check.

Training labels contain 75,824 S1-to-target links for 22,037 sampled S1 rows, an average of **3.44 links per S1**. Of these, 36,483 are S2 and 39,341 are S3. For training a pair classifier, construct negatives from the candidate pool, especially same-country shared-name and shared-address hard negatives; random pairs alone are too easy. Keep all known positives for sampled S1 during validation and evaluate ranking/retrieval separately from final pair classification. If a target record appears in more than one S1 candidate set, avoid assuming a global one-to-one target assignment unless the labels demonstrate that constraint.

## 4. Recommended end-to-end pipeline

1. **Create validation groups from training S1 IDs.** Split S1 IDs, retaining all their true S2/S3 links. Avoid testing on label-enriched background as though it were an unbiased full source. For a credible candidate benchmark, query an index over the **full training S2/S3 files** or clearly label the result as a sample-only diagnostic. Preserve repeated names and near duplicates in validation.
2. **Normalize and index all three sources.** Keep raw fields, derived fields, source prefix, country, and missingness flags. Build disk-backed, bounded postings or approximate nearest-neighbor indexes. Ensure all transforms are fitted on training data where they learn statistics; deterministic text normalization can run on test.
3. **Generate candidates in several passes.** Start with same-country search because all sampled true pairs agree, but quantify the recall trade-off before enforcing it exclusively. Combine informative name tokens, character n-gram neighbors, and address/location keys; include a name-only route when address is missing. Deduplicate by `(source1_id, target_id)`. Measure pair recall, S1 coverage, average/p95/max candidates per S1, and total model-scored pairs. Count recall separately for S2, S3, India, US, missing-address pairs, and low-overlap positives. France has no labeled recall; inspect its candidate count distribution and examples.
4. **Score each candidate pair.** Begin with a fast, interpretable baseline such as gradient-boosted trees over token/character similarities, normalized equality, edit distance, address-number agreement, locality/postal overlap, source indicator, country agreement, and missingness. Source-specific calibration or thresholds may help because S2/S3 address quality differs. Compare with an allowed embedding or reranking model only after measuring incremental F0.5 against runtime and memory.
5. **Tune the decision rule for the actual metric.** Select thresholds on held-out S1 groups by **macro F0.5**, not pairwise accuracy. Check precision and recall, singleton accuracy, and false merges among same-name businesses. No automatic top-1: retain all credible links and allow zero links. Recheck performance at several thresholds and across sources/countries.
6. **Run full test inference and validate artifacts.** Stream or batch through every test S1 ID. Write exact headers and tabs, unique S2/S3 IDs only, and blank lists for singletons. Write the exact final candidate pool scored by the model; verify predicted links are a subset. Run `python3 student_resource/utils/validate_submission.py --matching <path>/matching_results.tsv --candidate <path>/candidate_pairs.tsv --test-dir student_resource/dataset/test` (adjust paths to the actual data location, and use `--check-ids` if resources permit). Package reproducible code and methodology.

## 5. Experiments to run next, in order

| Experiment | Why it matters | Decision metric |
| --- | --- | --- |
| Full-train candidate benchmark for several blocking passes | Current analysis explicitly skipped blocking experiments; recall ceiling is unknown | True-pair recall at fixed candidates/S1 and runtime/RAM |
| Hard-negative pair classifier baseline | Positive similarity alone cannot set thresholds | Held-out macro F0.5, precision, singleton accuracy |
| Address-missing and weak-overlap ablation | Prevents systematic misses in known difficult pairs | Slice pair recall and F0.5 |
| Strict vs soft country gating | Sampled labels show 100% agreement; test has unseen France | Held-out pair recall and France candidate coverage |
| Source-specific feature/threshold comparison | S3 addresses have weaker overlap | S2/S3 precision and recall; aggregate macro F0.5 |
| Test-scale dry run | 11.7m test records create an engineering constraint | Wall time, peak memory, total scored pairs, validator PASS |

## 6. Analysis limitations and cautions

- The existing `true_match_similarity.tsv` measure `token_jaccard_milli` has means/medians in **0–1 units**, but its reported `std` is roughly 257/249/337/322, evidently scaled differently. This report uses only its mean/median and percentiles. Recompute standard deviations before using them.
- Normalized duplicate groups are review candidates, not proof of identity. Exact text can also describe two distinct businesses at different locations.
- No held-out precision, recall, macro F0.5, blocking recall, or France label quality is available in `analysis_output`. The charts are descriptive and must not be presented as model performance.
- Full-file row counts were scanned, but most text, missingness, label, and duplicate numbers are sampled. The training target sample is deliberately enriched with positives, so pair prevalence and source-level proportions cannot be extrapolated directly.

**Rebuild charts:** `python3 basic_python_scripts/niwesh_dataset_codes/analysis_output/make_ps_report_charts.py` from repository root. Source tables: `dataset_summary.tsv`, `country_summary.tsv`, `column_summary.tsv`, `ground_truth_summary.tsv`, and `true_match_similarity.tsv` in this directory.
