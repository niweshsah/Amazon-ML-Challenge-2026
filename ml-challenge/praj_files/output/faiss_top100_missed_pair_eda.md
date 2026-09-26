# FAISS Top-100 Missed-Pair EDA

## Question and scope

This analysis asks why known training matches are absent from the multilingual MiniLM FAISS top-100 candidates. It uses a deterministic sample of 50,000 missed true pairs from each target source (S2 and S3), with normalized names and addresses. Similarity is token Jaccard: shared unique tokens divided by the union of unique tokens. “Strong” means Jaccard >= 0.5; “weak” means < 0.2. The underlying missed-pair list contains 1,637,493 pairs; the rates below describe the 100,000-pair sample, not every miss.

## Quantified patterns

| Signal among missed pairs | S2 (n=50,000) | S3 (n=50,000) |
|---|---:|---:|
| Exact normalized name | 7.3% | 10.5% |
| Exact normalized address | 7.2% | 2.7% |
| Either name or address missing on one side | 17.6% | 13.6% |
| Name strong, address weak | 16.8% | 21.5% |
| Name weak, address strong | 35.4% | 19.8% |
| Both name and address strong | 17.0% | 14.2% |
| Both name and address weak | 3.1% | 4.1% |
| Country disagreement | 0.0% | 0.0% |
| Median name token overlap | 0.20 | 0.50 |
| Median address token overlap | 0.55 | 0.40 |

The four name/address strength buckets do not sum to 100% because intermediate Jaccard values (0.2–0.5) are omitted. “Either field missing” is a separate field-quality measure. Country agreement in this sample suggests country mismatch is not a common explanation here.

## What the evidence suggests

- **Name variation is a major issue, especially for S2.** In 35.4% of sampled S2 misses, address overlap is strong while name overlap is weak. For S3, 21.5% have the reverse pattern: strong name overlap but weak address overlap.
- **Address formatting and component order vary.** Examples include street/city components moving, unit numbers appearing on only one side, and locality differences despite shared street tokens. Exact address rates are especially low for S3.
- **Multilingual and transliteration differences are plausible, but not directly quantified here.** These input fields are already cleaned; the existing sample metrics measure lexical overlap, not detected language or writing system. Transliteration can keep both records in Latin script while spelling differs, so script mismatch alone would undercount it.
- **Some pairs are difficult in both fields.** Only 3.1% (S2) and 4.1% (S3) have both name and address Jaccard below 0.2, so many misses retain useful evidence in at least one field. A candidate retriever that treats name and address separately or unions lexical and embedding retrieval may recover some of these.

## Interpretation and next step

These are missed-pair diagnostics, not a causal attribution of FAISS errors. Exact FAISS search ranks each pair by embedding similarity; a true pair outside rank 100 can still be semantically correct. The sample supports the user's multilinguality hypothesis as a likely contributor, alongside field-specific spelling, completeness, and address variation. To measure language mismatch itself, run language/script detection on both sides (including raw text if available), then compare its rate against a same-size sample of retrieved true pairs. Do not interpret candidate-pair recall as final leaderboard F0.5: the final matching/ranking pipeline and its false positives also determine that score.

## Supporting files

- `faiss_top100_missed_pair_eda_summary.tsv`: source-level sample statistics.
- `faiss_top100_missed_pair_eda_examples.tsv`: representative missed-pair examples and overlap scores.
- `faiss_top100_missed_pair_human_review.tsv`: five examples for each of five overlap patterns, separately for S2 and S3 (50 rows total). It includes both record IDs, names and addresses, predominant Unicode scripts for each field, tokens found only on each side, and a review hint identifying the weaker field. Script labels are descriptive annotations for these examples, not a prevalence estimate.
- `faiss_top100_missed_true_pairs.tsv`: full list of missed true pairs.

The token-only columns are a comparison aid, not proof that an extra token is erroneous; reviewers should check the full field values and entity context. Predominant script can expose cross-script pairs such as Latin versus Devanagari, but it cannot identify language or catch same-script transliteration differences.
