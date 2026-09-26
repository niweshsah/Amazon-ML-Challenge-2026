# FAISS top-100 language mismatch analysis

This report checks whether different writing systems are overrepresented among the training true pairs that FAISS missed. It profiles **all 1,637,493 missed pairs** and compares them with a deterministic reservoir sample of retrieved true pairs (**30,000 per source**).

Script labels come from Unicode character names. A script mismatch means the predominant alphabetic script differs between the two records for the same field; the percentage denominator includes only pairs with a detectable script on both sides. “Non-Latin present” means at least one compared field contains a non-Latin script. This does not detect same-script transliteration (for example, transliterated words written in Latin on both sides).

## Results

| Pair group | Source | Pairs | Name script mismatch | Address script mismatch | Either field mismatch | Non-Latin in name pair | Non-Latin in address pair | Exact clean name | Exact clean address |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| missed_true_pairs | S2 | 708,873 | 35.6% (252,385) | 0.1% (653) | 35.6% (252,573) | 35.9% (254,522) | 16.5% (116,811) | 7.3% (51,400) | 7.2% (51,073) |
| missed_true_pairs | S3 | 928,620 | 16.7% (155,266) | 0.2% (1,733) | 16.9% (156,706) | 17.4% (161,554) | 12.0% (111,866) | 10.4% (96,427) | 2.6% (24,499) |
| retrieved_true_sample | S2 | 30,000 | 2.9% (875) | 0.0% (0) | 2.9% (875) | 3.2% (946) | 8.1% (2,423) | 28.2% (8,471) | 19.7% (5,920) |
| retrieved_true_sample | S3 | 30,000 | 1.2% (369) | 0.0% (5) | 1.2% (374) | 1.5% (435) | 7.8% (2,354) | 29.0% (8,700) | 5.0% (1,493) |

## Interpretation

Compare the `missed_true_pairs` rows with `retrieved_true_sample` rows within each source. A higher script-mismatch rate among misses supports cross-script language variation as a retrieval weakness. The two sources should be read separately because their data and baseline recall differ.

FAISS used `IndexFlatIP`, so these pairs were not lost to approximate-index search error: their dense cosine score ranked below the top 100. The embeddings were built from concatenated cleaned name and address text, without separate field weights. Language mismatch is one measured contributor; ranking competition and within-script spelling/address variation remain possible causes.

The retrieval comparison sample is a reservoir sample of the true pairs actually present in each top-100 candidate file. Rates are descriptive; differences should be interpreted with their denominators in the table.

Representative cross-script examples are in the summary TSV’s companion examples file from the earlier EDA. This report and the full missed-pair list contain training entity IDs and text only.
