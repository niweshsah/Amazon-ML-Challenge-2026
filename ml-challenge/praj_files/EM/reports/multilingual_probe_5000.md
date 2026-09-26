# Cross-script candidate recall probe

## Setup

- Seed 42 reservoir sample of 5,000 Source 1 training records.
- Full Source 2 and Source 3 training files scanned; 17,118 sampled ground-truth pairs.
- Baseline: existing BGE-M3 FAISS candidates, 100 per source/query.
- Additional route: ICU `Any-Latin; Latin-ASCII` romanization of non-Latin targets, name word and 4-gram blocks, and address number blocks; at most 100 added candidates per source/query.
- Labels are used only to score retrieval and to classify true-pair scripts, never to rank candidates.

## Candidate results

| Candidate route | True pairs found | Pair recall | All-match retention on non-singletons | Perfect-matcher macro F0.5 ceiling | Candidates |
|---|---:|---:|---:|---:|---:|
| BGE-M3 | 15,009 | 87.68% | 70.82% | 94.17% | 1,000,000 |
| Romanized blocks | 1,318 | 7.70% overall | 1.14% | 18.45% | 975,813 |
| Union | 15,225 | 88.94% | 72.74% | 94.96% | 1,966,523 |

The romanized route searches non-Latin targets only. Its low overall recall is expected. Its incremental value is 216 true pairs over BGE-M3, but it nearly doubles the sampled candidate count, so it needs stronger ranking or pruning before full deployment.

| True-pair target name script | True pairs | BGE-M3 recall | Union recall | New true pairs |
|---|---:|---:|---:|---:|
| Latin | 15,955 | 93.59% | 93.90% | 49 |
| Devanagari | 653 | 7.81% | 19.14% | 74 |
| Tamil | 66 | 6.06% | 21.21% | 10 |
| Other non-Latin | 444 | 4.73% | 23.42% | 83 |

Non-Latin name pairs account for 1,163 of 17,118 true pairs (6.8%), but 1,087 of BGE-M3's 2,109 missed pairs (51.5%). The Tamil slice contains only 66 true pairs, so its percentage is less stable than the larger slices.

Source 2 contained 836,231 records with non-Latin name or address text; Source 3 contained 686,482. The additional route performed 32,987,554 and 26,079,641 distinct block comparisons for Sources 2 and 3 respectively on the 5,000-query sample.

## Diagnosis

Among the 653 Devanagari true pairs, 581 share at least one romanized name or address block with their Source 1 query, yet only 93 appear in the additional route's top 100. Similar counts are 51 shared / 12 retrieved for Tamil, and 385 shared / 91 retrieved for other non-Latin scripts. This indicates substantial ranking and candidate-budget loss after block coverage.

ICU romanization is systematic but differs from English spellings in these records. For example, `स्वस्तिक कंसल्टेंट्स` becomes `svastika kansaltentsa`, while its Source 1 match is `Swastik Consultants`. The next probe should use native-to-Roman transliteration trained for common English spellings (for example, IndicXlit), and combine name similarity with strong address evidence before applying top-k. Compare this route at the same total candidate budget and report recall by script.

The existing `paper_blocking_probe.py` uses ASCII conversion and discards Indic characters. Its earlier results are not a cross-script evaluation.

## Reproduce

```bash
python3 praj_files/EM/multilingual_candidate_probe.py \
  --sample-size 5000 --seed 42 --top-k 100 --max-query-frequency 30 \
  --output praj_files/EM/reports/multilingual_probe_5000.tsv
```
