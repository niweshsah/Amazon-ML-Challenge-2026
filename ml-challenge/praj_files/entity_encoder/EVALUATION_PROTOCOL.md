# Evaluation protocol and observed zero-shot results

## 100,000-query full-source BGE-M3 retrieval audit

`evaluate_bge_100k.py` selected exactly 100,000 training T1 IDs by a fixed
Blake2b ordering. It excluded T1 hashes in `[0, 0.135)`: the first 0.12 is the
encoder training anchor partition and the next 0.015 contains the small model
selection slice. For these 100,000 T1 IDs, 94,359 have at least one ground
truth match and 5,641 are singletons. Their ground truth has 346,483 T2/T3
links.

The script audited previously generated, complete zero-shot BGE-M3 candidate
files. The underlying indexes cover all 5,034,616 training T2 and 5,285,603
training T3 rows; both candidate files contain all 2,206,821 training T1 rows.
It checked the stored index model, IVF-PQ type, source file names and fragment
sizes before scanning the candidates. Each file has the top 100 ranked
candidates **within that source**, so a T1 can have 200 total candidates.
The index uses cosine-compatible normalized BGE-M3 1,024-dimensional dense
vectors with FAISS IVF-PQ approximate search. The original run used 128-token
inputs from `business_name_clean` and `business_address_clean` with field
labels and country. The older cleaner can break Indic combining marks. This
audit measures that *whole pipeline*, including its preprocessing and
approximate index, not intrinsic BGE-M3 quality with raw Unicode text or
exact nearest-neighbor search.

Observed:

| Measure | Result |
| --- | ---: |
| True links retrieved in top 100 per source | 302,555 / 346,483 = 87.32% |
| Non-singleton T1 with at least one retrieved true link | 97.45% |
| Non-singleton T1 with every true link retrieved | 70.59% |
| True-link rate among all 20M candidates | 1.51% |
| Oracle macro F0.5 ceiling given this candidate set | 0.9430 |
| India pair recall | 76.45% |
| US pair recall | 94.56% |
| India queries with every true link | 52.36% |
| US queries with every true link | 82.68% |

The oracle F0.5 ceiling assumes perfect knowledge of which retrieved IDs are
true matches and correct empty predictions for singletons. It is an upper
bound, **not an achieved matching score**. These files contain no similarity
scores, so no similarity threshold, calibrated F0.5, or cross-source global
ranking can be computed from them. `best_per_source_rank_mrr` in the JSON uses
the best rank from either source list, not a globally merged ranking.

Full machine-readable results are in
`artifacts/zero_shot_bge_full_pool_100000.json`.

## Small raw-Unicode model-selection pool

The separate `prepare.py` validation set has 400 T1 queries, including 25
singletons. Its candidate pool contains every known match for those queries,
about 12k random T2/T3 distractors, and 6,076 additional same-name or
same-first-token targets from other entities (17,650 distinct targets total).
It uses raw Unicode names and addresses and 64-token inputs. The model scores
every query against every target in this small pool, computes Recall@1/5/10/50
and MRR, and searches similarity thresholds 0.30 to 0.95 in steps of 0.025
for the best empty-truth-aware macro F0.5. Subsets include Devanagari,
non-ASCII, missing fields, and single-field agreement.

This pool is useful for selecting a checkpoint and detecting regressions. It
is much easier than the 10.32M-target search above: known positives are
inserted and the pool is small. Its F0.5 is therefore not a leaderboard
estimate. Zero-shot results with the same 64-token setting were:

| Model | Recall@5 | MRR | Best macro F0.5 |
| --- | ---: | ---: | ---: |
| BAAI/bge-m3 | 0.9973 | 0.9963 | 0.9341 |
| karmx/Llama-Karmx-Indic-Embedding-bge-m3 | 0.9973 | 0.9976 | 0.9257 |

The Indic checkpoint's lower small-pool F0.5 does not rule it out. Its model
card reports better Indic passage retrieval, the full-source BGE audit has a
large India gap, and the user requested fine-tuning the Indic checkpoint.
The encoder training run uses that checkpoint and keeps the best measured
validation adapter.
