# Paper-inspired blocking probe

Reference: Papadakis et al., *How to reduce the search space of Entity Resolution: with Blocking or Nearest Neighbor search?*, arXiv:2202.12521v5.

## Setup

- Seed 42 reservoir sample of 1,000 Source 1 training records; 49 have no true matches.
- Search all 5,034,616 Source 2 and 5,285,603 Source 3 training records.
- Each method retains at most 100 candidates per query per target source.
- Paper-inspired probe: normalized name and address token blocks, exact normalized name blocks, optional name 4-gram blocks, inverse query block frequency weighting, per-query top-k pruning. This is an adaptation of the paper's blocking workflow, **not** a reproduction of its full meta-blocking algorithm.
- BGE-M3 baseline: existing `praj_files/output/candidate_pairs_bge_m3_s2.tsv` and `_s3.tsv`.
- No ground-truth labels are used during retrieval. Ground truth is read only for evaluation.

## Results

| Candidate source | True pairs found / 3,458 | Pair recall | Non-singleton queries with all matches | Oracle macro F0.5 | Candidates |
|---|---:|---:|---:|---:|---:|
| Paper-style blocks | 2,983 | 86.26% | 69.40% | 93.23% | 200,000 |
| BGE-M3 | 3,011 | 87.07% | 70.14% | 93.74% | 200,000 |
| Union | 3,168 | 91.61% | 80.65% | 95.88% | 378,416 |

Oracle macro F0.5 assumes a perfect matcher selects every true pair present in the candidate set and no false pairs. Singleton queries get 1.0 under the challenge rules. It is a ceiling for this sampled candidate set, not a prediction of achievable classifier performance or the leaderboard score.

The probe makes 119,618,083 distinct block comparisons for Source 2 and 133,672,262 for Source 3 on just 1,000 queries. This variant is too expensive to extrapolate directly to all Source 1 records. Before a full run, add target-side block frequency counts/purging and measure marginal recall against the existing BGE-M3 candidates at a fixed candidate budget.

## Reproduce

```bash
python3 praj_files/EM/paper_blocking_probe.py \
  --sample-size 1000 --seed 42 --top-k 100 --qgrams \
  --baseline-s2 praj_files/output/candidate_pairs_bge_m3_s2.tsv \
  --baseline-s3 praj_files/output/candidate_pairs_bge_m3_s3.tsv \
  --output praj_files/EM/reports/paper_blocking_probe_1000.tsv
```

The next useful diagnostic is to inspect the 290 true pairs still missing from the union, grouped by country, source, name/address missingness, and whether they share any normalized token or name 4-gram. That separates block coverage failures from top-k pruning failures.
