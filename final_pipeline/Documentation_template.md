# Business entity resolution methodology

## Methodology and data

We match Source 1 independently to records from Sources 2 and 3. Decisions permit zero, one, or multiple matches. Only provided business records and pretrained model assets are used; no business lookup, geocoding, or outside augmentation is performed. Country remains an unrestricted string, including unseen France labels.

Training samples Source 1 by exact percentage using seeded SHA-256 ranking; all links and singletons for selected groups are retained. Target pools contain all selected positives plus the configured percentage of remaining records, separately for Sources 2 and 3. Grouped splits use 70% fitting and 10% each development, calibration, and audit with largest-remainder rounding. Fitting supervision excludes all held-out positive targets, including candidates that would otherwise become negative examples. If a positive target is shared with a held-out group, it stays in the retrieval pool but is excluded from fitting. This protects supervision without using labels to change candidate recall.

## Candidate generation and blocking

The bi-encoder uses `karmx/Llama-Karmx-Indic-Embedding-bge-m3` with PEFT LoRA, CLS pooling, and L2 normalization. Inputs preserve raw Unicode names, addresses, and country. Separately, NFKC/casefolded Unicode views retain combining marks; transliterated views use Unidecode. Character-trigram TF-IDF searches both lexical views. Each target source independently yields Top-150 dense candidates and Top-150 positive-score lexical candidates. Their exact union is deduplicated, scored, and exported, preserving channel scores and ranks internally. Ground-truth matches are never injected into inference candidates. Supervised neural training retains positives separately when retrieval misses them.

Embedding shards are resumed only after key and checksum verification. Dense search is exact for small pools and compressed IVF-PQ for larger pools; training positions are sampled deterministically across all target shards. Lexical products and pair features use bounded batches. The disk-backed pair table avoids a complete pair matrix in RAM. Full test pools and every test query are processed regardless of training sample percentage.

## Model architecture and features

The feature allowlist includes dense and TF-IDF similarities; per-field normalized equality, sequence agreement, token Jaccard, transliterated sequence agreement and missingness; and address numeric agreement. Empty fields and empty numeric sets do not contribute equality evidence. Country is compared as text. IDs, labels, split assignments, sample tags, and pool roles are excluded.

XGBoost fits retrieval pairs from fitting groups with development early stopping. Training uses external-memory quantile matrices. The bi-encoder fits binary matching loss over explicit positive and lexical hard-negative pairs, avoiding false-negative assumptions between multiple positives in a batch. The cross-encoder defaults to `BAAI/bge-reranker-v2-m3`, trained with supervised binary matching loss over fitting positives and retrieval hard negatives. Neural models use development binary loss for checkpoint selection. Tokenizer/base, serialization, pooling and maximum-length compatibility are validated when loading adapters.

## Calibration and decisions

Calibration groups are hash-ranked into two disjoint halves. One half fits separate isotonic calibrators for XGBoost and cross-encoder scores. The other selects lower/upper XGBoost thresholds and a cross-encoder acceptance threshold over a configurable grid by complete-pipeline macro F0.5, including singletons and retrieval misses. Ties prefer higher global pair precision, then fewer routed pairs, then stable grid order. Below the lower threshold we reject; above the upper threshold we accept. Every intermediate pair, including boundary scores, is routed. Missing required cross-encoder assets stop routed inference. Model fingerprints bind the policy to the scored checkpoints.

## Evaluation, reproducibility and limitations

Audit groups are never used for fitting or calibration. Reporting includes candidate pair recall, macro F0.5, pair precision/recall, singleton results, country/missing-field/Unicode subgroups, target-source results, routing volume, stage runtime and peak process RSS. Metrics are explicitly labelled sampled training target pools and must not be interpreted as full-pool quality. A singleton scores 1 only with an empty prediction; otherwise its score is 0.

Configuration, input/model fingerprints, split manifests, checkpoints, calibration policy, atomic outputs, dependency lock, and migration inventory persist with the run. Local synthetic fixture tests exercise reproducibility, split isolation, Unicode, routing, singleton/multiple matches, unseen countries, exact candidate containment, recovery and stale-cache rejection. Fixture models are deterministic substitutes and do not establish trained-model quality. Real CUDA smoke, challenge training, full inference, licensed model review, and resource measurements must be run on the server. Model cards declare Apache-2.0 for the configured defaults; verify the exact server snapshots and any replacements against the challenge constraints.

## Submission

The two TSV headers are `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`. Both contain one row per complete test Source 1 entity, unique existing target IDs, and empty strings for empty lists. Every match is a candidate. Packaging validates these rules and reproduces the structure in `docs/PS.pdf`, including runnable code/environment and model assets.

## References

- [Challenge problem statement](docs/PS.pdf)
- [Indic encoder model card](https://huggingface.co/karmx/Llama-Karmx-Indic-Embedding-bge-m3)
- [BGE multilingual reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [XGBoost external-memory documentation](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/external_memory.html)
- Historical evidence: `docs/evidence/`; preservation checksums: `docs/migration_inventory.json`.
