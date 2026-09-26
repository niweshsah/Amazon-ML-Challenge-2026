# Dense encoder v2

This isolated experiment trains an Indic BGE-M3 bi-encoder and evaluates it on a **sampled target pool**. The final score is macro F0.5 on 80,000 held-out Source 1 queries. The threshold is selected using another 20,000 held-out queries. A separate 2,000-query development split selects the checkpoint. Every held-out true target is included in the evaluation pool, along with 500,000 fixed distractors. These numbers are not full-source results.

The base model is the cached [karmx/Llama-Karmx-Indic-Embedding-bge-m3](https://huggingface.co/karmx/Llama-Karmx-Indic-Embedding-bge-m3), pinned by the local Hugging Face snapshot. Raw Unicode name, address, and country are serialized with field labels. No transliteration is applied; cross-script results are reported as a measured subset.

From the repository root, inside `./run_qwen.sh`:

```bash
python3 praj_files/dense_encoder_v2/pipeline.py prepare
python3 praj_files/dense_encoder_v2/pipeline.py verify-data
python3 praj_files/dense_encoder_v2/pipeline.py train
python3 praj_files/dense_encoder_v2/pipeline.py resume  # only after an interrupted train
python3 praj_files/dense_encoder_v2/pipeline.py dev-eval
python3 praj_files/dense_encoder_v2/pipeline.py build-index
python3 praj_files/dense_encoder_v2/pipeline.py infer --top-k 20
python3 praj_files/dense_encoder_v2/pipeline.py score --top-k 20
python3 praj_files/dense_encoder_v2/pipeline.py validate
```

`bash praj_files/dense_encoder_v2/run_all.sh` runs all stages and captures terminal output in `artifacts/logs/terminal_<run-id>.log`. Each command also appends structured, UTC timestamped events with the same run ID to `artifacts/logs/events.jsonl`, including configuration, split counts, training loss and memory, the 50-step throughput estimate, checkpoint decisions, embedding throughput and truncation, index progress, search throughput, threshold, score, validator output, and failures. Artifacts in `artifacts/` include the split manifest, IDs and pair tags in Parquet, checkpoint adapter and optimizer/RNG state, vector caches, FAISS index, metrics, and candidate/match TSVs. The directory is ignored by Git.

The separate plot watcher refreshes figures every minute. It writes PNG and SVG files under `artifacts/plots/<run-id>/`: pipeline timing, live training loss and GPU memory, and development retrieval F0.5. At every saved checkpoint it creates `checkpoints/checkpoint_<step>.png` and `.svg`, plus `checkpoint_summary.csv`. After scoring, `validation_dashboard` shows Recall@K, the calibration threshold curve, held-out F0.5/precision/recall, and difficult subsets. Before scoring, validation panels explicitly say pending. Start the watcher after the run begins:

```bash
nohup python3 -u praj_files/dense_encoder_v2/plot_run.py --watch --interval 60 \
  > praj_files/dense_encoder_v2/artifacts/logs/plotter.log 2>&1 < /dev/null &
```

To regenerate a finished run manually:

```bash
python3 praj_files/dense_encoder_v2/plot_run.py --run-id <run-id>
```

Defaults are 200,000 pairs, batch size 128, 96 tokens, one epoch, an 80-minute training cap, Top-20, and IVF Flat search. `--pairs 250000` may be used only if the logged 50-step estimate and remaining evaluation budget permit it. The trainer automatically reduces to 64 tokens if the profile projects over the training cap; then pass `--max-length 64` to resume, build, and infer. Full-source indexing is outside this experiment.

Run focused tests with:

```bash
python3 -m unittest discover -s praj_files/dense_encoder_v2 -p 'test_*.py' -v
```

The generated `validation_dataset/` contains the evaluation IDs and lets the repository submission validator check the two TSVs, including target ID existence. These TSVs are diagnostic held-out outputs, not a test-set submission.
