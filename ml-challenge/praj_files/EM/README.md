# Entity matching preprocessing and BGE-M3 retrieval

This experiment keeps the original challenge dataset and existing prepared
datasets unchanged. Preprocessing and retrieval variants write to separate
directories under `praj_files/datasets` and `praj_files/output`.

## Install libpostal

The Python `postal` package wraps the native libpostal C library. On Debian or
Ubuntu, install the native build tools, compile/install libpostal and its parser
data, then install the Python binding:

```bash
apt-get update
apt-get install -y curl build-essential autoconf automake libtool pkg-config
git clone https://github.com/openvenues/libpostal.git /tmp/libpostal
cd /tmp/libpostal
./bootstrap.sh
./configure --datadir=/usr/local/share
make -j"$(nproc)"
make install
ldconfig
cd /workspace/ml-challenge
python3 -m pip install -r praj_files/EM/requirements-libpostal.txt
```

The libpostal build downloads its parser data. Keep enough free disk space for
that data and the additional preprocessed parquet files.

## Preprocess the original dataset

```bash
python3 praj_files/src/0_preprocess_libpostal.py \
  --input-dir student_resource/dataset \
  --output-dir praj_files/datasets/preprocessed_libpostal \
  --workers 4
```

The output keeps raw columns and adds `business_address_libpostal`,
`address_components_json`, parsed component columns, `address_parse_status`,
and a structured `combined_text_clean`. Rows retain source-file order. Failed
or empty parses fall back to the cleaned raw address; the parse status records
that fallback.

## Build Unicode-safe and Latin-script text views

The earlier Python cleanup expression removed Unicode combining marks, which
damaged some Indic-script text. Build a separate corrected dataset with the
original script retained and ICU transliterations added as
`business_name_latin` and `business_address_latin`:

```bash
python3 praj_files/src/blocking/7_build_latin_views.py \
  --input-dir praj_files/datasets/preprocessed \
  --output-dir praj_files/datasets/preprocessed_latin
```

This needs ICU's `uconv` command. Transliteration renders proper names in Latin
characters; it does not semantically translate names. The BGE text builder
includes both original and Latin views when those columns are present.

## Build BGE-M3 candidates

```bash
python3 -m pip install -r praj_files/EM/requirements-bge.txt
python3 praj_files/EM/6_bge_m3_faiss_search.py --top-k 100 --nprobe 512
```

This builds reusable compressed IVF-PQ indexes and writes separate candidate
TSVs for Sources 2 and 3 under `praj_files/output`. It defaults to the
libpostal-preprocessed data. The approximate-search `nprobe` can be increased
up to `--nlist` to inspect the recall/runtime tradeoff.

Run the transliteration experiment in an isolated output directory so baseline
indexes and candidates remain available for comparison:

```bash
python3 praj_files/EM/6_bge_m3_faiss_search.py \
  --data-dir praj_files/datasets/preprocessed_latin \
  --output-dir praj_files/output/bge_m3_latin_trial \
  --top-k 100 --nprobe 512
python3 praj_files/EM/faiss_retention_calc.py \
  --data-dir praj_files/datasets/preprocessed \
  --ground-truth praj_files/datasets/preprocessed/train_ground_truth.parquet \
  --s2-candidates praj_files/output/bge_m3_latin_trial/candidate_pairs_bge_m3_s2.tsv \
  --s3-candidates praj_files/output/bge_m3_latin_trial/candidate_pairs_bge_m3_s3.tsv \
  --top-ks 100 \
  --output praj_files/output/bge_m3_latin_trial_retention.tsv
```

Compare this pair recall with the baseline at the same `k`; do not infer final
leaderboard F0.5 from candidate recall alone.

Measure retention:

```bash
python3 praj_files/EM/faiss_retention_calc.py \
  --top-ks 5 10 20 30 50 75 100 \
  --s2-candidates praj_files/output/candidate_pairs_bge_m3_s2.tsv \
  --s3-candidates praj_files/output/candidate_pairs_bge_m3_s3.tsv
```

To compare a fixed candidate budget using both retrieval runs, interleave their
ranked lists and measure the resulting union:

```bash
python3 praj_files/EM/merge_candidate_generators.py --top-k 100
python3 praj_files/EM/faiss_retention_calc.py \
  --top-ks 5 10 20 30 50 75 100 \
  --s2-candidates praj_files/output/candidate_pairs_union_s2.tsv \
  --s3-candidates praj_files/output/candidate_pairs_union_s3.tsv \
  --output praj_files/faiss_retention_union_metrics.tsv
```
