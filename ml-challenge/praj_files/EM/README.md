# Entity matching preprocessing and BGE-M3 retrieval

This experiment keeps the original challenge dataset and the existing
`praj_files/datasets/preprocessed` output unchanged. The libpostal preprocessor
writes a separate dataset to `praj_files/datasets/preprocessed_libpostal`.

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

The output keeps raw columns and the current regex-cleaned name/address, then
adds `business_address_libpostal`, `address_components_json`, parsed component
columns, `address_parse_status`, and a structured `combined_text_clean`. Rows
are streamed in chunks and retain source-file order. Failed or empty parses
fall back to the cleaned raw address; the parse status records that fallback.

## Build BGE-M3 candidates

```bash
python3 -m pip install -r praj_files/EM/requirements-bge.txt
python3 praj_files/EM/6_bge_m3_faiss_search.py --top-k 100 --nprobe 512
```

This builds reusable compressed IVF-PQ indexes and writes separate candidate
TSVs for Sources 2 and 3 under `praj_files/output`. It defaults to the
libpostal-preprocessed data. The approximate-search `nprobe` can be increased
up to `--nlist` to inspect the recall/runtime tradeoff.

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
