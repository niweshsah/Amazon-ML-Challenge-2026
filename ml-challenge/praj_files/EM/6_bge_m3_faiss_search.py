"""Build BGE-M3 IVF-PQ indexes and retrieve training candidates per source.

This stores compressed FAISS indexes instead of full 1024-dimensional float
vectors. Candidate outputs are independent of the existing MiniLM FAISS files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import gc
from pathlib import Path
from typing import Any, Iterator

import faiss
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


SCRIPT_DIR = Path(__file__).resolve().parent
PRAJ_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PRAJ_DIR / "datasets" / "preprocessed_libpostal"
OUTPUT_DIR = PRAJ_DIR / "output"
MODEL_NAME = "BAAI/bge-m3"
TEXT_COLUMNS = ["entity_id", "business_name_clean", "business_address_clean", "country"]
DIMENSION = 1024


def copy_index_to_gpu(resources: Any, gpu_id: int, index: Any) -> Any:
    """Copy an IVF-PQ index using half-precision lookup tables for GPU limits."""
    options = faiss.GpuClonerOptions()
    # Some GPUs expose only 48 KB shared memory per block; the default
    # 64-subquantizer, 8-bit lookup table requires 64 KB in full precision.
    options.useFloat16 = True
    return faiss.index_cpu_to_gpu(resources, gpu_id, index, options)


def record_text(row: dict[str, Any]) -> str:
    name = str(row.get("business_name_clean") or "").strip()
    address = str(row.get("business_address_clean") or "").strip()
    name_latin = str(row.get("business_name_latin") or "").strip()
    address_latin = str(row.get("business_address_latin") or "").strip()
    parsed_address = str(row.get("business_address_libpostal") or "").strip()
    country = str(row.get("country") or "").strip()
    # Attribute labels keep the same surface string from being interpreted as
    # a name, address, or country depending on its position.
    latin_name_part = f" Latin name: {name_latin}." if name_latin and name_latin != name else ""
    latin_address_part = (
        f" Latin address: {address_latin}."
        if address_latin and address_latin != address
        else ""
    )
    parsed_part = f" Parsed address: {parsed_address}." if parsed_address else ""
    return (
        f"Business name: {name}.{latin_name_part} Address: {address}."
        f"{latin_address_part}{parsed_part} Country: {country}"
    ).strip()


def iter_records(path: Path) -> Iterator[list[dict[str, Any]]]:
    dataset = ds.dataset(path, format="parquet")
    available = set(dataset.schema.names)
    missing = set(TEXT_COLUMNS) - available
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    columns = list(TEXT_COLUMNS)
    if "business_address_libpostal" in available:
        columns.append("business_address_libpostal")
    columns.extend(
        column for column in ("business_name_latin", "business_address_latin")
        if column in available
    )
    for batch in dataset.to_batches(columns=columns, batch_size=20_000):
        yield batch.to_pylist()


def input_signature(path: Path) -> dict[str, Any]:
    # Parquet datasets produced by 0_preprocess.py are directories containing
    # one or more parquet fragments, so discover those fragments recursively.
    fragments = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not fragments:
        raise FileNotFoundError(f"No parquet fragments found in {path}")
    return {
        "path": str(path.resolve()),
        "fragments": [
            {"name": fragment.name, "size": fragment.stat().st_size, "mtime_ns": fragment.stat().st_mtime_ns}
            for fragment in fragments
        ],
    }


def encode(model: Any, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = np.ascontiguousarray(output["dense_vecs"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise ValueError(f"Expected BGE-M3 dense vectors with {DIMENSION} dimensions; got {vectors.shape}")
    faiss.normalize_L2(vectors)
    return vectors


def build_or_load_index(
    model: Any,
    source: int,
    source_dir: Path,
    args: argparse.Namespace,
) -> Any:
    index_path = args.output_dir / f"faiss_index_bge_m3_train_source{source}.index"
    metadata_path = Path(str(index_path) + ".json")
    expected_metadata = {
        "model": args.model,
        "dimension": DIMENSION,
        "index_type": "IndexIVFPQ",
        "metric": "inner_product",
        "nlist": args.nlist,
        "pq_subquantizers": args.pq_subquantizers,
        "pq_bits": args.pq_bits,
        "max_length": args.max_length,
        "source_data": input_signature(source_dir),
    }
    if not args.rebuild_index and index_path.exists() and metadata_path.exists():
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved == expected_metadata:
                print(f"[index S{source}] loading compressed index {index_path}", flush=True)
                return faiss.read_index(str(index_path))
        except (OSError, json.JSONDecodeError, RuntimeError):
            pass

    print(f"[index S{source}] selecting a deterministic sample for IVF-PQ training", flush=True)
    sample_texts: list[str] = []
    sample_modulus = max(2, int(args.sample_modulus))
    for rows in iter_records(source_dir):
        for row in rows:
            entity_id = str(row["entity_id"])
            digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
            if int.from_bytes(digest, "big") % sample_modulus == 0:
                sample_texts.append(record_text(row))
                if len(sample_texts) >= args.train_sample_size:
                    break
        if len(sample_texts) >= args.train_sample_size:
            break
    if len(sample_texts) < max(args.nlist, 256):
        raise ValueError(
            f"Only {len(sample_texts)} IVF-PQ training examples selected; need at least "
            f"max(nlist, 256)={max(args.nlist, 256)}. Lower --sample-modulus."
        )
    sample_vectors = encode(model, sample_texts, args.batch_size, args.max_length)
    del sample_texts

    quantizer = faiss.IndexFlatIP(DIMENSION)
    index = faiss.IndexIVFPQ(
        quantizer,
        DIMENSION,
        args.nlist,
        args.pq_subquantizers,
        args.pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    print(
        f"[index S{source}] training IVF-PQ on {len(sample_vectors):,} sampled vectors; "
        f"compressed code={args.pq_subquantizers * args.pq_bits // 8} bytes/vector",
        flush=True,
    )
    build_resources = None
    build_gpu_index = None
    if args.device.startswith("cuda"):
        build_resources = faiss.StandardGpuResources()
        gpu_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        build_gpu_index = copy_index_to_gpu(build_resources, gpu_id, index)
        index_to_fill = build_gpu_index
    else:
        index_to_fill = index
    index_to_fill.train(sample_vectors)
    del sample_vectors

    added = 0
    for rows in iter_records(source_dir):
        texts = [record_text(row) for row in rows]
        vectors = encode(model, texts, args.batch_size, args.max_length)
        index_to_fill.add(vectors)
        added += len(vectors)
        if added % 250_000 < len(vectors):
            print(f"[index S{source}] added {added:,} target vectors", flush=True)
        del texts, vectors
    if added != index_to_fill.ntotal:
        raise RuntimeError(
            f"Index row count mismatch for S{source}: added={added}, indexed={index_to_fill.ntotal}"
        )
    if build_gpu_index is not None:
        index = faiss.index_gpu_to_cpu(build_gpu_index)
        del build_gpu_index, index_to_fill, build_resources
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_index = Path(str(index_path) + ".tmp")
    faiss.write_index(index, str(temp_index))
    os.replace(temp_index, index_path)
    temp_metadata = Path(str(metadata_path) + ".tmp")
    temp_metadata.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    os.replace(temp_metadata, metadata_path)
    print(f"[index S{source}] saved compressed index {index_path} ({added:,} vectors)", flush=True)
    return index


def search_source(model: Any, source: int, args: argparse.Namespace) -> Path:
    source_dir = args.data_dir / f"train_source{source}.parquet"
    query_dir = args.data_dir / "train_source1.parquet"
    target_ids = pd.read_parquet(source_dir, columns=["entity_id"])["entity_id"].astype(str).to_numpy()
    query_ids = pd.read_parquet(query_dir, columns=["entity_id"])["entity_id"].astype(str).to_numpy()
    if len(target_ids) == 0 or len(query_ids) == 0:
        raise ValueError("Source 1 and target source must both contain records")

    cpu_index = build_or_load_index(model, source, source_dir, args)
    if cpu_index.d != DIMENSION or cpu_index.ntotal != len(target_ids):
        raise ValueError(
            f"S{source} index shape mismatch: {cpu_index.ntotal} vectors x {cpu_index.d}; "
            f"expected {len(target_ids)} x {DIMENSION}"
        )
    gpu_resources = None
    if args.device.startswith("cuda"):
        gpu_resources = faiss.StandardGpuResources()
        gpu_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        index = copy_index_to_gpu(gpu_resources, gpu_id, cpu_index)
        del cpu_index
    else:
        index = cpu_index
    index.nprobe = min(args.nprobe, args.nlist)
    output_path = args.output_dir / f"candidate_pairs_bge_m3_s{source}.tsv"
    temp_output = Path(str(output_path) + ".tmp")
    rows_done = 0
    with temp_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for rows in iter_records(query_dir):
            texts = [record_text(row) for row in rows]
            vectors = encode(model, texts, args.batch_size, args.max_length)
            _, indices = index.search(vectors, args.top_k)
            for query_id, row_indices in zip(query_ids[rows_done:rows_done + len(rows)], indices):
                candidate_ids = [target_ids[i] for i in row_indices if 0 <= i < len(target_ids)]
                writer.writerow([query_id, ",".join(candidate_ids)])
            rows_done += len(rows)
            if rows_done % 100_000 < len(rows):
                print(f"[search S{source}] queried {rows_done:,}/{len(query_ids):,}", flush=True)
            del texts, vectors, indices
    if rows_done != len(query_ids):
        raise RuntimeError(f"Wrote {rows_done} query rows, expected {len(query_ids)}")
    os.replace(temp_output, output_path)
    del index, gpu_resources
    print(f"[saved] {output_path}", flush=True)
    return output_path


def merge_source_outputs(s2_path: Path, s3_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "candidate_pairs_bge_m3.tsv"
    temp_path = Path(str(output_path) + ".tmp")
    with s2_path.open(encoding="utf-8", newline="") as s2_file, s3_path.open(
        encoding="utf-8", newline=""
    ) as s3_file, temp_path.open("w", encoding="utf-8", newline="") as output_file:
        s2_reader = csv.DictReader(s2_file, delimiter="\t")
        s3_reader = csv.DictReader(s3_file, delimiter="\t")
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        row_count = 0
        for s2_row, s3_row in zip(s2_reader, s3_reader):
            query_id = s2_row["source1_entity_id"]
            if query_id != s3_row["source1_entity_id"]:
                raise ValueError(f"S2/S3 candidate query order differs at row {row_count + 1}")
            candidates = [s2_row["candidate_entity_ids"], s3_row["candidate_entity_ids"]]
            writer.writerow([query_id, ",".join(value for value in candidates if value)])
            row_count += 1
        if next(s2_reader, None) is not None or next(s3_reader, None) is not None:
            raise ValueError("S2 and S3 candidate files contain different row counts")
    os.replace(temp_path, output_path)
    print(f"[saved] {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BGE-M3 dense retrieval candidates using compressed FAISS IVF-PQ indexes."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--pq-subquantizers", type=int, default=64)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--train-sample-size", type=int, default=100_000)
    parser.add_argument("--sample-modulus", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--faiss-threads", type=int, default=0, help="Set CPU FAISS thread count; 0 keeps the library default.")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write a combined TSV. This duplicates candidate IDs and needs several GB of disk space.",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.nlist <= 0 or args.nprobe <= 0:
        parser.error("top-k, nlist, and nprobe must be positive")
    if DIMENSION % args.pq_subquantizers != 0:
        parser.error("1024 must be divisible by --pq-subquantizers")
    if not 1 <= args.pq_bits <= 8:
        parser.error("--pq-bits must be between 1 and 8")
    if args.train_sample_size < args.nlist:
        parser.error("--train-sample-size must be at least --nlist")
    return args


def main() -> int:
    args = parse_args()
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "BGE-M3 requires FlagEmbedding. Install it with `pip install -U FlagEmbedding`."
        ) from exc
    if args.faiss_threads > 0:
        faiss.omp_set_num_threads(args.faiss_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} on {args.device} (fp16 when using CUDA)...", flush=True)
    model = BGEM3FlagModel(
        args.model,
        use_fp16=args.device.startswith("cuda"),
        devices=[args.device],
    )
    s2_path = search_source(model, 2, args)
    s3_path = search_source(model, 3, args)
    if args.write_combined:
        merge_source_outputs(s2_path, s3_path, args.output_dir)
    else:
        print("Per-source candidate TSVs are ready; skipped the extra combined TSV to save disk space.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
