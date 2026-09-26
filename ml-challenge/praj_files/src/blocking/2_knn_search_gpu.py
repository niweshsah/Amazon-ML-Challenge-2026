"""Generate exact GPU cosine top-k candidates from cached sparse TF-IDF matrices."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from cuml.neighbors import NearestNeighbors


PRAJ_DIR = Path(__file__).resolve().parents[2]


def load_ids(data_dir: Path, source: int) -> np.ndarray:
    frame = pd.read_parquet(data_dir / f"train_source{source}.parquet", columns=["entity_id"])
    return frame["entity_id"].astype(str).to_numpy()


def search_source(
    source1_matrix: sparse.csr_matrix,
    source1_ids: np.ndarray,
    source: int,
    data_dir: Path,
    tfidf_dir: Path,
    output_dir: Path,
    top_k: int,
    chunk_size: int,
) -> Path:
    target_ids = load_ids(data_dir, source)
    target_path = tfidf_dir / f"train_s{source}_tfidf.npz"
    target_matrix = sparse.load_npz(target_path).astype(np.float32, copy=False)
    if target_matrix.shape[0] != len(target_ids):
        raise ValueError(
            f"S{source} matrix has {target_matrix.shape[0]:,} rows but IDs have {len(target_ids):,}"
        )
    if source1_matrix.shape[0] != len(source1_ids):
        raise ValueError(
            f"S1 matrix has {source1_matrix.shape[0]:,} rows but IDs have {len(source1_ids):,}"
        )

    print(f"[S{source}] copying {target_matrix.shape} sparse target matrix to GPU", flush=True)
    target_gpu = cupy_sparse.csr_matrix(target_matrix)
    del target_matrix
    gc.collect()

    index = NearestNeighbors(
        n_neighbors=top_k,
        metric="cosine",
        algorithm="brute",
        output_type="cupy",
    )
    index.fit(target_gpu)
    del target_gpu
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"candidate_pairs_tfidf_s{source}.tsv"
    temporary_path = Path(str(output_path) + ".tmp")
    total = len(source1_ids)
    started = time.monotonic()
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                query_gpu = cupy_sparse.csr_matrix(source1_matrix[start:stop])
                _, neighbor_indices = index.kneighbors(query_gpu)
                cp.cuda.Stream.null.synchronize()
                indices = cp.asnumpy(neighbor_indices)
                for query_id, row_indices in zip(source1_ids[start:stop], indices):
                    valid_indices = row_indices[row_indices >= 0]
                    candidates = target_ids[valid_indices]
                    writer.writerow([query_id, ",".join(candidates)])
                del query_gpu, neighbor_indices, indices
                done = stop
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (total - done) / rate if rate else 0.0
                print(
                    f"[S{source}] queries {done:,}/{total:,} at {rate:.1f}/s; "
                    f"elapsed {elapsed / 60:.1f} min; ETA {eta / 60:.1f} min",
                    flush=True,
                )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        del index
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    print(f"[S{source}] saved {output_path}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PRAJ_DIR / "datasets/preprocessed")
    parser.add_argument("--tfidf-dir", type=Path, default=PRAJ_DIR / "datasets/tfidf")
    parser.add_argument("--output-dir", type=Path, default=PRAJ_DIR / "output")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    if args.top_k < 1 or args.chunk_size < 1:
        parser.error("--top-k and --chunk-size must be positive")

    source1_ids = load_ids(args.data_dir, 1)
    source1_matrix = sparse.load_npz(args.tfidf_dir / "train_s1_tfidf.npz").astype(
        np.float32, copy=False
    )
    if source1_matrix.shape[0] != len(source1_ids):
        raise ValueError(
            f"S1 matrix has {source1_matrix.shape[0]:,} rows but IDs have {len(source1_ids):,}"
        )
    for source in (2, 3):
        search_source(
            source1_matrix,
            source1_ids,
            source,
            args.data_dir,
            args.tfidf_dir,
            args.output_dir,
            args.top_k,
            args.chunk_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
