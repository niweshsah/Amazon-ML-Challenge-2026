"""Top-150 dense and character-trigram GPU retrieval with Parquet caching."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from .metrics import candidate_recall
from .normalizer import IndicXlitTextNormalizer, TextNormalizer


LOGGER = logging.getLogger(__name__)
RETRIEVAL_K = 150
MASTER_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("target_id", pa.string()),
        ("dense_score", pa.float32()),
        ("sparse_score", pa.float32()),
    ]
)


def _validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _scipy_to_torch_csr(matrix: sp.csr_matrix, device: str) -> Any:
    import torch

    matrix = matrix.tocsr()
    row_offsets = torch.from_numpy(matrix.indptr.astype(np.int64, copy=False))
    column_indices = torch.from_numpy(matrix.indices.astype(np.int64, copy=False))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_csr_tensor(
        row_offsets,
        column_indices,
        values,
        size=matrix.shape,
        dtype=torch.float32,
        device=device,
    )


@dataclass
class HybridGPURetriever:
    """Run exact GPU dense and sparse searches, then cache their candidate union."""

    model_name: str = "BAAI/bge-m3"
    device: str = "cuda:0"
    embedding_batch_size: int = 128
    dense_query_batch_size: int = 2_048
    sparse_query_batch_size: int = 8
    max_seq_length: int = 512
    max_tfidf_features: int | None = None
    transliterator: str = "sanscript"
    indicxlit_model_dir: Path | None = None

    def _load_and_normalize(
        self, t1_path: Path, t2_path: Path, output_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        t1 = pd.read_parquet(t1_path)
        t2 = pd.read_parquet(t2_path)
        _validate_columns(
            t1,
            {"entity_id", "business_name", "business_address", "country"},
            "t1",
        )
        _validate_columns(
            t2,
            {
                "entity_id",
                "business_name",
                "business_address",
                "country",
                "source",
                "pool_role",
            },
            "t2",
        )
        if t1["entity_id"].duplicated().any() or t2["entity_id"].duplicated().any():
            raise ValueError("entity_id values must be unique within t1 and t2")

        if self.transliterator == "indicxlit":
            normalizer = IndicXlitTextNormalizer(
                model_dir=self.indicxlit_model_dir
                or Path(__file__).resolve().parent / "indicxlit_model"
            )
        elif self.transliterator == "sanscript":
            normalizer = TextNormalizer()
        else:
            raise ValueError(
                f"Unsupported transliterator {self.transliterator!r}; "
                "choose 'sanscript' or 'indicxlit'."
            )
        LOGGER.info("Normalizing and transliterating %s query rows", f"{len(t1):,}")
        t1 = normalizer.transform(t1)
        LOGGER.info("Normalizing and transliterating %s target rows", f"{len(t2):,}")
        t2 = normalizer.transform(t2)
        t1[["entity_id", "full_text", "normalized_text"]].to_parquet(
            output_dir / "normalized_t1.parquet", index=False
        )
        t2[["entity_id", "full_text", "normalized_text"]].to_parquet(
            output_dir / "normalized_t2.parquet", index=False
        )
        return t1, t2

    def _dense_search(
        self, query_texts: list[str], target_texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            import faiss
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers, torch, and faiss-gpu"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the configured hybrid GPU retrieval")
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("The installed FAISS build has no GPU support")

        LOGGER.info("Loading dense encoder %s on %s", self.model_name, self.device)
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_length
        LOGGER.info("Encoding %s targets", f"{len(target_texts):,}")
        target_vectors = model.encode(
            target_texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        LOGGER.info("Encoding %s queries", f"{len(query_texts):,}")
        query_vectors = model.encode(
            query_texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        dimension = target_vectors.shape[1]
        cpu_index = faiss.IndexFlatIP(dimension)
        resources = faiss.StandardGpuResources()
        gpu_id = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
        index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        index.add(np.ascontiguousarray(target_vectors))
        score_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        for start in range(0, len(query_vectors), self.dense_query_batch_size):
            stop = min(start + self.dense_query_batch_size, len(query_vectors))
            scores, indices = index.search(
                np.ascontiguousarray(query_vectors[start:stop]), RETRIEVAL_K
            )
            score_parts.append(scores.astype(np.float32, copy=False))
            index_parts.append(indices.astype(np.int64, copy=False))
        dense_scores = np.concatenate(score_parts)
        dense_indices = np.concatenate(index_parts)

        del index, cpu_index, resources, model
        gc.collect()
        torch.cuda.empty_cache()
        return query_vectors, target_vectors, dense_scores, dense_indices

    def _sparse_search(
        self, query_texts: list[str], target_texts: list[str]
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
        try:
            import torch
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("Sparse retrieval requires torch and scikit-learn") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for sparse character-trigram retrieval")

        LOGGER.info("Fitting target character-trigram TF-IDF")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            lowercase=False,
            dtype=np.float32,
            norm="l2",
            max_features=self.max_tfidf_features,
        )
        target_matrix = vectorizer.fit_transform(target_texts).tocsr()
        query_matrix = vectorizer.transform(query_texts).tocsr()
        LOGGER.info(
            "Moving sparse target matrix (%s nonzeros) to %s",
            f"{target_matrix.nnz:,}",
            self.device,
        )
        target_transpose = target_matrix.transpose().tocsr()
        target_gpu_t = _scipy_to_torch_csr(target_transpose, self.device)
        del target_transpose
        score_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, query_matrix.shape[0], self.sparse_query_batch_size):
                stop = min(start + self.sparse_query_batch_size, query_matrix.shape[0])
                query_gpu = _scipy_to_torch_csr(query_matrix[start:stop], self.device)
                similarities = torch.sparse.mm(query_gpu, target_gpu_t).to_dense()
                scores, indices = torch.topk(
                    similarities, k=RETRIEVAL_K, dim=1, largest=True, sorted=True
                )
                score_parts.append(scores.cpu().numpy().astype(np.float32, copy=False))
                index_parts.append(indices.cpu().numpy().astype(np.int64, copy=False))
                del query_gpu, similarities, scores, indices
        del target_gpu_t
        torch.cuda.empty_cache()
        return (
            query_matrix,
            target_matrix,
            np.concatenate(score_parts),
            np.concatenate(index_parts),
        )

    @staticmethod
    def _write_master_pool(
        output_path: Path,
        query_ids: np.ndarray,
        target_ids: np.ndarray,
        query_vectors: np.ndarray,
        target_vectors: np.ndarray,
        dense_scores: np.ndarray,
        dense_indices: np.ndarray,
        query_sparse: sp.csr_matrix,
        target_sparse: sp.csr_matrix,
        sparse_scores: np.ndarray,
        sparse_indices: np.ndarray,
        rows_per_chunk: int = 256,
    ) -> None:
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary_path, MASTER_SCHEMA, compression="zstd")
        try:
            for chunk_start in range(0, len(query_ids), rows_per_chunk):
                chunk_stop = min(chunk_start + rows_per_chunk, len(query_ids))
                output_query_ids: list[str] = []
                output_target_ids: list[str] = []
                output_dense_scores: list[float] = []
                output_sparse_scores: list[float] = []
                for query_index in range(chunk_start, chunk_stop):
                    candidate_indices = np.unique(
                        np.concatenate(
                            (dense_indices[query_index], sparse_indices[query_index])
                        )
                    )
                    candidate_indices = candidate_indices[candidate_indices >= 0]
                    dense_lookup = dict(
                        zip(dense_indices[query_index], dense_scores[query_index])
                    )
                    dense_values = np.asarray(
                        [dense_lookup.get(index, np.nan) for index in candidate_indices],
                        dtype=np.float32,
                    )
                    missing_dense = np.isnan(dense_values)
                    dense_values[missing_dense] = (
                        target_vectors[candidate_indices[missing_dense]]
                        @ query_vectors[query_index]
                    )

                    sparse_lookup = dict(
                        zip(sparse_indices[query_index], sparse_scores[query_index])
                    )
                    sparse_values = np.asarray(
                        [sparse_lookup.get(index, np.nan) for index in candidate_indices],
                        dtype=np.float32,
                    )
                    missing_sparse = np.isnan(sparse_values)
                    sparse_values[missing_sparse] = query_sparse[query_index].dot(
                        target_sparse[candidate_indices[missing_sparse]].transpose()
                    ).toarray().ravel()
                    output_query_ids.extend(
                        [str(query_ids[query_index])] * len(candidate_indices)
                    )
                    output_target_ids.extend(target_ids[candidate_indices].astype(str).tolist())
                    output_dense_scores.extend(dense_values.astype(np.float32).tolist())
                    output_sparse_scores.extend(sparse_values.astype(np.float32).tolist())

                table = pa.Table.from_pydict(
                    {
                        "query_id": output_query_ids,
                        "target_id": output_target_ids,
                        "dense_score": output_dense_scores,
                        "sparse_score": output_sparse_scores,
                    },
                    schema=MASTER_SCHEMA,
                )
                writer.write_table(table)
                LOGGER.info(
                    "Serialized candidate unions for %s/%s queries",
                    f"{chunk_stop:,}",
                    f"{len(query_ids):,}",
                )
        finally:
            writer.close()
        os.replace(temporary_path, output_path)

    def retrieve(
        self,
        t1_path: Path,
        t2_path: Path,
        ground_truth_path: Path,
        output_dir: Path,
        min_candidate_recall: float = 0.99,
    ) -> dict[str, Any]:
        """Build and persist the union pool, returning timing and recall metrics."""
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        t1, t2 = self._load_and_normalize(t1_path, t2_path, output_dir)
        query_texts = t1["normalized_text"].tolist()
        target_texts = t2["normalized_text"].tolist()
        if len(t2) < RETRIEVAL_K:
            raise ValueError(f"t2 must contain at least {RETRIEVAL_K} rows")
        query_vectors, target_vectors, dense_scores, dense_indices = self._dense_search(
            query_texts, target_texts
        )
        query_sparse, target_sparse, sparse_scores, sparse_indices = self._sparse_search(
            query_texts, target_texts
        )
        master_path = output_dir / "retrieval_master_pool.parquet"
        self._write_master_pool(
            master_path,
            t1["entity_id"].astype(str).to_numpy(),
            t2["entity_id"].astype(str).to_numpy(),
            query_vectors,
            target_vectors,
            dense_scores,
            dense_indices,
            query_sparse,
            target_sparse,
            sparse_scores,
            sparse_indices,
        )
        ground_truth = pd.read_parquet(ground_truth_path)
        metrics = candidate_recall(str(master_path), ground_truth)
        metrics.update(
            {
                "retrieval_k_per_channel": RETRIEVAL_K,
                "master_pool_path": str(master_path),
                "retrieval_seconds": time.perf_counter() - started,
            }
        )
        (output_dir / "retrieval_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        LOGGER.info("Master-pool candidate recall: %.6f", metrics["candidate_recall"])
        if metrics["candidate_recall"] < min_candidate_recall:
            raise RuntimeError(
                "Candidate recall target was not met: "
                f"{metrics['candidate_recall']:.6f} < {min_candidate_recall:.6f}. "
                f"The reusable master pool remains at {master_path}."
            )
        return metrics
