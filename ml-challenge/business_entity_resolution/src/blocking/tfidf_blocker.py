import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from src.blocking.base import BaseBlocker


class TfidfBlocker(BaseBlocker):
    """Memory-bounded character n-gram blocker."""

    def __init__(self, ngram_range=(3, 5), n_features=2**18):
        self.vectorizer = HashingVectorizer(
            analyzer='char_wb',
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm='l2',
            dtype=np.float32,
        )
        self._corpus_chunks = None

    def fit(self, corpus_df: pd.DataFrame):
        """Keep a small in-memory corpus for backwards-compatible use."""
        self._corpus_chunks = [corpus_df[['entity_id', 'search_text']].copy()]
        return self

    def generate_candidates(
        self,
        queries_df: pd.DataFrame,
        top_k: int = 30,
        threshold: float = 0.15,
        batch_size: int = 256,
        corpus_chunks: Iterable[pd.DataFrame] | None = None,
        corpus_total_rows: int | None = None,
    ) -> pd.DataFrame:
        if batch_size <= 0:
            raise ValueError('batch_size must be greater than zero')
        if top_k <= 0:
            raise ValueError('top_k must be greater than zero')

        corpus_chunks = corpus_chunks or self._corpus_chunks
        if corpus_chunks is None:
            raise RuntimeError('Call fit() or provide corpus_chunks before retrieval')

        query_count = len(queries_df)
        best_scores = np.full((query_count, top_k), -np.inf, dtype=np.float32)
        best_indices = np.full((query_count, top_k), -1, dtype=np.int64)
        corpus_ids = []
        corpus_offset = 0
        started = time.perf_counter()

        for corpus_chunk_number, corpus_chunk in enumerate(corpus_chunks, start=1):
            chunk_ids = corpus_chunk['entity_id'].astype(str).to_numpy()
            corpus_ids.append(chunk_ids)
            corpus_vectors = self.vectorizer.transform(corpus_chunk['search_text'])

            for start in range(0, query_count, batch_size):
                end = min(start + batch_size, query_count)
                query_vectors = self.vectorizer.transform(
                    queries_df['search_text'].iloc[start:end]
                )
                similarities = query_vectors.dot(corpus_vectors.T).tocsr()

                for batch_row in range(end - start):
                    row = similarities.getrow(batch_row)
                    if not row.nnz:
                        continue

                    local_top_k = min(top_k, row.nnz)
                    local_order = np.argpartition(row.data, -local_top_k)[-local_top_k:]
                    local_order = local_order[np.argsort(row.data[local_order])[::-1]]
                    combined_scores = np.concatenate(
                        [best_scores[start + batch_row], row.data[local_order]]
                    )
                    combined_indices = np.concatenate(
                        [
                            best_indices[start + batch_row],
                            corpus_offset + row.indices[local_order],
                        ]
                    )
                    keep = np.argsort(combined_scores)[-top_k:][::-1]
                    best_scores[start + batch_row] = combined_scores[keep]
                    best_indices[start + batch_row] = combined_indices[keep]

            elapsed = time.perf_counter() - started
            processed_rows = corpus_offset + len(chunk_ids)
            eta_seconds = (
                elapsed * (corpus_total_rows / processed_rows - 1)
                if corpus_total_rows and processed_rows
                else 0
            )
            print(
                f'[timeline] corpus chunk {corpus_chunk_number} '
                f'({processed_rows:,}/{corpus_total_rows or "?"} corpus rows) | '
                f'elapsed {format_duration(elapsed)} | '
                f'ETA {format_duration(eta_seconds)}'
            )
            corpus_offset += len(chunk_ids)

        all_corpus_ids = np.concatenate(corpus_ids) if corpus_ids else np.array([])
        results = []
        for row_number, source1_id in enumerate(queries_df['entity_id']):
            candidate_ids = [
                all_corpus_ids[index]
                for score, index in zip(best_scores[row_number], best_indices[row_number])
                if index >= 0 and score >= threshold
            ]
            results.append(
                {
                    'source1_entity_id': source1_id,
                    'candidate_entity_ids': ','.join(dict.fromkeys(candidate_ids)),
                }
            )
        return pd.DataFrame(results)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for progress output."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours}h {minutes:02d}m {seconds:02d}s'
    return f'{minutes}m {seconds:02d}s'