from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_TRAIN_DIR = PROJECT_DIR / "challenge-dataset" / "dataset" / "train"
DEFAULT_CANDIDATES = REPO_DIR / "praj_files" / "output" / "candidate_pairs_faiss.tsv"
DEFAULT_EMBEDDING_DIR = REPO_DIR / "praj_files" / "datasets" / "dense_embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "gbdt"

BASE_FEATURE_NAMES = [
    "candidate_source2",
    "candidate_source3",
    "retrieval_rank",
    "country_match",
    "name_exact_match",
    "address_exact_match",
    "name_levenshtein_similarity",
    "name_jaro_winkler_similarity",
    "name_monge_elkan_similarity",
    "name_lcs_similarity",
    "address_levenshtein_similarity",
    "address_jaro_winkler_similarity",
    "address_monge_elkan_similarity",
    "address_lcs_similarity",
    "name_token_jaccard",
    "name_token_overlap_min_ratio",
    "name_token_overlap_count",
    "name_bigram_jaccard",
    "name_trigram_jaccard",
    "address_token_jaccard",
    "address_token_overlap_min_ratio",
    "address_token_overlap_count",
    "address_bigram_jaccard",
    "address_trigram_jaccard",
    "name_prefix_token_match",
    "name_suffix_token_match",
    "address_prefix_token_match",
    "address_suffix_token_match",
    "name_length_ratio",
    "address_length_ratio",
    "name_soundex_match",
    "address_soundex_match",
    "name_missing_left",
    "name_missing_right",
    "address_missing_left",
    "address_missing_right",
]


@dataclass(frozen=True)
class Entity:
    entity_id: str
    source: int
    name: str
    address: str
    country: str
    name_normalized: str
    address_normalized: str
    country_normalized: str
    name_tokens: tuple[str, ...]
    address_tokens: tuple[str, ...]
    name_token_set: frozenset[str]
    address_token_set: frozenset[str]
    combined_text: str
    row_index: int


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(char if char.isalnum() else " " for char in normalized).split()
    )


def prepare_entity(row: dict[str, str], source: int, row_index: int) -> Entity:
    name = row.get("business_name", "") or ""
    address = row.get("business_address", "") or ""
    country = row.get("country", "") or ""
    name_normalized = normalize_text(name)
    address_normalized = normalize_text(address)
    country_normalized = normalize_text(country)
    name_tokens = tuple(name_normalized.split())
    address_tokens = tuple(address_normalized.split())
    return Entity(
        entity_id=row["entity_id"],
        source=source,
        name=name,
        address=address,
        country=country,
        name_normalized=name_normalized,
        address_normalized=address_normalized,
        country_normalized=country_normalized,
        name_tokens=name_tokens,
        address_tokens=address_tokens,
        name_token_set=frozenset(name_tokens),
        address_token_set=frozenset(address_tokens),
        combined_text=f"{name_normalized} {address_normalized}".strip(),
        row_index=row_index,
    )


def select_query_ids(source1_path: Path, maximum: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0
    with source1_path.open(encoding="utf-8-sig", newline="") as source_file:
        scan_started = time.perf_counter()
        reader = csv.DictReader(source_file, delimiter="\t")
        require_columns(reader.fieldnames, {"entity_id"}, source1_path)
        for row in reader:
            entity_id = row["entity_id"]
            if maximum <= 0:
                reservoir.append(entity_id)
            elif count < maximum:
                reservoir.append(entity_id)
            else:
                selected_index = rng.randrange(count + 1)
                if selected_index < maximum:
                    reservoir[selected_index] = entity_id
            count += 1
            if count % 500_000 == 0:
                report_scan_progress("Source 1 query selection", count, scan_started, source_file)
    if not reservoir:
        raise ValueError(f"No Source 1 records found in {source1_path}")
    return reservoir


def require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
    actual = set(fieldnames or [])
    missing = required - actual
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def save_booster_atomic(booster: Any, path: Path) -> None:
    temporary_path = path.with_name(path.stem + ".tmp.json")
    booster.save_model(temporary_path)
    os.replace(temporary_path, path)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def report_scan_progress(stage: str, rows: int, started: float, source_file: Any) -> None:
    elapsed = time.perf_counter() - started
    try:
        total_bytes = os.fstat(source_file.fileno()).st_size
        scanned_bytes = min(source_file.buffer.tell(), total_bytes)
        fraction = scanned_bytes / max(total_bytes, 1)
        eta = elapsed * (total_bytes - scanned_bytes) / max(scanned_bytes, 1)
        eta_text = format_duration(eta)
    except (AttributeError, OSError, ValueError):
        fraction = 0.0
        eta_text = "estimating"
    print(
        f"[timeline] {stage}: {rows:,} rows, {fraction:.1%} read; "
        f"elapsed {format_duration(elapsed)}, ETA {eta_text}",
        flush=True,
    )


def load_ground_truth(path: Path, selected_ids: set[str]) -> dict[str, set[str]]:
    truth = {entity_id: set() for entity_id in selected_ids}
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as truth_file:
        scan_started = time.perf_counter()
        reader = csv.DictReader(truth_file, delimiter="\t")
        require_columns(reader.fieldnames, {"source1_entity_id", "matched_entity_ids"}, path)
        for row_number, row in enumerate(reader, start=1):
            if row_number % 500_000 == 0:
                report_scan_progress("Ground-truth scan", row_number, scan_started, truth_file)
            query_id = row["source1_entity_id"]
            if query_id not in selected_ids:
                continue
            if query_id in seen:
                raise ValueError(f"Duplicate ground-truth row for {query_id}")
            seen.add(query_id)
            truth[query_id] = {
                value.strip()
                for value in (row.get("matched_entity_ids", "") or "").split(",")
                if value.strip()
            }
    return truth


def load_candidate_lists(
    path: Path,
    selected_ids: set[str],
    top_k_per_source: int,
) -> dict[str, dict[str, int]]:
    candidates = {entity_id: {} for entity_id in selected_ids}
    with path.open(encoding="utf-8-sig", newline="") as candidate_file:
        scan_started = time.perf_counter()
        reader = csv.DictReader(candidate_file, delimiter="\t")
        require_columns(
            reader.fieldnames,
            {"source1_entity_id", "candidate_entity_ids"},
            path,
        )
        for row_number, row in enumerate(reader, start=1):
            query_id = row["source1_entity_id"]
            if query_id not in selected_ids:
                continue
            source_ranks = {2: 0, 3: 0}
            for candidate_id in (row.get("candidate_entity_ids", "") or "").split(","):
                candidate_id = candidate_id.strip()
                if candidate_id.startswith("S2-"):
                    source = 2
                elif candidate_id.startswith("S3-"):
                    source = 3
                else:
                    continue
                source_ranks[source] += 1
                if source_ranks[source] <= top_k_per_source:
                    candidates[query_id].setdefault(candidate_id, source_ranks[source])
            if row_number % 500_000 == 0:
                report_scan_progress("Candidate scan", row_number, scan_started, candidate_file)
    return candidates


def load_referenced_entities(
    train_dir: Path,
    query_ids: list[str],
    candidates: dict[str, dict[str, int]],
) -> tuple[dict[str, Entity], dict[int, int]]:
    wanted_by_source: dict[int, set[str]] = {1: set(query_ids), 2: set(), 3: set()}
    for query_candidates in candidates.values():
        for candidate_id in query_candidates:
            source = 2 if candidate_id.startswith("S2-") else 3
            wanted_by_source[source].add(candidate_id)

    entities: dict[str, Entity] = {}
    source_row_counts: dict[int, int] = {}
    for source in (1, 2, 3):
        source_path = train_dir / f"train_source{source}.tsv"
        found = 0
        row_count = 0
        with source_path.open(encoding="utf-8-sig", newline="") as source_file:
            scan_started = time.perf_counter()
            reader = csv.DictReader(source_file, delimiter="\t")
            require_columns(
                reader.fieldnames,
                {"entity_id", "business_name", "business_address", "country"},
                source_path,
            )
            for row_index, row in enumerate(reader):
                row_count = row_index + 1
                entity_id = row["entity_id"]
                if entity_id in wanted_by_source[source]:
                    if entity_id in entities:
                        raise ValueError(f"Duplicate entity ID found: {entity_id}")
                    entities[entity_id] = prepare_entity(row, source, row_index)
                    found += 1
                source_row_counts[source] = row_count
                if row_count % 1_000_000 == 0:
                    report_scan_progress(
                        f"Source {source} record scan", row_count, scan_started, source_file
                    )
        missing_ids = wanted_by_source[source] - entities.keys()
        if missing_ids:
            examples = sorted(missing_ids)[:5]
            raise ValueError(
                f"{source_path} is missing {len(missing_ids):,} referenced IDs; examples: {examples}"
            )
        print(
            f"[data] source{source}: retained {found:,} referenced rows "
            f"from {source_row_counts[source]:,}",
            flush=True,
        )
    return entities, source_row_counts


def soundex(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    letters = re.sub(r"[^A-Za-z]", "", ascii_value).upper()
    if not letters:
        return ""
    codes = {
        "BFPV": "1",
        "CGJKQSXZ": "2",
        "DT": "3",
        "L": "4",
        "MN": "5",
        "R": "6",
    }
    encoded = [""]
    for letter in letters[1:]:
        code = next((digit for chars, digit in codes.items() if letter in chars), "0")
        if code != encoded[-1] and code != "0":
            encoded.append(code)
    return (letters[0] + "".join(encoded[1:]) + "000")[:4]


def jaccard(left: set[Any] | frozenset[Any], right: set[Any] | frozenset[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def overlap_min_ratio(
    left: set[str] | frozenset[str], right: set[str] | frozenset[str]
) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def token_ngrams(tokens: tuple[str, ...], size: int) -> set[tuple[str, ...]]:
    return {tokens[index:index + size] for index in range(len(tokens) - size + 1)}


def monge_elkan(left: tuple[str, ...], right: tuple[str, ...], jaro_winkler: Any) -> float:
    if not left or not right:
        return 0.0

    def directional_average(source: tuple[str, ...], target: tuple[str, ...]) -> float:
        return sum(
            max(jaro_winkler.normalized_similarity(token, other) for other in target)
            for token in source
        ) / len(source)

    return (directional_average(left, right) + directional_average(right, left)) / 2


def feature_row(
    left: Entity,
    right: Entity,
    rank: int,
    levenshtein: Any,
    jaro_winkler: Any,
    lcs_sequence: Any,
) -> list[float]:
    name_overlap = left.name_token_set & right.name_token_set
    address_overlap = left.address_token_set & right.address_token_set
    name_length_ratio = min(len(left.name_normalized), len(right.name_normalized)) / max(
        len(left.name_normalized), len(right.name_normalized), 1
    )
    address_length_ratio = min(len(left.address_normalized), len(right.address_normalized)) / max(
        len(left.address_normalized), len(right.address_normalized), 1
    )
    return [
        float(right.source == 2),
        float(right.source == 3),
        float(rank),
        float(bool(left.country_normalized) and left.country_normalized == right.country_normalized),
        float(bool(left.name_normalized) and left.name_normalized == right.name_normalized),
        float(bool(left.address_normalized) and left.address_normalized == right.address_normalized),
        levenshtein.normalized_similarity(left.name_normalized, right.name_normalized),
        jaro_winkler.normalized_similarity(left.name_normalized, right.name_normalized),
        monge_elkan(left.name_tokens, right.name_tokens, jaro_winkler),
        lcs_sequence.normalized_similarity(left.name_normalized, right.name_normalized),
        levenshtein.normalized_similarity(left.address_normalized, right.address_normalized),
        jaro_winkler.normalized_similarity(left.address_normalized, right.address_normalized),
        monge_elkan(left.address_tokens, right.address_tokens, jaro_winkler),
        lcs_sequence.normalized_similarity(left.address_normalized, right.address_normalized),
        jaccard(left.name_token_set, right.name_token_set),
        overlap_min_ratio(left.name_token_set, right.name_token_set),
        float(len(name_overlap)),
        jaccard(token_ngrams(left.name_tokens, 2), token_ngrams(right.name_tokens, 2)),
        jaccard(token_ngrams(left.name_tokens, 3), token_ngrams(right.name_tokens, 3)),
        jaccard(left.address_token_set, right.address_token_set),
        overlap_min_ratio(left.address_token_set, right.address_token_set),
        float(len(address_overlap)),
        jaccard(token_ngrams(left.address_tokens, 2), token_ngrams(right.address_tokens, 2)),
        jaccard(token_ngrams(left.address_tokens, 3), token_ngrams(right.address_tokens, 3)),
        float(bool(left.name_tokens and right.name_tokens and left.name_tokens[0] == right.name_tokens[0])),
        float(bool(left.name_tokens and right.name_tokens and left.name_tokens[-1] == right.name_tokens[-1])),
        float(bool(left.address_tokens and right.address_tokens and left.address_tokens[0] == right.address_tokens[0])),
        float(bool(left.address_tokens and right.address_tokens and left.address_tokens[-1] == right.address_tokens[-1])),
        name_length_ratio,
        address_length_ratio,
        float(bool(soundex(left.name_normalized)) and soundex(left.name_normalized) == soundex(right.name_normalized)),
        float(bool(soundex(left.address_normalized)) and soundex(left.address_normalized) == soundex(right.address_normalized)),
        float(not left.name_normalized),
        float(not right.name_normalized),
        float(not left.address_normalized),
        float(not right.address_normalized),
    ]


def build_tfidf_vectorizers(
    records: dict[str, Entity],
    training_entity_ids: set[str],
    max_documents: int,
    seed: int,
) -> tuple[Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    ordered_ids = sorted(training_entity_ids)
    if len(ordered_ids) > max_documents:
        stride = math.ceil(len(ordered_ids) / max_documents)
        ordered_ids = ordered_ids[::stride][:max_documents]
    documents = [records[entity_id].combined_text for entity_id in ordered_ids]
    if not documents:
        raise ValueError("No text documents are available to fit TF-IDF features")
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )
    for vectorizer in (char_vectorizer, word_vectorizer):
        try:
            vectorizer.fit(documents)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            vectorizer.fit(["entity record text", "entity matching text"])
    print(f"[tfidf] fitted on {len(documents):,} training-side entity documents", flush=True)
    return char_vectorizer, word_vectorizer


def rowwise_tfidf_cosine(vectorizer: Any, left_texts: list[str], right_texts: list[str]) -> np.ndarray:
    left_matrix = vectorizer.transform(left_texts)
    right_matrix = vectorizer.transform(right_texts)
    return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).reshape(-1).astype(np.float32)


def build_feature_matrix(
    query_ids: list[str],
    query_indexes: np.ndarray,
    candidate_ids: list[str],
    candidate_ranks: np.ndarray,
    records: dict[str, Entity],
    vectorizers: tuple[Any, Any],
    embedding_arrays: dict[int, np.ndarray],
    batch_size: int,
    cache_path: Path,
    run_signature: str,
    resume: bool,
) -> tuple[np.ndarray, list[str]]:
    try:
        from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein
    except ImportError as exc:
        raise RuntimeError(
            "RapidFuzz is required. Install the training dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    char_vectorizer, word_vectorizer = vectorizers
    feature_names = BASE_FEATURE_NAMES + ["tfidf_char_cosine", "tfidf_word_cosine"]
    if embedding_arrays:
        feature_names.append("dense_embedding_cosine")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path.with_suffix(".json")
    partial_path = cache_path.with_name(cache_path.stem + ".partial.npy")
    if resume and cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("signature") == run_signature
            and manifest.get("complete")
            and manifest.get("rows") == len(candidate_ids)
            and manifest.get("feature_names") == feature_names
        ):
            print(f"[features] reusing completed matrix {cache_path}", flush=True)
            return np.load(cache_path, mmap_mode="r"), feature_names

    completed_rows = 0
    if resume and partial_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("signature") == run_signature
            and manifest.get("rows") == len(candidate_ids)
            and manifest.get("feature_names") == feature_names
        ):
            features = np.load(partial_path, mmap_mode="r+")
            if features.shape == (len(candidate_ids), len(feature_names)):
                completed_rows = min(
                    int(manifest.get("completed_rows", 0)) // batch_size * batch_size,
                    len(candidate_ids),
                )
                print(
                    f"[features] resuming at pair {completed_rows:,}/{len(candidate_ids):,}",
                    flush=True,
                )
            else:
                features = np.lib.format.open_memmap(
                    partial_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(candidate_ids), len(feature_names)),
                )
        else:
            features = np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(candidate_ids), len(feature_names)),
            )
    else:
        features = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(candidate_ids), len(feature_names)),
        )
    atomic_write_json(
        manifest_path,
        {
            "signature": run_signature,
            "complete": False,
            "rows": len(candidate_ids),
            "completed_rows": completed_rows,
            "feature_names": feature_names,
        },
    )
    feature_started = time.perf_counter()
    last_checkpoint_rows = completed_rows
    for start in range(completed_rows, len(candidate_ids), batch_size):
        stop = min(start + batch_size, len(candidate_ids))
        left_records = [records[query_ids[index]] for index in query_indexes[start:stop]]
        right_records = [records[entity_id] for entity_id in candidate_ids[start:stop]]
        block = np.empty((stop - start, len(feature_names)), dtype=np.float32)
        for row_index, (left, right, rank) in enumerate(
            zip(left_records, right_records, candidate_ranks[start:stop])
        ):
            block[row_index, :len(BASE_FEATURE_NAMES)] = feature_row(
                left, right, int(rank), Levenshtein, JaroWinkler, LCSseq
            )
        left_texts = [record.combined_text for record in left_records]
        right_texts = [record.combined_text for record in right_records]
        block[:, len(BASE_FEATURE_NAMES)] = rowwise_tfidf_cosine(
            char_vectorizer, left_texts, right_texts
        )
        block[:, len(BASE_FEATURE_NAMES) + 1] = rowwise_tfidf_cosine(
            word_vectorizer, left_texts, right_texts
        )
        if embedding_arrays:
            dense_column = len(feature_names) - 1
            dense_values = np.zeros(stop - start, dtype=np.float32)
            for source in (2, 3):
                positions = [
                    index
                    for index, record in enumerate(right_records)
                    if record.source == source
                ]
                if not positions:
                    continue
                left_indexes = np.fromiter(
                    (left_records[index].row_index for index in positions),
                    dtype=np.int64,
                    count=len(positions),
                )
                right_indexes = np.fromiter(
                    (right_records[index].row_index for index in positions),
                    dtype=np.int64,
                    count=len(positions),
                )
                left_vectors = np.asarray(embedding_arrays[1][left_indexes], dtype=np.float32)
                right_vectors = np.asarray(embedding_arrays[source][right_indexes], dtype=np.float32)
                dot = np.einsum("ij,ij->i", left_vectors, right_vectors)
                denominator = np.linalg.norm(left_vectors, axis=1) * np.linalg.norm(right_vectors, axis=1)
                dense_values[positions] = dot / np.maximum(denominator, 1e-12)
            block[:, dense_column] = dense_values
        features[start:stop] = block
        if stop - last_checkpoint_rows >= batch_size * 25 or stop == len(candidate_ids):
            features.flush()
            atomic_write_json(
                manifest_path,
                {
                    "signature": run_signature,
                    "complete": False,
                    "rows": len(candidate_ids),
                    "completed_rows": stop,
                    "feature_names": feature_names,
                },
            )
            last_checkpoint_rows = stop
        if stop % 100_000 < batch_size or stop == len(candidate_ids):
            elapsed = time.perf_counter() - feature_started
            processed = stop - completed_rows
            eta = elapsed * (len(candidate_ids) - stop) / max(processed, 1)
            print(
                f"[timeline] features {stop:,}/{len(candidate_ids):,} pairs; "
                f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}",
                flush=True,
            )
    features.flush()
    del features
    os.replace(partial_path, cache_path)
    atomic_write_json(
        manifest_path,
        {
            "signature": run_signature,
            "complete": True,
            "rows": len(candidate_ids),
            "completed_rows": len(candidate_ids),
            "feature_names": feature_names,
        },
    )
    print(f"[features] checkpointed matrix at {cache_path}", flush=True)
    return np.load(cache_path, mmap_mode="r"), feature_names


def load_embedding_arrays(
    embedding_dir: Path,
    source_row_counts: dict[int, int],
    enabled: bool,
) -> dict[int, np.ndarray]:
    if not enabled:
        print("[embeddings] disabled by command line", flush=True)
        return {}
    arrays: dict[int, np.ndarray] = {}
    for source in (1, 2, 3):
        path = embedding_dir / f"train_s{source}_embeddings.npy"
        if not path.exists():
            print(f"[embeddings] missing {path}; dense feature will be omitted", flush=True)
            return {}
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] != source_row_counts[source]:
            raise ValueError(
                f"Embedding rows in {path} ({array.shape[0] if array.ndim else 0}) do not match "
                f"the {source_row_counts[source]} rows in the corresponding source TSV"
            )
        arrays[source] = array
    dimensions = {array.shape[1] for array in arrays.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ across sources: {sorted(dimensions)}")
    print(
        f"[embeddings] enabled {next(iter(dimensions))}-dimensional cosine features from {embedding_dir}",
        flush=True,
    )
    return arrays


def evaluate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    query_indexes: np.ndarray,
    truth_counts: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    query_count = len(truth_counts)
    true_positive = np.bincount(
        query_indexes[predictions & (labels == 1)], minlength=query_count
    ).astype(np.float64)
    predicted_count = np.bincount(query_indexes[predictions], minlength=query_count).astype(np.float64)
    false_positive = predicted_count - true_positive
    false_negative = truth_counts.astype(np.float64) - true_positive
    denominator = 1.25 * true_positive + false_positive + 0.25 * false_negative
    per_query_f05 = np.divide(
        1.25 * true_positive,
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0,
    )
    total_true_positive = float(true_positive.sum())
    total_predicted = float(predicted_count.sum())
    total_truth = float(truth_counts.sum())
    return {
        "threshold": float(threshold),
        "macro_f0.5": float(per_query_f05.mean()) if query_count else 0.0,
        "pair_precision": total_true_positive / total_predicted if total_predicted else 0.0,
        "pair_recall": total_true_positive / total_truth if total_truth else 1.0,
        "predicted_pairs": total_predicted,
        "true_positive_pairs": total_true_positive,
    }


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    query_indexes: np.ndarray,
    truth_counts: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    thresholds = [round(value, 3) for value in np.arange(0.05, 1.0, 0.05)]
    thresholds.extend([0.975, 0.99])
    results = [
        evaluate_threshold(probabilities, labels, query_indexes, truth_counts, threshold)
        for threshold in thresholds
    ]
    best = max(results, key=lambda item: (item["macro_f0.5"], item["pair_precision"], item["threshold"]))
    return best, results


def checkpoint_signature(args: argparse.Namespace, query_ids: list[str]) -> str:
    input_paths = [
        args.train_dir / "train_source1.tsv",
        args.train_dir / "train_source2.tsv",
        args.train_dir / "train_source3.tsv",
        args.train_dir / "train_ground_truth.tsv",
        args.candidates,
    ]
    if not args.no_dense_features:
        input_paths.extend(args.embedding_dir / f"train_s{source}_embeddings.npy" for source in (1, 2, 3))
    inputs = {}
    for path in input_paths:
        if path.exists():
            stat = path.stat()
            inputs[str(path.resolve())] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            inputs[str(path.resolve())] = None
    query_hash = hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest()
    settings = {
        "inputs": inputs,
        "query_hash": query_hash,
        "top_k_per_source": args.top_k_per_source,
        "validation_fraction": args.validation_fraction,
        "max_tfidf_documents": args.max_tfidf_documents,
        "dense_features": not args.no_dense_features,
        "n_estimators": args.n_estimators,
        "early_stopping_rounds": args.early_stopping_rounds,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "feature_schema": BASE_FEATURE_NAMES,
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def xgboost_parameters(device: str, seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "nthread": n_jobs if n_jobs > 0 else max(1, os.cpu_count() or 1),
        "seed": seed,
    }


def verify_gpu_backend(device: str, min_free_gpu_gb: float, seed: int, n_jobs: int) -> None:
    if device == "cpu":
        print("[gpu] CPU explicitly selected; XGBoost will not use the GPU", flush=True)
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA verification requires PyTorch in this environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Run the trainer inside the SSH/Docker environment with NVIDIA GPU access."
        )

    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    if free_gb < min_free_gpu_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB of {total_gb:.2f} GiB GPU memory is free; "
            f"at least {min_free_gpu_gb:.2f} GiB is required. Other GPU workloads may be active."
        )

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required. Install the training dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    probe_features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    probe_labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    try:
        probe_matrix = xgb.DMatrix(probe_features, label=probe_labels)
        probe_model = xgb.train(
            xgboost_parameters("cuda", seed, 1), probe_matrix, num_boost_round=1
        )
        booster_config = json.loads(probe_model.save_config())
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not complete its CUDA smoke fit. Install a CUDA-enabled "
            "XGBoost build compatible with this machine before training."
        ) from exc
    actual_device = booster_config.get("learner", {}).get("generic_param", {}).get("device", "")
    if not actual_device.startswith("cuda"):
        raise RuntimeError(
            f"XGBoost smoke fit used {actual_device or 'an unknown device'}, not CUDA; "
            "refusing to continue with CPU fallback."
        )
    del probe_model
    torch.cuda.empty_cache()
    print(
        f"[gpu] verified XGBoost CUDA training on {device_name}; "
        f"{free_gb:.2f}/{total_gb:.2f} GiB was free before the probe",
        flush=True,
    )


def train_with_checkpoints(
    training_matrix: Any,
    target_rounds: int,
    checkpoint_dir: Path,
    stage: str,
    params: dict[str, Any],
    checkpoint_every: int,
    resume: bool,
    validation_matrix: Any | None = None,
    validation_labels: np.ndarray | None = None,
    early_stopping_rounds: int = 0,
) -> tuple[Any, int | None, float | None]:
    import xgboost as xgb

    latest_path = checkpoint_dir / f"{stage}_latest.json"
    best_path = checkpoint_dir / f"{stage}_best.json"
    state_path = checkpoint_dir / f"{stage}_state.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    booster = None
    completed_rounds = 0
    if resume and latest_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("target_rounds") == target_rounds and state.get("stage") == stage:
            booster = xgb.Booster()
            booster.load_model(latest_path)
            completed_rounds = booster.num_boosted_rounds()
            print(
                f"[checkpoint] resuming {stage} from round {completed_rounds:,}/{target_rounds:,}",
                flush=True,
            )
            if state.get("complete"):
                result = booster
                if validation_matrix is not None and best_path.exists():
                    result = xgb.Booster()
                    result.load_model(best_path)
                return result, state.get("best_round"), state.get("best_validation_logloss")
        else:
            state = {}
            booster = None
            completed_rounds = 0

    best_loss = float(state.get("best_validation_logloss", math.inf))
    best_round = state.get("best_round")
    stale_rounds = int(state.get("stale_rounds", 0))
    start_round = completed_rounds
    stage_started = time.perf_counter()

    if validation_matrix is not None and booster is not None and completed_rounds > int(
        state.get("completed_rounds", 0)
    ):
        current_probabilities = np.clip(booster.predict(validation_matrix), 1e-7, 1 - 1e-7)
        current_loss = float(
            -np.mean(
                validation_labels * np.log(current_probabilities)
                + (1 - validation_labels) * np.log(1 - current_probabilities)
            )
        )
        if current_loss < best_loss:
            best_loss = current_loss
            best_round = completed_rounds
            stale_rounds = 0
            save_booster_atomic(booster, best_path)
        else:
            stale_rounds += completed_rounds - int(state.get("completed_rounds", 0))

    stopped_early = False
    while completed_rounds < target_rounds:
        chunk_rounds = min(checkpoint_every, target_rounds - completed_rounds)
        evaluations = [(validation_matrix, "validation")] if validation_matrix is not None else []
        booster = xgb.train(
            params,
            training_matrix,
            num_boost_round=chunk_rounds,
            evals=evaluations,
            xgb_model=booster,
            verbose_eval=False,
        )
        completed_rounds = booster.num_boosted_rounds()
        if validation_matrix is not None:
            probabilities = np.clip(booster.predict(validation_matrix), 1e-7, 1 - 1e-7)
            validation_loss = float(
                -np.mean(
                    validation_labels * np.log(probabilities)
                    + (1 - validation_labels) * np.log(1 - probabilities)
                )
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_round = completed_rounds
                stale_rounds = 0
                save_booster_atomic(booster, best_path)
            else:
                stale_rounds += chunk_rounds

        save_booster_atomic(booster, latest_path)
        state = {
            "stage": stage,
            "complete": False,
            "completed_rounds": completed_rounds,
            "target_rounds": target_rounds,
            "best_round": best_round,
            "best_validation_logloss": best_loss if math.isfinite(best_loss) else None,
            "stale_rounds": stale_rounds,
        }
        if validation_matrix is not None and best_round is None:
            state["best_validation_logloss"] = None
        atomic_write_json(state_path, state)

        elapsed = time.perf_counter() - stage_started
        rounds_this_run = completed_rounds - start_round
        eta = elapsed * (target_rounds - completed_rounds) / max(rounds_this_run, 1)
        print(
            f"[timeline] {stage} {completed_rounds:,}/{target_rounds:,} rounds; "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}",
            flush=True,
        )
        if validation_matrix is not None and early_stopping_rounds > 0 and stale_rounds >= early_stopping_rounds:
            stopped_early = True
            break

    if booster is None:
        raise RuntimeError(f"No XGBoost rounds were run for stage {stage}")
    save_booster_atomic(booster, latest_path)
    state.update(
        {
            "stage": stage,
            "complete": True,
            "completed_rounds": completed_rounds,
            "target_rounds": target_rounds,
            "best_round": best_round,
            "best_validation_logloss": best_loss if math.isfinite(best_loss) else None,
            "stale_rounds": stale_rounds,
            "stopped_early": stopped_early,
        }
    )
    atomic_write_json(state_path, state)
    if validation_matrix is not None and best_path.exists():
        best_booster = xgb.Booster()
        best_booster.load_model(best_path)
        return best_booster, best_round, best_loss
    return booster, completed_rounds, None


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"[timeline] started {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    verify_gpu_backend(args.device, args.min_free_gpu_gb, args.seed, args.n_jobs)
    train_dir = args.train_dir.resolve()
    source1_path = train_dir / "train_source1.tsv"
    ground_truth_path = train_dir / "train_ground_truth.tsv"
    for path in (source1_path, ground_truth_path, args.candidates):
        if not path.exists():
            raise FileNotFoundError(f"Required training input does not exist: {path}")

    print(f"[train] selecting queries from {source1_path}", flush=True)
    query_ids = select_query_ids(source1_path, args.max_queries, args.seed)
    selected_ids = set(query_ids)
    print(f"[train] selected {len(query_ids):,} Source 1 queries", flush=True)
    run_signature = checkpoint_signature(args, query_ids)
    checkpoint_dir = args.output_dir / "checkpoints" / run_signature[:16]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] directory: {checkpoint_dir}", flush=True)
    truth = load_ground_truth(ground_truth_path, selected_ids)
    candidates = load_candidate_lists(args.candidates, selected_ids, args.top_k_per_source)
    entities, source_row_counts = load_referenced_entities(train_dir, query_ids, candidates)

    query_index_by_id = {entity_id: index for index, entity_id in enumerate(query_ids)}
    pair_query_indexes: list[int] = []
    pair_candidate_ids: list[str] = []
    pair_candidate_ranks: list[int] = []
    pair_labels: list[int] = []
    for query_id in query_ids:
        query_index = query_index_by_id[query_id]
        for candidate_id, rank in candidates[query_id].items():
            pair_query_indexes.append(query_index)
            pair_candidate_ids.append(candidate_id)
            pair_candidate_ranks.append(rank)
            pair_labels.append(int(candidate_id in truth[query_id]))

    if not pair_labels:
        raise ValueError("No candidate pairs found for the selected training queries")
    labels = np.asarray(pair_labels, dtype=np.uint8)
    query_indexes = np.asarray(pair_query_indexes, dtype=np.int32)
    candidate_ranks = np.asarray(pair_candidate_ranks, dtype=np.int16)
    truth_counts_all = np.fromiter(
        (len(truth[entity_id]) for entity_id in query_ids), dtype=np.int32, count=len(query_ids)
    )
    retrieved_positive_count = int(labels.sum())
    total_truth_count = int(truth_counts_all.sum())
    candidate_recall = retrieved_positive_count / total_truth_count if total_truth_count else 1.0
    print(
        f"[pairs] {len(labels):,} retrieved pairs; {retrieved_positive_count:,} positives; "
        f"candidate recall {candidate_recall:.4%}",
        flush=True,
    )
    if len(np.unique(labels)) < 2:
        raise ValueError("The sampled candidate pairs contain only one class; increase --max-queries")

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.seed
    )
    query_rows = np.arange(len(query_ids), dtype=np.int32)
    training_query_rows, validation_query_rows = next(
        splitter.split(query_rows, groups=query_rows)
    )
    training_query_mask = np.zeros(len(query_ids), dtype=bool)
    training_query_mask[training_query_rows] = True
    training_rows = np.flatnonzero(training_query_mask[query_indexes])
    validation_rows = np.flatnonzero(~training_query_mask[query_indexes])
    if len(np.unique(labels[training_rows])) < 2 or len(np.unique(labels[validation_rows])) < 2:
        raise ValueError(
            "A grouped train/validation partition contains only one class; "
            "increase --max-queries or change --validation-fraction"
        )
    training_entity_ids = set()
    for row_index in training_rows:
        training_entity_ids.add(query_ids[query_indexes[row_index]])
        training_entity_ids.add(pair_candidate_ids[row_index])
    vectorizers = build_tfidf_vectorizers(
        entities, training_entity_ids, args.max_tfidf_documents, args.seed
    )
    embedding_arrays = load_embedding_arrays(
        args.embedding_dir, source_row_counts, not args.no_dense_features
    )
    feature_cache_path = (
        args.output_dir / "feature_cache" / f"features-{run_signature[:16]}.npy"
    )
    feature_matrix, feature_names = build_feature_matrix(
        query_ids=query_ids,
        query_indexes=query_indexes,
        candidate_ids=pair_candidate_ids,
        candidate_ranks=candidate_ranks,
        records=entities,
        vectorizers=vectorizers,
        embedding_arrays=embedding_arrays,
        batch_size=args.feature_batch_size,
        cache_path=feature_cache_path,
        run_signature=run_signature,
        resume=not args.no_resume,
    )

    device = args.device
    print(f"[train] fitting XGBoost on {device}", flush=True)
    import xgboost as xgb

    validation_query_indexes = query_indexes[validation_rows]
    validation_query_ids = validation_query_rows.tolist()
    validation_truth_counts = truth_counts_all[validation_query_ids]
    validation_local_index = {global_index: local for local, global_index in enumerate(validation_query_ids)}
    validation_groups = np.fromiter(
        (validation_local_index[index] for index in validation_query_indexes),
        dtype=np.int32,
        count=len(validation_query_indexes),
    )
    training_matrix = xgb.DMatrix(
        feature_matrix[training_rows],
        label=labels[training_rows],
        feature_names=feature_names,
    )
    validation_matrix = xgb.DMatrix(
        feature_matrix[validation_rows],
        label=labels[validation_rows],
        feature_names=feature_names,
    )
    params = xgboost_parameters(device, args.seed, args.n_jobs)
    validation_model, best_round, best_validation_logloss = train_with_checkpoints(
        training_matrix=training_matrix,
        target_rounds=args.n_estimators,
        checkpoint_dir=checkpoint_dir,
        stage="validation",
        params=params,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        validation_matrix=validation_matrix,
        validation_labels=labels[validation_rows],
        early_stopping_rounds=args.early_stopping_rounds,
    )
    validation_probabilities = validation_model.predict(validation_matrix)
    best_threshold, threshold_results = choose_threshold(
        validation_probabilities,
        labels[validation_rows],
        validation_groups,
        validation_truth_counts,
    )
    final_estimators = int(best_round) if best_round is not None else args.n_estimators
    print(
        f"[validation] macro F0.5={best_threshold['macro_f0.5']:.5f} "
        f"at threshold={best_threshold['threshold']:.3f}; "
        f"pair precision={best_threshold['pair_precision']:.4f}, "
        f"pair recall={best_threshold['pair_recall']:.4f}",
        flush=True,
    )

    del training_matrix, validation_matrix, validation_model
    final_training_matrix = xgb.DMatrix(
        feature_matrix, label=labels, feature_names=feature_names
    )
    final_model, _, _ = train_with_checkpoints(
        training_matrix=final_training_matrix,
        target_rounds=final_estimators,
        checkpoint_dir=checkpoint_dir,
        stage="final",
        params=params,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    model_path = args.output_dir / "xgb_pair_matcher.json"
    save_booster_atomic(final_model, model_path)
    try:
        import joblib

        joblib.dump(
            {"char": vectorizers[0], "word": vectorizers[1]},
            args.output_dir / "tfidf_vectorizers.joblib",
            compress=3,
        )
    except ImportError as exc:
        raise RuntimeError("joblib is required to save the fitted TF-IDF vectorizers") from exc

    importance_path = args.output_dir / "feature_importance.tsv"
    gain_importance = final_model.get_score(importance_type="gain")
    importance = sorted(
        ((name, gain_importance.get(name, 0.0)) for name in feature_names),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    with importance_path.open("w", encoding="utf-8", newline="") as importance_file:
        writer = csv.writer(importance_file, delimiter="\t")
        writer.writerow(["feature", "importance"])
        writer.writerows((name, float(value)) for name, value in importance)

    validation_candidate_positive_count = int(labels[validation_rows].sum())
    validation_truth_count = int(validation_truth_counts.sum())
    metrics = {
        "model": "xgboost.train",
        "train_dir": str(train_dir),
        "candidate_file": str(args.candidates.resolve()),
        "embedding_dir": str(args.embedding_dir.resolve()) if embedding_arrays else None,
        "query_count": len(query_ids),
        "candidate_pair_count": len(labels),
        "positive_candidate_pairs": retrieved_positive_count,
        "candidate_recall": candidate_recall,
        "validation_query_count": len(validation_query_ids),
        "validation_pair_count": len(validation_rows),
        "validation_candidate_recall": (
            validation_candidate_positive_count / validation_truth_count
            if validation_truth_count
            else 1.0
        ),
        "validation_best_threshold": best_threshold,
        "threshold_grid": threshold_results,
        "best_iteration": best_round,
        "best_validation_logloss": best_validation_logloss,
        "final_estimators": final_estimators,
        "device": device,
        "seed": args.seed,
        "feature_names": feature_names,
        "run_signature": run_signature,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "feature_cache_path": str(feature_cache_path.resolve()),
        "model_path": str(model_path.resolve()),
        "feature_importance_path": str(importance_path.resolve()),
        "runtime_seconds": time.perf_counter() - started,
    }
    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[saved] model: {model_path}", flush=True)
    print(f"[saved] metrics: {metrics_path}", flush=True)
    print(f"[saved] feature importance: {importance_path}", flush=True)
    print(f"[timeline] complete; total elapsed {format_duration(metrics['runtime_seconds'])}", flush=True)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost pair matcher from retrieved business-entity candidates."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50_000,
        help="Reservoir-sample this many Source 1 training entities; use 0 for all entities.",
    )
    parser.add_argument("--top-k-per-source", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--max-tfidf-documents", type=int, default=200_000)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Persist a resumable model checkpoint every this many boosting rounds.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-free-gpu-gb", type=float, default=2.0)
    parser.add_argument("--no-dense-features", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing feature/model checkpoints and start the current run fresh.",
    )
    args = parser.parse_args(argv)
    if args.max_queries < 0:
        parser.error("--max-queries must be 0 or a positive integer")
    if args.top_k_per_source <= 0:
        parser.error("--top-k-per-source must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.feature_batch_size <= 0 or args.max_tfidf_documents <= 0:
        parser.error("feature batch size and TF-IDF document limit must be positive")
    if args.min_free_gpu_gb < 0:
        parser.error("--min-free-gpu-gb cannot be negative")
    if args.n_estimators <= 0 or args.early_stopping_rounds < 0 or args.checkpoint_every <= 0:
        parser.error("boosting rounds/checkpoint interval must be positive and early stopping nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
