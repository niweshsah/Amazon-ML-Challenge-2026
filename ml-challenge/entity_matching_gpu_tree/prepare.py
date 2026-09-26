"""Build cached retrieval candidates and labeled pair features from Parquet."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from config import Config
from text_features import COMMON_NAME, Record, grams, make_record, pair_features


LOGGER = logging.getLogger(__name__)


def stable_hash(value: str, seed: int) -> int:
    data = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def split_for(query_id: str, seed: int) -> str:
    bucket = stable_hash(query_id, seed) % 100
    return "fit" if bucket < 70 else "early" if bucket < 80 else "threshold" if bucket < 90 else "audit"


def cache_signature(config: Config) -> str:
    manifest = (config.dataset / "manifest.json").read_bytes()
    settings = {
        "seed": config.seed,
        "max_queries": config.max_queries,
        "max_candidates_per_source": config.max_candidates_per_source,
        "feature_batch_size": config.feature_batch_size,
        "version": 5,
    }
    digest = hashlib.sha256(manifest + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def cache_dir(config: Config) -> Path:
    return config.output / "cache" / cache_signature(config)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _raw_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def load_inputs(config: Config) -> tuple[list[Record], list[Record], dict[str, set[str]]]:
    started = time.monotonic()
    t1_rows = _raw_rows(config.dataset / "t1.parquet")
    t1_rows.sort(key=lambda row: stable_hash(row["entity_id"], config.seed))
    t1_rows = t1_rows[:config.max_queries]
    queries = [make_record(row, "S1") for row in t1_rows]
    LOGGER.info("Loaded %s T1 records in %.1fs", len(queries), time.monotonic() - started)
    t2_rows = _raw_rows(config.dataset / "t2.parquet")
    targets = [make_record(row, row["source"]) for row in t2_rows]
    LOGGER.info("Loaded %s T2 records in %.1fs", len(targets), time.monotonic() - started)
    selected = {record.entity_id for record in queries}
    truth = {
        row["source1_entity_id"]: set(row["matched_entity_ids"])
        for row in _raw_rows(config.dataset / "ground_truth.parquet")
        if row["source1_entity_id"] in selected
    }
    if set(truth) != selected:
        raise ValueError("Ground truth does not cover every selected T1")
    return queries, targets, truth


def save_record_cache(path: Path, records: list[Record]) -> None:
    if path.exists():
        return
    from dataclasses import asdict
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist([asdict(row) for row in records]), temporary, compression="zstd")
    os.replace(temporary, path)


class Retriever:
    """Bounded inverted-index retrieval; no label or pool-role access."""

    def __init__(self, targets: list[Record]) -> None:
        self.targets = targets
        self.index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self.name_exact: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, record in enumerate(targets):
            country = record.country.casefold()
            if record.name:
                self.name_exact[country, record.name].append(idx)
            for token in set(record.compact_name.split()):
                if len(token) >= 2:
                    self.index[country, "n", token].append(idx)
            if record.name_script not in {"LATIN", "NONE"}:
                for token in set(record.transliterated_name.split()):
                    if len(token) >= 3 and token not in COMMON_NAME:
                        self.index[country, "t", token].append(idx)
                for fragment in grams(record.transliterated_name, 4):
                    if len(fragment.strip()) >= 3:
                        self.index[country, "h", fragment].append(idx)
            for token in set(record.address.split()):
                if token.isdigit() or len(token) >= 5:
                    self.index[country, "a", token].append(idx)
            for fragment in grams(record.folded_name, 4):
                if len(fragment.strip()) >= 3:
                    self.index[country, "g", fragment].append(idx)
        LOGGER.info("Retrieval index built: %s keys", len(self.index))

    def retrieve(self, query: Record, per_source: int) -> list[tuple[int, float]]:
        country = query.country.casefold()
        scores: dict[int, float] = defaultdict(float)
        for idx in self.name_exact.get((country, query.name), ()):  # exact path
            scores[idx] += 20.0
        for kind, tokens, take, limit, weight in (
            ("n", set(query.compact_name.split()), 5, 5_000, 2.0),
            ("t", set(query.folded_name.split()) - COMMON_NAME, 4, 3_000, 1.5),
            ("a", {t for t in query.address.split() if t.isdigit() or len(t) >= 5}, 4, 2_500, 1.0),
            ("g", grams(query.folded_name, 4), 5, 1_000, 0.65),
            ("h", grams(query.folded_name, 4), 5, 1_500, 0.70),
        ):
            candidates = []
            for token in tokens:
                posting = self.index.get((country, kind, token))
                if posting and len(posting) <= limit:
                    candidates.append((len(posting), token, posting))
            for count, _, posting in sorted(candidates)[:take]:
                contribution = weight * math.log1p(1_000_000 / count)
                for idx in posting:
                    scores[idx] += contribution
        # Each source receives its own cap, preserving multi-match output.
        by_source: dict[str, list[tuple[int, float]]] = {"S2": [], "S3": []}
        for idx, score in scores.items():
            by_source[self.targets[idx].source].append((idx, score))
        selected = []
        for source in ("S2", "S3"):
            pool = heapq.nlargest(
                max(200, per_source * 5), by_source[source],
                key=lambda pair: (pair[1], self.targets[pair[0]].entity_id),
            )
            ranked = []
            for idx, score in pool:
                target = self.targets[idx]
                name_score = max(
                    fuzz.WRatio(query.folded_name, target.folded_name),
                    fuzz.WRatio(query.folded_name, target.transliterated_name),
                ) / 100.0
                address_score = fuzz.token_set_ratio(query.address, target.address) / 100.0 if query.address and target.address else 0.0
                rerank_score = 4.0 * name_score + 2.0 * address_score + 0.3 * math.log1p(score)
                ranked.append((idx, rerank_score, score))
            by_initial = sorted(ranked, key=lambda pair: (-pair[2], self.targets[pair[0]].entity_id))
            by_fuzzy = sorted(ranked, key=lambda pair: (-pair[1], self.targets[pair[0]].entity_id))
            keep: dict[int, float] = {}
            for idx, rerank_score, _ in by_initial[:max(1, per_source * 3 // 4)]:
                keep[idx] = rerank_score
            for idx, rerank_score, _ in by_fuzzy:
                if len(keep) >= per_source:
                    break
                keep[idx] = rerank_score
            selected.extend(sorted(keep.items(), key=lambda pair: (-pair[1], self.targets[pair[0]].entity_id)))
        return selected


def build_frequencies(queries: list[Record], targets: list[Record]) -> dict[str, Counter[str]]:
    frequencies = {key: Counter() for key in ("name", "address", "name_token", "address_token")}
    # Unsupervised corpus frequencies are available for every retrieval target at inference.
    from itertools import chain
    for record in chain(queries, targets):
        frequencies["name"][record.name] += 1
        frequencies["address"][record.address] += 1
        frequencies["name_token"].update(set(record.name.split()))
        frequencies["address_token"].update(set(record.address.split()))
    return frequencies


def prepare(config: Config) -> dict:
    destination = cache_dir(config)
    manifest_path = destination / "prepare_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("signature") == cache_signature(config):
            LOGGER.info("Reusing complete feature cache %s", destination)
            return manifest
    destination.mkdir(parents=True, exist_ok=True)
    parts = destination / "parts"
    parts.mkdir(exist_ok=True)
    queries, targets, truth = load_inputs(config)
    save_record_cache(destination / "t1_processed.parquet", queries)
    save_record_cache(destination / "t2_processed.parquet", targets)
    retriever = Retriever(targets)
    frequencies = build_frequencies(queries, targets)
    feature_names: list[str] = []
    counts = Counter()
    singleton_count = 0
    start_time = time.monotonic()
    for start in range(0, len(queries), config.feature_batch_size):
        batch = queries[start:start + config.feature_batch_size]
        part_path = parts / f"part_{start:06d}.parquet"
        rows = []
        for local, query in enumerate(batch):
            query_index = start + local
            expected = truth[query.entity_id]
            singleton_count += int(not expected)
            retrieved = retriever.retrieve(query, config.max_candidates_per_source)
            found = 0
            for target_index, score in retrieved:
                target = targets[target_index]
                label = int(target.entity_id in expected)
                found += label
                features = pair_features(query, target, frequencies, score)
                if not feature_names:
                    feature_names = list(features)
                rows.append({
                    "source1_entity_id": query.entity_id,
                    "candidate_entity_id": target.entity_id,
                    "query_index": query_index,
                    "split": split_for(query.entity_id, config.seed),
                    "label": label,
                    "retrieval_score": score,
                    "raw_name_t1": query.raw_name,
                    "raw_name_t2": target.raw_name,
                    "raw_address_t1": query.raw_address,
                    "raw_address_t2": target.raw_address,
                    "normalized_name_t1": query.name,
                    "normalized_name_t2": target.name,
                    "normalized_address_t1": query.address,
                    "normalized_address_t2": target.address,
                    "country_t1": query.country,
                    "country_t2": target.country,
                    "source_t2": target.source,
                    **features,
                })
            counts["pairs"] += len(retrieved)
            counts["retrieved_positives"] += found
            counts["ground_truth_positives"] += len(expected)
            counts[f"{split_for(query.entity_id, config.seed)}_queries"] += 1
            if (query_index + 1) % config.retrieval_progress_every == 0:
                LOGGER.info("Prepared %s/%s T1; %s pairs; pair recall %.4f; elapsed %.1fs", query_index + 1, len(queries), counts["pairs"], counts["retrieved_positives"] / max(1, counts["ground_truth_positives"]), time.monotonic() - start_time)
        if rows:
            temporary = part_path.with_suffix(".tmp.parquet")
            pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd", row_group_size=10_000)
            os.replace(temporary, part_path)
        LOGGER.info("Saved %s (%s rows)", part_path, len(rows))
    query_rows = [{
        "source1_entity_id": query.entity_id,
        "query_index": idx,
        "split": split_for(query.entity_id, config.seed),
        "truth_count": len(truth[query.entity_id]),
        "truth_ids": sorted(truth[query.entity_id]),
        "country": query.country,
        "name_script": query.name_script,
    } for idx, query in enumerate(queries)]
    pq.write_table(pa.Table.from_pylist(query_rows), destination / "queries.parquet", compression="zstd")
    manifest = {
        "complete": True,
        "signature": cache_signature(config),
        "config": config.serializable(),
        "feature_names": feature_names,
        "counts": dict(counts),
        "singletons": singleton_count,
        "candidate_recall": counts["retrieved_positives"] / max(1, counts["ground_truth_positives"]),
        "dataset_manifest_sha256": hashlib.sha256((config.dataset / "manifest.json").read_bytes()).hexdigest(),
        "seconds": time.monotonic() - start_time,
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info("Feature cache complete: %s", destination)
    return manifest
