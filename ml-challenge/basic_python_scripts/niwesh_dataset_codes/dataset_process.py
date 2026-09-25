#!/usr/bin/env python3
"""Profile the three-source business entity resolution dataset without altering it.

Example (from the repository root)::

    python3 ml-challenge/dataset-explore/dataset_analysis.py \
        --data-root ../student_resource/student_resource/dataset

The default pass is exhaustive for counts and bounded for displayed examples. Pairwise
similarity experiments require --expensive-analysis. No network access is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import mmap
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unicodedata
from typing import Any, Iterator

import numpy as np
import pandas as pd


LOG = logging.getLogger("dataset_analysis")
FIELDS = ("business_name", "business_address")
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
SOURCES = ("source1", "source2", "source3")
ID_RE = re.compile(r"^S[123]-[0-9]+$")
NUMBER_RE = re.compile(r"\b\d+[A-Za-z]?\b", re.UNICODE)
POSTAL_RE = re.compile(r"\b(?:\d{5}(?:-\d{4})?|\d{6})\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
NAME_VARIANTS = (("ltd", "limited"), ("pvt", "private"),
                 ("corp", "corporation"), ("inc", "incorporated"),
                 ("llc", "llc"), ("&", "and"))
ADDRESS_VARIANTS = (("rd", "road"), ("st", "street"),
                    ("ave", "avenue"), ("blvd", "boulevard"))


@dataclass(frozen=True)
class AnalysisConfig:
    data_root: Path
    output_dir: Path
    backend: str = "auto"
    chunk_size: int = 20_000
    sample_size: int = 5_000
    random_seed: int = 42
    top_k_words: int = 100
    top_k_values: int = 50
    tokenizer: str = "unicode"
    min_token_length: int = 2
    min_skew_df: int = 5
    skew_candidate_multiplier: int = 5
    ngram_min: int = 1
    ngram_max: int = 1
    char_ngram_width: int = 3
    short_length: int = 3
    long_length: int = 200
    punctuation_ratio: float = 0.30
    duplicate_limit: int = 50
    example_rows: int = 30
    similarity_threshold: float = 0.35
    rare_token_max_df: int = 1_000
    nonmatch_sample_limit: int = 1_000
    prefix_length: int = 4
    plots: bool = True
    expensive_analysis: bool = False


@dataclass
class Output:
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, row: dict[str, Any]) -> None:
        self.tables.setdefault(name, []).append(row)

    def warn(self, message: str) -> None:
        LOG.warning(message)
        self.warnings.append(message)


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Directory containing train/ and test/ directories")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "analysis_output")
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--top-k-words", type=int, default=100)
    parser.add_argument("--top-k-values", type=int, default=50)
    parser.add_argument("--tokenizer", choices=("unicode", "whitespace"), default="unicode")
    parser.add_argument("--min-token-length", type=int, default=2)
    parser.add_argument("--min-skew-df", type=int, default=5)
    parser.add_argument("--skew-candidate-multiplier", type=int, default=5)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=1,
                        help="Maximum token n-gram size; use 2 or 3 for more expensive analysis")
    parser.add_argument("--char-ngram-width", type=int, default=3)
    parser.add_argument("--short-length", type=int, default=3)
    parser.add_argument("--long-length", type=int, default=200)
    parser.add_argument("--punctuation-ratio", type=float, default=0.30)
    parser.add_argument("--duplicate-limit", type=int, default=50)
    parser.add_argument("--example-rows", type=int, default=30)
    parser.add_argument("--similarity-threshold", type=float, default=0.35)
    parser.add_argument("--rare-token-max-df", type=int, default=1_000)
    parser.add_argument("--nonmatch-sample-limit", type=int, default=1_000)
    parser.add_argument("--prefix-length", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--expensive-analysis", action="store_true")
    args = parser.parse_args()
    for name in ("chunk_size", "sample_size", "top_k_words", "top_k_values",
                 "min_token_length", "min_skew_df", "skew_candidate_multiplier",
                 "ngram_min", "ngram_max",
                 "char_ngram_width", "short_length", "long_length", "duplicate_limit",
                 "example_rows", "rare_token_max_df", "nonmatch_sample_limit",
                 "prefix_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ngram_max < args.ngram_min or args.ngram_max > 3:
        parser.error("Require 1 <= --ngram-min <= --ngram-max <= 3")
    if not 0 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be in [0, 1]")
    if not 0 <= args.punctuation_ratio <= 1:
        parser.error("--punctuation-ratio must be in [0, 1]")
    return AnalysisConfig(
        data_root=args.data_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        backend=args.backend, chunk_size=args.chunk_size,
        sample_size=args.sample_size, random_seed=args.random_seed,
        top_k_words=args.top_k_words, top_k_values=args.top_k_values,
        tokenizer=args.tokenizer,
        min_token_length=args.min_token_length, min_skew_df=args.min_skew_df,
        skew_candidate_multiplier=args.skew_candidate_multiplier,
        ngram_min=args.ngram_min, ngram_max=args.ngram_max,
        char_ngram_width=args.char_ngram_width, short_length=args.short_length,
        long_length=args.long_length, punctuation_ratio=args.punctuation_ratio,
        duplicate_limit=args.duplicate_limit,
        example_rows=args.example_rows,
        similarity_threshold=args.similarity_threshold,
        rare_token_max_df=args.rare_token_max_df,
        nonmatch_sample_limit=args.nonmatch_sample_limit,
        prefix_length=args.prefix_length, plots=not args.no_plots,
        expensive_analysis=args.expensive_analysis,
    )


def discover_dataset(root: Path) -> dict[tuple[str, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files = {(split, source): root / split / f"{split}_{source}.tsv"
             for split in ("train", "test") for source in SOURCES}
    files[("train", "ground_truth")] = root / "train" / "train_ground_truth.tsv"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected TSV files:\n" + "\n".join(missing))
    for (split, source), path in files.items():
        expected = TRUTH_COLUMNS if source == "ground_truth" else SOURCE_COLUMNS
        try:
            actual = tuple(pd.read_csv(path, sep="\t", nrows=0,
                                       encoding="utf-8-sig").columns)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(f"Cannot read TSV header in {path}: {exc}") from exc
        if actual != expected:
            raise ValueError(f"Unexpected columns in {path}: {actual}; expected {expected}")
    return files


class Backend:
    """Keep optional GPU imports and supported vector operations in one place."""

    def __init__(self, preference: str, output: Output) -> None:
        self.name = "cpu"
        self.preference = preference
        self.output = output
        self.cudf: Any = None
        self.cupy: Any = None
        if preference == "cpu":
            return
        try:
            import cudf  # type: ignore[import-not-found]
            import cupy  # type: ignore[import-not-found]
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device found")
            self.cudf, self.cupy = cudf, cupy
            self.name = "gpu"
        except Exception as exc:
            if preference == "gpu":
                raise RuntimeError(f"--backend gpu requires cuDF, CuPy and a CUDA GPU: {exc}") from exc
            output.warn(f"GPU backend unavailable ({exc}); using CPU")

    def lengths(self, values: pd.Series) -> np.ndarray:
        """Use cuDF for large string-length batches; retain CPU for Unicode tokenization."""
        if self.name == "gpu" and len(values) >= 10_000:
            try:
                return self.cudf.Series(values.astype(str).tolist()).str.len().to_pandas().to_numpy()
            except Exception as exc:
                if self.preference == "gpu":
                    raise RuntimeError(f"GPU string-length operation failed: {exc}") from exc
                self.output.warn(f"GPU string-length operation failed ({exc}); using CPU")
                self.name = "cpu"
        return values.str.len().to_numpy(dtype=np.int64)


def chunks(path: Path, config: AnalysisConfig) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False,
                               encoding="utf-8-sig", chunksize=config.chunk_size,
                               on_bad_lines="error")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Malformed or unreadable TSV {path}: {exc}") from exc


def contains_null_byte(path: Path) -> bool:
    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            return False
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return mapped.find(b"\x00") >= 0


def unicode_words(value: str) -> list[str]:
    """Keep Unicode letters, combining marks, and numbers within the same token."""
    words: list[str] = []
    current: list[str] = []
    for char in value:
        if unicodedata.category(char)[0] in "LMN":
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(unicode_words(value))


def word_tokens(value: str, strategy: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value)
    return unicode_words(value) if strategy == "unicode" else value.split()


def tokens(value: str, min_length: int, strategy: str = "unicode") -> list[str]:
    return [word for word in word_tokens(value.casefold(), strategy)
            if len(word) >= min_length]


def ngrams(words: list[str], minimum: int, maximum: int) -> Iterator[tuple[int, str]]:
    for size in range(minimum, maximum + 1):
        for pos in range(len(words) - size + 1):
            yield size, " ".join(words[pos:pos + size])


def safe_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def value_missing(value: str) -> bool:
    return not value.strip()


def format_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class Stats:
    """Exact bounded-cardinality length histogram with mergeable moments."""

    def __init__(self) -> None:
        self.hist: Counter[int] = Counter()
        self.count = 0
        self.total = 0
        self.total_sq = 0

    def add(self, number: int) -> None:
        self.hist[number] += 1
        self.count += 1
        self.total += number
        self.total_sq += number * number

    def percentile(self, fraction: float) -> float | None:
        if not self.count:
            return None
        target = (self.count - 1) * fraction
        left, right = math.floor(target), math.ceil(target)
        positions: dict[int, int] = {}
        seen = 0
        for value, count in sorted(self.hist.items()):
            for index in (left, right):
                if index not in positions and seen <= index < seen + count:
                    positions[index] = value
            seen += count
            if len(positions) == 2 or left == right and left in positions:
                break
        return positions[left] + (positions[right] - positions[left]) * (target - left)

    def row(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        mean = self.total / self.count
        result = {"count": self.count, "min": min(self.hist), "max": max(self.hist),
                  "mean": mean, "median": self.percentile(0.5),
                  "std": math.sqrt(max(0, self.total_sq / self.count - mean * mean))}
        result.update({f"p{p}": self.percentile(p / 100) for p in PERCENTILES})
        return result


def make_index(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        CREATE TABLE records (
            split TEXT NOT NULL, source TEXT NOT NULL, entity_id TEXT NOT NULL,
            business_name TEXT NOT NULL, business_address TEXT NOT NULL,
            country TEXT NOT NULL, norm_name TEXT NOT NULL, norm_address TEXT NOT NULL,
            sorted_name TEXT NOT NULL, sorted_address TEXT NOT NULL
        );
        CREATE TABLE truth (
            source1_entity_id TEXT NOT NULL, matched_entity_id TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE truth_rows (
            source1_entity_id TEXT NOT NULL, match_count INTEGER NOT NULL,
            source2_count INTEGER NOT NULL, source3_count INTEGER NOT NULL,
            matched_entity_ids TEXT NOT NULL
        );
        CREATE TABLE token_counts (
            split TEXT NOT NULL, source TEXT NOT NULL, country TEXT NOT NULL,
            field TEXT NOT NULL, n INTEGER NOT NULL, token TEXT NOT NULL,
            frequency INTEGER NOT NULL, document_frequency INTEGER NOT NULL,
            PRIMARY KEY (split, source, country, field, n, token)
        ) WITHOUT ROWID;
    """)
    return db


def index_records(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE INDEX records_id ON records(split, entity_id);
        CREATE INDEX records_name ON records(split, norm_name);
        CREATE INDEX records_address ON records(split, norm_address);
        CREATE INDEX truth_s1 ON truth(source1_entity_id);
        CREATE INDEX truth_match ON truth(matched_entity_id);
        CREATE INDEX truth_rows_s1 ON truth_rows(source1_entity_id);
        CREATE INDEX token_by_value ON token_counts(field, n, token, split);
    """)


def issue(out: Output, split: str, source: str, kind: str,
          detail: str, entity_id: str = "") -> None:
    counts = out.summary.setdefault("quality_issue_counts", {})
    key = f"{split}/{source}/{kind}"
    counts[key] = counts.get(key, 0) + 1
    if counts[key] <= 30:
        out.add("data_quality_issues", {"split": split, "source": source,
                                        "issue": kind, "entity_id": entity_id,
                                        "detail": detail})


def sample_add(bucket: list[Any], item: Any, seen: int,
               limit: int, rng: np.random.Generator) -> None:
    """Uniform bounded reservoir for examples, independent of row order."""
    if len(bucket) < limit:
        bucket.append(item)
    else:
        position = int(rng.integers(seen))
        if position < limit:
            bucket[position] = item


def profile_sources(files: dict[tuple[str, str], Path], config: AnalysisConfig,
                    backend: Backend, db: sqlite3.Connection, out: Output) -> None:
    rng = np.random.default_rng(config.random_seed)
    length_stats: dict[tuple[str, str, str, str, str], Stats] = defaultdict(Stats)
    missing_counts: Counter[tuple[str, str, str, str]] = Counter()
    country_counts: Counter[tuple[str, str, str]] = Counter()
    char_counts: Counter[tuple[str, str, str, str]] = Counter()
    variant_counts: Counter[tuple[str, str, str, str]] = Counter()

    for split in ("train", "test"):
        for source in SOURCES:
            path = files[(split, source)]
            LOG.info("Profiling %s", path)
            if contains_null_byte(path):
                issue(out, split, source, "null_byte_in_file", str(path))
            row_count = 0
            memory_bytes = 0
            empty: Counter[str] = Counter()
            whitespace: Counter[str] = Counter()
            nulls: Counter[str] = Counter()
            examples: list[dict[str, Any]] = []
            for frame in chunks(path, config):
                memory_bytes += int(frame.memory_usage(index=False, deep=True).sum())
                for column in SOURCE_COLUMNS:
                    nulls[column] += int(frame[column].isna().sum())
                    empty[column] += int(frame[column].eq("").sum())
                    whitespace[column] += int((frame[column].ne("") &
                                               frame[column].str.fullmatch(r"\s+")).sum())
                name_lengths = backend.lengths(frame["business_name"].fillna(""))
                address_lengths = backend.lengths(frame["business_address"].fillna(""))
                record_rows: list[tuple[str, ...]] = []
                token_acc: Counter[tuple[str, str, str, str, int, str]] = Counter()
                token_doc: Counter[tuple[str, str, str, str, int, str]] = Counter()
                for index, raw in enumerate(frame.itertuples(index=False, name=None)):
                    entity_id, name, address, country = map(safe_text, raw)
                    row_count += 1
                    if any(pd.isna(value) for value in raw):
                        issue(out, split, source, "null_or_short_row",
                              f"parsed row {row_count} has a null field", entity_id)
                    if not entity_id.strip():
                        issue(out, split, source, "empty_id", "ID is blank", entity_id)
                    elif not ID_RE.fullmatch(entity_id) or not entity_id.startswith(
                            f"S{SOURCES.index(source) + 1}-"):
                        issue(out, split, source, "unexpected_id", entity_id, entity_id)
                    for column, value in zip(SOURCE_COLUMNS, (entity_id, name, address, country)):
                        if value != value.strip():
                            issue(out, split, source, "edge_whitespace", column, entity_id)
                        if "\ufffd" in value or "\x00" in value or any(
                                unicodedata.category(char) == "Cc" and char not in "\t\n\r"
                                for char in value):
                            issue(out, split, source, "strange_character", column, entity_id)
                        if value == column:
                            issue(out, split, source, "embedded_header_value", column, entity_id)
                    if entity_id == "entity_id" and name == "business_name":
                        issue(out, split, source, "embedded_header_row", "Repeated header", entity_id)
                    n_name, n_address = normalized(name), normalized(address)
                    sorted_name = " ".join(sorted(n_name.split()))
                    sorted_address = " ".join(sorted(n_address.split()))
                    record_rows.append((split, source, entity_id, name, address, country,
                                        n_name, n_address, sorted_name, sorted_address))
                    country_counts[(split, source, country)] += 1
                    present = tuple(int(not value_missing(v)) for v in (name, address, country))
                    combination = f"name={present[0]},address={present[1]},country={present[2]}"
                    for grouping in (country, "*"):
                        missing_counts[(split, source, grouping, combination)] += 1
                    for field_name, value, length in ((FIELDS[0], name, int(name_lengths[index])),
                                                      (FIELDS[1], address, int(address_lengths[index]))):
                        words = tokens(value, config.min_token_length, config.tokenizer)
                        for grouping in (country, "*"):
                            prefix = (split, source, grouping, field_name)
                            length_stats[(*prefix, "characters")].add(length)
                            length_stats[(*prefix, "tokens")].add(len(words))
                            length_stats[(*prefix, "unique_tokens")].add(len(set(words)))
                        pattern_prefix = (split, source, field_name)
                        if value and len(words) == 1:
                            char_counts[(*pattern_prefix, "single_token")] += 1
                        if value.isnumeric():
                            char_counts[(*pattern_prefix, "numeric_only")] += 1
                        if value.isalpha():
                            char_counts[(*pattern_prefix, "alphabetic_only")] += 1
                        if length <= config.short_length and value.strip():
                            char_counts[(*pattern_prefix, f"very_short_le_{config.short_length}")] += 1
                        if length >= config.long_length:
                            char_counts[(*pattern_prefix, f"very_long_ge_{config.long_length}")] += 1
                        letters = sum(char.isalpha() for char in value)
                        punctuation = sum(unicodedata.category(char)[0] in "PS" for char in value)
                        char_counts[(*pattern_prefix, "letters")] += letters
                        char_counts[(*pattern_prefix, "digits")] += sum(char.isdigit() for char in value)
                        char_counts[(*pattern_prefix, "punctuation")] += punctuation
                        char_counts[(*pattern_prefix, "uppercase")] += sum(char.isupper() for char in value)
                        char_counts[(*pattern_prefix, "lowercase")] += sum(char.islower() for char in value)
                        char_counts[(*pattern_prefix, "non_ascii")] += sum(ord(char) > 127 for char in value)
                        char_counts[(*pattern_prefix, "repeated_punctuation")] += int(bool(
                            re.search(r"([^\w\s])\1+", value)))
                        char_counts[(*pattern_prefix,
                                     f"punctuation_heavy_gt_{config.punctuation_ratio:.0%}")] += int(
                            bool(length and punctuation / length > config.punctuation_ratio))
                        char_counts[(*pattern_prefix, "postal_like")] += int(bool(POSTAL_RE.search(value)))
                        char_counts[(*pattern_prefix, "phone_like")] += int(bool(PHONE_RE.search(value)))
                        char_counts[(*pattern_prefix, "address_number")] += int(bool(NUMBER_RE.search(value)))
                        char_counts[(*pattern_prefix, "multiple_spaces")] += int("  " in value)
                        char_counts[(*pattern_prefix, "contains_tab_or_newline")] += int(
                            "\t" in value or "\n" in value)
                        for char in value:
                            if unicodedata.category(char)[0] in "PS":
                                char_counts[(*pattern_prefix, f"punct:{char}")] += 1
                            if ord(char) > 127:
                                script = unicodedata.name(char, "UNKNOWN").split()[0]
                                char_counts[(*pattern_prefix, f"script:{script}")] += 1
                        variants = NAME_VARIANTS if field_name == "business_name" else ADDRESS_VARIANTS
                        word_set = set(unicode_words(value.casefold()))
                        for left, right in variants:
                            if left in word_set or left == "&" and "&" in value:
                                variant_counts[(*pattern_prefix, left)] += 1
                            if right in word_set:
                                variant_counts[(*pattern_prefix, right)] += 1
                        for token_field, token_list in ((field_name, words),
                                                        (f"{field_name}_raw", word_tokens(value, config.tokenizer))):
                            items = ngrams(token_list, 1,
                                           config.ngram_max if token_field == field_name else 1)
                            for n, token in items:
                                key = (split, source, country, token_field, n, token)
                                token_acc[key] += 1
                            for n, token in set(ngrams(token_list,
                                                       1,
                                                       config.ngram_max if token_field == field_name else 1)):
                                token_doc[(split, source, country, token_field, n, token)] += 1
                    sample_add(examples, {"split": split, "source": source,
                                          "entity_id": entity_id, "business_name": name,
                                          "business_address": address, "country": country},
                               row_count, config.example_rows, rng)
                db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)", record_rows)
                db.executemany("""INSERT INTO token_counts VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(split,source,country,field,n,token) DO UPDATE SET
                    frequency=frequency+excluded.frequency,
                    document_frequency=document_frequency+excluded.document_frequency""",
                    ((*key, frequency, token_doc[key]) for key, frequency in token_acc.items()))
                db.commit()
                LOG.info("%s: %s rows", path.name, f"{row_count:,}")
            out.add("dataset_summary", {"split": split, "source": source,
                                        "path": str(path), "file_size_bytes": path.stat().st_size,
                                        "rows": row_count, "columns": len(SOURCE_COLUMNS),
                                        "column_names": ",".join(SOURCE_COLUMNS),
                                        "approx_memory_bytes": memory_bytes})
            for column in SOURCE_COLUMNS:
                out.add("column_summary", {"split": split, "source": source,
                                           "column": column, "dtype": "string",
                                           "null_count": nulls[column],
                                           "null_percent": 100 * nulls[column] / max(1, row_count),
                                           "empty_count": empty[column],
                                           "whitespace_only_count": whitespace[column],
                                           "missing_count": nulls[column] + empty[column] +
                                                            whitespace[column],
                                           "missing_percent": 100 * (nulls[column] + empty[column] +
                                                                     whitespace[column]) / max(1, row_count)})
            out.tables.setdefault("example_rows", []).extend(examples)

    for (split, source, country, combination), count in sorted(missing_counts.items()):
        denominator = sum(v for (s, src, c, _), v in missing_counts.items()
                          if (s, src, c) == (split, source, country))
        out.add("missing_values", {"split": split, "source": source, "country": country,
                                   "combination": combination, "rows": count,
                                   "percent": 100 * count / denominator})
    for (split, source, country), count in sorted(country_counts.items()):
        out.add("country_summary", {"split": split, "source": source,
                                    "country": country, "rows": count})
    for (split, source, country, field_name, measure), stats in sorted(length_stats.items()):
        out.add("string_statistics", {"split": split, "source": source,
                                      "country": country, "field": field_name,
                                      "measure": measure, **stats.row()})
    for (split, source, field_name, pattern), count in sorted(char_counts.items()):
        out.add("character_patterns", {"split": split, "source": source,
                                       "field": field_name, "pattern": pattern, "count": count})
    for (split, source, field_name, variant), count in sorted(variant_counts.items()):
        out.add("normalization_patterns", {"split": split, "source": source,
                                           "field": field_name, "variant": variant, "count": count})


def profile_ground_truth(path: Path, config: AnalysisConfig,
                         db: sqlite3.Connection, out: Output) -> None:
    LOG.info("Profiling %s", path)
    if contains_null_byte(path):
        issue(out, "train", "ground_truth", "null_byte_in_file", str(path))
    rows = 0
    pairs = 0
    counts: Counter[int] = Counter()
    s2_counts: Counter[int] = Counter()
    s3_counts: Counter[int] = Counter()
    source_patterns: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    empty: Counter[str] = Counter()
    whitespace: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    memory_bytes = 0
    for frame in chunks(path, config):
        memory_bytes += int(frame.memory_usage(index=False, deep=True).sum())
        for column in TRUTH_COLUMNS:
            nulls[column] += int(frame[column].isna().sum())
            empty[column] += int(frame[column].eq("").sum())
            whitespace[column] += int((frame[column].ne("") &
                                       frame[column].str.fullmatch(r"\s+")).sum())
        truth_rows: list[tuple[str, int, int, int, str]] = []
        pair_rows: list[tuple[str, str, str]] = []
        for s1_raw, matches_raw in frame.itertuples(index=False, name=None):
            s1, raw_matches = safe_text(s1_raw), safe_text(matches_raw)
            rows += 1
            for column, value in ((TRUTH_COLUMNS[0], s1), (TRUTH_COLUMNS[1], raw_matches)):
                if value not in examples[column] and len(examples[column]) < 5:
                    examples[column].append(value)
            if not ID_RE.fullmatch(s1) or not s1.startswith("S1-"):
                issue(out, "train", "ground_truth", "malformed_s1_id", s1, s1)
            ids = [] if not raw_matches.strip() else [part.strip() for part in raw_matches.split(",")]
            if len(ids) != len(set(ids)):
                issue(out, "train", "ground_truth", "duplicate_match_in_list", raw_matches[:200], s1)
            if "" in ids:
                issue(out, "train", "ground_truth", "empty_match_id", raw_matches[:200], s1)
            s2_count = s3_count = 0
            for matched in ids:
                if not ID_RE.fullmatch(matched):
                    issue(out, "train", "ground_truth", "malformed_match_id", matched, s1)
                if matched.startswith("S1-"):
                    issue(out, "train", "ground_truth", "source1_as_match", matched, s1)
                source = "source2" if matched.startswith("S2-") else (
                    "source3" if matched.startswith("S3-") else "unknown")
                s2_count += source == "source2"
                s3_count += source == "source3"
                pair_rows.append((s1, matched, source))
            truth_rows.append((s1, len(ids), s2_count, s3_count, raw_matches))
            pairs += len(ids)
            counts[len(ids)] += 1
            s2_counts[s2_count] += 1
            s3_counts[s3_count] += 1
            pattern = "singleton" if not ids else (
                "both_s2_s3" if s2_count and s3_count else
                "only_s2" if s2_count else "only_s3" if s3_count else "unknown")
            source_patterns[pattern] += 1
        db.executemany("INSERT INTO truth_rows VALUES (?,?,?,?,?)", truth_rows)
        db.executemany("INSERT INTO truth VALUES (?,?,?)", pair_rows)
        db.commit()
        LOG.info("%s: %s rows", path.name, f"{rows:,}")
    out.add("dataset_summary", {"split": "train", "source": "ground_truth",
                                "path": str(path), "file_size_bytes": path.stat().st_size,
                                "rows": rows, "columns": 2,
                                "column_names": ",".join(TRUTH_COLUMNS),
                                "approx_memory_bytes": memory_bytes})
    for column in TRUTH_COLUMNS:
        out.add("column_summary", {"split": "train", "source": "ground_truth",
                                   "column": column, "dtype": "string",
                                   "null_count": nulls[column], "empty_count": empty[column],
                                   "null_percent": 100 * nulls[column] / max(1, rows),
                                   "whitespace_only_count": whitespace[column],
                                   "missing_count": nulls[column] + empty[column] +
                                                    whitespace[column],
                                   "example_values": json.dumps(examples[column], ensure_ascii=False),
                                   "missing_percent": 100 * (nulls[column] + empty[column] +
                                                             whitespace[column]) / max(1, rows)})
    for amount, n_rows in sorted(counts.items()):
        out.add("ground_truth_summary", {"measure": "match_count_distribution",
                                         "category": amount, "count": n_rows})
    for source, distribution in (("source2", s2_counts), ("source3", s3_counts)):
        for amount, n_rows in sorted(distribution.items()):
            out.add("ground_truth_summary", {"measure": f"{source}_match_count_distribution",
                                             "category": amount, "count": n_rows})
    for category, count in sorted(source_patterns.items()):
        out.add("ground_truth_summary", {"measure": "source_pattern",
                                         "category": category, "count": count})
    for name, value in (("source1_rows", rows), ("match_pairs", pairs),
                        ("singletons", counts[0]), ("singleton_percent", 100 * counts[0] / max(1, rows)),
                        ("one_match", counts[1]),
                        ("multiple_matches", sum(v for k, v in counts.items() if k > 1)),
                        ("max_matches", max(counts, default=0))):
        out.add("ground_truth_summary", {"measure": name, "category": "all", "count": value})


def query_one(db: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    result = db.execute(sql, params).fetchone()
    return result[0] if result else None


def validate_ids(db: sqlite3.Connection, out: Output) -> None:
    checks = (
        ("duplicate_record_id", """SELECT split, source, entity_id, COUNT(*) n
            FROM records GROUP BY split, source, entity_id HAVING n > 1"""),
        ("id_across_files", """SELECT MIN(split), MIN(source), entity_id, COUNT(*) n
            FROM records GROUP BY entity_id HAVING n > 1"""),
        ("duplicate_truth_s1", """SELECT 'train', 'ground_truth', source1_entity_id, COUNT(*) n
            FROM truth_rows GROUP BY source1_entity_id HAVING n > 1"""),
        ("duplicate_truth_row", """SELECT 'train', 'ground_truth', source1_entity_id, COUNT(*) n
            FROM truth_rows GROUP BY source1_entity_id, matched_entity_ids HAVING n > 1"""),
        ("truth_s1_missing", """SELECT 'train', 'ground_truth', t.source1_entity_id, 1
            FROM truth_rows t LEFT JOIN records r ON r.split='train' AND r.source='source1'
              AND r.entity_id=t.source1_entity_id WHERE r.entity_id IS NULL"""),
        ("truth_match_missing", """SELECT 'train', 'ground_truth', t.matched_entity_id, 1
            FROM truth t LEFT JOIN records r ON r.split='train' AND r.entity_id=t.matched_entity_id
              AND r.source=t.source WHERE r.entity_id IS NULL"""),
        ("match_assigned_to_multiple_s1", """SELECT 'train', 'ground_truth', matched_entity_id,
            COUNT(DISTINCT source1_entity_id) n FROM truth GROUP BY matched_entity_id HAVING n > 1"""),
    )
    for kind, sql in checks:
        count = 0
        for split, source, entity_id, number in db.execute(sql):
            count += 1
            issue(out, split, source, kind, f"occurrences={number}", entity_id)
        LOG.info("%s: %s", kind, f"{count:,}")
    missing_gt = db.execute("""SELECT r.entity_id FROM records r
        LEFT JOIN truth_rows t ON t.source1_entity_id=r.entity_id
        WHERE r.split='train' AND r.source='source1' AND t.source1_entity_id IS NULL""")
    for (entity_id,) in missing_gt:
        issue(out, "train", "source1", "missing_truth_row", "No ground-truth row", entity_id)


def duplicate_analysis(db: sqlite3.Connection, config: AnalysisConfig, out: Output) -> None:
    definitions = (
        ("exact_row", "entity_id, business_name, business_address, country"),
        ("identical_content_different_id", "business_name, business_address, country"),
        ("raw_name", "business_name"), ("raw_address", "business_address"),
        ("raw_name_address", "business_name, business_address"),
        ("normalized_name", "norm_name"),
        ("normalized_address", "norm_address"),
        ("normalized_name_address", "norm_name, norm_address"),
        ("sorted_name", "sorted_name"),
    )
    for split in ("train", "test"):
        for source in SOURCES:
            total_rows = next(row["rows"] for row in out.tables["dataset_summary"]
                              if row["split"] == split and row["source"] == source)
            for label, columns in definitions:
                where = f"split=? AND source=?"
                if label in ("raw_name", "raw_address", "normalized_name",
                             "normalized_address", "sorted_name"):
                    where += f" AND {columns}<>''"
                elif label == "normalized_name_address":
                    where += " AND (norm_name<>'' OR norm_address<>'')"
                elif label == "raw_name_address":
                    where += " AND (business_name<>'' OR business_address<>'')"
                grouped = f"SELECT COUNT(*) n FROM records WHERE {where} GROUP BY {columns} HAVING n>1"
                groups, excess = db.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(n-1),0) FROM ({grouped})",
                    (split, source)).fetchone()
                out.add("duplicate_summary", {"split": split, "source": source,
                                              "type": label, "groups": groups,
                                              "extra_records": excess,
                                              "duplicate_rate_percent": 100 * excess / max(1, total_rows)})
                if label == "exact_row":
                    for summary in out.tables["dataset_summary"]:
                        if summary["split"] == split and summary["source"] == source:
                            summary["duplicate_rows"] = excess
                            break
                if label in ("exact_row", "normalized_name", "normalized_address",
                             "normalized_name_address", "sorted_name"):
                    sql = (f"SELECT {columns}, COUNT(*) n FROM records WHERE {where} "
                           f"GROUP BY {columns} HAVING n>1 ORDER BY n DESC LIMIT ?")
                    for row in db.execute(sql, (split, source, config.duplicate_limit)):
                        out.add("suspicious_duplicates", {"split": split, "source": source,
                                                          "type": label,
                                                          "key": " | ".join(map(str, row[:-1])),
                                                          "records": row[-1]})
        for label, columns in definitions[1:]:
            first = columns.split(",")[0].strip()
            grouped = (f"SELECT COUNT(*) n FROM records WHERE split=? AND {first}<>'' "
                       f"GROUP BY {columns} HAVING COUNT(DISTINCT source)>1")
            groups = query_one(db, f"SELECT COUNT(*) FROM ({grouped})", (split,))
            out.add("duplicate_summary", {"split": split, "source": "across_sources",
                                          "type": label, "groups": groups,
                                          "extra_records": ""})
            if label in ("raw_name_address", "normalized_name_address"):
                sql = (f"SELECT {columns}, COUNT(*) n, GROUP_CONCAT(DISTINCT source) "
                       f"FROM records WHERE split=? AND {first}<>'' "
                       f"GROUP BY {columns} HAVING COUNT(DISTINCT source)>1 "
                       f"ORDER BY n DESC LIMIT ?")
                for row in db.execute(sql, (split, config.duplicate_limit)):
                    out.add("suspicious_duplicates", {
                        "split": split, "source": "across_sources", "type": label,
                        "key": " | ".join(map(str, row[:-2])),
                        "records": row[-2], "sources": row[-1]})
    for split in ("train", "test"):
        for source in SOURCES:
            duplicate_ids = query_one(db, """SELECT COALESCE(SUM(n-1),0) FROM (
                SELECT COUNT(*) n FROM records WHERE split=? AND source=?
                GROUP BY entity_id HAVING n>1)""", (split, source))
            for summary in out.tables["dataset_summary"]:
                if summary["split"] == split and summary["source"] == source:
                    summary["duplicate_entity_ids"] = duplicate_ids
                    break
            for column, expression in (("entity_id", "entity_id"),
                                       ("business_name", "business_name"),
                                       ("business_address", "business_address"),
                                       ("country", "country")):
                unique = query_one(db, f"SELECT COUNT(DISTINCT {expression}) FROM records "
                                       "WHERE split=? AND source=?", (split, source))
                for row in out.tables["column_summary"]:
                    if row["split"] == split and row["source"] == source and row["column"] == column:
                        row["unique_values"] = unique
                        break
                if column != "entity_id":
                    for value, frequency in db.execute(
                            f"SELECT {expression}, COUNT(*) n FROM records "
                            f"WHERE split=? AND source=? GROUP BY {expression} "
                            "ORDER BY n DESC LIMIT ?", (split, source, config.top_k_values)):
                        out.add("common_values", {"split": split, "source": source,
                                                  "column": column, "value": value,
                                                  "frequency": frequency})
    for row in out.tables["column_summary"]:
        if row["source"] == "ground_truth":
            column = "source1_entity_id" if row["column"] == "source1_entity_id" else "matched_entity_ids"
            row["unique_values"] = query_one(
                db, f"SELECT COUNT(DISTINCT {column}) FROM truth_rows")
    conflicts = db.execute("""SELECT split, source, norm_name,
        COUNT(DISTINCT norm_address) addresses, COUNT(*) records
        FROM records WHERE norm_name<>'' GROUP BY split, source, norm_name
        HAVING addresses>1 ORDER BY records DESC LIMIT ?""", (config.duplicate_limit,))
    for split, source, name, addresses, records in conflicts:
        out.add("suspicious_duplicates", {"split": split, "source": source,
                                          "type": "name_conflicting_addresses",
                                          "key": name, "records": records,
                                          "distinct_addresses": addresses})
    truth_duplicates = query_one(db, """SELECT COALESCE(SUM(n-1),0) FROM (
        SELECT COUNT(*) n FROM truth_rows GROUP BY source1_entity_id, matched_entity_ids
        HAVING n>1)""")
    for summary in out.tables["dataset_summary"]:
        if summary["source"] == "ground_truth":
            summary["duplicate_rows"] = truth_duplicates
            summary["duplicate_entity_ids"] = query_one(db, """SELECT COALESCE(SUM(n-1),0)
                FROM (SELECT COUNT(*) n FROM truth_rows GROUP BY source1_entity_id
                HAVING n>1)""")
            break


def export_tokens(db: sqlite3.Connection, config: AnalysisConfig, out: Output) -> None:
    """Export bounded Top-K token tables while counting the full vocabulary on disk."""
    for field_name in (*FIELDS, *(f"{field}_raw" for field in FIELDS)):
        table = "common_name_tokens" if "name" in field_name else "common_address_tokens"
        for n in range(1, config.ngram_max + 1):
            if field_name.endswith("_raw") and n != 1:
                continue
            if not field_name.endswith("_raw") and n < config.ngram_min:
                continue
            rows = db.execute("""SELECT token, SUM(frequency) tf, SUM(document_frequency) df
                FROM token_counts WHERE field=? AND n=? GROUP BY token
                ORDER BY tf DESC LIMIT ?""", (field_name, n, config.top_k_words))
            for token, tf, df in rows:
                out.add(table, {"field": field_name, "n": n, "split": "all",
                                "source": "all", "country": "all", "token": token,
                                "frequency": tf, "document_frequency": df})
            vocabulary, singletons = db.execute("""SELECT COUNT(*),
                COALESCE(SUM(tf=1),0) FROM (
                  SELECT SUM(frequency) tf FROM token_counts WHERE field=? AND n=?
                  GROUP BY token)""", (field_name, n)).fetchone()
            out.add("vocabulary_summary", {"field": field_name, "n": n,
                                           "vocabulary_size": vocabulary,
                                           "singleton_tokens": singletons})
            for split in ("train", "test"):
                for source in SOURCES:
                    rows = db.execute("""SELECT token, SUM(frequency) tf,
                        SUM(document_frequency) df FROM token_counts
                        WHERE split=? AND source=? AND field=? AND n=?
                        GROUP BY token ORDER BY tf DESC LIMIT ?""",
                        (split, source, field_name, n, config.top_k_words))
                    for token, tf, df in rows:
                        out.add(table, {"field": field_name, "n": n, "split": split,
                                        "source": source, "country": "all", "token": token,
                                        "frequency": tf, "document_frequency": df})
                    countries = [row[0] for row in db.execute("""SELECT DISTINCT country
                        FROM token_counts WHERE split=? AND source=? AND field=? AND n=?""",
                        (split, source, field_name, n))]
                    for country in countries:
                        rows = db.execute("""SELECT token, frequency, document_frequency
                            FROM token_counts WHERE split=? AND source=? AND country=?
                              AND field=? AND n=? ORDER BY frequency DESC LIMIT ?""",
                            (split, source, country, field_name, n, config.top_k_words))
                        for token, tf, df in rows:
                            out.add(table, {"field": field_name, "n": n, "split": split,
                                            "source": source, "country": country,
                                            "token": token, "frequency": tf,
                                            "document_frequency": df})

    # The comparison uses normalized unigrams only. Counts are exact, output is bounded.
    for field_name in FIELDS:
        train_vocab = query_one(db, """SELECT COUNT(*) FROM (SELECT token FROM token_counts
            WHERE split='train' AND field=? AND n=1 GROUP BY token)""", (field_name,))
        test_vocab = query_one(db, """SELECT COUNT(*) FROM (SELECT token FROM token_counts
            WHERE split='test' AND field=? AND n=1 GROUP BY token)""", (field_name,))
        unseen = query_one(db, """SELECT COUNT(*) FROM (
            SELECT token FROM token_counts WHERE split='test' AND field=? AND n=1
            GROUP BY token EXCEPT SELECT token FROM token_counts
            WHERE split='train' AND field=? AND n=1 GROUP BY token)""",
            (field_name, field_name))
        out.add("train_test_comparison", {"measure": "vocabulary",
                                          "field": field_name, "train": train_vocab,
                                          "test": test_vocab, "test_unseen": unseen,
                                          "test_unseen_percent": 100 * unseen / max(1, test_vocab)})
        distribution: Counter[str] = Counter()
        rare_examples = 0
        for token, frequency in db.execute("""SELECT token, SUM(frequency) tf
            FROM token_counts WHERE field=? AND n=1 GROUP BY token""", (field_name,)):
            bucket = "1" if frequency == 1 else ("2-5" if frequency <= 5 else
                     "6-20" if frequency <= 20 else "21-100" if frequency <= 100 else ">100")
            distribution[bucket] += 1
            if frequency == 1 and rare_examples < config.top_k_values:
                out.add("rare_tokens", {"field": field_name, "token": token,
                                        "frequency": frequency})
                rare_examples += 1
        for bucket, amount in distribution.items():
            out.add("token_frequency_distribution", {"field": field_name,
                                                     "frequency_bucket": bucket,
                                                     "vocabulary_count": amount})
        for split in ("train", "test"):
            global_rows = sum(r["rows"] for r in out.tables["dataset_summary"]
                              if r["split"] == split and r["source"] in SOURCES)
            for source in SOURCES:
                source_rows = next(r["rows"] for r in out.tables["dataset_summary"]
                                   if r["split"] == split and r["source"] == source)
                groups = [(source, "all", source_rows)]
                groups.extend((source, r["country"], r["rows"])
                              for r in out.tables["country_summary"]
                              if r["split"] == split and r["source"] == source)
                for group_source, country, group_rows in groups:
                    country_clause = "" if country == "all" else "AND country=?"
                    parameters: tuple[Any, ...] = (split, field_name, group_source)
                    if country != "all":
                        parameters += (country,)
                    sql = f"""SELECT token, SUM(document_frequency) df
                        FROM token_counts WHERE split=? AND field=? AND source=?
                        AND n=1 {country_clause} GROUP BY token
                        HAVING df>=? ORDER BY df DESC LIMIT ?"""
                    candidates = db.execute(sql, parameters + (
                        config.min_skew_df,
                        config.top_k_words * config.skew_candidate_multiplier)).fetchall()
                    ranked = []
                    for token, local_df in candidates:
                        global_df = query_one(db, """SELECT SUM(document_frequency)
                            FROM token_counts WHERE field=? AND n=1 AND token=? AND split=?""",
                            (field_name, token, split)) or 0
                        lift = local_df * global_rows / max(1, global_df * group_rows)
                        ranked.append((lift, token, local_df, global_df))
                    for lift, token, local_df, global_df in sorted(ranked, reverse=True)[:config.top_k_values]:
                        out.add("token_disproportion", {"split": split, "source": group_source,
                                                        "country": country, "field": field_name,
                                                        "token": token, "group_df": local_df,
                                                        "global_df": global_df, "lift": lift,
                                                        "candidate_scope": "top_by_group_df"})


def jaccard(left: str, right: str, min_length: int, strategy: str = "unicode") -> float:
    a, b = set(tokens(left, min_length, strategy)), set(tokens(right, min_length, strategy))
    return len(a & b) / len(a | b) if a or b else 1.0


def char_ngrams(value: str, width: int = 3) -> set[str]:
    value = normalized(value)
    if not value:
        return set()
    return {value[i:i + width] for i in range(max(1, len(value) - width + 1))}


def char_jaccard(left: str, right: str, width: int = 3) -> float:
    a, b = char_ngrams(left, width), char_ngrams(right, width)
    return len(a & b) / len(a | b) if a or b else 1.0


def true_match_analysis(db: sqlite3.Connection, config: AnalysisConfig,
                        out: Output) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(config.random_seed)
    sample: list[tuple[str, ...]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    stats: dict[tuple[str, str, str], Stats] = defaultdict(Stats)
    pair_counts: Counter[str] = Counter()
    missing_pairs: Counter[tuple[str, str]] = Counter()
    sql = """SELECT t.source, t.source1_entity_id, t.matched_entity_id,
        a.business_name, a.business_address, a.country,
        b.business_name, b.business_address, b.country
        FROM truth t JOIN records a ON a.split='train' AND a.source='source1'
          AND a.entity_id=t.source1_entity_id
        JOIN records b ON b.split='train' AND b.source=t.source
          AND b.entity_id=t.matched_entity_id
        WHERE t.source IN ('source2','source3')"""
    for pair in db.execute(sql):
        source, s1_id, match_id, name_a, address_a, country_a, name_b, address_b, country_b = pair
        pair_counts[source] += 1
        sample_add(sample, pair, sum(pair_counts.values()), config.sample_size, rng)
        stats[(source, "other", "country_agreement")].add(int(country_a == country_b))
        missing_key = (f"s1_name={int(bool(name_a.strip()))},"
                       f"other_name={int(bool(name_b.strip()))},"
                       f"s1_address={int(bool(address_a.strip()))},"
                       f"other_address={int(bool(address_b.strip()))}")
        missing_pairs[(source, missing_key)] += 1
        for field_name, a, b in (("business_name", name_a, name_b),
                                 ("business_address", address_a, address_b)):
            norm_a, norm_b = normalized(a), normalized(b)
            set_a, set_b = set(tokens(a, config.min_token_length, config.tokenizer)), set(
                tokens(b, config.min_token_length, config.tokenizer))
            overlap = len(set_a & set_b)
            measures = {
                "exact_equality": int(a == b),
                "normalized_equality": int(norm_a == norm_b),
                "token_jaccard_milli": round(jaccard(a, b, config.min_token_length,
                                                      config.tokenizer) * 1_000),
                "shared_tokens": overlap,
                "length_difference": abs(len(a) - len(b)),
                "missing_both": int(value_missing(a) and value_missing(b)),
                "missing_one": int(value_missing(a) != value_missing(b)),
            }
            for measure, value in measures.items():
                stats[(source, field_name, measure)].add(value)
            category = "exact" if a == b else (
                "normalized_equal" if norm_a == norm_b else
                "missing" if value_missing(a) or value_missing(b) else
                "low_similarity" if measures["token_jaccard_milli"] <=
                config.similarity_threshold * 1_000 else "fuzzy")
            categories = [category]
            if a != b and norm_a and norm_b and sorted(norm_a.split()) == sorted(norm_b.split()):
                categories.append("word_order")
            variants = NAME_VARIANTS if field_name == "business_name" else ADDRESS_VARIANTS
            if any((left in set_a and right in set_b) or
                   (right in set_a and left in set_b) for left, right in variants):
                categories.append("abbreviation_or_variant")
            for kind in categories:
                key = f"{field_name}_{kind}"
                seen[key] += 1
                sample_add(examples[key], {"category": key, "source": source,
                                           "source1_entity_id": s1_id,
                                           "matched_entity_id": match_id,
                                           "source1_value": a, "matched_value": b,
                                           "jaccard": measures["token_jaccard_milli"] / 1_000},
                           seen[key], config.example_rows, rng)
    for (source, field_name, measure), aggregate in sorted(stats.items()):
        row = {"source": source, "field": field_name, "measure": measure,
               **aggregate.row()}
        if measure.endswith("_milli"):
            for name in ("mean", "median", "min", "max", "p1", "p5", "p25", "p50",
                         "p75", "p95", "p99"):
                row[name] = row[name] / 1_000 if row.get(name) is not None else None
        out.add("true_match_similarity", row)
    for category, rows in sorted(examples.items()):
        out.tables.setdefault("true_match_examples", []).extend(rows)
    for source, count in pair_counts.items():
        out.add("ground_truth_summary", {"measure": "joined_true_pairs",
                                         "category": source, "count": count})
    for (source, pattern), count in sorted(missing_pairs.items()):
        out.add("true_match_missingness", {"source": source,
                                           "availability_pattern": pattern, "pairs": count})
    return sample


def compare_splits(out: Output) -> None:
    rows = out.tables.get("dataset_summary", [])
    by_source = {(r["split"], r["source"]): r for r in rows}
    for source in SOURCES:
        train = by_source[("train", source)]["rows"]
        test = by_source[("test", source)]["rows"]
        out.add("train_test_comparison", {"measure": "rows", "field": source,
                                          "train": train, "test": test,
                                          "test_to_train_ratio": test / max(1, train)})
    countries = defaultdict(lambda: [0, 0])
    for row in out.tables.get("country_summary", []):
        countries[row["country"]][0 if row["split"] == "train" else 1] += row["rows"]
    for country, (train, test) in sorted(countries.items()):
        out.add("train_test_comparison", {"measure": "country_rows", "field": country,
                                          "train": train, "test": test,
                                          "test_unseen": bool(test and not train)})
    for table, measure_keys, value_key in (
        ("column_summary", ("column",), "missing_percent"),
        ("string_statistics", ("field", "measure"), "mean"),
        ("duplicate_summary", ("type",), "duplicate_rate_percent"),
    ):
        for source in SOURCES:
            a = [r for r in out.tables.get(table, []) if r["split"] == "train"
                 and r["source"] == source and r.get("country", "*") == "*"]
            b = [r for r in out.tables.get(table, []) if r["split"] == "test"
                 and r["source"] == source and r.get("country", "*") == "*"]
            b_index = {tuple(r.get(key) for key in measure_keys): r for r in b}
            for row in a:
                key = tuple(row.get(name) for name in measure_keys)
                if key in b_index:
                    out.add("train_test_comparison", {"measure": table + "." + value_key,
                                                      "field": source + "/" + "/".join(map(str, key)),
                                                      "train": row.get(value_key),
                                                      "test": b_index[key].get(value_key)})
    patterns = out.tables.get("character_patterns", [])
    for source in SOURCES:
        train = {(r["field"], r["pattern"]): r["count"] for r in patterns
                 if r["split"] == "train" and r["source"] == source}
        test = {(r["field"], r["pattern"]): r["count"] for r in patterns
                if r["split"] == "test" and r["source"] == source}
        for field_name, pattern in sorted(train.keys() | test.keys()):
            out.add("train_test_comparison", {
                "measure": "character_pattern_count",
                "field": f"{source}/{field_name}/{pattern}",
                "train": train.get((field_name, pattern), 0),
                "test": test.get((field_name, pattern), 0)})


def expensive_experiments(db: sqlite3.Connection, sample: list[tuple[str, ...]],
                          config: AnalysisConfig, out: Output) -> None:
    """Bound all pair comparisons by a reservoir sampled from training true pairs."""
    if not sample:
        out.warn("No valid joined true pairs available for optional experiments")
        return
    try:
        from rapidfuzz.fuzz import ratio as edit_ratio  # type: ignore[import-not-found]
    except ImportError:
        edit_ratio = None
        out.warn("rapidfuzz unavailable; edit similarity omitted")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
    except ImportError:
        TfidfVectorizer = None
        out.warn("scikit-learn unavailable; TF-IDF cosine omitted")
    rng = np.random.default_rng(config.random_seed)
    rare_cache: dict[str, int] = {}

    def rare_df(token: str) -> int:
        if token not in rare_cache:
            rare_cache[token] = query_one(db, """SELECT COALESCE(SUM(document_frequency),0)
                FROM token_counts WHERE split='train' AND field='business_name'
                AND n=1 AND token=?""", (token,)) or 0
        return rare_cache[token]

    rule_hits: Counter[str] = Counter()
    pair_text: dict[str, list[tuple[str, str]]] = {field: [] for field in FIELDS}
    for source, s1_id, match_id, name_a, addr_a, country_a, name_b, addr_b, country_b in sample:
        norm_a, norm_b = normalized(name_a), normalized(name_b)
        addr_norm_a, addr_norm_b = normalized(addr_a), normalized(addr_b)
        a_tokens = set(tokens(name_a, config.min_token_length, config.tokenizer))
        b_tokens = set(tokens(name_b, config.min_token_length, config.tokenizer))
        a_numbers, b_numbers = set(NUMBER_RE.findall(addr_a)), set(NUMBER_RE.findall(addr_b))
        rules = {
            "exact_normalized_name": bool(norm_a and norm_a == norm_b),
            "shared_rare_name_token": any(0 < rare_df(token) <= config.rare_token_max_df
                                          for token in a_tokens & b_tokens),
            "shared_address_number": bool(a_numbers & b_numbers),
            "name_prefix": bool(norm_a and norm_b and
                                norm_a[:config.prefix_length] == norm_b[:config.prefix_length]),
            "exact_normalized_address": bool(addr_norm_a and addr_norm_a == addr_norm_b),
        }
        for rule, hit in rules.items():
            rule_hits[rule] += hit
        rule_hits["any_rule"] += any(rules.values())
        for field_name, a, b in ((FIELDS[0], name_a, name_b), (FIELDS[1], addr_a, addr_b)):
            pair_text[field_name].append((a, b))
            result: dict[str, Any] = {
                "source": source, "source1_entity_id": s1_id,
                "matched_entity_id": match_id, "field": field_name,
                "char_ngram_jaccard": char_jaccard(a, b, config.char_ngram_width),
                "token_jaccard": jaccard(a, b, config.min_token_length, config.tokenizer),
            }
            if edit_ratio is not None:
                result["edit_similarity"] = edit_ratio(a, b) / 100
            out.add("sampled_pair_similarity", result)

    if TfidfVectorizer is not None:
        for field_name, pairs in pair_text.items():
            texts = [normalized(value) for pair in pairs for value in pair]
            if not any(texts):
                continue
            try:
                matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                         min_df=1).fit_transform(texts)
                cosine = np.asarray(matrix[0::2].multiply(matrix[1::2]).sum(axis=1)).ravel()
                rows = [row for row in out.tables["sampled_pair_similarity"]
                        if row["field"] == field_name]
                for row, score in zip(rows, cosine):
                    row["tfidf_cosine"] = float(score)
            except ValueError as exc:
                out.warn(f"TF-IDF skipped for {field_name}: {exc}")

    for rule, hits in sorted(rule_hits.items()):
        out.add("blocking_recall_sample", {"rule": rule, "sampled_true_pairs": len(sample),
                                           "retained_true_pairs": hits,
                                           "estimated_recall": hits / len(sample)})

    # Build an illustrative non-match baseline from independently sampled source records.
    bounds = {}
    for source in ("source1", "source2", "source3"):
        bounds[source] = db.execute("""SELECT MIN(rowid), MAX(rowid) FROM records
            WHERE split='train' AND source=?""", (source,)).fetchone()
    for index in range(min(config.sample_size, config.nonmatch_sample_limit)):
        left_source, right_source = "source1", "source2" if index % 2 == 0 else "source3"
        picks = []
        for source in (left_source, right_source):
            low, high = bounds[source]
            if low is None:
                break
            position = int(rng.integers(low, high + 1))
            record = db.execute("""SELECT entity_id, business_name, business_address, country
                FROM records WHERE rowid>=? AND split='train' AND source=? LIMIT 1""",
                (position, source)).fetchone()
            if record is not None:
                picks.append(record)
        if len(picks) != 2:
            continue
        a, b = picks
        if query_one(db, """SELECT 1 FROM truth WHERE source1_entity_id=?
            AND matched_entity_id=? LIMIT 1""", (a[0], b[0])):
            continue
        out.add("sampled_nonmatch_similarity", {
            "source1_entity_id": a[0], "other_entity_id": b[0],
            "country_agreement": a[3] == b[3],
            "name_token_jaccard": jaccard(a[1], b[1], config.min_token_length,
                                           config.tokenizer),
            "address_token_jaccard": jaccard(a[2], b[2], config.min_token_length,
                                              config.tokenizer),
        })

    # Probe a bounded number of indexed name-prefix blocks for plausible false merges.
    near_examples = 0
    for pair in sample[:config.example_rows]:
        _, s1_id, _, name, address, _, _, _, _ = pair
        prefix = normalized(name)[:config.prefix_length]
        if len(prefix) < config.prefix_length:
            continue
        candidates = db.execute("""SELECT entity_id, business_name, business_address
            FROM records WHERE split='train' AND source IN ('source2','source3')
              AND norm_name>=? AND norm_name<? LIMIT ?""",
            (prefix, prefix + "\uffff", config.top_k_values))
        for candidate_id, candidate_name, candidate_address in candidates:
            if query_one(db, """SELECT 1 FROM truth WHERE source1_entity_id=?
                AND matched_entity_id=? LIMIT 1""", (s1_id, candidate_id)):
                continue
            score = char_jaccard(name, candidate_name, config.char_ngram_width)
            if score < config.similarity_threshold:
                continue
            out.add("near_duplicate_examples", {
                "source1_entity_id": s1_id, "candidate_entity_id": candidate_id,
                "source1_name": name, "candidate_name": candidate_name,
                "source1_address": address, "candidate_address": candidate_address,
                "name_char_jaccard": score,
                "address_token_jaccard": jaccard(address, candidate_address,
                                                  config.min_token_length, config.tokenizer)})
            near_examples += 1
            if near_examples >= config.duplicate_limit:
                break
        if near_examples >= config.duplicate_limit:
            break

    for split in ("train", "test"):
        db.create_function("first_number", 1,
                           lambda value: (match.group() if (match := NUMBER_RE.search(value))
                                          else ""), deterministic=True)
        db.create_function("postal_like", 1,
                           lambda value: (match.group() if (match := POSTAL_RE.search(value))
                                          else ""), deterministic=True)
        for key_name, expression in (
            ("name_first_token", "substr(norm_name,1,instr(norm_name||' ',' ')-1)"),
            ("name_prefix", f"substr(norm_name,1,{config.prefix_length})"),
            ("normalized_name", "norm_name"),
            ("normalized_address", "norm_address"),
            ("address_number", "first_number(business_address)"),
            ("postal_like", "postal_like(business_address)"),
        ):
            sql = f"""SELECT key, COUNT(*) n FROM
                (SELECT {expression} AS key FROM records WHERE split=?)
                WHERE key<>'' GROUP BY key ORDER BY n DESC LIMIT ?"""
            for key, count in db.execute(sql, (split, config.top_k_values)):
                out.add("large_blocks", {"split": split, "key_type": key_name,
                                         "key": key, "records": count})


def add_column_examples(out: Output, config: AnalysisConfig) -> None:
    examples = defaultdict(list)
    for row in out.tables.get("example_rows", []):
        for column in SOURCE_COLUMNS:
            key = (row["split"], row["source"], column)
            value = row[column]
            if value not in examples[key] and len(examples[key]) < min(5, config.example_rows):
                examples[key].append(value)
    for row in out.tables.get("column_summary", []):
        key = (row["split"], row["source"], row["column"])
        row["example_values"] = json.dumps(examples[key], ensure_ascii=False)


def save_tables(out: Output, directory: Path) -> None:
    empty_headers = {
        "data_quality_issues": ["split", "source", "issue", "entity_id", "detail"],
        "suspicious_duplicates": ["split", "source", "type", "key", "records"],
        "true_match_examples": ["category", "source", "source1_entity_id",
                                "matched_entity_id", "source1_value", "matched_value"],
        "rare_tokens": ["field", "token", "frequency"],
        "token_disproportion": ["split", "source", "country", "field", "token", "lift"],
    }
    for name in empty_headers:
        out.tables.setdefault(name, [])
    for name, rows in sorted(out.tables.items()):
        if not rows and name not in empty_headers:
            continue
        columns = (list(dict.fromkeys(key for row in rows for key in row)) if rows
                   else empty_headers[name])
        with (directory / f"{name}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def make_plots(out: Output, directory: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        out.warn("matplotlib unavailable; plots omitted")
        return
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)

    def save_bar(rows: list[dict[str, Any]], labels: list[str], values: list[float],
                 title: str, name: str, horizontal: bool = False) -> None:
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(max(8, min(16, len(labels) * 0.5)), 6))
        if horizontal:
            ax.barh(labels, values)
        else:
            ax.bar(labels, values)
            ax.tick_params(axis="x", rotation=40)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=140)
        plt.close(fig)

    files = [row for row in out.tables.get("dataset_summary", [])
             if row["source"] != "ground_truth"]
    save_bar(files, [f"{r['split']}/{r['source']}" for r in files],
             [r["rows"] for r in files], "Rows by file", "rows_by_file.png")
    country = [row for row in out.tables.get("country_summary", [])]
    by_country: Counter[str] = Counter()
    for row in country:
        by_country[row["country"]] += row["rows"]
    top = by_country.most_common(20)
    save_bar(top, [name or "<blank>" for name, _ in top],
             [count for _, count in top], "Records by country", "countries.png", True)
    missing = [row for row in out.tables.get("column_summary", [])
               if row["column"] in (*FIELDS, "country")]
    save_bar(missing, [f"{r['split']}/{r['source']}/{r['column']}" for r in missing],
             [r["missing_percent"] for r in missing], "Missing values (%)",
             "missing_values.png", True)
    truth = [row for row in out.tables.get("ground_truth_summary", [])
             if row["measure"] == "match_count_distribution"]
    save_bar(truth, [str(r["category"]) for r in truth],
             [r["count"] for r in truth], "Ground-truth matches per Source 1 entity",
             "match_counts.png")
    length = [row for row in out.tables.get("string_statistics", [])
              if row["country"] == "*" and row["measure"] == "characters"]
    save_bar(length, [f"{r['split']}/{r['source']}/{r['field']}" for r in length],
             [r.get("median", 0) for r in length], "Median character lengths",
             "string_lengths.png", True)
    similarities = [row for row in out.tables.get("true_match_similarity", [])
                    if row["measure"] == "token_jaccard_milli"]
    save_bar(similarities, [f"{r['source']}/{r['field']}" for r in similarities],
             [r.get("median", 0) for r in similarities],
             "Median true-match token Jaccard", "true_match_similarity.png")
    distribution = [row for row in out.tables.get("token_frequency_distribution", [])
                    if row["field"] == "business_name"]
    save_bar(distribution, [r["frequency_bucket"] for r in distribution],
             [r["vocabulary_count"] for r in distribution],
             "Business-name token frequency distribution", "token_frequency.png")
    split_rows = [row for row in out.tables.get("train_test_comparison", [])
                  if row["measure"] == "rows"]
    if split_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = np.arange(len(split_rows))
        ax.bar(positions - 0.2, [r["train"] for r in split_rows], width=0.4, label="train")
        ax.bar(positions + 0.2, [r["test"] for r in split_rows], width=0.4, label="test")
        ax.set_xticks(positions, [r["field"] for r in split_rows])
        ax.legend()
        ax.set_title("Train versus test rows")
        fig.tight_layout()
        fig.savefig(figures / "train_test_rows.png", dpi=140)
        plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "No rows available."
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value).replace("|", "\\|").replace("\n", " ")[:100])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Output, config: AnalysisConfig, backend: Backend) -> None:
    summary = out.tables.get("dataset_summary", [])
    source_summary = [row for row in summary if row["source"] != "ground_truth"]
    truth = out.tables.get("ground_truth_summary", [])
    singleton = next((r["count"] for r in truth if r["measure"] == "singleton_percent"), None)
    unseen = [r["field"] for r in out.tables.get("train_test_comparison", [])
              if r["measure"] == "country_rows" and r.get("test_unseen")]
    duplicate = [r for r in out.tables.get("duplicate_summary", [])
                 if r["source"] == "source1" and r["type"] == "normalized_name_address"]
    similarities = [r for r in out.tables.get("true_match_similarity", [])
                    if r["measure"] in ("exact_equality", "normalized_equality",
                                        "token_jaccard_milli")]
    name_tokens = [r for r in out.tables.get("common_name_tokens", [])
                   if r["field"] == "business_name" and r["n"] == 1 and
                   r["source"] == "all"]
    skew = sorted([r for r in out.tables.get("token_disproportion", [])
                   if r["field"] == "business_name" and r["country"] != "all"],
                  key=lambda row: row["lift"], reverse=True)
    lines = [
        "# Business Entity Resolution Dataset Analysis", "",
        f"Dataset root: `{config.data_root}`. Backend: `{backend.name}`.",
        "Counts and inventories use all rows. Displayed examples are bounded reservoirs; "
        "the optional similarity and blocking estimates use sampled training pairs.", "",
        "## Dataset inventory", "",
        markdown_table(source_summary, ["split", "source", "rows", "file_size_bytes",
                                        "approx_memory_bytes"]), "",
        "Detailed file and column profiles: `dataset_summary.tsv`, `column_summary.tsv`, "
        "and `common_values.tsv`.", "",
        "## Missingness and text", "",
        markdown_table([r for r in out.tables.get("column_summary", [])
                        if r["column"] in (*FIELDS, "country")],
                       ["split", "source", "column", "missing_percent", "unique_values"]), "",
        "See `missing_values.tsv`, `string_statistics.tsv`, `character_patterns.tsv`, "
        "`normalization_patterns.tsv`, and the token tables for source and country detail.", "",
        "## Tokens and blocking clues", "",
        markdown_table(name_tokens, ["token", "frequency", "document_frequency"], 10), "",
        "Source/country-associated name tokens among each group's most frequent candidates "
        "(lift is relative document frequency):", "",
        markdown_table(skew, ["split", "source", "country", "token", "lift"], 10), "",
        "Frequent tokens may create large candidate blocks; rare tokens may help recall. "
        "Inspect `token_disproportion.tsv`, `rare_tokens.tsv`, and "
        "`token_frequency_distribution.tsv` before selecting rules.", "",
        "## Ground truth and true matches", "",
        f"Singleton share: {singleton:.2f}% of ground-truth rows."
        if singleton is not None else "Singleton share unavailable.", "",
        markdown_table([r for r in truth if r["measure"] in
                        ("source1_rows", "match_pairs", "singletons", "one_match",
                         "multiple_matches", "max_matches")],
                       ["measure", "count"]), "",
        markdown_table(similarities, ["source", "field", "measure", "mean", "median"], 20), "",
        "Examples are in `true_match_examples.tsv`; inspect low-similarity examples before "
        "choosing normalization or blocking rules.", "",
        "## Duplicates, blocking, and train/test shift", "",
        markdown_table(duplicate, ["split", "source", "type", "groups", "extra_records"]), "",
        f"Countries present in test and absent in train: {', '.join(unseen) if unseen else 'none found'}.",
        "See `duplicate_summary.tsv`, `suspicious_duplicates.tsv`, "
        "`train_test_comparison.tsv`, and `data_quality_issues.tsv`.", "",
    ]
    if config.expensive_analysis:
        lines.extend(["Sampled blocking rule recall and large-block examples are in "
                      "`blocking_recall_sample.tsv` and `large_blocks.tsv`. "
                      "These estimates are diagnostic, not a trained matcher.", ""])
    else:
        lines.extend(["Pairwise near-duplicate search, edit/TF-IDF similarity, and blocking "
                      "experiments were skipped. Enable `--expensive-analysis` to run them.", ""])
    lines.extend(["## Data-quality warnings", ""])
    counts = out.summary.get("quality_issue_counts", {})
    if counts:
        lines.extend([f"- `{kind}`: {count:,}" for kind, count in sorted(counts.items())])
    else:
        lines.append("No issues were flagged by the implemented checks.")
    lines.extend([f"- {warning}" for warning in out.warnings])
    lines.extend(["", "## Interpretation notes", "",
                  "A normalized duplicate is a review candidate, not proof of identity. "
                  "A true pair can have different country labels or weak text overlap. "
                  "Randomly paired records in the optional baseline are illustrative "
                  "and are not guaranteed negatives. Empty and whitespace-only fields "
                  "are reported separately; literal 'NA' is preserved as text. "
                  "No external data was consulted.", ""])
    (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_args()
    LOG.info("[1/12] Discovering dataset")
    files = discover_dataset(config.data_root)
    if config.output_dir == config.data_root or config.data_root in config.output_dir.parents:
        raise ValueError("Output directory must be outside the dataset root")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out = Output()
    LOG.info("[2/12] Selecting backend")
    backend = Backend(config.backend, out)
    with tempfile.TemporaryDirectory(prefix="dataset-analysis-", dir=config.output_dir) as temporary:
        db = make_index(Path(temporary) / "index.sqlite3")
        try:
            LOG.info("[3/12] Profiling source TSVs")
            profile_sources(files, config, backend, db, out)
            LOG.info("[4/12] Parsing ground truth")
            profile_ground_truth(files[("train", "ground_truth")], config, db, out)
            LOG.info("[5/12] Indexing records")
            index_records(db)
            LOG.info("[6/12] Validating IDs and ground-truth references")
            validate_ids(db, out)
            LOG.info("[7/12] Analyzing duplicates")
            duplicate_analysis(db, config, out)
            LOG.info("[8/12] Summarizing tokens")
            export_tokens(db, config, out)
            LOG.info("[9/12] Joining true matches")
            sample = true_match_analysis(db, config, out)
            LOG.info("[10/12] Comparing train and test")
            compare_splits(out)
            if config.expensive_analysis:
                LOG.info("[11/12] Running bounded similarity and blocking experiments")
                expensive_experiments(db, sample, config, out)
            else:
                LOG.info("[11/12] Skipping optional expensive analysis")
        finally:
            db.close()
    LOG.info("[12/12] Writing outputs")
    add_column_examples(out, config)
    save_tables(out, config.output_dir)
    if config.plots:
        make_plots(out, config.output_dir)
    truth_summary = out.tables.get("ground_truth_summary", [])
    unseen_countries = [row["field"] for row in out.tables.get("train_test_comparison", [])
                        if row["measure"] == "country_rows" and row.get("test_unseen")]
    out.summary.update({
        "dataset_root": str(config.data_root), "output_dir": str(config.output_dir),
        "backend": backend.name, "files": out.tables.get("dataset_summary", []),
        "warnings": out.warnings, "expensive_analysis": config.expensive_analysis,
        "key_findings": {
            "total_source_rows": sum(row["rows"] for row in out.tables.get("dataset_summary", [])
                                     if row["source"] in SOURCES),
            "singleton_percent": next((row["count"] for row in truth_summary
                                       if row["measure"] == "singleton_percent"), None),
            "test_only_countries": unseen_countries,
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(config).items()},
    })
    (config.output_dir / "summary.json").write_text(
        json.dumps(out.summary, ensure_ascii=False, indent=2, default=format_number),
        encoding="utf-8")
    write_report(out, config, backend)
    LOG.info("Profiled %s source records; singleton share: %s%%",
             f'{out.summary["key_findings"]["total_source_rows"]:,}',
             out.summary["key_findings"]["singleton_percent"])
    LOG.info("Analysis complete. Report: %s", config.output_dir / "report.md")
    LOG.info("Machine-readable tables and summary: %s", config.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as error:
        LOG.error("Analysis failed: %s", error)
        sys.exit(1)
