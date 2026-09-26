"""Chunked preprocessing with existing text cleanup plus libpostal address parsing.

Writes a separate parquet dataset so the raw files and the current
``datasets/preprocessed`` outputs remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_DIR / "student_resource" / "dataset"
DEFAULT_OUTPUT_DIR = REPO_DIR / "praj_files" / "datasets" / "preprocessed_libpostal"
PARSE_ADDRESS: Any = None
COMPONENT_ORDER = (
    "house_number", "road", "unit", "level", "staircase", "entrance", "po_box",
    "postcode", "suburb", "city_district", "city", "island", "state_district",
    "state", "country_region", "country", "world_region", "near", "category",
)
OUTPUT_COMPONENTS = (
    "house_number", "road", "unit", "level", "staircase", "entrance", "po_box",
    "postcode", "suburb", "city_district", "city", "island", "state_district",
    "state", "country_region", "country", "world_region", "near", "category",
)


def init_parser() -> None:
    global PARSE_ADDRESS
    try:
        from postal.parser import parse_address
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'postal' is unavailable. Install libpostal and its Python "
            "binding first; see praj_files/EM/README.md."
        ) from exc
    PARSE_ADDRESS = parse_address


def normalize_component(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in value).split())


def parse_one(address: str) -> tuple[str, str, str, dict[str, str]]:
    """Return canonical labeled text, JSON components, status, and component fields."""
    raw_address = address or ""
    fallback = normalize_component(raw_address)
    fields = {f"address_{name}": "" for name in OUTPUT_COMPONENTS}
    if not raw_address.strip():
        return "", "{}", "empty", fields

    try:
        parsed_pairs = PARSE_ADDRESS(raw_address)
        grouped: dict[str, list[str]] = {}
        for value, label in parsed_pairs:
            normalized = normalize_component(str(value))
            if normalized:
                grouped.setdefault(str(label), [])
                if normalized not in grouped[str(label)]:
                    grouped[str(label)].append(normalized)
        components = {label: " ".join(values) for label, values in grouped.items()}
        if not components:
            return fallback, "{}", "fallback_empty", fields

        canonical_labels = [label for label in COMPONENT_ORDER if label in components]
        canonical_labels.extend(sorted(set(components) - set(canonical_labels)))
        canonical = " ; ".join(f"{label}: {components[label]}" for label in canonical_labels)
        fields.update(
            {
                f"address_{label}": components.get(label, "")
                for label in OUTPUT_COMPONENTS
            }
        )
        return canonical, json.dumps(components, ensure_ascii=False, sort_keys=True), "parsed", fields
    except Exception:
        # Keep a usable address even when the native parser rejects an unusual row.
        return fallback, "{}", "fallback_error", fields


def clean_text_column(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.lower()
    replacements_before_punctuation = {
        r"&": " and ", r"@": " at ", r"\bc/o\b": " care of ", r"\bs/o\b": " son of ",
        r"\bd/o\b": " daughter of ", r"\bw/o\b": " wife of ", r"\bopp\b\.?": " opposite ",
    }
    for pattern, replacement in replacements_before_punctuation.items():
        cleaned = cleaned.str.replace(pattern, replacement, regex=True)
    cleaned = cleaned.str.replace(r"[^\w\s]", " ", regex=True)
    replacements = {
        r"\bprivate\b": "pvt", r"\blimited\b": "ltd", r"\bcorporation\b": "corp",
        r"\bcompany\b": "co", r"\bincorporated\b": "inc", r"\broad\b": "rd",
        r"\bstreet\b": "st", r"\bavenue\b": "ave", r"\blane\b": "ln",
        r"\bfloor\b": "fl", r"\bapartment\b": "apt", r"\bbuilding\b": "bldg",
        r"\bnr\s": "near ",
    }
    for pattern, replacement in replacements.items():
        cleaned = cleaned.str.replace(pattern, replacement, regex=True)
    return cleaned.str.replace(r"\s+", " ", regex=True).str.strip()


def source_files(input_dir: Path, train_only: bool = False) -> list[Path]:
    files = [
        input_dir / "train" / f"train_source{source}.tsv" for source in (1, 2, 3)
    ]
    files.append(input_dir / "train" / "train_ground_truth.tsv")
    if not train_only:
        files.extend(input_dir / "test" / f"test_source{source}.tsv" for source in (1, 2, 3))
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input TSVs: " + ", ".join(str(path) for path in missing))
    return files


def process_file(
    input_path: Path,
    output_dir: Path,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    output_path = output_dir / input_path.with_suffix(".parquet").name
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists; pass --overwrite to replace it", flush=True)
        return

    print(f"[preprocess] {input_path.name}", flush=True)
    started = time.perf_counter()
    temporary_path = Path(str(output_path) + ".tmp")
    writer: pq.ParquetWriter | None = None
    status_counts: Counter[str] = Counter()
    rows_written = 0

    pool: ProcessPoolExecutor | None = None
    if "source" in input_path.name and "ground_truth" not in input_path.name and workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=init_parser)
    elif "source" in input_path.name and "ground_truth" not in input_path.name:
        init_parser()

    try:
        chunks = pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            chunksize=batch_size,
        )
        for frame in chunks:
            if "business_name" in frame.columns:
                frame["business_name_clean"] = clean_text_column(frame["business_name"])
            if "business_address" in frame.columns:
                frame["business_address_clean"] = clean_text_column(frame["business_address"])

            if "business_address" in frame.columns:
                addresses = frame["business_address"].tolist()
                if pool is not None:
                    parsed_rows = list(pool.map(parse_one, addresses, chunksize=256))
                elif "source" in input_path.name:
                    parsed_rows = [parse_one(address) for address in addresses]
                else:
                    parsed_rows = []
                if parsed_rows:
                    frame["business_address_libpostal"] = [item[0] for item in parsed_rows]
                    frame["address_components_json"] = [item[1] for item in parsed_rows]
                    frame["address_parse_status"] = [item[2] for item in parsed_rows]
                    status_counts.update(item[2] for item in parsed_rows)
                    for component in OUTPUT_COMPONENTS:
                        frame[f"address_{component}"] = [item[3][f"address_{component}"] for item in parsed_rows]
                else:
                    frame["business_address_libpostal"] = frame["business_address_clean"]
                    frame["address_components_json"] = "{}"
                    frame["address_parse_status"] = "not_parsed"

            if "business_name_clean" in frame.columns and "business_address_clean" in frame.columns:
                parsed_address = frame.get("business_address_libpostal", frame["business_address_clean"])
                frame["combined_text_clean"] = (
                    "Business name: " + frame["business_name_clean"]
                    + " Address: " + frame["business_address_clean"]
                    + " Parsed address: " + parsed_address
                    + " Country: " + frame.get("country", "")
                ).str.strip()

            for column in frame.columns:
                frame[column] = frame[column].astype("string").fillna("")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows_written += len(frame)
            if rows_written % 100_000 < len(frame):
                print(f"[preprocess] {input_path.name}: {rows_written:,} rows", flush=True)
        if writer is None:
            raise ValueError(f"No rows found in {input_path}")
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if pool is not None:
            pool.shutdown(wait=True)

    if status_counts:
        parsed = status_counts["parsed"]
        fallback = status_counts["fallback_empty"] + status_counts["fallback_error"]
        print(
            f"[parse] {input_path.name}: parsed={parsed:,}, fallback={fallback:,}, "
            f"empty={status_counts['empty']:,}",
            flush=True,
        )
    print(
        f"[saved] {output_path} ({rows_written:,} rows, "
        f"{time.perf_counter() - started:.1f}s)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create separate train/test parquet files with libpostal-parsed addresses."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4, help="Address parser worker processes")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Process training sources and ground truth only; useful for retention experiments.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 1:
        parser.error("--batch-size and --workers must be positive")

    try:
        files = source_files(args.input_dir, train_only=args.train_only)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for input_path in files:
            process_file(input_path, args.output_dir, args.batch_size, args.workers, args.overwrite)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
