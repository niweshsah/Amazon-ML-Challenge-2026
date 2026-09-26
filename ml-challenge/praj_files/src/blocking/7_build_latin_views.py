"""Create a separate Unicode-safe dataset with an ICU Latin transliteration view.

Original script text is retained. The added Latin columns are transliterations,
not semantic translations, so proper names remain identifiable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from multilingual_text import clean_text, latin_transliterate


PRAJ_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PRAJ_DIR / "datasets" / "preprocessed"
DEFAULT_OUTPUT_DIR = PRAJ_DIR / "datasets" / "preprocessed_latin"
DEFAULT_DATASETS = (
    "train_source1.parquet", "train_source2.parquet", "train_source3.parquet",
    "test_source1.parquet", "test_source2.parquet", "test_source3.parquet",
)


def build_dataset(
    input_path: Path,
    output_root: Path,
    batch_size: int,
    max_rows: int | None,
    overwrite: bool,
) -> Path:
    dataset = ds.dataset(input_path, format="parquet")
    output_dir = output_root / input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "part-00000.parquet"
    temporary_path = Path(str(output_path) + ".tmp")
    if output_path.exists() and not overwrite and max_rows is None:
        print(f"[skip] {output_path} already exists; pass --overwrite", flush=True)
        return output_dir

    columns = dataset.schema.names
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
            frame = batch.to_pydict()
            row_count = len(frame[columns[0]]) if columns else 0
            if max_rows is not None:
                row_count = min(row_count, max_rows - written)
                if row_count <= 0:
                    break
                frame = {key: values[:row_count] for key, values in frame.items()}

            names = [clean_text(value) for value in frame.get("business_name", [""] * row_count)]
            addresses = [clean_text(value) for value in frame.get("business_address", [""] * row_count)]
            name_latin = [clean_text(value) for value in latin_transliterate(names)]
            address_latin = [clean_text(value) for value in latin_transliterate(addresses)]
            if "business_name" in frame:
                frame["business_name_clean"] = names
                frame["business_name_latin"] = name_latin
            if "business_address" in frame:
                frame["business_address_clean"] = addresses
                frame["business_address_latin"] = address_latin
            if "business_name" in frame and "business_address" in frame:
                frame["combined_text_clean"] = [
                    f"{name} {address}".strip() for name, address in zip(names, addresses)
                ]

            table = pa.Table.from_pydict(frame)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
            written += row_count
            if written % 100_000 < row_count:
                print(f"[{input_path.name}] wrote {written:,} rows", flush=True)
            if max_rows is not None and written >= max_rows:
                break
        if writer is None:
            raise ValueError(f"No records found in {input_path}")
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"[saved] {output_path} ({written:,} rows)", flush=True)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-rows", type=int, help="Limit each dataset for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_rows is not None and args.max_rows < 1):
        parser.error("--batch-size and --max-rows must be positive")

    for name in args.datasets:
        source = args.input_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        build_dataset(source, args.output_dir, args.batch_size, args.max_rows, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
