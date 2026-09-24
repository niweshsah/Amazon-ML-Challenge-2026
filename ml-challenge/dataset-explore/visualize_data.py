"""Explore the business entity TSV files with pandas and matplotlib.

Example:
	python dataset-explore/visualize_data.py \
		--data-dir challenge-dataset \
		--output-dir dataset-explore/plots
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# The container image may have a non-writable home directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_COLUMNS = [
	"entity_id",
	"business_name",
	"business_address",
	"country",
]
CHUNK_SIZE = 100_000


def find_tsv_files(data_dir: Path) -> list[Path]:
	"""Return TSV files whose first line looks like the expected header."""
	valid_files = []
	for path in sorted(data_dir.glob("*.tsv")):
		try:
			header = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig")
		except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
			logging.warning("Skipping %s: %s", path.name, error)
			continue

		if list(header.columns) == EXPECTED_COLUMNS:
			valid_files.append(path)
		else:
			logging.warning("Skipping %s: unexpected columns %s", path.name, list(header.columns))

	if not valid_files:
		raise FileNotFoundError(
			f"No TSV files with columns {EXPECTED_COLUMNS} were found in {data_dir}"
		)
	return valid_files


def read_dataset(path: Path, sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Collect exact counts and a bounded sample from one large TSV file."""
	chunks = []
	total_rows = 0
	for chunk in pd.read_csv(
		path,
		sep="\t",
		dtype="string",
		keep_default_na=False,
		encoding="utf-8-sig",
		chunksize=CHUNK_SIZE,
	):
		chunk["source"] = path.stem
		total_rows += len(chunk)
		remaining = sample_size - sum(len(item) for item in chunks)
		if remaining > 0:
			chunks.append(chunk.head(remaining))

	sample = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
	logging.info("Loaded %s rows from %s", f"{total_rows:,}", path.name)
	return sample, pd.DataFrame({"source": [path.stem], "rows": [total_rows]})


def add_lengths(frame: pd.DataFrame) -> pd.DataFrame:
	"""Add fields used by the distribution plots."""
	frame = frame.copy()
	for column in ("business_name", "business_address"):
		frame[f"{column}_length"] = frame[column].str.len()
	return frame


def create_plots(frame: pd.DataFrame, row_counts: pd.DataFrame, output_dir: Path) -> None:
	"""Write overview, country, and text-length plots to output_dir."""
	output_dir.mkdir(parents=True, exist_ok=True)
	frame = add_lengths(frame)

	country_counts = frame["country"].value_counts().head(20).sort_values()
	figure, axes = plt.subplots(2, 2, figsize=(16, 11))
	row_counts.plot.bar(x="source", y="rows", ax=axes[0, 0], legend=False, color="#1f6f8b")
	axes[0, 0].set_title("Rows per source file")
	axes[0, 0].set_xlabel("")
	axes[0, 0].set_ylabel("Rows")

	country_counts.plot.barh(ax=axes[0, 1], color="#d95f02")
	axes[0, 1].set_title("Top countries in sampled rows")
	axes[0, 1].set_xlabel("Rows")
	axes[0, 1].set_ylabel("")

	frame["business_name_length"].plot.hist(bins=40, ax=axes[1, 0], color="#4daf4a")
	axes[1, 0].set_title("Business-name length")
	axes[1, 0].set_xlabel("Characters")

	frame["business_address_length"].plot.hist(bins=40, ax=axes[1, 1], color="#984ea3")
	axes[1, 1].set_title("Business-address length")
	axes[1, 1].set_xlabel("Characters")

	figure.suptitle("Business Entity Dataset Overview", fontsize=16)
	figure.tight_layout()
	figure.savefig(output_dir / "dataset_overview.png", dpi=150)
	plt.close(figure)

	source_country = (
		frame.groupby(["source", "country"], dropna=False)
		.size()
		.reset_index(name="rows")
		.sort_values("rows", ascending=False)
		.groupby("source")
		.head(10)
	)
	pivot = source_country.pivot(index="country", columns="source", values="rows").fillna(0)
	pivot.sort_values(pivot.columns[0], ascending=True).plot.barh(figsize=(12, 8))
	plt.title("Top countries by source file")
	plt.xlabel("Rows in sample")
	plt.tight_layout()
	plt.savefig(output_dir / "countries_by_source.png", dpi=150)
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, default=Path("challenge-dataset"))
	parser.add_argument("--output-dir", type=Path, default=Path("dataset-explore/plots"))
	parser.add_argument("--sample-size", type=int, default=100_000)
	parser.add_argument(
		"--show-rows",
		type=int,
		default=5,
		help="Number of rows to print from each source file (use 0 to hide rows)",
	)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	if args.sample_size <= 0:
		parser.error("--sample-size must be greater than zero")
	if args.show_rows < 0:
		parser.error("--show-rows cannot be negative")

	samples = []
	counts = []
	for path in find_tsv_files(args.data_dir):
		sample, count = read_dataset(path, args.sample_size)
		samples.append(sample)
		counts.append(count)

	sampled_data = pd.concat(samples, ignore_index=True)
	row_counts = pd.concat(counts, ignore_index=True)
	create_plots(sampled_data, row_counts, args.output_dir)

	print("\nDataset summary")
	print(row_counts.to_string(index=False))
	print(f"\nSampled rows: {len(sampled_data):,}")
	print("\nMissing values in sample:")
	print(sampled_data[EXPECTED_COLUMNS].replace("", pd.NA).isna().sum().to_string())
	if args.show_rows:
		rows_to_show = sampled_data.groupby("source", sort=False).head(args.show_rows)
		print(f"\nFirst {args.show_rows} rows from each source:")
		print(rows_to_show[["source", *EXPECTED_COLUMNS]].to_string(index=False))
	print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
	main()
