"""Rebuild the PS-focused figures from the existing analysis TSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "ps_figures"
COLORS = {"US": "#315da8", "India": "#eb9c36", "France": "#49a982"}


def read_tsv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.18,
    })


def dataset_figure() -> None:
    inventory = read_tsv("dataset_summary.tsv")
    countries = read_tsv("country_summary.tsv")
    missing = read_tsv("column_summary.tsv")
    truth = read_tsv("ground_truth_summary.tsv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["S1", "S2", "S3"]
    x = np.arange(3)
    for offset, split in [(-0.18, "train"), (0.18, "test")]:
        rows = [next(r for r in inventory if r["split"] == split and r["source"] == f"source{i}") for i in (1, 2, 3)]
        values = [int(r["input_rows"]) / 1e6 for r in rows]
        bars = ax.bar(x + offset, values, 0.34, label=split.title(), color="#315da8" if split == "train" else "#49a982")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Full input file rows (millions)")
    ax.set_ylim(0, 6.0)
    ax.set_title("24.23 million input rows across six source files")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for offset, split in [(-0.18, "train"), (0.18, "test")]:
        totals = [sum(int(r["rows"]) for r in countries if r["split"] == split and r["source"] == f"source{i}") for i in (1, 2, 3)]
        bottom = np.zeros(3)
        for country in ("US", "India", "France"):
            values = np.array([sum(int(r["rows"]) for r in countries if r["split"] == split and r["source"] == f"source{i}" and r["country"] == country) / totals[i - 1] * 100 for i in (1, 2, 3)])
            ax.bar(x + offset, values, 0.34, bottom=bottom, color=COLORS[country], label=country if split == "train" else None)
            bottom += values
    country_positions = [position + offset for position in x for offset in (-0.18, 0.18)]
    country_labels = [f"{source}\n{split}" for source in labels for split in ("train", "test")]
    ax.set_xticks(country_positions, country_labels)
    ax.set_ylabel("Share of analyzed records (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Test introduces France; country mix shifts")
    ax.legend(frameon=False, ncol=3, loc="upper center")

    ax = axes[1, 0]
    for offset, split in [(-0.18, "train"), (0.18, "test")]:
        values = [float(next(r for r in missing if r["split"] == split and r["source"] == f"source{i}" and r["column"] == "business_address")["missing_percent"]) for i in (1, 2, 3)]
        bars = ax.bar(x + offset, values, 0.34, color="#315da8" if split == "train" else "#49a982")
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Missing business address (%)")
    ax.set_ylim(0, 5)
    ax.set_title("S2/S3 need a name-only fallback")

    ax = axes[1, 1]
    distribution = sorted(((int(r["category"]), int(r["count"])) for r in truth if r["measure"] == "match_count_distribution"), key=lambda pair: pair[0])
    bars = ax.bar([str(k) for k, _ in distribution], [v / 22037 * 100 for _, v in distribution], color=["#e37761" if k == 0 else "#315da8" for k, _ in distribution])
    ax.set_xlabel("True S2/S3 matches per Source 1 entity")
    ax.set_ylabel("Training Source 1 sample (%)")
    ax.set_title("94.25% have at least one match; many have several")
    ax.set_ylim(0, 26)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    fig.suptitle("Dataset scale and label structure | seeded 1% analysis", fontsize=16, fontweight="bold")
    fig.savefig(FIGURES / "ps_dataset_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def matching_figure() -> None:
    similarity = read_tsv("true_match_similarity.tsv")
    truth = read_tsv("ground_truth_summary.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    ax = axes[0]
    fields = ["business_name", "business_address"]
    x = np.arange(2)
    for offset, source in [(-0.18, "source2"), (0.18, "source3")]:
        vals = [float(next(r for r in similarity if r["source"] == source and r["field"] == field and r["measure"] == "normalized_equality")["mean"]) * 100 for field in fields]
        bars = ax.bar(x + offset, vals, 0.34, label=source.title(), color="#315da8" if source == "source2" else "#49a982")
        ax.bar_label(bars, fmt="%.1f%%", padding=2)
    ax.set_xticks(x, ["Name", "Address"])
    ax.set_ylim(0, 29)
    ax.set_ylabel("True pairs with normalized equality (%)")
    ax.set_title("Exact normalized text catches a minority")
    ax.legend(frameon=False)

    ax = axes[1]
    for offset, source in [(-0.18, "source2"), (0.18, "source3")]:
        vals = [float(next(r for r in similarity if r["source"] == source and r["field"] == field and r["measure"] == "token_jaccard_milli")["median"]) for field in fields]
        bars = ax.bar(x + offset, vals, 0.34, color="#315da8" if source == "source2" else "#49a982")
        ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_xticks(x, ["Name", "Address"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Median token Jaccard among true pairs")
    ax.set_title("Source 3 address overlap is weaker")

    ax = axes[2]
    categories = ["Both S2 + S3", "S2 only", "S3 only", "Singleton"]
    names = ["both_s2_s3", "only_s2", "only_s3", "singleton"]
    vals = [int(next(r for r in truth if r["measure"] == "source_pattern" and r["category"] == name)["count"]) / 22037 * 100 for name in names]
    bars = ax.barh(categories[::-1], vals[::-1], color=["#315da8", "#49a982", "#eb9c36", "#e37761"][::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 95)
    ax.set_xlabel("Training Source 1 sample (%)")
    ax.set_title("Recover matches from both sources")

    fig.suptitle("What the training labels imply for matching | 75,824 true pairs", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "ps_matching_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    style()
    dataset_figure()
    matching_figure()
