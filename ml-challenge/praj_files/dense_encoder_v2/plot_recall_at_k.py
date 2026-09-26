"""Render high-resolution coverage and candidate-quality plots for K=1..50."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(root: Path) -> tuple[list[dict], dict]:
    json_path = root / "recall_at_k_1_to_50.json"
    csv_path = root / "recall_at_k_1_to_50.csv"
    if json_path.exists():
        report = json.loads(json_path.read_text(encoding="utf-8"))
        return report["rows"], report
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key, value in row.items():
                row[key] = float(value) if key.endswith("_pct") or key.startswith("mean_") else int(value)
        return rows, {"label": "sampled-pool", "query_count": None, "target_pool_size": None}
    raise FileNotFoundError(f"No recall_at_k_1_to_50.json or .csv under {root}")


def plot(root: Path, output: Path) -> None:
    rows, report = load_rows(root)
    k = np.array([r["k"] for r in rows])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                         "figure.dpi": 140, "savefig.dpi": 240})
    title_suffix = f"Sampled-pool · {report.get('query_count', '?')} queries · {report.get('target_pool_size', '?')} targets"

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for field, label, color in [
        ("pair_recall_pct", "Ground-truth pair recall", "#1769aa"),
        ("any_match_recall_pct", "Any true match per query", "#e87500"),
        ("all_match_recall_pct", "All true matches per query", "#238636"),
    ]:
        ax.plot(k, [r[field] for r in rows], label=label, linewidth=2.4, color=color)
    ax.set(title=f"Retrieval coverage vs candidate depth K\n{title_suffix}", xlabel="Candidates retained per query (K)",
           ylabel="Coverage (%)", xlim=(1, 50), ylim=(0, 101))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.grid(True, alpha=.25)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(output / "recall_coverage_vs_k.png", bbox_inches="tight")
    fig.savefig(output / "recall_coverage_vs_k.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.plot(k, [r["singleton_accuracy_pct"] for r in rows], label="Singleton target found", linewidth=2.4, color="#7b2cbf")
    ax.plot(k, [r["multi_match_complete_pct"] for r in rows], label="All targets found (multi-match queries)", linewidth=2.4, color="#008c95")
    ax.set(title=f"Complete-query coverage by query type\n{title_suffix}", xlabel="Candidates retained per query (K)",
           ylabel="Queries fully recovered (%)", xlim=(1, 50), ylim=(0, 101))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.grid(True, alpha=.25)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(output / "complete_query_recovery_vs_k.png", bbox_inches="tight")
    fig.savefig(output / "complete_query_recovery_vs_k.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.plot(k, [r["candidate_precision_pct"] for r in rows], label="Candidate precision", linewidth=2.4, color="#bf3989")
    ax.set(title=f"Candidate precision vs candidate depth\n{title_suffix}", xlabel="Candidates retained per query (K)",
           ylabel="True pairs / candidate slots (%)", xlim=(1, 50))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", frameon=True)
    fig.savefig(output / "candidate_precision_vs_k.png", bbox_inches="tight")
    fig.savefig(output / "candidate_precision_vs_k.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    found = np.array([r["ground_truth_pairs_found"] for r in rows])
    total = rows[0]["total_ground_truth_pairs"]
    ax.plot(k, found, label="True pairs retrieved", linewidth=2.4, color="#1769aa")
    ax.axhline(total, color="#555", linestyle="--", linewidth=1.4, label=f"Total ground-truth pairs ({total:,})")
    ax.set(title=f"Ground-truth pair yield vs K\n{title_suffix}", xlabel="Candidates retained per query (K)",
           ylabel="Ground-truth pairs", xlim=(1, 50))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.grid(True, alpha=.25)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(output / "ground_truth_pairs_found_vs_k.png", bbox_inches="tight")
    fig.savefig(output / "ground_truth_pairs_found_vs_k.svg", bbox_inches="tight")
    plt.close(fig)

    description = {
        "source": str(root), "label": report.get("label", "sampled-pool"),
        "plots": [
            "recall_coverage_vs_k.png/.svg: pair recall, any-match and all-match recall",
            "complete_query_recovery_vs_k.png/.svg: singleton accuracy and multi-match completeness",
            "candidate_precision_vs_k.png/.svg: true pairs divided by all candidate slots",
            "ground_truth_pairs_found_vs_k.png/.svg: absolute ground-truth pair count",
        ],
    }
    (output / "plot_manifest.json").write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output), "plots": description["plots"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("praj_files/dense_encoder_v2/artifacts"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "plots" / "recall_at_k_1_to_50"
    plot(root, output)


if __name__ == "__main__":
    main()
