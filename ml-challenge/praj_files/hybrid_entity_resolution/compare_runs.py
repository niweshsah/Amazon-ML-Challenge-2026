"""Print a side-by-side summary of sanscript and IndicXlit run metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    metrics_path = path / "all_metrics.json"
    if not metrics_path.exists():
        retrieval_metrics_path = path / "retrieval_metrics.json"
        if retrieval_metrics_path.exists():
            return {"retrieval": json.loads(retrieval_metrics_path.read_text(encoding="utf-8"))}
        raise FileNotFoundError(f"No retrieval metrics found at {retrieval_metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanscript_dir", type=Path)
    parser.add_argument("indicxlit_dir", type=Path)
    args = parser.parse_args()
    runs = {
        "sanscript": load_metrics(args.sanscript_dir),
        "indicxlit": load_metrics(args.indicxlit_dir),
    }
    print("variant\tmaster_recall\ttop_k_recall\tvalidation_macro_f0.5\tvalidation_precision\tvalidation_recall")
    for name, metrics in runs.items():
        retrieval = metrics.get("retrieval", {})
        features = metrics.get("features", {})
        validation = metrics.get("reranker", {}).get("validation", {})
        print(
            f"{name}\t"
            f"{retrieval.get('candidate_recall', 'NA')}\t"
            f"{features.get('candidate_recall', 'NA')}\t"
            f"{validation.get('macro_f0_5', 'NA')}\t"
            f"{validation.get('precision', 'NA')}\t"
            f"{validation.get('recall', 'NA')}"
        )


if __name__ == "__main__":
    main()
