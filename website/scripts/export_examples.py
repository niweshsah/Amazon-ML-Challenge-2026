"""Export a bounded, labelled snapshot of business records for EntityLens."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

FIELDS = ("entity_id", "business_name", "business_address", "country")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def normalized(value):
    return unicodedata.normalize("NFKC", value).casefold().strip()


def export_examples(data_dir, output, max_queries, seed, synthetic):
    source_path = data_dir / "train_source1.tsv"
    count = sum(1 for _ in read_rows(source_path))
    def selection_key(record):
        return hashlib.sha256(f"{seed}:{record['entity_id']}".encode()).hexdigest()
    queries = heapq.nsmallest(min(count, max_queries), read_rows(source_path), key=selection_key)
    queries.sort(key=lambda record: record["entity_id"])
    selected_ids = {record["entity_id"] for record in queries}
    truth = {}
    for row in read_rows(data_dir / "train_ground_truth.tsv"):
        query_id = row["source1_entity_id"]
        if query_id in selected_ids:
            if query_id in truth:
                raise ValueError(f"Duplicate ground-truth entity: {query_id}")
            truth[query_id] = set(filter(None, row["matched_entity_ids"].split(",")))
    if set(truth) != selected_ids:
        raise ValueError("Ground truth must include every selected entity and singleton")
    positives = {target_id for links in truth.values() for target_id in links}
    found_positives = set()
    negative_heaps = {(query["entity_id"], source): [] for query in queries for source in (2, 3)}
    target_records = {}
    target_counts = {}
    for source in (2, 3):
        target_counts[str(source)] = 0
        for target in read_rows(data_dir / f"train_source{source}.tsv"):
            target_counts[str(source)] += 1
            target_id = target["entity_id"]
            if not target_id.startswith(f"S{source}-"):
                raise ValueError(f"Invalid source prefix: {target_id}")
            if target_id in positives:
                target_records[target_id] = {field: target[field] for field in FIELDS}
                found_positives.add(target_id)
            for query in queries:
                query_id = query["entity_id"]
                if target_id in truth[query_id]:
                    continue
                agreement = SequenceMatcher(None, normalized(query["business_name"]), normalized(target["business_name"])).ratio()
                country_agreement = normalized(query["country"]) == normalized(target["country"])
                heap = negative_heaps[(query_id, source)]
                entry = (country_agreement, agreement, target_id, target)
                if len(heap) < 4:
                    heapq.heappush(heap, entry)
                elif entry[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, entry)
    if found_positives != positives:
        raise ValueError("Some positive targets are absent from the input sources")
    examples = []
    for query in queries:
        query_id = query["entity_id"]
        comparisons = []
        for target_id in sorted(truth[query_id]):
            comparisons.append({"target": target_records[target_id], "source": int(target_id[1]), "is_match": True})
        for source in (2, 3):
            for _, _, _, target in sorted(negative_heaps[(query_id, source)], key=lambda entry: (-entry[1], entry[2])):
                comparisons.append({"target": {field: target[field] for field in FIELDS}, "source": source, "is_match": False})
        examples.append({"record": {field: query[field] for field in FIELDS}, "match_count": len(truth[query_id]), "comparisons": comparisons})
    snapshot = {
        "metadata": {"title": "Synthetic ground-truth examples" if synthetic else "Training ground-truth examples",
            "synthetic": synthetic, "generated_at": datetime.now(timezone.utc).isoformat(),
            "reference_count": len(queries), "true_links": sum(len(links) for links in truth.values()),
            "singletons": sum(not links for links in truth.values()),
            "countries": sorted({query["country"] for query in queries}), "target_source_counts": target_counts,
            "negative_selection": "Up to four unlinked targets per source, prioritizing country agreement and similar names. Labels come only from ground truth."},
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(examples)} reference records to {output}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "final_pipeline/.fixture/data")
    parser.add_argument("--output", type=Path, default=project_root / "website/dist/assets/examples.json")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=None)
    arguments = parser.parse_args()
    if arguments.max_queries < 1:
        parser.error("--max-queries must be positive")
    is_synthetic = arguments.synthetic if arguments.synthetic is not None else ".fixture" in arguments.data_dir.parts
    export_examples(arguments.data_dir, arguments.output, arguments.max_queries, arguments.seed, is_synthetic)
