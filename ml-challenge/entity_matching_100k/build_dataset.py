"""Build a reusable, raw-text entity-resolution sample from the challenge TSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SEED = 20260926
SAMPLE_SIZE = 100_000
CANDIDATE_SIZE = 500_000
BASE_SIZE = 70_000
PER_TAG = 3_000
DISTRACTORS_PER_SOURCE = 250_000
DISTRACTOR_HEAP_SIZE = 275_000
TAGS = (
    "singleton", "many_matches", "rare_indic", "devanagari", "mixed_script",
    "missing_t2_field", "short_noisy", "numeric_punctuation",
    "duplicate_like", "low_name_overlap",
)
FIELDS = ("entity_id", "business_name", "business_address", "country")
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F), "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}
RARE_INDIC = set(SCRIPT_RANGES) - {"Devanagari"}
WORD = re.compile(r"\w+", re.UNICODE)
LOG = logging.getLogger("entity_matching_100k")


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["source1_entity_id", "matched_entity_ids"] if "ground_truth" in path.name else list(FIELDS)
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed TSV row in {path}")
            yield row


def rank(value: str, domain: str) -> int:
    digest = hashlib.blake2b(f"{SEED}|{domain}|{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def add_smallest(heap: list[tuple[int, str]], limit: int, score: int, entity_id: str) -> None:
    entry = (-score, entity_id)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def scripts(value: str) -> set[str]:
    found: set[str] = set()
    for char in value:
        code = ord(char)
        if char.isalpha() and (code < 0x0250 or 0x1E00 <= code <= 0x1EFF):
            found.add("Latin")
        elif char.isalpha():
            for label, (start, end) in SCRIPT_RANGES.items():
                if start <= code <= end:
                    found.add(label)
                    break
            else:
                found.add("Other")
    return found


def duplicate_key(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def noisy(value: str) -> bool:
    if not value:
        return True
    punctuation = sum(not char.isalnum() and not char.isspace() for char in value)
    digits = sum(char.isdigit() for char in value)
    return punctuation >= max(3, len(value) // 5) or digits >= max(3, len(value) * 3 // 10)


def low_name_overlap(left: str, right: str) -> bool:
    a, b = set(WORD.findall(left.casefold())), set(WORD.findall(right.casefold()))
    return bool(a and b) and len(a & b) / len(a | b) <= 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(path: Path, data: list[dict], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(data, schema=schema)
    pq.write_table(table, path, compression="zstd", version="2.6", row_group_size=100_000)


def select_candidates(source_dir: Path) -> tuple[list[str], dict[str, dict]]:
    heap: list[tuple[int, str]] = []
    for count, row in enumerate(rows(source_dir / "train_source1.tsv"), 1):
        entity_id = row["entity_id"]
        add_smallest(heap, CANDIDATE_SIZE, rank(entity_id, "t1"), entity_id)
        if count % 500_000 == 0:
            LOG.info("T1 hash scan: %d records", count)
    if len(heap) != CANDIDATE_SIZE:
        raise ValueError("Fewer than 500,000 T1 records")
    ordered = [entity_id for _, entity_id in sorted((-score, entity_id) for score, entity_id in heap)]
    chosen = set(ordered)
    source1 = {row["entity_id"]: row for row in rows(source_dir / "train_source1.tsv") if row["entity_id"] in chosen}
    if len(source1) != CANDIDATE_SIZE:
        raise ValueError("Selected T1 IDs are missing or duplicated")
    LOG.info("Selected %d candidate T1 records", len(source1))
    return ordered, source1


def candidate_truth(source_dir: Path, candidate_ids: set[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    truth: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for count, row in enumerate(rows(source_dir / "train_ground_truth.tsv"), 1):
        sid = row["source1_entity_id"]
        if sid not in candidate_ids:
            continue
        targets = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        if sid in truth or len(targets) != len(set(targets)):
            raise ValueError(f"Duplicate truth mapping for {sid}")
        truth[sid] = targets
        for tid in targets:
            if not tid.startswith(("S2-", "S3-")) or tid in owner:
                raise ValueError(f"Invalid or multiply owned positive T2: {tid}")
            owner[tid] = sid
        if count % 500_000 == 0:
            LOG.info("Ground-truth scan: %d rows, %d candidate positive IDs", count, len(owner))
    if set(truth) != candidate_ids:
        raise ValueError(f"Missing {len(candidate_ids - set(truth))} candidate ground-truth rows")
    LOG.info("Candidate ground truth: %d T1 rows, %d positive T2 IDs", len(truth), len(owner))
    return truth, owner


def scan_t2(source_dir: Path, source1: dict[str, dict], owner: dict[str, str]) -> tuple[dict[str, set[str]], dict[int, list[tuple[int, str]]], int]:
    flags: dict[str, set[str]] = defaultdict(set)
    heaps: dict[int, list[tuple[int, str]]] = {2: [], 3: []}
    found_positives = 0
    for source in (2, 3):
        for count, row in enumerate(rows(source_dir / f"train_source{source}.tsv"), 1):
            tid = row["entity_id"]
            add_smallest(heaps[source], DISTRACTOR_HEAP_SIZE, rank(tid, "distractor"), tid)
            sid = owner.get(tid)
            if count % 1_000_000 == 0:
                LOG.info("Source %d feature scan: %d rows, %d candidate positives found", source, count, found_positives)
            if sid is None:
                continue
            found_positives += 1
            name, address = row["business_name"], row["business_address"]
            name_scripts = scripts(name)
            all_scripts = name_scripts | scripts(address)
            indic = all_scripts & set(SCRIPT_RANGES)
            if indic & RARE_INDIC:
                flags[sid].add("rare_indic")
            if "Devanagari" in indic:
                flags[sid].add("devanagari")
            if "Latin" in name_scripts and name_scripts & set(SCRIPT_RANGES):
                flags[sid].add("mixed_script")
            if not name or not address:
                flags[sid].add("missing_t2_field")
            if len(name) <= 8 or len(address) <= 12:
                flags[sid].add("short_noisy")
            if noisy(name) or noisy(address):
                flags[sid].add("numeric_punctuation")
            if "Latin" in name_scripts and not name_scripts & set(SCRIPT_RANGES):
                if low_name_overlap(source1[sid]["business_name"], name):
                    flags[sid].add("low_name_overlap")
    if found_positives != len(owner):
        raise ValueError(f"Only {found_positives}/{len(owner)} candidate positives found in T2 files")
    LOG.info("T2 feature scan complete: %d candidate positives found", found_positives)
    return flags, heaps, found_positives


def choose_t1(ordered: list[str], source1: dict[str, dict], truth: dict[str, list[str]], flags: dict[str, set[str]]) -> tuple[list[str], dict[str, str], Counter]:
    names = Counter(duplicate_key(row["business_name"]) for row in source1.values() if row["business_name"])
    addresses = Counter(duplicate_key(row["business_address"]) for row in source1.values() if row["business_address"])
    for sid, row in source1.items():
        name, address = row["business_name"], row["business_address"]
        if not truth[sid]:
            flags[sid].add("singleton")
        if len(truth[sid]) >= 7:
            flags[sid].add("many_matches")
        if len(name) <= 8 or len(address) <= 12 or not name or not address:
            flags[sid].add("short_noisy")
        if noisy(name) or noisy(address):
            flags[sid].add("numeric_punctuation")
        if names[duplicate_key(name)] > 1 or addresses[duplicate_key(address)] > 1:
            flags[sid].add("duplicate_like")
    selected: dict[str, str] = {sid: "representative" for sid in ordered[:BASE_SIZE]}
    LOG.info("Representative T1 allocation: %d", len(selected))
    for tag in TAGS:
        count = 0
        for sid in ordered:
            if sid not in selected and tag in flags.get(sid, ()):
                selected[sid] = tag
                count += 1
                if count == PER_TAG:
                    break
        LOG.info("Diversity allocation %s: %d records", tag, count)
    for sid in ordered:
        if len(selected) == SAMPLE_SIZE:
            break
        selected.setdefault(sid, "representative_fill")
    if len(selected) != SAMPLE_SIZE:
        raise ValueError(f"Selected {len(selected)} instead of 100,000 T1 rows")
    LOG.info("Final T1 sample: %d records", len(selected))
    return sorted(selected), selected, Counter(tag for sid in selected for tag in flags.get(sid, ()))


def pick_distractors(heaps: dict[int, list[tuple[int, str]]], positive_ids: set[str]) -> set[str]:
    result: set[str] = set()
    for source in (2, 3):
        ordered = sorted((-score, tid) for score, tid in heaps[source])
        selected = [tid for _, tid in ordered if tid not in positive_ids][:DISTRACTORS_PER_SOURCE]
        if len(selected) != DISTRACTORS_PER_SOURCE:
            raise ValueError(f"Insufficient Source {source} distractor heap; increase its size")
        result.update(selected)
        LOG.info("Selected %d Source %d distractors", len(selected), source)
    return result


def sample_negatives(selected_ids: list[str], source1: dict[str, dict], distractor_ids: set[str], targets: dict[str, dict]) -> list[dict]:
    by_source: dict[int, list[str]] = {source: sorted(tid for tid in distractor_ids if tid.startswith(f"S{source}-")) for source in (2, 3)}
    by_country: dict[tuple[int, str], list[str]] = defaultdict(list)
    for source in (2, 3):
        for tid in by_source[source]:
            by_country[(source, targets[tid]["country"])].append(tid)
    negatives: list[dict] = []
    for sid in selected_ids:
        country = source1[sid]["country"]
        used: set[str] = set()
        for source in (2, 3):
            for slot, kind in enumerate(("same_country", "same_country", "global")):
                pool = by_country[(source, country)] if kind == "same_country" else by_source[source]
                if not pool:
                    raise ValueError(f"No negative pool for Source {source}, country {country}")
                attempt = 0
                while True:
                    tid = pool[rank(f"{sid}|{source}|{slot}|{attempt}", "negative") % len(pool)]
                    attempt += 1
                    if tid not in used:
                        break
                    if attempt > 100:
                        raise ValueError(f"Cannot sample distinct negatives for {sid}")
                used.add(tid)
                negatives.append({"source1_entity_id": sid, "negative_entity_id": tid,
                                  "negative_kind": kind, "source": f"S{source}"})
    return negatives


def field_stats(records: list[dict]) -> dict:
    script_counts: Counter = Counter()
    for row in records:
        script_counts.update(scripts(row["business_name"] + " " + row["business_address"]))
    return {
        "rows": len(records),
        "country": dict(sorted(Counter(row["country"] for row in records).items())),
        "empty_fields": {field: sum(not row[field] for row in records) for field in FIELDS[1:]},
        "scripts_record_counts": dict(sorted(script_counts.items())),
        "short_names_le_8": sum(len(row["business_name"]) <= 8 for row in records),
        "non_ascii_names": sum(any(ord(ch) > 127 for ch in row["business_name"]) for row in records),
        "non_ascii_addresses": sum(any(ord(ch) > 127 for ch in row["business_address"]) for row in records),
    }


def build(source_dir: Path, output_dir: Path, force: bool = False) -> dict:
    start = time.perf_counter()
    files = ("t1.parquet", "t2.parquet", "ground_truth.parquet", "positives.parquet", "negatives.parquet", "selection.parquet", "statistics.json", "manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and any((output_dir / filename).exists() for filename in files):
        raise FileExistsError(f"Dataset already exists at {output_dir}; use --force to rebuild")
    LOG.info("Build started: source=%s output=%s seed=%d", source_dir, output_dir, SEED)
    ordered, source1 = select_candidates(source_dir)
    truth, owner = candidate_truth(source_dir, set(ordered))
    flags, heaps, candidate_positive_count = scan_t2(source_dir, source1, owner)
    selected_ids, selection_reason, tag_counts = choose_t1(ordered, source1, truth, flags)
    positives = [
        {"source1_entity_id": sid, "matched_entity_id": tid, "ground_truth_position": index}
        for sid in selected_ids for index, tid in enumerate(truth[sid])
    ]
    positive_ids = {row["matched_entity_id"] for row in positives}
    distractor_ids = pick_distractors(heaps, positive_ids)
    needed = positive_ids | distractor_ids
    targets: dict[str, dict] = {}
    LOG.info("Extracting %d distinct T2 records", len(needed))
    for source in (2, 3):
        for count, row in enumerate(rows(source_dir / f"train_source{source}.tsv"), 1):
            if row["entity_id"] in needed:
                if row["entity_id"] in targets:
                    raise ValueError(f"Duplicate T2 ID {row['entity_id']}")
                targets[row["entity_id"]] = row
            if count % 1_000_000 == 0:
                LOG.info("Source %d extraction: %d rows scanned, %d T2 records loaded", source, count, len(targets))
    if set(targets) != needed:
        raise ValueError(f"Missing {len(needed - set(targets))} selected T2 records")
    negatives = sample_negatives(selected_ids, source1, distractor_ids, targets)
    LOG.info("Generated %d positives and %d negatives", len(positives), len(negatives))
    t1_rows = [source1[sid] for sid in selected_ids]
    t2_rows = [dict(targets[tid], source=f"S{tid[1]}", pool_role="positive" if tid in positive_ids else "distractor") for tid in sorted(needed)]
    truth_rows = [{"source1_entity_id": sid, "matched_entity_ids": truth[sid]} for sid in selected_ids]
    selection_rows = [{"source1_entity_id": sid, "selection_reason": selection_reason[sid],
                       "sampling_tags": sorted(flags.get(sid, ()))} for sid in selected_ids]
    raw_schema = pa.schema([(field, pa.string()) for field in FIELDS])
    write_parquet(output_dir / "t1.parquet", t1_rows, raw_schema)
    write_parquet(output_dir / "t2.parquet", t2_rows, raw_schema.append(pa.field("source", pa.string())).append(pa.field("pool_role", pa.string())))
    write_parquet(output_dir / "ground_truth.parquet", truth_rows, pa.schema([("source1_entity_id", pa.string()), ("matched_entity_ids", pa.list_(pa.string()))]))
    write_parquet(output_dir / "positives.parquet", positives, pa.schema([("source1_entity_id", pa.string()), ("matched_entity_id", pa.string()), ("ground_truth_position", pa.int32())]))
    write_parquet(output_dir / "negatives.parquet", negatives, pa.schema([("source1_entity_id", pa.string()), ("negative_entity_id", pa.string()), ("negative_kind", pa.string()), ("source", pa.string())]))
    write_parquet(output_dir / "selection.parquet", selection_rows, pa.schema([("source1_entity_id", pa.string()), ("selection_reason", pa.string()), ("sampling_tags", pa.list_(pa.string()))]))
    LOG.info("Wrote six canonical Parquet tables")
    stats = {
        "seed": SEED, "candidate_t1_count": len(ordered), "candidate_positive_count": candidate_positive_count,
        "t1": field_stats(t1_rows), "t2": field_stats(t2_rows),
        "positive_pairs": len(positives), "positive_t2_by_source": dict(sorted(Counter(tid[:2] for tid in positive_ids).items())),
        "negative_pairs": len(negatives), "distractors": len(distractor_ids),
        "singletons": sum(not truth[sid] for sid in selected_ids),
        "match_count_distribution": dict(sorted(Counter(len(truth[sid]) for sid in selected_ids).items())),
        "sampling_tags": dict(sorted(tag_counts.items())),
        "selection_reasons": dict(sorted(Counter(selection_reason.values()).items())),
    }
    (output_dir / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_files = [source_dir / name for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv")]
    manifest = {
        "format_version": 1, "seed": SEED, "source_files": {path.name: sha256(path) for path in source_files},
        "output_files": {name: sha256(output_dir / name) for name in files if name != "manifest.json"},
        "parameters": {"sample_size": SAMPLE_SIZE, "candidate_size": CANDIDATE_SIZE,
                       "representative_size": BASE_SIZE, "diversity_per_tag": PER_TAG,
                       "distractors_per_source": DISTRACTORS_PER_SOURCE, "negatives_per_t1": 6},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Build complete in %.1f seconds: %d T1, %d T2", time.perf_counter() - start, len(t1_rows), len(t2_rows))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "student_resource/dataset/train")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true", help="Replace an existing generated dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = build(args.source_dir.resolve(), args.output_dir.resolve(), args.force)
    print(json.dumps({"t1": stats["t1"]["rows"], "t2": stats["t2"]["rows"],
                      "positives": stats["positive_pairs"], "negatives": stats["negative_pairs"]}))


if __name__ == "__main__":
    main()
