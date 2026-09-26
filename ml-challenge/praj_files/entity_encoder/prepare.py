from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import config, record_text, rows, save_json, source_path, value


def bucket(entity_id: str) -> float:
    digest = hashlib.blake2b(entity_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def name_key(row: dict) -> str:
    return " ".join(re.findall(r"\w+", value(row.get("business_name")).casefold()))


def tags(query: dict, target: dict) -> list[str]:
    name_a, name_b = name_key(query), name_key(target)
    addr_a = value(query.get("business_address")).casefold()
    addr_b = value(target.get("business_address")).casefold()
    out = []
    if not addr_a or not addr_b or not name_a or not name_b:
        out.append("missing_field")
    if any(ord(c) > 127 for c in value(target.get("business_name")) + addr_b):
        out.append("non_ascii")
    if re.search(r"[\u0900-\u097f]", value(target.get("business_name")) + addr_b):
        out.append("devanagari")
    if name_a == name_b and addr_a != addr_b:
        out.append("same_name_changed_address")
    if addr_a == addr_b and name_a != name_b:
        out.append("same_address_changed_name")
    if name_a != name_b:
        out.append("name_variant")
    if addr_a != addr_b:
        out.append("address_variant")
    if not out:
        out.append("exact")
    return out


def write_parquet(path: Path, items: list[dict], batch: int = 50000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, len(items), batch):
            table = pa.Table.from_pylist(items[start:start + batch])
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()


def prepare(cfg: dict) -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    train_truth: dict[str, list[str]] = {}
    val_truth: dict[str, list[str]] = {}
    val_limit = cfg["max_validation_queries"]
    for row in rows(Path(cfg["data_dir"]) / "train_ground_truth.parquet",
                    ["source1_entity_id", "matched_entity_ids"]):
        anchor = row["source1_entity_id"]
        b = bucket(anchor)
        ids = [x for x in value(row["matched_entity_ids"]).split(",") if x]
        if b < cfg["anchor_fraction"]:
            train_truth[anchor] = ids
        elif b < cfg["anchor_fraction"] + cfg["validation_fraction"] and len(val_truth) < val_limit:
            val_truth[anchor] = ids
    # Drop labels whose target is also attached to an anchor in the other split.
    train_targets = {t for ids in train_truth.values() for t in ids}
    for anchor in list(val_truth):
        if any(t in train_targets for t in val_truth[anchor]):
            del val_truth[anchor]
    anchor_ids = set(train_truth) | set(val_truth)
    anchor_rows = {}
    for row in rows(source_path(cfg, "train", 1),
                    ["entity_id", "business_name", "business_address", "country"]):
        if row["entity_id"] in anchor_ids:
            anchor_rows[row["entity_id"]] = row
    required = train_targets | {t for ids in val_truth.values() for t in ids}
    target_rows = {}
    distractors = {}
    for source in (2, 3):
        for row in rows(source_path(cfg, "train", source),
                        ["entity_id", "business_name", "business_address", "country"]):
            entity_id = row["entity_id"]
            if entity_id in required:
                target_rows[entity_id] = row
            elif bucket(entity_id) < cfg["validation_distractors"] / 11_000_000:
                distractors[entity_id] = row
    missing = required - target_rows.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} ground truth target IDs; sample: {list(missing)[:3]}")
    train_truth = {k: v for k, v in train_truth.items() if k in anchor_rows}
    val_truth = {k: v for k, v in val_truth.items() if k in anchor_rows}
    pairs = []
    for anchor, ids in train_truth.items():
        q = anchor_rows[anchor]
        for target_id in ids:
            t = target_rows[target_id]
            label_tags = tags(q, t)
            # Exact duplicates receive the smallest sampling priority.
            weight = 0.3 if label_tags == ["exact"] else 1.0
            if "missing_field" in label_tags: weight *= 2.5
            if "non_ascii" in label_tags: weight *= 2.0
            if "devanagari" in label_tags: weight *= 1.5
            if "same_name_changed_address" in label_tags or "same_address_changed_name" in label_tags:
                weight *= 1.8
            pairs.append((anchor, target_id, label_tags, weight))
    if not pairs:
        raise ValueError("No positive pairs selected")
    if len(pairs) > cfg["max_train_pairs"]:
        weights = np.fromiter((p[3] for p in pairs), dtype=np.float64)
        selected = np.random.default_rng(cfg["seed"]).choice(
            len(pairs), size=cfg["max_train_pairs"], replace=False, p=weights / weights.sum())
        pairs = [pairs[int(i)] for i in selected]
    rng.shuffle(pairs)
    by_name = collections.defaultdict(list)
    by_token = collections.defaultdict(list)
    by_country = collections.defaultdict(list)
    for target_id in train_targets:
        row = target_rows[target_id]
        country = value(row.get("country"))
        name = name_key(row)
        by_name[(country, name)].append(target_id)
        by_token[(country, name.split(" ")[0] if name else "")].append(target_id)
        by_country[country].append(target_id)
    train_data = []
    negative_types = collections.Counter()
    for anchor, target_id, label_tags, _ in pairs:
        q = anchor_rows[anchor]
        target = target_rows[target_id]
        country = value(q.get("country"))
        name = name_key(q)
        options = [("same_name", by_name.get((country, name), [])),
                   ("same_first_token", by_token.get((country, name.split(" ")[0] if name else ""), [])),
                   ("same_country", by_country.get(country, []))]
        excluded = set(train_truth[anchor])
        negative_id = None
        negative_type = None
        for label, pool in options:
            if not pool: continue
            for _ in range(12):
                candidate = rng.choice(pool)
                if candidate not in excluded:
                    negative_id, negative_type = candidate, label
                    break
            if negative_id: break
        if not negative_id:
            continue
        negative_types[negative_type] += 1
        train_data.append({"anchor_id": anchor, "positive_id": target_id,
                           "negative_id": negative_id, "anchor": record_text(q),
                           "positive": record_text(target),
                           "negative": record_text(target_rows[negative_id]),
                           "tags": ",".join(label_tags), "negative_type": negative_type})
    write_parquet(out / "train_pairs.parquet", train_data)
    val_queries = [{"entity_id": k, "text": record_text(anchor_rows[k]),
                    "truth": ",".join(v),
                    "tags": ",".join(sorted({tag for target_id in v
                                             for tag in tags(anchor_rows[k], target_rows[target_id])}
                                            or {"singleton"}))}
                   for k, v in val_truth.items()]
    val_target_ids = {t for ids in val_truth.values() for t in ids}
    val_target_ids.update(distractors)
    hard_validation_count = 0
    for anchor, true_ids in val_truth.items():
        query = anchor_rows[anchor]
        country = value(query.get("country"))
        name = name_key(query)
        seen = set(true_ids)
        for pool, limit in ((by_name.get((country, name), []), 5),
                            (by_token.get((country, name.split(" ")[0] if name else ""), []), 20)):
            candidates = [x for x in pool if x not in seen]
            if len(candidates) > limit:
                candidates = rng.sample(candidates, limit)
            val_target_ids.update(candidates)
            hard_validation_count += len(candidates)
    val_targets = [{"entity_id": k, "text": record_text(target_rows.get(k, distractors.get(k)))}
                   for k in sorted(val_target_ids)]
    write_parquet(out / "val_queries.parquet", val_queries)
    write_parquet(out / "val_targets.parquet", val_targets)
    counts = collections.Counter(tag for p in pairs for tag in p[2])
    report = {"train_anchors": len(train_truth), "validation_anchors": len(val_truth),
              "positive_pairs_available": sum(map(len, train_truth.values())),
              "train_pairs_written": len(train_data), "validation_targets": len(val_targets),
              "hard_validation_negatives_added": hard_validation_count,
              "sampled_pair_tags": counts, "negative_types": negative_types,
              "split": "deterministic blake2b on T1 ID; cross-split shared targets removed",
              "seed": cfg["seed"]}
    save_json(out / "sampling_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    print(json.dumps(prepare(config(args.config)), indent=2))
