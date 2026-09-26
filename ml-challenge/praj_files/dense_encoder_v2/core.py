"""Data and metric primitives for the sampled-pool dense encoder experiment."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

SEED = 42
MODEL = "karmx/Llama-Karmx-Indic-Embedding-bge-m3"
MODEL_REVISION = "64eb64c1c6bf92d1f56948eb80061523c3b6a87f"
DATA = Path(__file__).resolve().parents[2] / "student_resource/dataset/train"
FIELDS = ("business_name", "business_address", "country")
SCRIPTS = {"devanagari": (0x0900, 0x097F), "bengali": (0x0980, 0x09FF),
           "gurmukhi": (0x0A00, 0x0A7F), "gujarati": (0x0A80, 0x0AFF),
           "oriya": (0x0B00, 0x0B7F), "tamil": (0x0B80, 0x0BFF),
           "telugu": (0x0C00, 0x0C7F), "kannada": (0x0C80, 0x0CFF),
           "malayalam": (0x0D00, 0x0D7F)}


def digest(value: str, salt: str = "split") -> int:
    return int.from_bytes(hashlib.blake2b(f"{SEED}:{salt}:{value}".encode(), digest_size=8).digest(), "big")


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def serialize(row: dict) -> str:
    return " | ".join(f"{label}: {clean(row.get(field))}" for label, field in
                      (("Name", FIELDS[0]), ("Address", FIELDS[1]), ("Country", FIELDS[2]))
                     if clean(row.get(field)))


def script_tags(text: str) -> set[str]:
    found = {name for name, (lo, hi) in SCRIPTS.items() if any(lo <= ord(c) <= hi for c in text)}
    if any("LATIN" in unicodedata.name(c, "") for c in text if c.isalpha()):
        found.add("latin")
    return found


def tags(query: dict, target: dict | None = None) -> list[str]:
    values = [clean(query.get(f)) for f in FIELDS]
    combined = " ".join(values + ([clean(target.get(f)) for f in FIELDS] if target else []))
    script_text = " ".join(values[:2] + ([clean(target.get(f)) for f in FIELDS[:2]] if target else []))
    scripts = script_tags(script_text)
    out = set(scripts)
    if len(scripts) > 1:
        out.add("mixed_script")
    if any(not x for x in values) or (target and any(not clean(target.get(f)) for f in FIELDS)):
        out.add("missing_field")
    if len(values[0]) < 8 or len(values[1]) < 10:
        out.add("short_text")
    if re.search(r"[^\w\s.,/&'-]", combined, re.UNICODE):
        out.add("noisy_text")
    if target:
        a, b = clean(query.get("business_name")).casefold(), clean(target.get("business_name")).casefold()
        if a and b and a == b:
            out.add("same_name")
        elif a and b and (a in b or b in a):
            out.add("near_duplicate")
        if script_tags(a) != script_tags(b):
            out.add("cross_script")
    return sorted(out)


def f05(truth: set[str], prediction: set[str]) -> float:
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    if not tp:
        return 0.0
    precision, recall = tp / len(prediction), tp / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def metrics(queries: list[dict], ranked_ids: list[list[str]], scores: np.ndarray,
            threshold: float, top_k: int) -> dict:
    n = len(queries)
    truths = [set(filter(None, q["truth"].split(","))) for q in queries]
    candidates = [ids[:top_k] for ids in ranked_ids]
    predictions = [{tid for tid, score in zip(ids, row) if score >= threshold}
                   for ids, row in zip(candidates, scores)]
    hit_counts = [len(t & set(c)) for t, c in zip(truths, candidates)]
    nonempty = [i for i, t in enumerate(truths) if t]
    recall_at = {str(k): sum(bool(t & set(ids[:k])) for t, ids in zip(truths, ranked_ids) if t) /
                 max(1, len(nonempty)) for k in (1, 5, 10, 20)}
    mrr = sum(next((1 / (j + 1) for j, tid in enumerate(ranked_ids[i]) if tid in truths[i]), 0)
              for i in nonempty) / max(1, len(nonempty))
    tp = sum(len(t & p) for t, p in zip(truths, predictions))
    predicted = sum(map(len, predictions))
    positives = sum(map(len, truths))
    per_query = [f05(t, p) for t, p in zip(truths, predictions)]
    groups = {}
    for tag in sorted({tag for q in queries for tag in q.get("tags", "").split(",") if tag} | {"singleton"}):
        ix = [i for i, q in enumerate(queries) if tag in q.get("tags", "").split(",") or
              (tag == "singleton" and not truths[i])]
        if ix:
            groups[tag] = {"queries": len(ix), "macro_f0.5": float(np.mean([per_query[i] for i in ix])),
                           "any_match_retention": sum(bool(truths[i] & set(candidates[i])) for i in ix if truths[i]) /
                           max(1, sum(bool(truths[i]) for i in ix))}
    return {"queries": n, "true_pairs": positives, "threshold": threshold, "top_k": top_k,
            "macro_f0.5": float(np.mean(per_query)) if n else 0.0,
            "precision": tp / max(1, predicted), "recall": tp / max(1, positives),
            "recall_at": recall_at, "mrr": mrr,
            "pair_retention": sum(hit_counts) / max(1, positives),
            "any_match_retention": sum(bool(hit_counts[i]) for i in nonempty) / max(1, len(nonempty)),
            "all_match_retention": sum(hit_counts[i] == len(truths[i]) for i in nonempty) / max(1, len(nonempty)),
            "singleton_accuracy": sum(not predictions[i] for i in range(n) if not truths[i]) /
                                  max(1, n - len(nonempty)), "difficult_subsets": groups}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
