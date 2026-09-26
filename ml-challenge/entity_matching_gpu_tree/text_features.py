"""Unicode-aware record normalization and pairwise numeric features."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


ADDRESS_ALIASES = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "apt": "apartment",
    "fl": "floor", "no": "number",
}
NAME_ALIASES = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "&": "and",
}
COMMON_NAME = frozenset({
    "limited", "private", "llc", "ltd", "inc", "incorporated", "corporation",
    "corp", "pvt", "company", "co", "the", "and", "services", "group",
})
COMMON_ADDRESS = frozenset({
    "road", "street", "avenue", "drive", "lane", "near", "city", "state",
    "floor", "building", "block", "no", "number", "india", "usa",
})
NUMBER_RE = re.compile(r"\d+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def normalize(value: str) -> str:
    """Keep letters and digits from every Unicode script; preserve raw elsewhere."""
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    return " ".join("".join(
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in value
    ).split())


def ascii_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def compact(value: str, aliases: dict[str, str], common: frozenset[str]) -> str:
    tokens = [aliases.get(token, token) for token in value.split()]
    useful = [token for token in tokens if token not in common]
    return " ".join(useful or tokens)


def script(value: str) -> str:
    scripts: set[str] = set()
    for char in value:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        current = next((s for s in (
            "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "KANNADA",
            "MALAYALAM", "ORIYA", "TAMIL", "TELUGU", "ARABIC", "CYRILLIC",
            "LATIN",
        ) if s in name), "OTHER")
        scripts.add(current)
        if len(scripts) > 1:
            return "MIXED"
    return next(iter(scripts), "NONE")


def transliterate_name(value: str, name_script: str) -> str:
    """Use the installed MIT-licensed Indic transliterator when applicable."""
    if name_script in {"LATIN", "NONE", "MIXED", "OTHER"}:
        return ascii_fold(value)
    try:
        from indic_transliteration import sanscript
        source = getattr(sanscript, name_script)
        return normalize(ascii_fold(sanscript.transliterate(value, source, sanscript.IAST)))
    except (ImportError, AttributeError, ValueError):
        return ascii_fold(value)


@dataclass(frozen=True)
class Record:
    entity_id: str
    raw_name: str
    raw_address: str
    country: str
    source: str
    name: str
    address: str
    folded_name: str
    folded_address: str
    compact_name: str
    compact_address: str
    name_script: str
    transliterated_name: str


def make_record(row: dict[str, str], source: str) -> Record:
    raw_name = row.get("business_name", "") or ""
    raw_address = row.get("business_address", "") or ""
    name = normalize(raw_name)
    address = normalize(raw_address)
    name_script = script(raw_name)
    return Record(
        entity_id=row["entity_id"], raw_name=raw_name,
        raw_address=raw_address, country=row.get("country", "") or "",
        source=source, name=name, address=address,
        folded_name=ascii_fold(name), folded_address=ascii_fold(address),
        compact_name=compact(name, NAME_ALIASES, COMMON_NAME),
        compact_address=compact(address, ADDRESS_ALIASES, COMMON_ADDRESS),
        name_script=name_script,
        transliterated_name=transliterate_name(name, name_script),
    )


def grams(value: str, size: int) -> set[str]:
    padded = f" {value} "
    return {padded[i:i + size] for i in range(max(0, len(padded) - size + 1))}


def overlap(left: set[str], right: set[str]) -> tuple[float, float, float]:
    common = len(left & right)
    union = len(left | right)
    return (
        common / union if union else 0.0,
        common / min(len(left), len(right)) if left and right else 0.0,
        float(common),
    )


def similarity_features(prefix: str, left: str, right: str) -> dict[str, float]:
    lt, rt = set(left.split()), set(right.split())
    jaccard, containment, common = overlap(lt, rt)
    char3, _, _ = overlap(grams(left, 3), grams(right, 3))
    char4, _, _ = overlap(grams(left, 4), grams(right, 4))
    minimum = min(len(left), len(right))
    maximum = max(len(left), len(right))
    both_present = bool(left and right)
    return {
        f"{prefix}_equal": float(both_present and left == right),
        f"{prefix}_missing_left": float(not left),
        f"{prefix}_missing_right": float(not right),
        f"{prefix}_missing_both": float(not left and not right),
        f"{prefix}_length_left": float(len(left)),
        f"{prefix}_length_right": float(len(right)),
        f"{prefix}_length_difference": float(maximum - minimum),
        f"{prefix}_length_ratio": minimum / maximum if maximum else 0.0,
        f"{prefix}_token_jaccard": jaccard,
        f"{prefix}_token_containment": containment,
        f"{prefix}_common_tokens": common,
        f"{prefix}_char3_jaccard": char3,
        f"{prefix}_char4_jaccard": char4,
        f"{prefix}_levenshtein": Levenshtein.normalized_similarity(left, right) if both_present else 0.0,
        f"{prefix}_jaro_winkler": JaroWinkler.normalized_similarity(left, right) if both_present else 0.0,
        f"{prefix}_token_sort": fuzz.token_sort_ratio(left, right) / 100 if both_present else 0.0,
        f"{prefix}_token_set": fuzz.token_set_ratio(left, right) / 100 if both_present else 0.0,
        f"{prefix}_prefix_token": float(bool(lt and rt and left.split()[0] == right.split()[0])),
    }


def pair_features(left: Record, right: Record, frequencies: dict[str, Counter[str]], retrieval_score: float) -> dict[str, float]:
    features: dict[str, float] = {
        "target_source3": float(right.source == "S3"),
        "country_equal": float(bool(left.country and right.country and left.country == right.country)),
        "name_raw_equal": float(bool(left.raw_name and left.raw_name == right.raw_name)),
        "address_raw_equal": float(bool(left.raw_address and left.raw_address == right.raw_address)),
        "retrieval_score": float(retrieval_score),
        "script_equal": float(left.name_script == right.name_script),
        "script_mixed": float("MIXED" in (left.name_script, right.name_script)),
        "script_target_nonlatin": float(right.name_script not in {"LATIN", "NONE"}),
        "name_nonascii_left": float(not left.raw_name.isascii()),
        "name_nonascii_right": float(not right.raw_name.isascii()),
        "name_transliteration_equal": float(bool(left.transliterated_name and left.transliterated_name == right.transliterated_name)),
        "name_transliteration_similarity": fuzz.ratio(left.transliterated_name, right.transliterated_name) / 100 if left.transliterated_name and right.transliterated_name else 0.0,
    }
    for prefix, first, second in (
        ("name", left.name, right.name),
        ("address", left.address, right.address),
        ("name_folded", left.folded_name, right.folded_name),
        ("address_folded", left.folded_address, right.folded_address),
        ("name_compact", left.compact_name, right.compact_name),
        ("address_compact", left.compact_address, right.compact_address),
    ):
        # The full suite on primary fields, equality and token overlap on alternate forms.
        if prefix in {"name", "address"}:
            features.update(similarity_features(prefix, first, second))
        else:
            jaccard, containment, count = overlap(set(first.split()), set(second.split()))
            features.update({
                f"{prefix}_equal": float(bool(first and second and first == second)),
                f"{prefix}_token_jaccard": jaccard,
                f"{prefix}_token_containment": containment,
                f"{prefix}_common_tokens": count,
            })
    left_numbers, right_numbers = set(NUMBER_RE.findall(left.address)), set(NUMBER_RE.findall(right.address))
    features["address_number_jaccard"], features["address_number_containment"], features["address_shared_numbers"] = overlap(left_numbers, right_numbers)
    features["address_number_conflict"] = float(bool(left_numbers and right_numbers and not left_numbers & right_numbers))
    for kind, pattern in (("email", EMAIL_RE), ("domain", DOMAIN_RE), ("phone", PHONE_RE)):
        left_values = set(pattern.findall(left.raw_name + " " + left.raw_address))
        right_values = set(pattern.findall(right.raw_name + " " + right.raw_address))
        features[f"{kind}_present_left"] = float(bool(left_values))
        features[f"{kind}_present_right"] = float(bool(right_values))
        features[f"{kind}_exact_overlap"] = float(bool(left_values & right_values))
        features[f"{kind}_conflict"] = float(bool(left_values and right_values and not left_values & right_values))
    for field, lvalue, rvalue in (
        ("name", left.name, right.name), ("address", left.address, right.address),
    ):
        counts = frequencies[field]
        features[f"{field}_left_log_frequency"] = math.log1p(counts[lvalue]) if lvalue else 0.0
        features[f"{field}_right_log_frequency"] = math.log1p(counts[rvalue]) if rvalue else 0.0
        features[f"{field}_min_token_frequency"] = min(
            (math.log1p(frequencies[f"{field}_token"][token]) for token in set(rvalue.split())),
            default=0.0,
        )
    return features
