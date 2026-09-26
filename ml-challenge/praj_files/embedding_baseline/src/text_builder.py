from __future__ import annotations

import re
from collections.abc import Iterable

import pyarrow as pa


_SPACE_RE = re.compile(r"\s+")


def clean_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return _SPACE_RE.sub(" ", text)


def build_text(name: object, address: object, country: object, representation: str) -> str:
    values = {
        "name_only": [name],
        "name_address": [name, address],
        "name_address_country": [name, address, country],
    }
    if representation not in values:
        raise ValueError(f"Unknown text representation: {representation}")
    return " | ".join(part for part in map(clean_value, values[representation]) if part)


def build_texts(table: pa.Table, representation: str) -> list[str]:
    cols = table.to_pydict()
    return [
        build_text(name, address, country, representation)
        for name, address, country in zip(
            cols["business_name"], cols["business_address"], cols["country"]
        )
    ]


def deduplicate_texts(texts: Iterable[str]) -> tuple[list[str], list[int]]:
    unique: list[str] = []
    indexes: dict[str, int] = {}
    inverse: list[int] = []
    for text in texts:
        index = indexes.get(text)
        if index is None:
            index = len(unique)
            indexes[text] = index
            unique.append(text)
        inverse.append(index)
    return unique, inverse

