"""Unicode-safe cleanup and ICU-based Latin transliteration helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from collections.abc import Sequence


def _unicode_mark_class() -> str:
    """Build compact regex ranges for all Unicode combining marks."""
    codepoints = [
        codepoint for codepoint in range(0x110000)
        if unicodedata.category(chr(codepoint)).startswith("M")
    ]
    ranges: list[tuple[int, int]] = []
    start = previous = codepoints[0]
    for codepoint in codepoints[1:]:
        if codepoint == previous + 1:
            previous = codepoint
            continue
        ranges.append((start, previous))
        start = previous = codepoint
    ranges.append((start, previous))

    def escape(codepoint: int) -> str:
        return f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"

    return "".join(
        escape(start) + ("-" + escape(end) if end > start else "")
        for start, end in ranges
    )


_PUNCTUATION_RE = re.compile(r"[^\w\s" + _unicode_mark_class() + r"]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_REPLACEMENTS = (
    (re.compile(r"&"), " and "),
    (re.compile(r"@"), " at "),
    (re.compile(r"\bc/o\b"), " care of "),
    (re.compile(r"\bs/o\b"), " son of "),
    (re.compile(r"\bd/o\b"), " daughter of "),
    (re.compile(r"\bw/o\b"), " wife of "),
    (re.compile(r"\bopp\.?"), " opposite "),
    (re.compile(r"\bprivate\b"), " pvt "),
    (re.compile(r"\blimited\b"), " ltd "),
    (re.compile(r"\bcorporation\b"), " corp "),
    (re.compile(r"\bcompany\b"), " co "),
    (re.compile(r"\bincorporated\b"), " inc "),
    (re.compile(r"\broad\b"), " rd "),
    (re.compile(r"\bstreet\b"), " st "),
    (re.compile(r"\bavenue\b"), " ave "),
    (re.compile(r"\blane\b"), " ln "),
    (re.compile(r"\bfloor\b"), " fl "),
    (re.compile(r"\bapartment\b"), " apt "),
    (re.compile(r"\bbuilding\b"), " bldg "),
)


def clean_text(value: object) -> str:
    """Normalize text while preserving Unicode letters and combining marks."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    for pattern, replacement in _REPLACEMENTS[:7]:
        text = pattern.sub(replacement, text)
    text = _PUNCTUATION_RE.sub(" ", text)
    for pattern, replacement in _REPLACEMENTS[7:]:
        text = pattern.sub(replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def latin_transliterate(values: Sequence[str], executable: str = "uconv") -> list[str]:
    """Transliterate a batch to ASCII using ICU, preserving row alignment."""
    binary = shutil.which(executable)
    if not binary:
        raise RuntimeError(
            "ICU's `uconv` executable is required (for example, install the `icu-devtools` package)."
        )
    # Input columns are names/addresses: embedded line breaks are whitespace, not record boundaries.
    safe_values = [str(value or "").replace("\r", " ").replace("\n", " ") for value in values]
    if not safe_values:
        return []
    process = subprocess.run(
        [binary, "-x", "Any-Latin; Latin-ASCII"],
        input="\n".join(safe_values) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    result = process.stdout.splitlines()
    if len(result) != len(safe_values):
        raise RuntimeError(
            f"ICU returned {len(result)} lines for {len(safe_values)} input values"
        )
    normalized: list[str] = []
    for value in result:
        value = re.sub(r"\bpra['’]?\s*iveta\b", "private", value, flags=re.IGNORECASE)
        value = re.sub(r"\blimiteda\b", "limited", value, flags=re.IGNORECASE)
        value = re.sub(r"\bela['’]?\s*elapi\b", "llp", value, flags=re.IGNORECASE)
        value = re.sub(r"\bela['’]?\s*elasi\b", "llc", value, flags=re.IGNORECASE)
        normalized.append(" ".join(value.split()))
    return normalized
