"""Indic transliteration and deterministic text normalization."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


_SCRIPT_RANGES = (
    ("\u0900", "\u097f", "DEVANAGARI"),
    ("\u0980", "\u09ff", "BENGALI"),
    ("\u0a00", "\u0a7f", "GURMUKHI"),
    ("\u0a80", "\u0aff", "GUJARATI"),
    ("\u0b00", "\u0b7f", "ORIYA"),
    ("\u0b80", "\u0bff", "TAMIL"),
    ("\u0c00", "\u0c7f", "TELUGU"),
    ("\u0c80", "\u0cff", "KANNADA"),
    ("\u0d00", "\u0d7f", "MALAYALAM"),
)
_INDIC_PATTERN = re.compile(
    "[" + "".join(f"{start}-{end}" for start, end, _ in _SCRIPT_RANGES) + "]"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_INDICXLIT_LANGUAGES = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "ORIYA": "or",
    "TAMIL": "ta",
    "TELUGU": "te",
    "KANNADA": "kn",
    "MALAYALAM": "ml",
}


@dataclass
class TextNormalizer:
    """Merge fields, transliterate Indic runs with sanscript, and normalize."""

    _transliterate: Callable[[str, str, str], str] | None = field(
        default=None, init=False, repr=False
    )
    _schemes: dict[str, str] | None = field(default=None, init=False, repr=False)
    _latin_scheme: str | None = field(default=None, init=False, repr=False)

    def _load_transliterator(self) -> None:
        if self._transliterate is not None:
            return
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
        except ImportError as exc:
            raise RuntimeError(
                "Indic text was found, but indic-transliteration is not installed. "
                "Install the dependencies from requirements.txt."
            ) from exc

        self._transliterate = transliterate
        self._schemes = {
            name: getattr(sanscript, name)
            for _, _, name in _SCRIPT_RANGES
        }
        self._latin_scheme = sanscript.ITRANS

    @staticmethod
    def merge_text(name: Any, address: Any) -> str:
        """Join nonempty name and address values with exactly one space."""
        parts = []
        for value in (name, address):
            if value is not None and not pd.isna(value):
                value_text = str(value).strip()
                if value_text:
                    parts.append(value_text)
        return " ".join(parts)

    def transliterate_indic(self, text: str) -> str:
        """Transliterate each native-script run while leaving Latin text intact."""
        if not _INDIC_PATTERN.search(text):
            return text
        self._load_transliterator()
        assert self._transliterate is not None
        assert self._schemes is not None
        assert self._latin_scheme is not None

        result = text
        for start, end, scheme_name in _SCRIPT_RANGES:
            run_pattern = re.compile(f"[{start}-{end}]+")
            source_scheme = self._schemes[scheme_name]
            result = run_pattern.sub(
                lambda match: self._transliterate(
                    match.group(0), source_scheme, self._latin_scheme
                ),
                result,
            )
        return result

    def normalize(self, text: Any) -> str:
        """Return lowercase, diacritic-free, punctuation-free normalized text."""
        if text is None or pd.isna(text):
            return ""
        transliterated = self.transliterate_indic(str(text))
        decomposed = unicodedata.normalize("NFKD", transliterated.lower())
        without_marks = "".join(
            character for character in decomposed
            if not unicodedata.combining(character)
        )
        alphanumeric = "".join(
            character if character.isalnum() or character.isspace() else " "
            for character in without_marks
        )
        return _WHITESPACE_PATTERN.sub(" ", alphanumeric).strip()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add ``full_text`` and ``normalized_text`` to an entity frame."""
        missing = {"business_name", "business_address"} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing text columns: {sorted(missing)}")
        output = frame.copy()
        output["full_text"] = [
            self.merge_text(name, address)
            for name, address in zip(
                output["business_name"], output["business_address"]
            )
        ]
        output["normalized_text"] = self.normalize_many(output["full_text"].tolist())
        return output

    def normalize_many(self, texts: list[str]) -> list[str]:
        transliterated = [self.transliterate_indic(text) for text in texts]
        return [self._normalize_romanized(text) for text in transliterated]

    @staticmethod
    def _normalize_romanized(text: Any) -> str:
        if text is None or pd.isna(text):
            return ""
        decomposed = unicodedata.normalize("NFKD", str(text).lower())
        without_marks = "".join(
            character for character in decomposed
            if not unicodedata.combining(character)
        )
        alphanumeric = "".join(
            character if character.isalnum() or character.isspace() else " "
            for character in without_marks
        )
        return _WHITESPACE_PATTERN.sub(" ", alphanumeric).strip()


@dataclass
class IndicXlitTextNormalizer(TextNormalizer):
    """AI4Bharat IndicXlit normalizer using its local Indic-to-English model."""

    model_dir: Path = Path("praj_files/hybrid_entity_resolution/indicxlit_model")
    beam: int = 4
    batch_size: int = 256
    _word_cache: dict[tuple[str, str], str] = field(
        default_factory=dict, init=False, repr=False
    )

    def _model_paths(self) -> tuple[Path, Path]:
        model_dir = self.model_dir.resolve()
        data_dir = model_dir / "corpus-bin"
        checkpoint = model_dir / "transformer" / "indicxlit.pt"
        required = [data_dir, checkpoint, model_dir / "lang_list.txt"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "IndicXlit model files are missing: " + ", ".join(missing)
                + ". Download and extract the AI4Bharat Indic-to-English release; "
                "see README.md."
            )
        return data_dir, checkpoint

    def _transliterate_words(self, language: str, words: list[str]) -> dict[str, str]:
        unique_words = list(dict.fromkeys(words))
        uncached = [
            word for word in unique_words if (language, word) not in self._word_cache
        ]
        if not uncached:
            return {
                word: self._word_cache[(language, word)] for word in unique_words
            }

        data_dir, checkpoint = self._model_paths()
        lang_pairs = (
            "as-en,bn-en,brx-en,gom-en,gu-en,hi-en,kn-en,ks-en,mai-en,ml-en,"
            "mni-en,mr-en,ne-en,or-en,pa-en,sa-en,sd-en,si-en,ta-en,te-en,ur-en"
        )
        command = [
            "fairseq-interactive",
            str(data_dir),
            "--path",
            str(checkpoint),
            "--task",
            "translation_multi_simple_epoch",
            "--beam",
            str(self.beam),
            "--nbest",
            "1",
            "--source-lang",
            language,
            "--target-lang",
            "en",
            "--encoder-langtok",
            "src",
            "--lang-dict",
            str(self.model_dir.resolve() / "lang_list.txt"),
            "--lang-pairs",
            lang_pairs,
            "--batch-size",
            str(self.batch_size),
            "--buffer-size",
            str(max(self.batch_size, len(uncached))),
        ]
        encoded_lines = [
            " ".join(list(word.lower()))
            for word in uncached
        ]
        try:
            result = subprocess.run(
                command,
                input="\n".join(encoded_lines) + "\n",
                text=True,
                capture_output=True,
                cwd=self.model_dir.resolve(),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "IndicXlit inference needs the fairseq-interactive command. "
                "Install the IndicXlit/Fairseq inference dependencies in this environment."
            ) from exc
        if result.returncode:
            raise RuntimeError(
                "IndicXlit inference failed for language "
                f"{language}: {result.stderr[-4000:]}"
            )

        hypotheses: dict[int, str] = {}
        for line in result.stdout.splitlines():
            if not line.startswith("H-"):
                continue
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                continue
            try:
                row_index = int(parts[0][2:])
            except ValueError:
                continue
            # The upstream IndicXlit inference removes character-spaced output.
            hypotheses[row_index] = "".join(parts[2].split())

        if len(hypotheses) != len(uncached):
            raise RuntimeError(
                f"IndicXlit returned {len(hypotheses)} of {len(uncached)} "
                f"hypotheses for {language}."
            )
        for index, word in enumerate(uncached):
            self._word_cache[(language, word)] = hypotheses[index]
        return {word: self._word_cache[(language, word)] for word in unique_words}

    def normalize_many(self, texts: list[str]) -> list[str]:
        output = list(texts)
        for start, end, script_name in _SCRIPT_RANGES:
            language = _INDICXLIT_LANGUAGES[script_name]
            pattern = re.compile(f"[{start}-{end}]+")
            words = [
                match.group(0)
                for text in output
                for match in pattern.finditer(text)
            ]
            if not words:
                continue
            translated = self._transliterate_words(language, words)
            output = [
                pattern.sub(
                    lambda match: translated[match.group(0)],
                    text,
                )
                for text in output
            ]
        return [self._normalize_romanized(text) for text in output]

    def normalize(self, text: Any) -> str:
        """Normalize one value (bulk transforms use batched inference)."""
        if text is None or pd.isna(text):
            return ""
        return self.normalize_many([str(text)])[0]
