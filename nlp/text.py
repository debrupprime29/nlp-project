"""Dependency-free Unicode text, token-span, and protection contracts.

All offsets returned here are half-open Python string offsets into the NFC
``processing_text`` returned by :func:`normalize_processing_text`.  They are
Unicode code-point offsets, not UTF-8 byte offsets.  An exact raw query must be
stored separately when byte-for-byte preservation matters.

The protection policy is deliberately conservative.  It protects values whose
accidental alteration is especially likely to change an information need:
numbers and dates, negation, URLs and email addresses, acronyms, and likely
proper names.  It does *not* protect question words such as ``kyu`` or ``keno``;
later query-view generation must be able to transliterate those words while the
exact original query remains available as a separate view.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


MAX_TEXT_CHARACTERS = 131_072
JOINERS = {"\u200c", "\u200d"}


class TextPreparationError(ValueError):
    """Raised when text or a declared span violates the shared contract."""


@dataclass(frozen=True)
class TokenSpan:
    """One script-aware token in NFC processing text."""

    text: str
    start: int
    end: int
    script: str


@dataclass(frozen=True)
class ProtectedSpan:
    """A half-open processing-text span that rewriting must preserve exactly."""

    start: int
    end: int
    reasons: tuple[str, ...]


def normalize_processing_text(text: str) -> str:
    """Return bounded NFC processing text without changing other characters."""

    if not isinstance(text, str):
        raise TextPreparationError("text must be a string")
    if "\x00" in text:
        raise TextPreparationError("text contains a NUL byte")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise TextPreparationError(
            f"text exceeds the {MAX_TEXT_CHARACTERS}-character safety limit"
        )
    return unicodedata.normalize("NFC", text)


def script_bucket(character: str) -> str:
    """Return the supported script bucket for one Unicode character."""

    if not isinstance(character, str) or len(character) != 1:
        raise TextPreparationError("script_bucket expects one character")
    codepoint = ord(character)
    if 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF:
        return "devanagari"
    if 0x0980 <= codepoint <= 0x09FF:
        return "bengali"
    category = unicodedata.category(character)
    if category.startswith("N"):
        return "numeric"
    name = unicodedata.name(character, "")
    if "LATIN" in name:
        return "latin"
    if category.startswith("L"):
        return "other_letter"
    return "other"


def _token_script(characters: Sequence[str]) -> str:
    scripts = {
        script_bucket(character)
        for character in characters
        if unicodedata.category(character).startswith(("L", "N"))
    }
    if not scripts:
        return "other"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "mixed"


def tokenize_with_spans(text: str) -> tuple[TokenSpan, ...]:
    """Tokenize NFC text while preserving Indic marks and exact offsets.

    Script transitions create boundaries even without whitespace.  Combining
    marks and ZWJ/ZWNJ remain attached to the preceding token, which avoids
    damaging Indic orthography.
    """

    processing_text = normalize_processing_text(text)
    spans: list[TokenSpan] = []
    token_start: int | None = None
    token_characters: list[str] = []
    active_script: str | None = None

    def finish(end: int) -> None:
        nonlocal token_start, token_characters, active_script
        if token_start is not None and token_characters:
            token_text = "".join(token_characters)
            spans.append(
                TokenSpan(
                    text=token_text,
                    start=token_start,
                    end=end,
                    script=_token_script(token_characters),
                )
            )
        token_start = None
        token_characters = []
        active_script = None

    for index, character in enumerate(processing_text):
        category = unicodedata.category(character)
        if category.startswith(("L", "N")):
            current_script = script_bucket(character)
            if (
                token_start is not None
                and active_script is not None
                and current_script != active_script
            ):
                finish(index)
            if token_start is None:
                token_start = index
            token_characters.append(character)
            active_script = current_script
            continue
        if category.startswith("M") and token_start is not None:
            token_characters.append(character)
            continue
        if character in JOINERS and token_start is not None:
            token_characters.append(character)
            continue
        finish(index)
    finish(len(processing_text))
    return tuple(spans)


_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
_EMAIL_PATTERN = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
_DATE_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,4}[-/.]\d{1,2}(?:[-/.]\d{1,4})?)(?!\w)"
)
_CONTRACTION_PATTERN = re.compile(
    r"(?i)(?<!\w)(?:[a-z]+n['’]t|cannot)(?!\w)"
)

_NEGATIONS = frozenset(
    {
        # English
        "no",
        "not",
        "never",
        "none",
        "neither",
        "without",
        # Romanized Hindi/Bengali alternatives.  Script is not treated as a
        # language decision; these are protected solely because they are common
        # negation forms.
        "na",
        "nahi",
        "nahin",
        "mat",
        "bina",
        "noy",
        # Hindi
        "न",
        "नहीं",
        "नहि",
        "मत",
        "बिना",
        # Bengali
        "না",
        "নয়",
        "নয়",
        "নেই",
        "বিনা",
    }
)


def _coerce_explicit_span(
    value: Any, processing_text: str, position: int
) -> ProtectedSpan:
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
        raw_reasons = value.get("reasons", value.get("reason", "explicit"))
        expected_text = value.get("text")
    elif isinstance(value, (tuple, list)) and len(value) in {2, 3}:
        start, end = value[0], value[1]
        raw_reasons = value[2] if len(value) == 3 else "explicit"
        expected_text = None
    else:
        raise TextPreparationError(
            f"explicit span {position} must be an object or a 2/3-item sequence"
        )
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TextPreparationError(f"explicit span {position} offsets must be integers")
    if start < 0 or end <= start or end > len(processing_text):
        raise TextPreparationError(
            f"explicit span {position} is outside NFC processing text"
        )
    if expected_text is not None and str(expected_text) != processing_text[start:end]:
        raise TextPreparationError(
            f"explicit span {position} text does not match NFC processing text"
        )
    if isinstance(raw_reasons, str):
        reasons = (raw_reasons.strip() or "explicit",)
    elif isinstance(raw_reasons, (tuple, list, set, frozenset)):
        reasons = tuple(sorted({str(reason).strip() for reason in raw_reasons if str(reason).strip()}))
        if not reasons:
            reasons = ("explicit",)
    else:
        raise TextPreparationError(f"explicit span {position} reasons are invalid")
    return ProtectedSpan(start, end, reasons)


def _merge_protected_spans(spans: Iterable[ProtectedSpan]) -> tuple[ProtectedSpan, ...]:
    ordered = sorted(spans, key=lambda item: (item.start, item.end, item.reasons))
    merged: list[ProtectedSpan] = []
    for span in ordered:
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = ProtectedSpan(
            start=previous.start,
            end=max(previous.end, span.end),
            reasons=tuple(sorted(set(previous.reasons) | set(span.reasons))),
        )
    return tuple(merged)


def protection_spans(
    text: str, explicit_spans: Sequence[Any] = ()
) -> tuple[ProtectedSpan, ...]:
    """Return conservative rewrite protections for NFC processing text.

    Explicit offsets are interpreted against NFC text.  If callers derived
    offsets from non-NFC raw input, they must normalize or remap them first.
    """

    processing_text = normalize_processing_text(text)
    if explicit_spans and processing_text != text:
        raise TextPreparationError(
            "explicit spans require input that is already NFC-normalized"
        )
    spans: list[ProtectedSpan] = []
    spans.extend(
        _coerce_explicit_span(value, processing_text, position)
        for position, value in enumerate(explicit_spans)
    )
    for match in _URL_PATTERN.finditer(processing_text):
        spans.append(ProtectedSpan(match.start(), match.end(), ("url",)))
    for match in _EMAIL_PATTERN.finditer(processing_text):
        spans.append(ProtectedSpan(match.start(), match.end(), ("email",)))
    for match in _DATE_PATTERN.finditer(processing_text):
        spans.append(ProtectedSpan(match.start(), match.end(), ("date",)))
    for match in _CONTRACTION_PATTERN.finditer(processing_text):
        spans.append(ProtectedSpan(match.start(), match.end(), ("negation",)))

    for token in tokenize_with_spans(processing_text):
        reasons: set[str] = set()
        folded = token.text.casefold()
        if any(character.isdigit() for character in token.text):
            reasons.add("number")
        if folded in _NEGATIONS:
            reasons.add("negation")
        cased_letters = [character for character in token.text if character.isalpha()]
        if (
            len(cased_letters) >= 2
            and all(character.isupper() for character in cased_letters)
        ):
            reasons.add("acronym")
        elif token.text.istitle() and any(character.isalpha() for character in token.text):
            reasons.add("name_heuristic")
        if reasons:
            spans.append(
                ProtectedSpan(token.start, token.end, tuple(sorted(reasons)))
            )
    return _merge_protected_spans(spans)


__all__ = [
    "MAX_TEXT_CHARACTERS",
    "ProtectedSpan",
    "TextPreparationError",
    "TokenSpan",
    "normalize_processing_text",
    "protection_spans",
    "script_bucket",
    "tokenize_with_spans",
]
