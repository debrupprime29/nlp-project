"""Protected, bounded query hypotheses with explicit transliteration resources.

The exact raw original is always a separate view.  Rewrite offsets refer to NFC
processing text, never raw bytes.  Native hypotheses contain changed text only;
uncertain tokens and protected spans survive unchanged.  Transliteration scores
are resource frequencies or ranks, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from data.make_splits import ALL_SPLITS, _atomic_write_jsonl
from data.prepare_corpus import CorpusPreparationError, canonical_language, iter_jsonl
from nlp.language_id import (
    LABELS,
    LanguageIdError,
    ScriptEvidenceBaseline,
    TokenLanguageIdentifier,
    TokenPrediction,
)
from nlp.text import (
    JOINERS,
    ProtectedSpan,
    TextPreparationError,
    TokenSpan,
    normalize_processing_text,
    protection_spans,
    script_bucket,
    tokenize_with_spans,
)
from retrieval.char_tfidf import NORMALIZATION_VERSION, is_roman_script_view


SCHEMA_VERSION = 1
MAX_QUERY_CHARACTERS = 4096
TARGET_SCRIPTS = {"hi": "devanagari", "bn": "bengali"}
LEXICON_CANDIDATE_POLICY = "safe_target_score_ordered_top64_v1"


class QueryViewError(ValueError):
    """Invalid query spans, model outputs, resources, or view configuration."""


ViewError = QueryViewError


@dataclass(frozen=True)
class TransliterationCandidate:
    text: str
    raw_score: float
    backend: str
    revision: str


@dataclass(frozen=True)
class RewriteEdit:
    start: int
    end: int
    original: str
    replacement: str
    language: str
    backend: str
    revision: str
    raw_score: float


@dataclass(frozen=True)
class QueryView:
    kind: str
    text: str
    edits: tuple[RewriteEdit, ...] = ()
    raw_score: float = 0.0


@dataclass(frozen=True)
class QueryViews:
    original_text: str
    processing_text: str
    tokens: tuple[TokenPrediction, ...]
    protections: tuple[ProtectedSpan, ...]
    original: QueryView
    native_views: tuple[QueryView, ...]
    roman_view: QueryView | None
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ViewConfig:
    max_native_views: int = 2
    max_candidates_per_token: int = 2
    max_backend_calls: int = 64
    min_probability: float = 0.60

    def validate(self) -> None:
        for name, lower, upper in (
            ("max_native_views", 0, 2),
            ("max_candidates_per_token", 1, 8),
            ("max_backend_calls", 0, 64),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise QueryViewError(f"{name} must be an integer in [{lower}, {upper}]")
        if not _finite(self.min_probability) or not 0 <= self.min_probability <= 1:
            raise QueryViewError("min_probability must be finite and in [0, 1]")


class Transliterator(Protocol):
    def native_candidates(self, token: str, target_language: str, limit: int) -> Sequence[TransliterationCandidate]: ...
    def roman_candidates(self, token: str, source_language: str, limit: int) -> Sequence[TransliterationCandidate]: ...


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _add_scores(left: float, right: float) -> float:
    total = left + right
    if not _finite(total):
        raise QueryViewError("combined transliteration raw_score must remain finite")
    return total


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QueryViewError("resource metadata must be finite, JSON-serializable data") from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _view_key(text: str) -> str:
    return " ".join(normalize_processing_text(text).split())


def _letter_token(text: Any, script: str) -> bool:
    if not isinstance(text, str) or not text or len(text) > MAX_QUERY_CHARACTERS:
        return False
    try:
        normalized = normalize_processing_text(text)
    except TextPreparationError:
        return False
    if normalized != text:
        return False
    spans = tokenize_with_spans(text)
    if len(spans) != 1 or spans[0].start != 0 or spans[0].end != len(text):
        return False
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    return bool(letters) and all(script_bucket(character) == script for character in letters) and all(
        (
            unicodedata.category(character).startswith(("L", "M")) or character in JOINERS
        )
        and not (
            unicodedata.category(character).startswith("M")
            and script_bucket(character) in {"latin", "devanagari", "bengali"}
            and script_bucket(character) != script
        )
        for character in text
    )


class LexiconTransliterator:
    """Attested, supplied word pairs; no builtin lexicon or implicit download.

    Each record needs language (hi/bn), native, romanized, source, revision,
    and optional positive count.  Frequencies only order alternatives; they do
    not express a calibrated posterior over query meaning.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for position, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise QueryViewError(f"lexicon record {position} must be an object")
            try:
                language = canonical_language(str(record.get("language", "")))
            except CorpusPreparationError as exc:
                raise QueryViewError(str(exc)) from exc
            if language not in TARGET_SCRIPTS:
                raise QueryViewError("transliteration lexicon supports hi and bn only")
            native, romanized = record.get("native"), record.get("romanized")
            if not _letter_token(native, TARGET_SCRIPTS[language]) or not _letter_token(romanized, "latin"):
                raise QueryViewError("lexicon native/romanized values must be complete tokens in the declared scripts")
            for field in ("source", "revision"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise QueryViewError(f"lexicon {field} must be a nonempty string")
            count = record.get("count", 1)
            if not _finite(count) or count <= 0:
                raise QueryViewError("lexicon count must be finite and positive")
            normalized.append({"language": language, "native": native, "romanized": romanized, "source": record["source"].strip(), "revision": record["revision"].strip(), "count": count})
        self._records = tuple(json.loads(value) for value in sorted({_canonical(record).decode("utf-8") for record in normalized}))
        self._manifest = {
            "backend": "supplied_lexicon", "schema_version": SCHEMA_VERSION,
            "record_count": len(self._records), "resource_sha256": _fingerprint(self._records),
            "sources": sorted({record["source"] for record in self._records}),
            "revisions": sorted({record["revision"] for record in self._records}),
            "score_type": "log_attestation_count_uncalibrated",
            "candidate_policy": LEXICON_CANDIDATE_POLICY,
            "max_candidate_response": 64,
        }
        # Build both directions once. Full resources have many thousands of
        # pairs, and corpus Romanization must not scan them all for every word.
        # Records retain their canonical order and provenance; candidate
        # validation, deduplication, ordering, and limits remain shared below.
        native_buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        roman_buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in self._records:
            native_buckets.setdefault((record["language"], record["romanized"].casefold()), []).append(record)
            roman_buckets.setdefault((record["language"], record["native"].casefold()), []).append(record)
        self._native_lookup = {key: tuple(values) for key, values in native_buckets.items()}
        self._roman_lookup = {key: tuple(values) for key, values in roman_buckets.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> LexiconTransliterator:
        try:
            return cls(list(iter_jsonl(Path(path))))
        except CorpusPreparationError as exc:
            raise QueryViewError(str(exc)) from exc

    @property
    def manifest(self) -> Mapping[str, Any]:
        return json.loads(_canonical(self._manifest))

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.manifest

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._manifest)

    def _candidates(self, token: str, language: str, limit: int, *, native: bool) -> tuple[TransliterationCandidate, ...]:
        if language not in TARGET_SCRIPTS:
            raise QueryViewError("transliteration language must be hi or bn")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise QueryViewError("candidate limit must be an integer in [1, 8]")
        target_field = "native" if native else "romanized"
        key = normalize_processing_text(token).casefold()
        lookup = self._native_lookup if native else self._roman_lookup
        script = TARGET_SCRIPTS[language] if native else "latin"
        # Supplied resources can legitimately have more than 64 spellings for
        # a word or include targets that query rewriting may never introduce
        # (notably negation). Filter those targets, select each spelling's best
        # provenance, then bound the ranked resource response. The generic
        # validator stays fail-closed for arbitrary/upstream model responses.
        distinct: dict[str, TransliterationCandidate] = {}
        for record in lookup.get((language, key), ()):
            target = record[target_field]
            if not _letter_token(target, script) or any("negation" in span.reasons for span in protection_spans(target)):
                continue
            candidate = TransliterationCandidate(target, math.log(record["count"]), "lexicon:" + record["source"], record["revision"])
            previous = distinct.get(target)
            if previous is None or (-candidate.raw_score, candidate.backend, candidate.revision) < (-previous.raw_score, previous.backend, previous.revision):
                distinct[target] = candidate
        candidates = sorted(distinct.values(), key=lambda candidate: (-candidate.raw_score, candidate.text, candidate.backend, candidate.revision))[:64]
        return _validated_candidates(candidates, script=script, limit=limit)

    def native_candidates(self, token: str, target_language: str, limit: int) -> tuple[TransliterationCandidate, ...]:
        return self._candidates(token, target_language, limit, native=True)

    def roman_candidates(self, token: str, source_language: str, limit: int) -> tuple[TransliterationCandidate, ...]:
        return self._candidates(token, source_language, limit, native=False)


class IndicXlitTransliterator:
    """Lazy optional AI4Bharat adapter with explicit model-download consent.

    The upstream package can fetch pretrained assets while constructing its
    engine, so construction is forbidden unless allow_model_download=True.
    The adapter records package/default-weight provenance honestly; its word
    ranks are not beam log probabilities and are never labeled calibrated.
    """

    def __init__(self, *, allow_model_download: bool = False, beam_width: int = 4) -> None:
        if allow_model_download is not True:
            raise QueryViewError("IndicXlit requires explicit allow_model_download=True because upstream may download model assets")
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 2 <= beam_width <= 64:
            raise QueryViewError("IndicXlit beam_width must be an integer in [2, 64]")
        self.beam_width = beam_width
        self._engines: dict[str, Any] = {}
        try:
            version = importlib.metadata.version("ai4bharat-transliteration")
        except importlib.metadata.PackageNotFoundError:
            version = "not_installed"
        self._manifest = {
            "backend": "IndicXlit", "package": "ai4bharat-transliteration", "package_version": version,
            "checkpoint_revision": "upstream_package_default_unpinned", "beam_width": beam_width,
            "rescore": False, "allow_model_download": True, "score_type": "negative_rank_uncalibrated",
            "source": "https://github.com/AI4Bharat/IndicXlit",
        }

    @property
    def manifest(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.manifest

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._manifest)

    def _engine(self, direction: str, language: str) -> Any:
        key = language if direction == "native" else "indic"
        if key not in self._engines:
            try:
                from ai4bharat.transliteration import XlitEngine
            except ImportError as exc:
                raise QueryViewError("IndicXlit is optional; install ai4bharat-transliteration to use this backend") from exc
            try:
                if direction == "native":
                    self._engines[key] = XlitEngine(language, src_script_type="roman", beam_width=self.beam_width, rescore=False)
                else:
                    self._engines[key] = XlitEngine(src_script_type="indic", beam_width=self.beam_width, rescore=False)
            except Exception as exc:
                raise QueryViewError(f"could not initialize IndicXlit {direction} engine: {exc}") from exc
        return self._engines[key]

    def _candidates(self, token: str, language: str, limit: int, *, native: bool) -> tuple[TransliterationCandidate, ...]:
        if language not in TARGET_SCRIPTS:
            raise QueryViewError("IndicXlit supports project languages hi and bn only")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise QueryViewError("candidate limit must be an integer in [1, 8]")
        engine = self._engine("native" if native else "roman", language)
        try:
            response = engine.translit_word(token, lang_code=language, topk=limit)
        except Exception as exc:
            raise QueryViewError(f"IndicXlit word inference failed: {exc}") from exc
        if isinstance(response, Mapping):
            response = response.get(language)
        if not isinstance(response, (list, tuple)):
            raise QueryViewError("IndicXlit returned an invalid candidate response")
        revision = f"{self._manifest['package_version']}:upstream_default_unpinned"
        candidates = [TransliterationCandidate(text, -float(rank), "IndicXlit", revision) for rank, text in enumerate(response)]
        return _validated_candidates(candidates, script=TARGET_SCRIPTS[language] if native else "latin", limit=limit)

    def native_candidates(self, token: str, target_language: str, limit: int) -> tuple[TransliterationCandidate, ...]:
        return self._candidates(token, target_language, limit, native=True)

    def roman_candidates(self, token: str, source_language: str, limit: int) -> tuple[TransliterationCandidate, ...]:
        return self._candidates(token, source_language, limit, native=False)


def _validated_candidates(response: Any, *, script: str, limit: int) -> tuple[TransliterationCandidate, ...]:
    if not isinstance(response, (tuple, list)) or len(response) > 64:
        raise QueryViewError("transliterator response must be a bounded list/tuple of candidates")
    unique: dict[str, TransliterationCandidate] = {}
    for candidate in response:
        if not isinstance(candidate, TransliterationCandidate):
            raise QueryViewError("transliterator entries must be TransliterationCandidate records")
        if not _letter_token(candidate.text, script):
            raise QueryViewError("transliteration candidate must be one NFC word in the requested target script")
        if any("negation" in span.reasons for span in protection_spans(candidate.text)):
            raise QueryViewError("transliteration candidate introduces a protected negation token")
        if not _finite(candidate.raw_score):
            raise QueryViewError("transliteration candidate raw_score must be finite")
        if any(not isinstance(value, str) or not value.strip() for value in (candidate.backend, candidate.revision)):
            raise QueryViewError("transliteration candidates require backend and revision provenance")
        previous = unique.get(candidate.text)
        if previous is None or (-candidate.raw_score, candidate.backend, candidate.revision) < (-previous.raw_score, previous.backend, previous.revision):
            unique[candidate.text] = candidate
    return tuple(sorted(unique.values(), key=lambda candidate: (-candidate.raw_score, candidate.text, candidate.backend, candidate.revision))[:limit])


def _validated_predictions(text: str, identifier: Any) -> tuple[TokenPrediction, ...]:
    try:
        predictions = tuple(identifier.predict(text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise QueryViewError(f"language identifier prediction failed: {exc}") from exc
    expected = tokenize_with_spans(text)
    if len(predictions) != len(expected):
        raise QueryViewError("language identifier must predict every shared token exactly once")
    for expected_span, prediction in zip(expected, predictions):
        if not isinstance(prediction, TokenPrediction) or prediction.span != expected_span:
            raise QueryViewError("language identifier spans must match the shared NFC tokenization exactly")
        probabilities = prediction.probabilities
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(LABELS):
            raise QueryViewError("language probabilities must contain en, hi, bn, other")
        if any(not _finite(value) or not 0 <= value <= 1 for value in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise QueryViewError("language probabilities must be finite, nonnegative, and sum to one")
        if prediction.top_label not in LABELS or prediction.decision_label not in LABELS or not isinstance(prediction.uncertain, bool):
            raise QueryViewError("invalid language decision labels or uncertainty flag")
        if probabilities[prediction.top_label] < max(probabilities.values()) - 1e-12:
            raise QueryViewError("language top_label conflicts with its probabilities")
        if not _finite(prediction.entropy) or prediction.entropy < 0:
            raise QueryViewError("language entropy must be finite and nonnegative")
        expected_entropy = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
        if not math.isclose(prediction.entropy, expected_entropy, rel_tol=1e-6, abs_tol=1e-8):
            raise QueryViewError("language entropy conflicts with its probabilities")
        if prediction.uncertain and prediction.decision_label != "other":
            raise QueryViewError("uncertain language predictions must abstain as other")
        if not prediction.uncertain and prediction.decision_label != prediction.top_label:
            raise QueryViewError("confident language decision must equal top_label")
    return predictions


def _explicit_protections(raw: str, processing: str, explicit: Sequence[Any]) -> list[Any]:
    result = []
    for value in explicit:
        if isinstance(value, ProtectedSpan):
            value = {**asdict(value), "coordinate_space": "nfc_text"}
        if isinstance(value, Mapping):
            value = dict(value)
            space = value.get("coordinate_space")
            if space is not None and space != "nfc_text":
                raise QueryViewError("explicit protection coordinates must be nfc_text")
            if raw != processing and space != "nfc_text" and "text" not in value:
                raise QueryViewError("non-NFC raw input requires explicit nfc_text coordinates or matching expected text")
        elif raw != processing:
            raise QueryViewError("non-NFC raw input requires explicit protections with NFC coordinate metadata")
        result.append(value)
    return result


def _metadata(resource: Any, *, identifier: bool = False) -> Mapping[str, Any]:
    if resource is None:
        return {"backend": "none"}
    # Defaults to getattr are evaluated eagerly: do not even materialize a
    # large training manifest when a content-bound inference summary exists.
    manifest = getattr(resource, "inference_metadata", None)
    if not isinstance(manifest, Mapping):
        manifest = getattr(resource, "manifest", None)
    if not isinstance(manifest, Mapping):
        manifest = getattr(resource, "metadata", None)
    if isinstance(manifest, Mapping):
        result = dict(manifest)
    else:
        result = {"class": f"{type(resource).__module__}.{type(resource).__qualname__}"}
        config = getattr(resource, "config", None)
        if is_dataclass(config):
            result["config"] = asdict(config)
    if identifier:
        result["probability_calibration"] = "heldout_temperature" if result.get("calibration") else "uncalibrated_or_unknown"
    _canonical(result)
    return result


def _render(text: str, edits: Sequence[RewriteEdit]) -> str:
    pieces = []
    cursor = 0
    for edit in edits:
        if edit.start < cursor or text[edit.start:edit.end] != edit.original:
            raise QueryViewError("rewrite edits have conflicting processing-text offsets")
        pieces.extend((text[cursor:edit.start], edit.replacement))
        cursor = edit.end
    pieces.append(text[cursor:])
    output = "".join(pieces)
    if len(output) > MAX_QUERY_CHARACTERS:
        raise QueryViewError("generated query exceeds the query character safety limit")
    return output


def build_query_views(
    text: str,
    language_identifier: Any = None,
    transliterator: Transliterator | None = None,
    explicit_protections: Sequence[Any] = (),
    config: ViewConfig = ViewConfig(),
) -> QueryViews:
    """Build query-only hypotheses; no relevance labels or corpus fallback.

    Explicit protection offsets are into NFC processing text.  For a non-NFC
    original, mappings must carry coordinate_space='nfc_text' or an expected
    text value matching that NFC slice.  ProtectedSpan values always have this
    documented NFC coordinate contract.  The exact original is never rewritten.
    """

    if not isinstance(config, ViewConfig):
        raise QueryViewError("config must be a ViewConfig")
    config.validate()
    try:
        processing = normalize_processing_text(text)
        if len(processing) > MAX_QUERY_CHARACTERS:
            raise QueryViewError(f"query exceeds the {MAX_QUERY_CHARACTERS}-character safety limit")
        protections = protection_spans(processing, _explicit_protections(text, processing, explicit_protections))
    except TextPreparationError as exc:
        raise QueryViewError(str(exc)) from exc
    identifier = language_identifier if language_identifier is not None else ScriptEvidenceBaseline()
    predictions = _validated_predictions(processing, identifier)
    identifier_metadata = _metadata(identifier, identifier=True)
    backend_metadata = _metadata(transliterator)
    diagnostics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "config": asdict(config), "config_hash": _fingerprint(asdict(config)),
        "normalization_version": NORMALIZATION_VERSION, "offset_coordinate_space": "nfc_text_codepoints",
        "identifier": identifier_metadata, "identifier_fingerprint": _fingerprint(identifier_metadata),
        "transliterator": backend_metadata, "transliterator_fingerprint": _fingerprint(backend_metadata),
        "score_type": "additive_resource_scores_uncalibrated", "backend_calls": 0,
        "backend_cache_hits": 0, "backend_call_limit_reached": False,
        "intermediate_hypothesis_count": 0, "protected_attempted_rewrite_count": 0,
    }
    cache: dict[tuple[str, str, str], tuple[TransliterationCandidate, ...]] = {}
    def candidates(token: TokenSpan, language: str, *, native: bool) -> tuple[TransliterationCandidate, ...]:
        if transliterator is None:
            return ()
        key = ("native" if native else "roman", language, token.text)
        if key in cache:
            diagnostics["backend_cache_hits"] += 1
            return cache[key]
        if diagnostics["backend_calls"] >= config.max_backend_calls:
            diagnostics["backend_call_limit_reached"] = True
            return ()
        diagnostics["backend_calls"] += 1
        method = transliterator.native_candidates if native else transliterator.roman_candidates
        try:
            response = method(token.text, language, config.max_candidates_per_token)
        except QueryViewError:
            raise
        except Exception as exc:
            raise QueryViewError(f"transliteration backend failed: {exc}") from exc
        validated = _validated_candidates(response, script=TARGET_SCRIPTS[language] if native else "latin", limit=config.max_candidates_per_token)
        cache[key] = validated
        return validated
    def protected(token: TokenSpan) -> bool:
        return any(token.start < span.end and token.end > span.start for span in protections)
    def confident(prediction: TokenPrediction, language: str) -> bool:
        return not prediction.uncertain and prediction.decision_label == language and prediction.probabilities[language] >= config.min_probability
    def edit(token: TokenSpan, language: str, candidate: TransliterationCandidate) -> RewriteEdit:
        return RewriteEdit(token.start, token.end, token.text, candidate.text, language, candidate.backend, candidate.revision, float(candidate.raw_score))
    beam: list[tuple[tuple[RewriteEdit, ...], float]] = [((), 0.0)]
    if config.max_native_views:
        for prediction in predictions:
            token, language = prediction.span, prediction.decision_label
            if token.script != "latin" or language not in TARGET_SCRIPTS or not confident(prediction, language):
                continue
            if protected(token):
                diagnostics["protected_attempted_rewrite_count"] += 1
                continue
            options = candidates(token, language, native=True)
            if not options:
                continue
            expanded = [(edits + (edit(token, language, candidate),), _add_scores(score, candidate.raw_score)) for edits, score in beam for candidate in options]
            diagnostics["intermediate_hypothesis_count"] += len(expanded)
            distinct: dict[str, tuple[tuple[RewriteEdit, ...], float]] = {}
            for edits, score in sorted(expanded, key=lambda item: (-item[1], _render(processing, item[0]))):
                distinct.setdefault(_view_key(_render(processing, edits)), (edits, score))
            beam = list(distinct.values())[:config.max_native_views]
    native_views = tuple(
        QueryView("native", _render(processing, edits), edits, float(score))
        for edits, score in beam
        if edits and _view_key(_render(processing, edits)) != _view_key(processing)
    )
    roman_edits: list[RewriteEdit] = []
    roman_score = 0.0
    roman_reason: str | None = None
    for prediction in predictions:
        token = prediction.span
        if token.script in {"latin", "numeric"} or not any(unicodedata.category(character).startswith("L") for character in token.text):
            continue
        language = {"devanagari": "hi", "bengali": "bn"}.get(token.script)
        if language is None:
            roman_reason = "unsupported_non_latin_script"
            break
        if protected(token):
            roman_reason = "protected_native_token"
            break
        if not confident(prediction, language):
            roman_reason = "uncertain_native_token"
            break
        options = candidates(token, language, native=False)
        if not options:
            roman_reason = "backend_call_limit" if diagnostics["backend_call_limit_reached"] else "no_romanization_candidate"
            break
        roman_edits.append(edit(token, language, options[0]))
        roman_score = _add_scores(roman_score, options[0].raw_score)
    roman_text = _render(processing, roman_edits)
    if roman_reason is None and not is_roman_script_view(roman_text):
        roman_reason = "no_complete_roman_content"
    roman_view = QueryView("romanized", roman_text, tuple(roman_edits), float(roman_score)) if roman_reason is None else None
    script_counts = Counter(prediction.span.script for prediction in predictions)
    script_character_counts: Counter[str] = Counter()
    for prediction in predictions:
        script_character_counts[prediction.span.script] += len(prediction.span.text)
    script_character_count = sum(script_character_counts.values())
    decisions = [prediction.decision_label for prediction in predictions if not prediction.uncertain and prediction.decision_label != "other"]
    diagnostics.update({
        "native_available": bool(native_views), "native_view_count": len(native_views),
        "roman_available": roman_view is not None, "roman_unavailable_reason": roman_reason,
        "token_count": len(predictions), "protected_span_count": len(protections),
        "protection_reasons": dict(sorted(Counter(reason for span in protections for reason in span.reasons).items())),
        "mean_token_entropy": sum(prediction.entropy for prediction in predictions) / len(predictions) if predictions else 0.0,
        "uncertain_token_fraction": sum(prediction.uncertain for prediction in predictions) / len(predictions) if predictions else 0.0,
        "script_fractions": {script: count / script_character_count for script, count in sorted(script_character_counts.items())},
        "token_script_fractions": {script: count / len(predictions) for script, count in sorted(script_counts.items())},
        "language_switch_count": sum(left != right for left, right in zip(decisions, decisions[1:])),
        "native_changed_token_fraction": max((len(view.edits) for view in native_views), default=0) / len(predictions) if predictions else 0.0,
        "roman_changed_token_fraction": len(roman_edits) / len(predictions) if roman_view and predictions else 0.0,
    })
    _canonical(diagnostics)
    return QueryViews(text, processing, predictions, protections, QueryView("original", text), native_views, roman_view, diagnostics)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--queries", type=Path)
    parser.add_argument("--split", choices=ALL_SPLITS)
    parser.add_argument("--lid-model", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--backend", choices=("none", "lexicon", "indicxlit"))
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-native-views", type=int, default=2)
    parser.add_argument("--max-candidates-per-token", type=int, default=2)
    parser.add_argument("--max-backend-calls", type=int, default=64)
    parser.add_argument("--min-probability", type=float, default=0.60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists() and not args.force:
            raise QueryViewError(f"refusing to overwrite {args.output}; pass --force")
        identifier = TokenLanguageIdentifier.load_json(args.lid_model) if args.lid_model else ScriptEvidenceBaseline()
        backend = args.backend or ("lexicon" if args.lexicon else "none")
        if backend == "lexicon":
            if args.lexicon is None:
                raise QueryViewError("lexicon backend requires --lexicon")
            transliterator: Transliterator | None = LexiconTransliterator.from_jsonl(args.lexicon)
        elif backend == "indicxlit":
            if args.lexicon:
                raise QueryViewError("--lexicon cannot be combined with indicxlit backend")
            transliterator = IndicXlitTransliterator(allow_model_download=args.allow_model_download)
        else:
            if args.lexicon:
                raise QueryViewError("--lexicon cannot be combined with backend none")
            transliterator = None
        config = ViewConfig(args.max_native_views, args.max_candidates_per_token, args.max_backend_calls, args.min_probability)
        config.validate()
        if args.queries:
            if not args.split or not args.output:
                raise QueryViewError("batch query views require --split and --output")
            from data.generate_variants import load_assigned_queries
            selected = [record for record in load_assigned_queries(args.queries) if record["split"] == args.split]
            if not selected:
                raise QueryViewError(f"no queries selected for split {args.split!r}")
            records = [
                {**query, **asdict(build_query_views(query["text"], identifier, transliterator, query.get("protected_spans", ()), config))}
                for query in selected
            ]
        else:
            records = [asdict(build_query_views(args.text, identifier, transliterator, config=config))]
        for record in records:
            _canonical(record)
        if args.output:
            _atomic_write_jsonl(args.output, records)
        else:
            sys.stdout.write("".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in records))
    except (QueryViewError, TextPreparationError, LanguageIdError, CorpusPreparationError, OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke tested
    raise SystemExit(main())
