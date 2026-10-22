"""Build a fixed, reproducible retrieval corpus from MIRACL-like files.

The implementation deliberately has no third-party dependencies.  It streams
the source corpus, retains every qrel-referenced passage, and selects remaining
distractors by a stable SHA-256 priority.  Hash-priority sampling is reproducible
even if the source file is reordered.

Supported passage JSONL fields include MIRACL's ``docid``, ``title`` and
``text`` as well as the canonical ``doc_id`` spelling.  Qrels may be JSONL or
three/four-column whitespace-delimited text.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
SUPPORTED_LANGUAGES = {"en", "hi", "bn"}
LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "bn": "bn",
    "ben": "bn",
    "bengali": "bn",
    "bangla": "bn",
}
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024


class CorpusPreparationError(ValueError):
    """Raised when source data would produce an invalid corpus artifact."""


@dataclass(frozen=True)
class Passage:
    """Canonical passage record required by the project data contract."""

    doc_id: str
    language: str
    title: str
    text: str
    romanized_text: str
    source: str
    source_doc_id: str
    romanization_status: str


@dataclass(frozen=True)
class Qrel:
    """Canonical relevance judgment."""

    query_id: str
    doc_id: str
    relevance: int


@dataclass(frozen=True)
class CorpusSource:
    """One language-specific corpus and its optional judgments."""

    language: str
    passages_path: Path
    qrels_path: Path | None = None
    source_name: str = "MIRACL"


@dataclass(frozen=True)
class PreparedCorpus:
    """In-memory result returned by :func:`prepare_corpus`."""

    passages: tuple[Passage, ...]
    qrels: tuple[Qrel, ...]
    id_map: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def canonical_language(value: str) -> str:
    """Return an ISO-like project language code or fail loudly."""

    normalized = value.strip().casefold()
    try:
        return LANGUAGE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise CorpusPreparationError(
            f"Unsupported language {value!r}; expected one of {allowed}"
        ) from exc


def _clean_id(value: Any, field: str) -> str:
    if value is None:
        raise CorpusPreparationError(f"Missing required {field}")
    result = str(value).strip()
    if not result:
        raise CorpusPreparationError(f"{field} must not be empty")
    if any(unicodedata.category(ch) == "Cc" for ch in result):
        raise CorpusPreparationError(f"{field} contains a control character")
    return result


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _canonical_passage(
    record: Mapping[str, Any], *, language: str, source: str, prefix_ids: bool
) -> Passage:
    raw_id = _clean_id(
        _first(
            record,
            ("source_doc_id", "doc_id", "docid", "document_id", "id", "_id"),
        ),
        "doc_id",
    )
    raw_text = _first(record, ("text", "passage", "contents", "body"))
    if raw_text is None:
        raise CorpusPreparationError(f"Passage {raw_id!r} is missing text")
    text = unicodedata.normalize("NFC", str(raw_text))
    if not text.strip():
        raise CorpusPreparationError(f"Passage {raw_id!r} has empty text")
    if "\x00" in text:
        raise CorpusPreparationError(f"Passage {raw_id!r} contains a NUL byte")
    title_value = _first(record, ("title", "heading"))
    title = unicodedata.normalize("NFC", "" if title_value is None else str(title_value))

    supplied_romanized = record.get("romanized_text")
    if supplied_romanized is not None:
        romanized_text = unicodedata.normalize("NFC", str(supplied_romanized))
        romanization_status = str(record.get("romanization_status", "supplied"))
    elif language == "en":
        romanized_text = text
        romanization_status = "identity"
    else:
        # Transliteration is a later module.  An empty value satisfies the stable
        # record shape while making the missing transformation explicit.
        romanized_text = ""
        romanization_status = "pending"

    doc_id = (
        raw_id
        if not prefix_ids or raw_id.startswith(f"{language}:")
        else f"{language}:{raw_id}"
    )
    return Passage(
        doc_id=doc_id,
        language=language,
        title=title,
        text=text,
        romanized_text=romanized_text,
        source=source,
        source_doc_id=raw_id,
        romanization_status=romanization_status,
    )


def iter_jsonl(path: Path, *, max_line_bytes: int = MAX_JSONL_LINE_BYTES) -> Iterator[dict[str, Any]]:
    """Yield JSON objects with useful, bounded-input error messages."""

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > max_line_bytes:
                raise CorpusPreparationError(
                    f"{path}:{line_number} exceeds the {max_line_bytes}-byte line limit"
                )
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CorpusPreparationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise CorpusPreparationError(f"Expected an object at {path}:{line_number}")
            yield record


def _parse_relevance(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise CorpusPreparationError(f"Boolean relevance is invalid at {context}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CorpusPreparationError(f"Invalid relevance {value!r} at {context}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise CorpusPreparationError(f"Relevance must be an integer at {context}")
    if number < 0:
        raise CorpusPreparationError(f"Relevance must be nonnegative at {context}")
    return number


def load_qrels(path: Path) -> list[Qrel]:
    """Read canonical/MIRACL JSONL or TREC-style qrels."""

    qrels: list[Qrel] = []
    seen: set[tuple[str, str]] = set()
    if path.suffix.casefold() in {".jsonl", ".json"}:
        records: Iterable[Mapping[str, Any]] = iter_jsonl(path)
        for line_number, record in enumerate(records, start=1):
            query_id = _clean_id(
                _first(record, ("query_id", "queryid", "qid")), "query_id"
            )
            doc_id = _clean_id(
                _first(record, ("doc_id", "docid", "document_id", "corpus_id")),
                "doc_id",
            )
            relevance = _parse_relevance(
                _first(record, ("relevance", "score", "rel")),
                context=f"{path}:{line_number}",
            )
            key = (query_id, doc_id)
            if key in seen:
                raise CorpusPreparationError(f"Duplicate qrel {key!r} in {path}")
            seen.add(key)
            qrels.append(Qrel(query_id, doc_id, relevance))
        return qrels

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) == 3:
                query_id, doc_id, relevance_value = fields
            elif len(fields) == 4:
                query_id, _iteration, doc_id, relevance_value = fields
            else:
                raise CorpusPreparationError(
                    f"Expected 3 or 4 qrel columns at {path}:{line_number}"
                )
            relevance = _parse_relevance(relevance_value, context=f"{path}:{line_number}")
            key = (_clean_id(query_id, "query_id"), _clean_id(doc_id, "doc_id"))
            if key in seen:
                raise CorpusPreparationError(f"Duplicate qrel {key!r} in {path}")
            seen.add(key)
            qrels.append(Qrel(key[0], key[1], relevance))
    return qrels


def _sample_priority(seed: int, language: str, source_doc_id: str) -> int:
    payload = f"v1\0{seed}\0{language}\0{source_doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _prepare_one_source(
    source: CorpusSource,
    *,
    target_size: int,
    seed: int,
    prefix_ids: bool,
) -> tuple[list[Passage], list[Qrel], list[dict[str, str]], dict[str, Any]]:
    language = canonical_language(source.language)
    if target_size <= 0:
        raise CorpusPreparationError("target_size must be positive")
    raw_qrels = load_qrels(source.qrels_path) if source.qrels_path else []
    judged_ids = {qrel.doc_id for qrel in raw_qrels}
    distractor_capacity = max(0, target_size - len(judged_ids))

    mandatory: dict[str, Passage] = {}
    # Negative priorities turn heapq into a bounded max-priority heap.  A SHA-256
    # collision between distinct IDs is practically irrelevant; doc_id remains
    # in the tuple to make ordering deterministic even in that event.
    distractors: list[tuple[int, str, Passage]] = []
    seen_source_ids: set[str] = set()
    source_count = 0

    for record in iter_jsonl(source.passages_path):
        passage = _canonical_passage(
            record,
            language=language,
            source=source.source_name,
            prefix_ids=prefix_ids,
        )
        source_count += 1
        if passage.source_doc_id in seen_source_ids:
            raise CorpusPreparationError(
                f"Duplicate passage ID {passage.source_doc_id!r} in {source.passages_path}"
            )
        seen_source_ids.add(passage.source_doc_id)
        if passage.source_doc_id in judged_ids:
            mandatory[passage.source_doc_id] = passage
            continue
        if distractor_capacity == 0:
            continue
        priority = _sample_priority(seed, language, passage.source_doc_id)
        entry = (-priority, passage.source_doc_id, passage)
        if len(distractors) < distractor_capacity:
            heapq.heappush(distractors, entry)
        elif priority < -distractors[0][0]:
            heapq.heapreplace(distractors, entry)

    missing = sorted(judged_ids - mandatory.keys())
    if missing:
        preview = ", ".join(repr(item) for item in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise CorpusPreparationError(
            f"{len(missing)} qrel document(s) missing from {source.passages_path}: "
            f"{preview}{suffix}"
        )

    selected = list(mandatory.values()) + [entry[2] for entry in distractors]
    selected.sort(key=lambda passage: passage.doc_id)
    mapping = {
        passage.source_doc_id: passage.doc_id
        for passage in selected
    }
    rewritten_qrels = [
        Qrel(qrel.query_id, mapping[qrel.doc_id], qrel.relevance)
        for qrel in raw_qrels
    ]
    rewritten_qrels.sort(key=lambda qrel: (qrel.query_id, qrel.doc_id))
    id_map = [
        {
            "language": language,
            "source_doc_id": passage.source_doc_id,
            "doc_id": passage.doc_id,
        }
        for passage in selected
    ]
    summary = {
        "language": language,
        "source": source.source_name,
        "passages_path": str(source.passages_path),
        "qrels_path": str(source.qrels_path) if source.qrels_path else None,
        "source_passage_count": source_count,
        "judged_passage_count": len(judged_ids),
        "selected_passage_count": len(selected),
        "requested_target_size": target_size,
        "effective_target_size": max(target_size, len(judged_ids)),
        "target_expanded_for_judgments": len(judged_ids) > target_size,
        "selected_distractor_count": len(selected) - len(judged_ids),
        "target_shortfall": max(
            0, max(target_size, len(judged_ids)) - len(selected)
        ),
    }
    return selected, rewritten_qrels, id_map, summary


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _records_digest(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_corpus(
    sources: Sequence[CorpusSource],
    *,
    target_size: int = 50_000,
    seed: int = 13,
    dataset_version: str = "unspecified",
    prefix_doc_ids: bool | None = None,
) -> PreparedCorpus:
    """Prepare one fixed corpus from one or more language sources.

    ``prefix_doc_ids`` defaults to true for pooled (multi-language) corpora and
    false for a single-language track.
    """

    if not sources:
        raise CorpusPreparationError("At least one corpus source is required")
    languages = [canonical_language(source.language) for source in sources]
    if len(set(languages)) != len(languages):
        raise CorpusPreparationError("Provide at most one source per language")
    if prefix_doc_ids is None:
        prefix_doc_ids = len(sources) > 1

    all_passages: list[Passage] = []
    all_qrels: list[Qrel] = []
    all_id_map: list[Mapping[str, str]] = []
    summaries: list[Mapping[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_qrels: set[tuple[str, str]] = set()

    for source in sorted(sources, key=lambda item: canonical_language(item.language)):
        passages, qrels, id_map, summary = _prepare_one_source(
            source,
            target_size=target_size,
            seed=seed,
            prefix_ids=prefix_doc_ids,
        )
        for passage in passages:
            if passage.doc_id in seen_doc_ids:
                raise CorpusPreparationError(
                    f"Document ID collision in pooled corpus: {passage.doc_id!r}"
                )
            seen_doc_ids.add(passage.doc_id)
        for qrel in qrels:
            key = (qrel.query_id, qrel.doc_id)
            if key in seen_qrels:
                raise CorpusPreparationError(f"Duplicate pooled qrel: {key!r}")
            seen_qrels.add(key)
        all_passages.extend(passages)
        all_qrels.extend(qrels)
        all_id_map.extend(id_map)
        summaries.append(summary)

    all_passages.sort(key=lambda passage: passage.doc_id)
    all_qrels.sort(key=lambda qrel: (qrel.query_id, qrel.doc_id))
    all_id_map.sort(key=lambda item: (item["language"], item["source_doc_id"]))

    passage_records = [asdict(passage) for passage in all_passages]
    qrel_records = [asdict(qrel) for qrel in all_qrels]
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "seed": seed,
        "target_size_per_language": target_size,
        "prefix_doc_ids": prefix_doc_ids,
        "sampling": "sha256_priority_v1",
        "preprocessing": {
            "unicode_normalization": "NFC",
            "romanization": "identity_en_pending_hi_bn",
        },
        "languages": sorted(languages),
    }
    config_hash = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    manifest: dict[str, Any] = {
        **config,
        "config_hash": config_hash,
        "passage_count": len(all_passages),
        "qrel_count": len(all_qrels),
        "corpus_sha256": _records_digest(passage_records),
        "qrels_sha256": _records_digest(qrel_records),
        "sources": summaries,
    }
    return PreparedCorpus(
        passages=tuple(all_passages),
        qrels=tuple(all_qrels),
        id_map=tuple(all_id_map),
        manifest=manifest,
    )


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_prepared_corpus(
    prepared: PreparedCorpus, output_dir: Path, *, overwrite: bool = False
) -> Mapping[str, Path]:
    """Atomically write corpus, qrels, ID map, and manifest artifacts."""

    paths = {
        "corpus": output_dir / "corpus.jsonl",
        "qrels": output_dir / "qrels.jsonl",
        "id_map": output_dir / "doc_id_map.jsonl",
        "manifest": output_dir / "corpus_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise CorpusPreparationError(
            f"Refusing to overwrite existing artifact(s): {names}; pass --force"
        )
    _atomic_write_jsonl(paths["corpus"], (asdict(item) for item in prepared.passages))
    _atomic_write_jsonl(paths["qrels"], (asdict(item) for item in prepared.qrels))
    _atomic_write_jsonl(paths["id_map"], prepared.id_map)
    _atomic_write_json(paths["manifest"], prepared.manifest)
    return paths


def _parse_source_spec(value: str, *, option: str) -> tuple[str, Path]:
    language, separator, raw_path = value.partition("=")
    if not separator or not language or not raw_path:
        raise argparse.ArgumentTypeError(f"{option} must use LANGUAGE=PATH")
    try:
        canonical = canonical_language(language)
    except CorpusPreparationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return canonical, Path(raw_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passages",
        action="append",
        required=True,
        metavar="LANGUAGE=PATH",
        help="language-specific passage JSONL; repeat for a pooled corpus",
    )
    parser.add_argument(
        "--qrels",
        action="append",
        default=[],
        metavar="LANGUAGE=PATH",
        help="language-specific qrels; repeat as needed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset-version", default="unspecified")
    parser.add_argument(
        "--prefix-doc-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="prefix IDs with language (default: yes for pooled corpora)",
    )
    parser.add_argument("--source-name", default="MIRACL")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        passage_specs = [
            _parse_source_spec(value, option="--passages") for value in args.passages
        ]
        qrel_pairs = [
            _parse_source_spec(value, option="--qrels") for value in args.qrels
        ]
        qrel_languages = [language for language, _path in qrel_pairs]
        if len(qrel_languages) != len(set(qrel_languages)):
            raise CorpusPreparationError("Provide at most one qrels file per language")
        qrel_specs = dict(qrel_pairs)
        sources = [
            CorpusSource(
                language=language,
                passages_path=path,
                qrels_path=qrel_specs.get(language),
                source_name=args.source_name,
            )
            for language, path in passage_specs
        ]
        unused_qrels = sorted(set(qrel_specs) - {language for language, _ in passage_specs})
        if unused_qrels:
            raise CorpusPreparationError(
                f"Qrels supplied without passages for: {', '.join(unused_qrels)}"
            )
        prepared = prepare_corpus(
            sources,
            target_size=args.target_size,
            seed=args.seed,
            dataset_version=args.dataset_version,
            prefix_doc_ids=args.prefix_doc_ids,
        )
        paths = write_prepared_corpus(prepared, args.output_dir, overwrite=args.force)
    except (CorpusPreparationError, argparse.ArgumentTypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
