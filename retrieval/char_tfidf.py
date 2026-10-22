"""Character 3--5-gram TF-IDF retrieval over explicit Romanized text.

Only the canonical passage ``romanized_text`` field is indexed. Empty or
non-Roman fields are unavailable and are never replaced with native text. The
same NFC/casefold/whitespace normalization is applied to documents and queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import (
    CorpusPreparationError,
    _canonical_json_bytes,
    _clean_id,
    _first,
    canonical_language,
    iter_jsonl,
)
from retrieval.bm25 import (
    DEFAULT_TOP_K,
    MAX_QUERY_CHARACTERS,
    RetrievalQuery,
    RetrievalResult,
    _file_sha256,
    load_retrieval_queries,
    write_run_artifacts,
)


SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "nfc_casefold_collapse_whitespace_v1"


class CharTfidfError(ValueError):
    """Raised when character-index input or configuration is invalid."""


@dataclass(frozen=True)
class CharTfidfConfig:
    min_n: int = 3
    max_n: int = 5

    def validate(self) -> None:
        if isinstance(self.min_n, bool) or isinstance(self.max_n, bool):
            raise CharTfidfError("N-gram bounds must be integers")
        if self.min_n < 1:
            raise CharTfidfError("min_n must be at least 1")
        if self.max_n < self.min_n:
            raise CharTfidfError("max_n must be greater than or equal to min_n")
        if self.max_n > 16:
            raise CharTfidfError("max_n must not exceed 16")


@dataclass(frozen=True)
class CharDocument:
    doc_id: str
    romanized_text: str
    language: str
    source: str
    romanization_status: str


@dataclass(frozen=True)
class CharTfidfHit:
    doc_id: str
    rank: int
    score: float


def normalize_romanized_text(text: str, *, query: bool = False) -> str:
    """Apply the shared, conservative Romanized-text normalization."""

    if not isinstance(text, str):
        raise CharTfidfError("Romanized text must be a string")
    if "\x00" in text:
        raise CharTfidfError("Romanized text contains a NUL byte")
    if query and len(text) > MAX_QUERY_CHARACTERS:
        raise CharTfidfError(
            f"Query exceeds the {MAX_QUERY_CHARACTERS}-character safety limit"
        )
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    pieces: list[str] = []
    pending_space = False
    for character in normalized:
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("C"):
            pending_space = bool(pieces)
            continue
        if pending_space:
            pieces.append(" ")
            pending_space = False
        pieces.append(character)
    return "".join(pieces).strip()


def is_roman_script_view(text: str) -> bool:
    """Return whether all letters are Latin and the view has useful content."""

    normalized = normalize_romanized_text(text)
    has_letter_or_number = False
    for character in normalized:
        category = unicodedata.category(character)
        if category.startswith("L"):
            has_letter_or_number = True
            if "LATIN" not in unicodedata.name(character, ""):
                return False
        elif category.startswith("N"):
            has_letter_or_number = True
        elif category.startswith("M"):
            # Combining marks are allowed with Roman letters; NFC normally
            # composes the common cases.
            continue
    return has_letter_or_number


def character_ngrams(
    text: str, *, min_n: int = 3, max_n: int = 5
) -> tuple[str, ...]:
    """Return raw, overlapping character n-grams in deterministic order."""

    config = CharTfidfConfig(min_n, max_n)
    config.validate()
    normalized = normalize_romanized_text(text)
    grams: list[str] = []
    for width in range(min_n, max_n + 1):
        grams.extend(
            normalized[start : start + width]
            for start in range(max(0, len(normalized) - width + 1))
        )
    return tuple(grams)


class CharTfidfIndex:
    """Immutable sparse character TF-IDF cosine index."""

    def __init__(
        self,
        documents: Sequence[CharDocument],
        config: CharTfidfConfig = CharTfidfConfig(),
    ) -> None:
        config.validate()
        self.config = config
        ordered = sorted(documents, key=lambda document: document.doc_id)
        doc_ids = [document.doc_id for document in ordered]
        duplicates = sorted(
            doc_id for doc_id, count in Counter(doc_ids).items() if count > 1
        )
        if duplicates:
            raise CharTfidfError(
                f"Duplicate document ID(s): {', '.join(duplicates[:5])}"
            )

        status_counts = Counter(document.romanization_status for document in ordered)
        empty_count = non_roman_count = too_short_count = 0
        active: list[tuple[CharDocument, Counter[str]]] = []
        corpus_digest = hashlib.sha256()
        for document in ordered:
            corpus_digest.update(_canonical_json_bytes(asdict(document)))
            corpus_digest.update(b"\n")
            normalized = normalize_romanized_text(document.romanized_text)
            if not normalized:
                empty_count += 1
                continue
            if not is_roman_script_view(normalized):
                non_roman_count += 1
                continue
            counts = Counter(
                character_ngrams(
                    normalized, min_n=config.min_n, max_n=config.max_n
                )
            )
            if not counts:
                too_short_count += 1
                continue
            active.append((document, counts))

        document_frequency: Counter[str] = Counter()
        for _document, counts in active:
            document_frequency.update(counts.keys())
        active_count = len(active)
        self._idf = {
            term: math.log((1.0 + active_count) / (1.0 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._doc_ids = tuple(document.doc_id for document, _counts in active)
        for document_index, (_document, counts) in enumerate(active):
            weights = {
                term: frequency * self._idf[term]
                for term, frequency in counts.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in weights.values()))
            if norm <= 0 or not math.isfinite(norm):
                raise CharTfidfError("Document TF-IDF vector has an invalid norm")
            for term, weight in weights.items():
                postings[term].append((document_index, weight / norm))
        self._postings = {
            term: tuple(values) for term, values in sorted(postings.items())
        }

        index_config = {
            "normalization": NORMALIZATION_VERSION,
            "analyzer": "raw_character_ngrams",
            "min_n": config.min_n,
            "max_n": config.max_n,
            "tf": "raw_count",
            "idf": "log((1+N)/(1+df))+1",
            "normalization_after_weighting": "l2",
            "similarity": "cosine_via_inner_product",
            "input_field": "romanized_text",
            "native_text_fallback": False,
        }
        config_hash = hashlib.sha256(_canonical_json_bytes(index_config)).hexdigest()
        corpus_fingerprint = corpus_digest.hexdigest()
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "retriever": "character_tfidf",
            **index_config,
            "config_hash": config_hash,
            "source_document_count": len(ordered),
            "indexed_document_count": active_count,
            "empty_romanized_document_count": empty_count,
            "non_roman_document_count": non_roman_count,
            "too_short_document_count": too_short_count,
            "romanization_status_counts": dict(sorted(status_counts.items())),
            "vocabulary_size": len(self._postings),
            "posting_count": sum(len(values) for values in self._postings.values()),
            "corpus_fingerprint": corpus_fingerprint,
            "index_fingerprint": hashlib.sha256(
                f"{corpus_fingerprint}\0{config_hash}".encode("ascii")
            ).hexdigest(),
        }

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    def query_available(self, query_text: str) -> bool:
        normalized = normalize_romanized_text(query_text, query=True)
        if not normalized or not is_roman_script_view(normalized):
            return False
        return len(normalized) >= self.config.min_n

    def search(
        self, query_text: str, *, top_k: int = DEFAULT_TOP_K
    ) -> tuple[CharTfidfHit, ...]:
        if top_k < 0:
            raise CharTfidfError("top_k must be nonnegative")
        if top_k == 0 or not self._doc_ids or not self.query_available(query_text):
            return ()
        counts = Counter(
            character_ngrams(
                query_text,
                min_n=self.config.min_n,
                max_n=self.config.max_n,
            )
        )
        weights = {
            term: frequency * self._idf[term]
            for term, frequency in counts.items()
            if term in self._idf
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        if norm <= 0 or not math.isfinite(norm):
            return ()
        scores: dict[int, float] = defaultdict(float)
        for term, weight in weights.items():
            query_weight = weight / norm
            for document_index, document_weight in self._postings[term]:
                scores[document_index] += query_weight * document_weight
        ranked = sorted(
            (
                (document_index, score)
                for document_index, score in scores.items()
                if score > 0 and math.isfinite(score)
            ),
            key=lambda item: (-item[1], self._doc_ids[item[0]]),
        )[:top_k]
        return tuple(
            CharTfidfHit(self._doc_ids[document_index], rank, score)
            for rank, (document_index, score) in enumerate(ranked, start=1)
        )


def load_corpus(path: Path) -> list[CharDocument]:
    """Load the canonical prepared corpus without native-text fallback."""

    documents: list[CharDocument] = []
    seen: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            doc_id = _clean_id(
                _first(record, ("doc_id", "docid", "document_id")), "doc_id"
            )
            if doc_id in seen:
                raise CharTfidfError(f"Duplicate document ID {doc_id!r}")
            seen.add(doc_id)
            if "romanized_text" not in record:
                raise CharTfidfError(
                    f"Document {doc_id!r} is missing required romanized_text"
                )
            romanized = record["romanized_text"]
            if not isinstance(romanized, str):
                raise CharTfidfError(
                    f"Document {doc_id!r} romanized_text must be a string"
                )
            language = canonical_language(str(record.get("language", "")))
            documents.append(
                CharDocument(
                    doc_id=doc_id,
                    romanized_text=romanized,
                    language=language,
                    source=str(record.get("source", "unspecified")),
                    romanization_status=str(
                        record.get(
                            "romanization_status",
                            "supplied" if romanized else "unavailable",
                        )
                    ),
                )
            )
        except (CharTfidfError, CorpusPreparationError) as exc:
            raise CharTfidfError(f"{path}:{line_number}: {exc}") from exc
    return documents


def run_queries(
    index: CharTfidfIndex,
    queries: Sequence[RetrievalQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
    branch: str = "char_tfidf",
) -> tuple[RetrievalResult, ...]:
    branch = _clean_id(branch, "branch")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise CharTfidfError("Duplicate query IDs in retrieval input")
    results: list[RetrievalResult] = []
    for query in sorted(queries, key=lambda item: item.query_id):
        for hit in index.search(query.text, top_k=top_k):
            results.append(
                RetrievalResult(
                    query_id=query.query_id,
                    doc_id=hit.doc_id,
                    branch=branch,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
    return tuple(results)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=ALL_SPLITS, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--branch", default="char_tfidf")
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.top_k <= 0:
            raise CharTfidfError("top-k must be positive")
        config = CharTfidfConfig(args.min_n, args.max_n)
        documents = load_corpus(args.corpus)
        queries = load_retrieval_queries(args.queries, split=args.split)
        build_started = time.perf_counter()
        index = CharTfidfIndex(documents, config)
        build_seconds = time.perf_counter() - build_started
        active_queries = sum(index.query_available(query.text) for query in queries)
        search_started = time.perf_counter()
        branch = _clean_id(args.branch, "branch")
        results = run_queries(index, queries, top_k=args.top_k, branch=branch)
        search_seconds = time.perf_counter() - search_started
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "index": index.manifest,
            "corpus_path": str(args.corpus),
            "corpus_file_sha256": _file_sha256(args.corpus),
            "queries_path": str(args.queries),
            "queries_file_sha256": _file_sha256(args.queries),
            "selected_split": args.split,
            "branch": branch,
            "top_k": args.top_k,
            "query_count": len(queries),
            "active_roman_query_count": active_queries,
            "inactive_query_count": len(queries) - active_queries,
            "search_count": active_queries,
            "intermediate_candidate_count": len(results),
            "result_count": len(results),
            "build_seconds": build_seconds,
            "search_seconds": search_seconds,
        }
        write_run_artifacts(
            results,
            args.output,
            manifest,
            manifest_path,
            overwrite=args.force,
        )
    except (CharTfidfError, CorpusPreparationError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"run": str(args.output), "manifest": str(manifest_path)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(main())

