"""Pure-Python Okapi BM25 baseline for original and native query views.

The index combines title and passage term frequencies with a configurable title
weight and uses conservative NFC, case-folded, script-aware tokenization.  It
does not stem, remove stopwords, correct spelling, or rewrite the query.

This implementation is intentionally dependency-free and appropriate for unit
fixtures and the proposed reduced corpora.  It is not presented as an efficient
backend for the full 32.9-million-passage English MIRACL collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import (
    CorpusPreparationError,
    _canonical_json_bytes,
    _clean_id,
    _first,
    canonical_language,
    iter_jsonl,
)


SCHEMA_VERSION = 1
TOKENIZER_VERSION = "nfc_casefold_script_v1"
MAX_QUERY_CHARACTERS = 4_096
MAX_QUERY_TOKENS = 256
DEFAULT_TOP_K = 100


class BM25Error(ValueError):
    """Raised when index configuration or retrieval input is invalid."""


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    title_weight: float = 2.0

    def validate(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise BM25Error("k1 must be finite and greater than zero")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise BM25Error("b must be finite and between zero and one")
        if not math.isfinite(self.title_weight) or self.title_weight < 0:
            raise BM25Error("title_weight must be finite and nonnegative")


@dataclass(frozen=True)
class BM25Document:
    doc_id: str
    title: str
    text: str
    language: str
    source: str


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    text: str
    split: str | None = None


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    doc_id: str
    branch: str
    rank: int
    score: float


@dataclass(frozen=True)
class MultiViewSearch:
    hits: tuple[BM25Hit, ...]
    unique_view_count: int
    search_count: int
    intermediate_candidate_count: int


def _script_bucket(character: str) -> str:
    codepoint = ord(character)
    if 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF:
        return "devanagari"
    if 0x0980 <= codepoint <= 0x09FF:
        return "bengali"
    name = unicodedata.name(character, "")
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    return "other_letter"


def tokenize(text: str, *, query: bool = False) -> tuple[str, ...]:
    """Tokenize letters/numbers while retaining Indic combining marks.

    A script transition creates a boundary even without whitespace, so a string
    such as ``earthquakeक्यों`` becomes two tokens. ZWJ/ZWNJ are retained inside
    an existing token because they can be meaningful in Indic orthography.
    """

    if not isinstance(text, str):
        raise BM25Error("Text to tokenize must be a string")
    if "\x00" in text:
        raise BM25Error("Text contains a NUL byte")
    if query and len(text) > MAX_QUERY_CHARACTERS:
        raise BM25Error(
            f"Query exceeds the {MAX_QUERY_CHARACTERS}-character safety limit"
        )
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    tokens: list[str] = []
    current: list[str] = []
    current_script: str | None = None

    def flush() -> None:
        nonlocal current, current_script
        if current:
            tokens.append("".join(current))
            current = []
            current_script = None

    for character in normalized:
        category = unicodedata.category(character)
        if category.startswith("L"):
            script = _script_bucket(character)
            if current and current_script not in {None, "number", script}:
                flush()
            current.append(character)
            if current_script in {None, "number"}:
                current_script = script
        elif category.startswith("N"):
            current.append(character)
            if current_script is None:
                current_script = "number"
        elif category.startswith("M"):
            # A mark belongs to its preceding base.  Orphan marks are retained
            # as a token rather than silently deleted.
            current.append(character)
            if current_script is None:
                current_script = _script_bucket(character)
        elif character in {"\u200c", "\u200d"} and current:
            current.append(character)
        else:
            flush()
    flush()
    if query and len(tokens) > MAX_QUERY_TOKENS:
        raise BM25Error(f"Query exceeds the {MAX_QUERY_TOKENS}-token safety limit")
    return tuple(tokens)


class BM25Index:
    """Immutable in-memory BM25 index shared by original/native query views."""

    def __init__(self, documents: Sequence[BM25Document], config: BM25Config = BM25Config()):
        config.validate()
        sorted_documents = sorted(documents, key=lambda document: document.doc_id)
        doc_ids = [document.doc_id for document in sorted_documents]
        if len(doc_ids) != len(set(doc_ids)):
            duplicates = sorted(
                doc_id for doc_id, count in Counter(doc_ids).items() if count > 1
            )
            raise BM25Error(f"Duplicate document ID(s): {', '.join(duplicates[:5])}")
        self._documents = tuple(sorted_documents)
        self.config = config
        self._lengths: list[float] = []
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for index, document in enumerate(self._documents):
            title_tokens = tokenize(document.title)
            body_tokens = tokenize(document.text)
            title_terms = Counter(title_tokens)
            body_terms = Counter(body_tokens)
            weighted_tf: dict[str, float] = {}
            for term in title_terms.keys() | body_terms.keys():
                weighted_tf[term] = (
                    body_terms.get(term, 0)
                    + self.config.title_weight * title_terms.get(term, 0)
                )
            length = len(body_tokens) + self.config.title_weight * len(title_tokens)
            self._lengths.append(length)
            for term, frequency in weighted_tf.items():
                if frequency > 0:
                    postings[term].append((index, frequency))

        self._postings = {
            term: tuple(values) for term, values in sorted(postings.items())
        }
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        document_count = len(self._documents)
        self._idf = {
            term: math.log(
                1.0
                + (document_count - len(values) + 0.5) / (len(values) + 0.5)
            )
            for term, values in self._postings.items()
        }
        digest = hashlib.sha256()
        for document in self._documents:
            digest.update(_canonical_json_bytes(asdict(document)))
            digest.update(b"\n")
        index_config = {
            "tokenizer": TOKENIZER_VERSION,
            "k1": self.config.k1,
            "b": self.config.b,
            "title_weight": self.config.title_weight,
            "query_term_policy": "sum contribution once per query-token occurrence",
            "field_policy": "weighted_tf=body_tf+title_weight*title_tf",
        }
        config_hash = hashlib.sha256(_canonical_json_bytes(index_config)).hexdigest()
        corpus_fingerprint = digest.hexdigest()
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "retriever": "okapi_bm25",
            **index_config,
            "config_hash": config_hash,
            "formula": "idf=log(1+(N-df+0.5)/(df+0.5)); standard BM25 tf normalization",
            "document_count": document_count,
            "vocabulary_size": len(self._postings),
            "posting_count": sum(len(values) for values in self._postings.values()),
            "average_weighted_document_length": self._average_length,
            "corpus_fingerprint": corpus_fingerprint,
            "index_fingerprint": hashlib.sha256(
                f"{corpus_fingerprint}\0{config_hash}".encode("ascii")
            ).hexdigest(),
        }

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    def search(self, query_text: str, *, top_k: int = DEFAULT_TOP_K) -> tuple[BM25Hit, ...]:
        if top_k < 0:
            raise BM25Error("top_k must be nonnegative")
        if top_k == 0 or not self._documents:
            return ()
        query_terms = Counter(tokenize(query_text, query=True))
        if not query_terms or self._average_length <= 0:
            return ()
        scores: dict[int, float] = defaultdict(float)
        k1 = self.config.k1
        b = self.config.b
        for term, query_frequency in query_terms.items():
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self._idf[term]
            for document_index, term_frequency in postings:
                length_ratio = self._lengths[document_index] / self._average_length
                denominator = term_frequency + k1 * (1.0 - b + b * length_ratio)
                contribution = (
                    idf
                    * (term_frequency * (k1 + 1.0) / denominator)
                    * query_frequency
                )
                scores[document_index] += contribution
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._documents[item[0]].doc_id),
        )[:top_k]
        return tuple(
            BM25Hit(self._documents[document_index].doc_id, rank, score)
            for rank, (document_index, score) in enumerate(ranked, start=1)
        )

    def search_views(
        self,
        query_views: Sequence[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        per_view_top_k: int = DEFAULT_TOP_K,
        k_view: float = 60.0,
    ) -> MultiViewSearch:
        """Search deduplicated views and average within-branch RRF contributions."""

        if top_k < 0 or per_view_top_k < 0:
            raise BM25Error("candidate limits must be nonnegative")
        if not math.isfinite(k_view) or k_view < 0:
            raise BM25Error("k_view must be finite and nonnegative")
        unique: dict[tuple[str, ...], str] = {}
        for view in query_views:
            token_key = tokenize(view, query=True)
            if token_key and token_key not in unique:
                unique[token_key] = view
        if not unique or top_k == 0 or per_view_top_k == 0:
            return MultiViewSearch((), len(unique), 0, 0)
        contributions: dict[str, float] = defaultdict(float)
        intermediate_count = 0
        for view in unique.values():
            hits = self.search(view, top_k=per_view_top_k)
            intermediate_count += len(hits)
            for hit in hits:
                contributions[hit.doc_id] += 1.0 / (k_view + hit.rank)
        divisor = len(unique)
        combined = sorted(
            ((doc_id, value / divisor) for doc_id, value in contributions.items()),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        hits = tuple(
            BM25Hit(doc_id, rank, score)
            for rank, (doc_id, score) in enumerate(combined, start=1)
        )
        return MultiViewSearch(
            hits,
            unique_view_count=len(unique),
            search_count=len(unique),
            intermediate_candidate_count=intermediate_count,
        )


def load_corpus(path: Path) -> list[BM25Document]:
    documents: list[BM25Document] = []
    seen: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            doc_id = _clean_id(
                _first(record, ("doc_id", "docid", "document_id")), "doc_id"
            )
            if doc_id in seen:
                raise BM25Error(f"Duplicate document ID {doc_id!r}")
            seen.add(doc_id)
            raw_text = record.get("text")
            if raw_text is None:
                raise BM25Error(f"Document {doc_id!r} is missing text")
            text = unicodedata.normalize("NFC", str(raw_text))
            title = unicodedata.normalize("NFC", str(record.get("title", "")))
            if not text.strip() and not title.strip():
                raise BM25Error(f"Document {doc_id!r} has no indexable fields")
            if "\x00" in text or "\x00" in title:
                raise BM25Error(f"Document {doc_id!r} contains a NUL byte")
            language = canonical_language(str(record.get("language", "")))
            source = str(record.get("source", "unspecified"))
            documents.append(BM25Document(doc_id, title, text, language, source))
        except (BM25Error, CorpusPreparationError) as exc:
            raise BM25Error(f"{path}:{line_number}: {exc}") from exc
    return documents


def load_retrieval_queries(path: Path, *, split: str | None = None) -> list[RetrievalQuery]:
    if split is not None and split not in ALL_SPLITS:
        raise BM25Error(f"Unknown split {split!r}")
    queries: list[RetrievalQuery] = []
    seen: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            query_id = _clean_id(
                _first(record, ("query_id", "queryid", "qid")), "query_id"
            )
            if query_id in seen:
                raise BM25Error(f"Duplicate query ID {query_id!r}")
            seen.add(query_id)
            query_split_value = record.get("split")
            query_split = str(query_split_value) if query_split_value is not None else None
            if split is not None and query_split != split:
                continue
            raw_text = _first(record, ("text", "query", "query_text"))
            if raw_text is None:
                raise BM25Error(f"Query {query_id!r} is missing text")
            text = unicodedata.normalize("NFC", str(raw_text))
            # Validate safety limits now rather than halfway through a run.
            tokenize(text, query=True)
            queries.append(RetrievalQuery(query_id, text, query_split))
        except (BM25Error, CorpusPreparationError) as exc:
            raise BM25Error(f"{path}:{line_number}: {exc}") from exc
    if not queries:
        suffix = f" for split {split!r}" if split else ""
        raise BM25Error(f"No retrieval queries found in {path}{suffix}")
    return queries


def run_queries(
    index: BM25Index,
    queries: Sequence[RetrievalQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
    branch: str = "bm25_original",
) -> tuple[RetrievalResult, ...]:
    """Run original query text through a shared index in stable query order."""

    branch = _clean_id(branch, "branch")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise BM25Error("Duplicate query IDs in retrieval input")
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                )
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
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
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


def write_run_artifacts(
    results: Sequence[RetrievalResult],
    run_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    existing = [path for path in (run_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise BM25Error(
            "Refusing to overwrite existing artifact(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --force"
        )
    _atomic_write_jsonl(run_path, (asdict(result) for result in results))
    _atomic_write_json(manifest_path, manifest)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=ALL_SPLITS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--branch", default="bm25_original")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--title-weight", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.top_k <= 0:
            raise BM25Error("top-k must be positive")
        config = BM25Config(args.k1, args.b, args.title_weight)
        documents = load_corpus(args.corpus)
        queries = load_retrieval_queries(args.queries, split=args.split)
        build_started = time.perf_counter()
        index = BM25Index(documents, config)
        build_seconds = time.perf_counter() - build_started
        search_started = time.perf_counter()
        branch = _clean_id(args.branch, "branch")
        results = run_queries(
            index,
            queries,
            top_k=args.top_k,
            branch=branch,
        )
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
            "search_count": len(queries),
            "intermediate_candidate_count": len(results),
            "result_count": len(results),
            "build_seconds": build_seconds,
            "search_seconds": search_seconds,
            "latency_includes_view_generation": False,
        }
        write_run_artifacts(
            results,
            args.output,
            manifest,
            manifest_path,
            overwrite=args.force,
        )
    except (BM25Error, CorpusPreparationError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"run": str(args.output), "manifest": str(manifest_path)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
