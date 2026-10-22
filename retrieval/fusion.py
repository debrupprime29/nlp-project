"""Bounded multi-view retrieval and rank-only reciprocal rank fusion.

Both lexical branches share one original-text BM25 index.  Native views are
masked when token-equivalent to the original; missing branches contribute no
weight.  This module supports fixed weights and an inference weight-provider
hook; fitting a learned gate belongs to Module 10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import CorpusPreparationError, _clean_id, iter_jsonl
from nlp.language_id import LanguageIdError, TokenLanguageIdentifier
from nlp.query_views import (
    LexiconTransliterator, QueryViewError, QueryViews, ViewConfig, _metadata, build_query_views,
)
from retrieval.bm25 import (
    BM25Error, BM25Index, RetrievalQuery, RetrievalResult, _atomic_write_jsonl,
    _file_sha256, load_corpus as load_bm25_corpus, tokenize, write_run_artifacts,
)
from retrieval.char_tfidf import (
    CharTfidfError, CharTfidfIndex, load_corpus as load_char_corpus,
)
from retrieval.dense import (
    DenseConfig, DenseError, DenseIndex, INDEX_MANIFEST_NAME,
    _add_encoder_arguments, _encoder_from_arguments,
)


SCHEMA_VERSION = 1
BRANCHES = ("bm25_original", "bm25_native", "char_tfidf", "dense")


class FusionError(ValueError):
    """Invalid rankings, masks, weights, budgets, or incompatible indexes."""


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class FusionConfig:
    per_branch_top_k: int = 100
    per_view_top_k: int = 100
    final_top_k: int = 10
    k_view: float = 60.0
    k: float = 60.0

    def validate(self) -> None:
        for name in ("per_branch_top_k", "per_view_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
                raise FusionError(f"{name} must be an integer in [1, 100]")
        if isinstance(self.final_top_k, bool) or not isinstance(self.final_top_k, int) or not 0 <= self.final_top_k <= 400:
            raise FusionError("final_top_k must be an integer in [0, 400]")
        for name in ("k_view", "k"):
            value = getattr(self, name)
            if not _finite(value) or value < 0:
                raise FusionError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BranchHit:
    doc_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    doc_id: str
    rank: int
    score: float
    contributions: Mapping[str, float]


@dataclass(frozen=True)
class FusionResult:
    hits: tuple[FusedHit, ...]
    branch_results: Mapping[str, tuple[BranchHit, ...]]
    active_weights: Mapping[str, float]
    diagnostics: Mapping[str, Any]


def _ranking(hits: Sequence[Any], *, label: str) -> tuple[BranchHit, ...]:
    if isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence):
        raise FusionError(f"{label} ranking must be a sequence")
    checked: list[BranchHit] = []
    seen: set[str] = set()
    for expected_rank, hit in enumerate(hits, start=1):
        def field(name: str) -> Any:
            return hit.get(name) if isinstance(hit, Mapping) else getattr(hit, name, None)
        doc_id, rank, score = field("doc_id"), field("rank"), field("score")
        if not isinstance(doc_id, str) or not doc_id or doc_id != doc_id.strip() or any(ord(character) < 32 for character in doc_id):
            raise FusionError(f"{label} contains an invalid document ID")
        if doc_id in seen:
            raise FusionError(f"{label} contains duplicate document ID {doc_id!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
            raise FusionError(f"{label} ranks must be contiguous and ordered from 1")
        if not _finite(score):
            raise FusionError(f"{label} scores must be finite numbers")
        seen.add(doc_id)
        checked.append(BranchHit(doc_id, rank, float(score)))
    return tuple(checked)


def _view_key(text: str) -> str:
    if not isinstance(text, str) or "\x00" in text:
        raise FusionError("query views must be strings without NUL characters")
    return " ".join(unicodedata.normalize("NFC", text).split())


def average_rankings(
    view_rankings: Mapping[str, Sequence[Any]], *, config: FusionConfig = FusionConfig(),
) -> tuple[BranchHit, ...]:
    """Average per-view reciprocal ranks; raw scores never enter the sum.

    NFC and whitespace-equivalent view strings are deduplicated, preserving
    case.  Empty rankings remain in the divisor, as required by missing-zero
    average RRF.  The first equivalent view is retained.
    """
    config.validate()
    if not isinstance(view_rankings, Mapping):
        raise FusionError("view_rankings must be a mapping")
    unique: dict[str, tuple[BranchHit, ...]] = {}
    for text, hits in view_rankings.items():
        key = _view_key(text)
        checked = _ranking(hits, label="view")[:config.per_view_top_k]
        if key:
            unique.setdefault(key, checked)
    if not unique:
        return ()
    values: dict[str, float] = defaultdict(float)
    for hits in unique.values():
        for hit in hits:
            values[hit.doc_id] += 1.0 / (config.k_view + hit.rank) / len(unique)
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:config.per_branch_top_k]
    return tuple(BranchHit(doc_id, rank, score) for rank, (doc_id, score) in enumerate(ordered, 1))


def _weights(weights: Mapping[str, float] | None, active: Sequence[str]) -> dict[str, float]:
    supplied = {branch: 1.0 for branch in BRANCHES} if weights is None else weights
    if not isinstance(supplied, Mapping) or set(supplied) != set(BRANCHES):
        raise FusionError("explicit weights must contain exactly all four branch names")
    if any(not _finite(value) or value < 0 for value in supplied.values()):
        raise FusionError("branch weights must be finite and nonnegative")
    masked = {branch: float(supplied[branch]) for branch in active}
    if not masked:
        return {}
    # Scale first so even large finite user weights cannot overflow the sum.
    scale = max(masked.values())
    if scale <= 0:
        raise FusionError("active branches must receive positive total weight")
    scaled = {branch: weight / scale for branch, weight in masked.items()}
    total = math.fsum(scaled.values())
    return {branch: weight / total for branch, weight in scaled.items()}


def fuse_rankings(
    branch_rankings: Mapping[str, Sequence[Any]], weights: Mapping[str, float] | None = None,
    *, config: FusionConfig = FusionConfig(),
) -> FusionResult:
    """Fuse branch ranks after masking empty branches and normalizing weights."""
    config.validate()
    if not isinstance(branch_rankings, Mapping) or set(branch_rankings) - set(BRANCHES):
        raise FusionError("rankings must map only the four known branch names")
    rankings = {branch: _ranking(branch_rankings.get(branch, ()), label=branch)[:config.per_branch_top_k] for branch in BRANCHES}
    active = tuple(branch for branch in BRANCHES if rankings[branch])
    normalized = _weights(weights, active)
    contributions: dict[str, dict[str, float]] = defaultdict(dict)
    candidate_union = {hit.doc_id for hits in rankings.values() for hit in hits}
    for branch in active:
        if normalized[branch] <= 0:
            continue
        for hit in rankings[branch]:
            contributions[hit.doc_id][branch] = normalized[branch] / (config.k + hit.rank)
    scored = [(doc_id, math.fsum(parts.values()), parts) for doc_id, parts in contributions.items()]
    scored.sort(key=lambda item: (-item[1], item[0]))
    hits = tuple(FusedHit(doc_id, rank, score, dict(parts)) for rank, (doc_id, score, parts) in enumerate(scored[:config.final_top_k], 1))
    return FusionResult(hits, rankings, normalized, {
        "candidate_union_count": len(candidate_union),
        "branch_candidate_counts": {branch: len(hits) for branch, hits in rankings.items()},
        "fusion_policy": "weighted_reciprocal_rank_only",
    })


def _known_ids(index: Any) -> set[str] | None:
    if index is None:
        return None
    ids = getattr(index, "_doc_ids", None)
    if ids is not None:
        return set(ids)
    documents = getattr(index, "_documents", None)
    return {document.doc_id for document in documents} if documents is not None else None


class RetrievalPipeline:
    """Generate protected views, search four bounded branches, and fuse ranks."""

    def __init__(
        self, bm25_index: Any, char_index: Any = None, dense_index: Any = None,
        language_identifier: Any = None, transliterator: Any = None,
        fusion_config: FusionConfig = FusionConfig(), view_config: ViewConfig = ViewConfig(),
        weight_provider: Callable[[QueryViews, Mapping[str, tuple[BranchHit, ...]]], Mapping[str, float]] | None = None,
    ) -> None:
        fusion_config.validate()
        view_config.validate()
        if bm25_index is None or not callable(getattr(bm25_index, "search", None)):
            raise FusionError("a shared original-text BM25 index is required")
        for name, index in (("character", char_index), ("dense", dense_index)):
            if index is not None and not callable(getattr(index, "search", None)):
                raise FusionError(f"{name} index must support search")
        if weight_provider is not None and not callable(weight_provider):
            raise FusionError("weight_provider must be callable")
        self.bm25_index, self.char_index, self.dense_index = bm25_index, char_index, dense_index
        self.language_identifier, self.transliterator = language_identifier, transliterator
        self.config, self.view_config, self.weight_provider = fusion_config, view_config, weight_provider
        self._corpus_ids = _known_ids(bm25_index)
        if self._corpus_ids is not None:
            for name, index in (("character", char_index), ("dense", dense_index)):
                ids = _known_ids(index)
                if ids is not None and not ids <= self._corpus_ids:
                    raise FusionError(f"{name} index contains documents outside the BM25 corpus")
                if name == "dense" and ids is not None and ids != self._corpus_ids:
                    raise FusionError("dense and BM25 indexes must cover the same corpus")
        bm_manifest = dict(getattr(bm25_index, "manifest", {}))
        dense_manifest = dict(getattr(dense_index, "manifest", {})) if dense_index is not None else {}
        # Both original-text indexes hash the same canonical document fields.
        if bm_manifest.get("corpus_fingerprint") and dense_manifest.get("corpus_fingerprint") and bm_manifest["corpus_fingerprint"] != dense_manifest["corpus_fingerprint"]:
            raise FusionError("dense and BM25 corpus fingerprints differ")
        if callable(getattr(weight_provider, "assert_compatible", None)):
            weight_provider.assert_compatible(self.manifest)

    @property
    def manifest(self) -> Mapping[str, Any]:
        indexes = {name: dict(getattr(index, "manifest", {})) if index is not None else None for name, index in (("bm25", self.bm25_index), ("char_tfidf", self.char_index), ("dense", self.dense_index))}
        dense = indexes["dense"] or {}
        resources = {"language_identifier": _metadata(self.language_identifier, identifier=True), "transliterator": _metadata(self.transliterator)}
        payload = {"schema_version": SCHEMA_VERSION, "retriever": "multi_view_rrf", "branches": list(BRANCHES), "config": asdict(self.config), "view_config": asdict(self.view_config), "indexes": indexes, "resources": resources, "fixture_only": bool(dense.get("encoder", {}).get("fixture_only", False)), "gate_training_implemented": True}
        provider_metadata = getattr(self.weight_provider, "metadata", None)
        if provider_metadata is not None:
            payload["weight_provider"] = dict(provider_metadata)
        return {**payload, "pipeline_fingerprint": _fingerprint(payload)}

    def search(self, text: str, weights: Mapping[str, float] | None = None) -> FusionResult:
        started = time.perf_counter()
        views = build_query_views(text, self.language_identifier, self.transliterator, config=self.view_config)
        view_seconds = time.perf_counter() - started
        original_key = tokenize(views.original.text, query=True)
        native: dict[tuple[str, ...], str] = {}
        for view in views.native_views:
            key = tokenize(view.text, query=True)
            if key and key != original_key:
                native.setdefault(key, view.text)
        dense_views: dict[str, str] = {}
        for text_view in (views.original.text, *(view.text for view in views.native_views)):
            key = unicodedata.normalize("NFC", text_view)
            if key.strip():
                dense_views.setdefault(key, key)
        branches: dict[str, tuple[BranchHit, ...]] = {}
        traces: dict[str, Any] = {}
        def search_branch(branch: str, index: Any, query_texts: Sequence[str], reason: str | None = None) -> None:
            begin = time.perf_counter()
            per_view: dict[str, Sequence[Any]] = {}
            intermediate = 0
            if index is not None and getattr(index, "document_count", 1) > 0:
                for query_text in query_texts:
                    raw = _ranking(index.search(query_text, top_k=self.config.per_view_top_k), label=branch)
                    if len(raw) > self.config.per_view_top_k:
                        raise FusionError(f"{branch} exceeded its per-view candidate budget")
                    if self._corpus_ids is not None and any(hit.doc_id not in self._corpus_ids for hit in raw):
                        raise FusionError(f"{branch} retrieved a document outside the indexed corpus")
                    index_ids = _known_ids(index)
                    if index_ids is not None and any(hit.doc_id not in index_ids for hit in raw):
                        raise FusionError(f"{branch} retrieved a document outside its own index")
                    intermediate += len(raw)
                    per_view[query_text] = raw
            else:
                reason = reason or ("index_not_supplied" if index is None else "empty_index")
            # Dense views preserve case and exact NFC whitespace, matching E5.
            # Give the already deduplicated views opaque keys so average_rankings
            # cannot merge distinct model inputs due to whitespace differences.
            if branch == "dense":
                per_view = {f"view:{number}": hits for number, hits in enumerate(per_view.values())}
            hits = average_rankings(per_view, config=self.config)
            branches[branch] = hits
            traces[branch] = {"views": list(query_texts), "unique_view_count": len(query_texts), "search_count": len(per_view), "intermediate_candidate_count": intermediate, "candidate_count": len(hits), "seconds": time.perf_counter() - begin, "unavailable_reason": reason or ("empty_results" if not hits else None)}
        retrieval_start = time.perf_counter()
        search_branch("bm25_original", self.bm25_index, (views.original.text,) if original_key else (), "empty_query" if not original_key else None)
        search_branch("bm25_native", self.bm25_index, tuple(native.values()), "no_distinct_native_view" if not native else None)
        roman = views.roman_view.text if views.roman_view is not None else None
        char_available = roman is not None and bool(roman.strip()) and self.char_index is not None
        if char_available and callable(getattr(self.char_index, "query_available", None)):
            char_available = self.char_index.query_available(roman)
        search_branch("char_tfidf", self.char_index, (roman,) if char_available else (), "no_available_roman_view" if not char_available else None)
        search_branch("dense", self.dense_index, tuple(dense_views.values()))
        retrieval_seconds = time.perf_counter() - retrieval_start
        weight_source = "explicit" if weights is not None else "equal"
        if weights is None and self.weight_provider is not None:
            if callable(getattr(self.weight_provider, "assert_compatible", None)):
                self.weight_provider.assert_compatible(self.manifest)
            weights = self.weight_provider(views, branches)
            weight_source = "provider"
        fusion_start = time.perf_counter()
        fused = fuse_rankings(branches, weights, config=self.config)
        fusion_seconds = time.perf_counter() - fusion_start
        diagnostics = {**fused.diagnostics, "search_count": sum(value["search_count"] for value in traces.values()), "intermediate_candidate_count": sum(value["intermediate_candidate_count"] for value in traces.values()), "view_generation_seconds": view_seconds, "retrieval_seconds": retrieval_seconds, "fusion_seconds": fusion_seconds, "total_seconds": time.perf_counter() - started, "branch_diagnostics": traces, "views": {"original": views.original_text, "processing": views.processing_text, "native": [view.text for view in views.native_views], "roman": roman, "diagnostics": dict(views.diagnostics)}, "weight_source": weight_source, "fixture_only": self.manifest["fixture_only"]}
        return FusionResult(fused.hits, fused.branch_results, fused.active_weights, diagnostics)

    def run_queries(
        self, queries: Sequence[Any], *, weights: Mapping[str, float] | None = None,
        branch: str = "fusion_fixed",
    ) -> tuple[RetrievalResult, ...]:
        try:
            label = _clean_id(branch, "branch")
            ids = [_clean_id(query.query_id, "query_id") for query in queries]
        except CorpusPreparationError as exc:
            raise FusionError(str(exc)) from exc
        if len(ids) != len(set(ids)):
            raise FusionError("duplicate query IDs in retrieval input")
        if self.manifest["fixture_only"] and "fixture" not in label:
            raise FusionError("fusion with a fixture encoder requires a fixture run label")
        return tuple(RetrievalResult(query.query_id, hit.doc_id, label, hit.rank, hit.score) for query in sorted(queries, key=lambda item: item.query_id) for hit in self.search(query.text, weights).hits)


def run_queries(pipeline: RetrievalPipeline, queries: Sequence[Any], **kwargs: Any) -> tuple[RetrievalResult, ...]:
    return pipeline.run_queries(queries, **kwargs)


def load_queries(path: Path, *, split: str) -> list[RetrievalQuery]:
    """Load exact original text; processing normalization happens in the views."""
    if split not in ALL_SPLITS:
        raise FusionError("unknown retrieval split")
    queries: list[RetrievalQuery] = []
    seen: set[str] = set()
    for record in iter_jsonl(path):
        try:
            query_id = _clean_id(record.get("query_id", record.get("qid")), "query_id")
        except CorpusPreparationError as exc:
            raise FusionError(str(exc)) from exc
        if query_id in seen:
            raise FusionError(f"duplicate query ID {query_id!r}")
        seen.add(query_id)
        if record.get("split") != split:
            continue
        text = record.get("text", record.get("query_text", record.get("query")))
        if not isinstance(text, str):
            raise FusionError(f"query {query_id!r} is missing string text")
        queries.append(RetrievalQuery(query_id, text, split))
    if not queries:
        raise FusionError(f"no queries for split {split!r}")
    return sorted(queries, key=lambda query: query.query_id)


def _parse_weights(value: str | None) -> Mapping[str, float] | None:
    if value is None:
        return None
    try:
        result = json.loads(value if value.lstrip().startswith("{") else Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FusionError(f"could not read branch weights: {exc}") from exc
    _weights(result, BRANCHES)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--split", choices=ALL_SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--weights", help="JSON object or path with all four branch weights; use explicit zero to disable a branch")
    parser.add_argument("--gate-model", type=Path, help="Module 10 learned gate; incompatible frozen pipelines are rejected")
    parser.add_argument("--lid-model", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-branch-top-k", type=int, default=100)
    parser.add_argument("--per-view-top-k", type=int, default=100)
    parser.add_argument("--k-view", type=float, default=60.0)
    parser.add_argument("--k", type=float, default=60.0)
    parser.add_argument("--branch")
    parser.add_argument("--force", action="store_true")
    _add_encoder_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = FusionConfig(args.per_branch_top_k, args.per_view_top_k, args.top_k, args.k_view, args.k)
        config.validate()
        weights = _parse_weights(args.weights)
        if weights is not None and args.gate_model is not None:
            raise FusionError("choose --weights or --gate-model, not both")
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        paths = [args.output, manifest_path] + ([args.trace] if args.trace else [])
        if len({path.resolve() for path in paths}) != len(paths):
            raise FusionError("run, manifest, and trace paths must be distinct")
        inputs = [args.corpus, args.queries, args.lid_model, args.lexicon, args.gate_model]
        if args.weights is not None and not args.weights.lstrip().startswith("{"):
            inputs.append(Path(args.weights))
        if args.dense_index:
            inputs.extend((args.dense_index / INDEX_MANIFEST_NAME, args.dense_index / "vectors.jsonl"))
        if args.checkpoint is not None:
            inputs.extend(path for path in args.checkpoint.rglob("*") if path.is_file())
        for directory in (args.checkpoint, args.dense_index):
            if directory is not None and any(path.resolve().is_relative_to(directory.resolve()) for path in paths):
                raise FusionError("outputs must not be inside frozen checkpoint or dense-index directories")
        if {path.resolve() for path in paths}.intersection(path.resolve() for path in inputs if path is not None):
            raise FusionError("output paths must not overwrite input artifacts")
        if not args.force and any(path.exists() for path in paths):
            raise FusionError("refusing to overwrite existing artifacts; pass --force")
        dense = None
        if args.dense_index:
            dense = DenseIndex.load(args.dense_index, _encoder_from_arguments(args), expected_config=DenseConfig(max_length=args.max_length, batch_size=args.batch_size))
        pipeline = RetrievalPipeline(BM25Index(load_bm25_corpus(args.corpus)), CharTfidfIndex(load_char_corpus(args.corpus)), dense, TokenLanguageIdentifier.load_json(args.lid_model) if args.lid_model else None, LexiconTransliterator.from_jsonl(args.lexicon) if args.lexicon else None, config)
        if args.gate_model:
            from training.train_gate import LinearSoftmaxGate

            pipeline.weight_provider = LinearSoftmaxGate.load_json(args.gate_model, expected_pipeline_manifest=pipeline.manifest)
        queries = load_queries(args.queries, split=args.split)
        fixture = pipeline.manifest["fixture_only"]
        label = args.branch or ("fusion_adaptive_fixture" if fixture and args.gate_model else "fusion_fixture" if fixture else "fusion_adaptive" if args.gate_model else "fusion_fixed")
        if fixture and "fixture" not in label:
            raise FusionError("fusion with a fixture encoder requires a fixture run label")
        _clean_id(label, "branch")
        results: list[RetrievalResult] = []
        traces = []
        for query in queries:
            result = pipeline.search(query.text, weights)
            results.extend(RetrievalResult(query.query_id, hit.doc_id, label, hit.rank, hit.score) for hit in result.hits)
            traces.append({"query_id": query.query_id, "active_weights": dict(result.active_weights), "hits": [asdict(hit) for hit in result.hits], "branch_results": {branch: [asdict(hit) for hit in hits] for branch, hits in result.branch_results.items()}, "diagnostics": dict(result.diagnostics)})
        manifest = {"schema_version": SCHEMA_VERSION, "pipeline": pipeline.manifest, "branch": label, "split": args.split, "weights": dict(weights) if weights is not None else "learned_gate" if args.gate_model else "equal", "query_count": len(queries), "result_count": len(results), "fixture_only": fixture, "input_sha256": {str(path): _file_sha256(path) for path in inputs if path is not None}, "search_count": sum(trace["diagnostics"]["search_count"] for trace in traces), "intermediate_candidate_count": sum(trace["diagnostics"]["intermediate_candidate_count"] for trace in traces), "latency_seconds": {key: sum(trace["diagnostics"][key] for trace in traces) for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds")}, "trace": str(args.trace) if args.trace else None}
        write_run_artifacts(results, args.output, manifest, manifest_path, overwrite=args.force)
        if args.trace:
            _atomic_write_jsonl(args.trace, traces)
    except (ValueError, BM25Error, CharTfidfError, DenseError, QueryViewError, LanguageIdError, CorpusPreparationError, OSError, TypeError) as exc:
        print(f"fusion: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run": str(args.output), "manifest": str(manifest_path), "trace": str(args.trace) if args.trace else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
