"""Replay sealed fusion traces for a public-data, original-query CPU pilot.

No retriever/model is run here. Fixed weights are selected only on validation;
final evaluation requires that frozen, content-sealed selection. Stored serving
rankings are checked at their original depth, then rank fusion is replayed at
100 for Recall@100 without changing the serving/gate pipeline fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import _parse_relevance, iter_jsonl, Qrel
from evaluation.evaluate import (
    EvaluationReport, RunEntry, bootstrap_parent_groups, evaluate_run,
    load_evaluation_queries, load_run,
)
from retrieval.fusion import BRANCHES, FusionConfig, _ranking, fuse_rankings


SCHEMA_VERSION = 1
SCOPE = "public_cpu_verified_original_query_pilot"


class StudyError(ValueError):
    """Unsafe inputs, incompatible provenance, or leakage-prone selection."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _weights(value: Any) -> dict[str, float]:
    if (not isinstance(value, Mapping) or set(value) != set(BRANCHES)
            or any(not _finite(v) or v < 0 for v in value.values())
            or not math.isclose(math.fsum(value.values()), 1.0, abs_tol=1e-9)):
        raise StudyError("Weights must contain four finite nonnegative values summing to one")
    return {branch: float(value[branch]) for branch in BRANCHES}


def _safe_path(path: Path) -> Path:
    absolute = Path(path).absolute()
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise StudyError("Symlink inputs/outputs are not supported by sealed studies")
    return absolute.resolve()


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value) + b"\n")


def _write_run(path: Path, entries: Sequence[RunEntry]) -> None:
    with path.open("xb") as handle:
        for entry in entries:
            handle.write(_canonical(asdict(entry)) + b"\n")


def _queries(path: Path, split: str) -> tuple[list[Any], dict[str, Mapping[str, Any]]]:
    records = list(iter_jsonl(path))
    by_id: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, tuple[str, str]] = {}
    texts: dict[tuple[str, str], str] = {}
    for row in records:
        qid, group = row.get("query_id"), row.get("group_id")
        language, partition = row.get("benchmark_language"), row.get("split")
        if (not isinstance(qid, str) or not qid.strip() or qid != qid.strip() or qid in by_id
                or not isinstance(group, str) or not group.strip() or group != group.strip()
                or any(ord(character) < 32 for character in qid + group)
                or language not in ("en", "hi", "bn") or partition not in ALL_SPLITS):
            raise StudyError("Query IDs/groups/languages/splits must be canonical and unique")
        if (row.get("verified") is not True or row.get("review_status") != "verified"
                or row.get("variant_type") != "original" or row.get("parent_query_id") != qid
                or row.get("judgment_query_id") != qid or row.get("languages") != [language]):
            raise StudyError("This public study accepts only verified monolingual original queries")
        expected_source = "official_dev" if partition == "final_evaluation" else "official_train"
        if row.get("source_partition") != expected_source:
            raise StudyError("Official development queries must remain locked to final evaluation")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or "\0" in text:
            raise StudyError("Queries require nonempty text without NUL")
        signature = (partition, language)
        if group in groups and groups[group] != signature:
            raise StudyError("Information-need groups cannot cross splits or languages")
        groups[group] = signature
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        text_key = (language, normalized)
        if text_key in texts and texts[text_key] != group:
            raise StudyError("Exact normalized duplicate queries must share one information-need group")
        texts[text_key] = group
        by_id[qid] = row
    selected = load_evaluation_queries(path, split=split)
    return sorted(selected, key=lambda q: q.query_id), by_id


def _scoped_qrels(path: Path, wanted: set[str], corpus_ids: set[str]) -> list[Qrel]:
    # Ignore labels outside the evaluated partition before parsing relevance or
    # document IDs. Validation selection never consumes final query utilities.
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = iter_jsonl(path)
    else:
        def text_rows():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if not fields or fields[0].startswith("#"):
                        continue
                    if fields[0] not in wanted:
                        continue
                    if len(fields) not in (3, 4):
                        raise StudyError("Malformed selected qrel row")
                    yield {"query_id": fields[0], "doc_id": fields[-2], "relevance": fields[-1]}
        rows = text_rows()
    result = []
    seen = set()
    positive = set()
    for row in rows:
        qid = row.get("query_id", row.get("qid"))
        if qid not in wanted:
            continue
        doc = row.get("doc_id", row.get("docid"))
        relevance = _parse_relevance(row.get("relevance", row.get("rel")), context=str(path))
        if doc not in corpus_ids or (qid, doc) in seen:
            raise StudyError("Selected qrels must be unique and reference the fixed corpus")
        seen.add((qid, doc))
        result.append(Qrel(qid, doc, relevance))
        if relevance > 0:
            positive.add(qid)
    if positive != wanted:
        raise StudyError("Every selected original query requires positive judgments")
    return sorted(result, key=lambda q: (q.query_id, q.doc_id))


def _contains_fixture(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("fixture_only") is True or any(_contains_fixture(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_fixture(v) for v in value)
    return False


def _original_dense(run_path, metadata_path, *, pipeline, queries, queries_path, query_sha, split, corpus_ids, fixture):
    metadata = _json(metadata_path)
    if (metadata.get("index") != pipeline.get("indexes", {}).get("dense")
            or metadata.get("selected_split") != split
            or metadata.get("queries_file_sha256") != query_sha
            or Path(metadata.get("queries_path", "")).resolve() != queries_path
            or metadata.get("query_count") != len(queries)
            or metadata.get("fixture_only") != fixture
            or type(metadata.get("top_k")) is not int or not 1 <= metadata["top_k"] <= 100):
        raise StudyError("Original-only dense run has incompatible frozen index/query/split provenance")
    entries = load_run(run_path)
    wanted = {query.query_id for query in queries}
    counts = Counter()
    for entry in entries:
        if entry.query_id not in wanted or entry.doc_id not in corpus_ids or entry.branch != metadata.get("branch"):
            raise StudyError("Original-only dense run has unknown IDs or branch provenance")
        counts[entry.query_id] += 1
    if any(count > metadata["top_k"] for count in counts.values()) or metadata.get("result_count") != len(entries):
        raise StudyError("Original-only dense result count/depth mismatch")
    if not _finite(metadata.get("search_seconds")) or metadata["search_seconds"] < 0:
        raise StudyError("Original-only dense measured search costs are required")
    if type(metadata.get("search_count")) is not int or not 0 <= metadata["search_count"] <= len(queries):
        raise StudyError("Original-only dense measured search count is invalid")
    return entries, {key: metadata.get(key) for key in ("search_count", "search_seconds", "query_count", "result_count", "latency_includes_view_generation")}


def _input_hash(manifest: Mapping[str, Any], path: Path) -> str:
    hashes = manifest.get("input_sha256")
    if not isinstance(hashes, Mapping):
        raise StudyError("Fusion manifest lacks input SHA-256 provenance")
    matches = [value for name, value in hashes.items() if Path(name).resolve() == path]
    if len(matches) != 1 or matches[0] != _sha256(path):
        raise StudyError(f"Fusion input hash/path mismatch: {path}")
    return matches[0]


def _traces(path: Path, manifest: Mapping[str, Any], query_rows: Mapping[str, Any],
            corpus_ids: set[str], config: FusionConfig, fixture: bool) -> list[dict[str, Any]]:
    traces = []
    seen = set()
    for trace in iter_jsonl(path):
        qid = trace.get("query_id")
        if qid not in query_rows or qid in seen:
            raise StudyError("Trace requires exactly one record per evaluated query")
        seen.add(qid)
        raw = trace.get("branch_results")
        if not isinstance(raw, Mapping) or set(raw) != set(BRANCHES):
            raise StudyError("Trace must contain exactly four branch rankings")
        branches = {}
        for branch in BRANCHES:
            hits = _ranking(raw[branch], label=branch)
            if len(hits) > config.per_branch_top_k or len(hits) > 100:
                raise StudyError("Stored branch ranking exceeds the candidate budget")
            if any(hit.doc_id not in corpus_ids or hit.score < 0 for hit in hits):
                raise StudyError("Branch hits must reference the fixed corpus with nonnegative scores")
            if any(a.score < b.score for a, b in zip(hits, hits[1:])):
                raise StudyError("Branch scores must be nonincreasing in rank order")
            branches[branch] = hits
        active = {branch for branch in BRANCHES if branches[branch]}
        weights = trace.get("active_weights")
        if (not isinstance(weights, Mapping) or set(weights) != active
                or any(not _finite(v) or not 0 <= v <= 1 for v in weights.values())
                or (active and not math.isclose(math.fsum(weights.values()), 1, abs_tol=1e-9))):
            raise StudyError("Trace active weights must match nonempty branches and sum to one")
        complete_weights = {branch: float(weights.get(branch, 0)) for branch in BRANCHES}
        served = fuse_rankings(branches, complete_weights, config=config)
        stored = trace.get("hits")
        checked = _ranking(stored, label="served")
        if len(checked) != len(served.hits):
            raise StudyError("Serving trace depth disagrees with its frozen fusion configuration")
        for actual, expected, original in zip(checked, served.hits, stored):
            if actual.doc_id != expected.doc_id or not math.isclose(actual.score, expected.score, rel_tol=1e-10, abs_tol=1e-12):
                raise StudyError("Stored serving hits disagree with branch ranks and weights")
            contributions = original.get("contributions")
            if (not isinstance(contributions, Mapping) or set(contributions) != set(expected.contributions)
                    or any(not _finite(contributions[b]) or not math.isclose(contributions[b], v, rel_tol=1e-10, abs_tol=1e-12) for b, v in expected.contributions.items())):
                raise StudyError("Stored fusion contributions disagree with replay")
        diagnostics = trace.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("fixture_only") != fixture:
            raise StudyError("Trace diagnostics/fixture provenance missing or inconsistent")
        views = diagnostics.get("views")
        if (not isinstance(views, Mapping) or views.get("original") != query_rows[qid]["text"]
                or not isinstance(views.get("native"), list) or len(views["native"]) > 2
                or any(not isinstance(v, str) or "\0" in v for v in views["native"])):
            raise StudyError("Trace views do not match the exact original query")
        branch_info = diagnostics.get("branch_diagnostics")
        if not isinstance(branch_info, Mapping) or set(branch_info) != set(BRANCHES):
            raise StudyError("Per-branch measured call costs are required")
        for branch, info in branch_info.items():
            for key in ("search_count", "unique_view_count", "intermediate_candidate_count", "candidate_count"):
                if type(info.get(key)) is not int or info[key] < 0:
                    raise StudyError("Branch call/candidate counters must be nonnegative integers")
            if (not _finite(info.get("seconds")) or info["seconds"] < 0
                    or info["candidate_count"] != len(branches[branch])
                    or info["search_count"] > info["unique_view_count"]
                    or info["search_count"] > {"bm25_original": 1, "bm25_native": 2, "char_tfidf": 1, "dense": 3}[branch]
                    or not len(branches[branch]) <= info["intermediate_candidate_count"] <= info["search_count"] * config.per_view_top_k):
                raise StudyError("Branch measured costs violate stored budgets/rankings")
            if (not isinstance(info.get("views"), list) or any(not isinstance(v, str) for v in info["views"])
                    or len(info["views"]) != info["unique_view_count"]
                    or info["unique_view_count"] > {"bm25_original": 1, "bm25_native": 2, "char_tfidf": 1, "dense": 3}[branch]):
                raise StudyError("Branch view counts/types violate the trace contract")
            if not branches[branch] and not info.get("unavailable_reason"):
                raise StudyError("Masked branch requires an explicit unavailable reason")
        original_view = unicodedata.normalize("NFC", query_rows[qid]["text"])
        expected_dense_views = list(dict.fromkeys(unicodedata.normalize("NFC", v)
                                                 for v in [original_view, *views["native"]] if v.strip()))
        if (branch_info["dense"]["views"] != expected_dense_views
                or branch_info["bm25_original"]["views"] not in ([], [original_view])
                or any(v not in views["native"] for v in branch_info["bm25_native"]["views"])
                or branch_info["char_tfidf"]["views"] not in ([], [views.get("roman")])):
            raise StudyError("Per-branch inputs disagree with the protected query views")
        for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds"):
            if not _finite(diagnostics.get(key)) or diagnostics[key] < 0:
                raise StudyError("Measured trace latency must be finite and nonnegative")
        for key in ("search_count", "intermediate_candidate_count"):
            if type(diagnostics.get(key)) is not int or diagnostics[key] != sum(info[key] for info in branch_info.values()):
                raise StudyError("Trace total call/candidate counts disagree with branches")
        declared = manifest.get("weights")
        if declared == "equal" or isinstance(declared, Mapping):
            reference = fuse_rankings(branches, None if declared == "equal" else declared, config=config)
            if any(not math.isclose(weights[b], value, abs_tol=1e-9) for b, value in reference.active_weights.items()):
                raise StudyError("Trace weights disagree with the fusion run manifest")
        elif declared == "learned_gate":
            if diagnostics.get("weight_source") != "provider" or "weight_provider" not in manifest["pipeline"]:
                raise StudyError("Adaptive trace requires a learned provider's frozen provenance")
        else:
            raise StudyError("Unknown fusion weight provenance")
        traces.append({**trace, "branch_results": branches})
    if seen != set(query_rows) or manifest.get("query_count") != len(traces):
        raise StudyError("Trace/query manifest query counts or memberships disagree")
    if manifest.get("result_count") != sum(len(t["hits"]) for t in traces):
        raise StudyError("Fusion manifest serving-result count mismatch")
    for key in ("search_count", "intermediate_candidate_count"):
        if type(manifest.get(key)) is not int or manifest[key] != sum(t["diagnostics"][key] for t in traces):
            raise StudyError("Fusion manifest measured-call totals mismatch")
    for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds"):
        value = manifest.get("latency_seconds", {}).get(key)
        if not _finite(value) or value < 0 or not math.isclose(value, math.fsum(t["diagnostics"][key] for t in traces), rel_tol=1e-9, abs_tol=1e-8):
            raise StudyError("Fusion manifest measured-latency totals mismatch")
    return sorted(traces, key=lambda t: t["query_id"])


def _fused(traces: Sequence[Mapping[str, Any]], weights: Mapping[str, float] | str,
           config: FusionConfig, label: str) -> list[RunEntry]:
    result = []
    for trace in traces:
        current = ({branch: trace["active_weights"].get(branch, 0.0) for branch in BRANCHES}
                   if weights == "trace" else weights)
        # An explicitly disabled branch stays disabled on final queries, even
        # when it is the only branch returning candidates. No adaptive fallback.
        if not any(current[b] > 0 and trace["branch_results"][b] for b in BRANCHES):
            continue
        hits = fuse_rankings(trace["branch_results"], current, config=config).hits
        result.extend(RunEntry(trace["query_id"], hit.doc_id, label, hit.rank, hit.score) for hit in hits)
    return result


def _grid(resolution: int, branches: int = 4):
    if branches == 1:
        yield (resolution,)
    else:
        for value in range(resolution + 1):
            for tail in _grid(resolution - value, branches - 1):
                yield (value, *tail)


def select_fixed_weights(traces, queries, qrels, config: FusionConfig, *, resolution: int = 4) -> Mapping[str, Any]:
    """Select on validation only; this function must never receive final queries."""
    if not queries or any(q.split != "validation" for q in queries):
        raise StudyError("Fixed-weight selection is validation-only")
    if type(resolution) is not int or not 1 <= resolution <= 16:
        raise StudyError("Grid resolution must be an integer in [1,16]")
    # Match training.train_gate.compare_validation: equal is the incumbent;
    # a simplex point must improve language-macro nDCG by more than 1e-12.
    equal = {branch: 0.25 for branch in BRANCHES}
    equal_report = evaluate_run(queries, qrels, _fused(traces, equal, config, "fixed_selection"), bootstrap_samples=0)
    best = {"weights": equal, "language_macro_ndcg_at_10": equal_report.summary["macro_languages"]["ndcg_at_10"]}
    attempted = 0
    feasible = 0
    for point in _grid(resolution):
        attempted += 1
        weights = dict(zip(BRANCHES, (value / resolution for value in point)))
        if any(any(t["branch_results"][b] for b in BRANCHES)
               and not any(weights[b] > 0 and t["branch_results"][b] for b in BRANCHES) for t in traces):
            continue
        feasible += 1
        report = evaluate_run(queries, qrels, _fused(traces, weights, config, "fixed_selection"), bootstrap_samples=0)
        value = report.summary["macro_languages"]["ndcg_at_10"]
        if value > best["language_macro_ndcg_at_10"] + 1e-12:
            best = {"weights": weights, "language_macro_ndcg_at_10": value}
    return {**best, "grid_resolution": resolution, "attempted_candidates": attempted,
            "feasible_candidates": feasible, "selection_metric": "language_macro_actual_fused_ndcg_at_10",
            "initial_equal_candidate_evaluated": True, "improvement_tolerance": 1e-12,
            "tie_policy": "equal_first_then_lexicographic_simplex_tuple"}


def _evaluate(queries, qrels, run, groups, *, samples, seed, provenance):
    report = evaluate_run(queries, qrels, run, bootstrap_samples=0, seed=seed, provenance=provenance)
    # Originals may be duplicate parents in one information-need group. Keep
    # original parent IDs in per-query records but bootstrap the stronger group.
    bootstrap = bootstrap_parent_groups([replace(score, parent_query_id=groups[score.query_id])
                                        for score in report.per_query], samples=samples, seed=seed)
    bootstrap["unit"] = "information_need_group_id"
    return EvaluationReport({**report.summary, "bootstrap": bootstrap}, report.per_query)


def process_study(
    trace_path: Path, run_manifest_path: Path, queries_path: Path, qrels_path: Path,
    corpus_path: Path, output_dir: Path, *, split: str, selection_path: Path | None = None,
    grid_resolution: int = 4, bootstrap_samples: int = 1000, seed: int = 13,
    allow_fixture: bool = False, dense_original_run: Path | None = None,
    dense_original_manifest: Path | None = None,
) -> Mapping[str, Any]:
    """Validate traces, freeze validation choices, and publish a fresh study."""
    if split not in ("validation", "final_evaluation"):
        raise StudyError("Study split must be validation or final_evaluation")
    if type(grid_resolution) is not int or not 1 <= grid_resolution <= 16:
        raise StudyError("Grid resolution must be an integer in [1,16]")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int:
        raise StudyError("Bootstrap count/seed must be integers with a nonnegative count")
    if type(allow_fixture) is not bool:
        raise StudyError("Fixture opt-in must be boolean")
    trace_path, run_manifest_path, queries_path, qrels_path, corpus_path, output_dir = map(_safe_path,
        (trace_path, run_manifest_path, queries_path, qrels_path, corpus_path, output_dir))
    if selection_path is not None:
        selection_path = _safe_path(selection_path)
    inputs = [trace_path, run_manifest_path, queries_path, qrels_path, corpus_path] + ([selection_path] if selection_path else [])
    if dense_original_manifest is not None and dense_original_run is None:
        raise StudyError("An original dense manifest requires its ranking run")
    if dense_original_run is not None:
        dense_original_run = _safe_path(dense_original_run)
        dense_original_manifest = _safe_path(dense_original_manifest or dense_original_run.with_suffix(".manifest.json"))
        inputs.extend((dense_original_run, dense_original_manifest))
    if output_dir.exists() or any(path == output_dir or output_dir in path.parents for path in inputs):
        raise StudyError("Output must be fresh and must not contain or overwrite inputs")
    if (split == "final_evaluation") != (selection_path is not None):
        raise StudyError("Final evaluation requires a frozen validation selection; validation must select afresh")
    snapshots = {str(path): _sha256(path) for path in inputs}
    manifest = _json(run_manifest_path)
    pipeline = manifest.get("pipeline")
    if not isinstance(pipeline, dict) or pipeline.get("branches") != list(BRANCHES) or type(pipeline.get("fixture_only")) is not bool:
        raise StudyError("Complete fusion pipeline provenance is required")
    fingerprint = pipeline.get("pipeline_fingerprint")
    if fingerprint != _digest({k: v for k, v in pipeline.items() if k != "pipeline_fingerprint"}):
        raise StudyError("Pipeline fingerprint does not match its frozen metadata")
    fixture = _contains_fixture(pipeline) or manifest.get("fixture_only") is True
    if fixture and not allow_fixture:
        raise StudyError("Fixture traces are not real-data benchmark evidence; API opt-in is required")
    if pipeline["fixture_only"] != fixture or manifest.get("fixture_only") != fixture:
        raise StudyError("Inconsistent fixture metadata")
    config = FusionConfig(**pipeline["config"])
    config.validate()
    evaluation_config = replace(config, final_top_k=100)
    if manifest.get("split") != split or not manifest.get("trace") or Path(manifest["trace"]).resolve() != trace_path:
        raise StudyError("Fusion manifest split/advertised trace path mismatch")
    corpus_sha, query_sha = _input_hash(manifest, corpus_path), _input_hash(manifest, queries_path)
    if any(Path(name).resolve() == qrels_path for name in manifest["input_sha256"]):
        raise StudyError("Retrieval provenance must not include relevance judgments as an input")
    corpus_ids = set()
    for row in iter_jsonl(corpus_path):
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip() or doc_id in corpus_ids:
            raise StudyError("Corpus IDs must be present and unique")
        corpus_ids.add(doc_id)
    if not corpus_ids:
        raise StudyError("Corpus must not be empty")
    queries, all_rows = _queries(queries_path, split)
    selected_rows = {q.query_id: all_rows[q.query_id] for q in queries}
    traces = _traces(trace_path, manifest, selected_rows, corpus_ids, config, fixture)
    groups = {qid: row["group_id"] for qid, row in all_rows.items()}
    validation_ids = sorted(qid for qid, row in all_rows.items() if row["split"] == "validation")
    validation_groups = sorted({groups[qid] for qid in validation_ids})
    binding = {"pipeline_fingerprint": fingerprint, "corpus_sha256": corpus_sha,
               "query_source_sha256": query_sha, "scope": SCOPE, "fixture_only": fixture}
    if split == "validation":
        qrels = _scoped_qrels(qrels_path, set(selected_rows), corpus_ids)
        chosen = select_fixed_weights(traces, queries, qrels, evaluation_config, resolution=grid_resolution)
        selection = {"schema_version": SCHEMA_VERSION, "source_split": "validation", **binding, **chosen,
                     "validation_query_ids": validation_ids, "validation_group_ids": validation_groups,
                     "validation_qrels_scoped_sha256": _digest([asdict(q) for q in qrels]),
                     "serving_fusion_config": asdict(config), "evaluation_fusion_config": asdict(evaluation_config),
                     "profile": {"corpus_document_count": len(corpus_ids), "query_source_count": len(all_rows),
                                 "query_source_counts_by_split": dict(Counter(row["split"] for row in all_rows.values()))}}
        selection["selection_sha256"] = _digest(selection)
    else:
        selection = _json(selection_path)
        if (not isinstance(selection, dict) or selection.get("selection_sha256") != _digest({k: v for k, v in selection.items() if k != "selection_sha256"})
                or selection.get("source_split") != "validation" or selection.get("schema_version") != SCHEMA_VERSION):
            raise StudyError("Frozen validation selection content seal/source split is invalid")
        if (any(selection.get(key) != value for key, value in binding.items())
                or selection.get("validation_query_ids") != validation_ids
                or selection.get("validation_group_ids") != validation_groups
                or selection.get("serving_fusion_config") != asdict(config)
                or selection.get("evaluation_fusion_config") != asdict(evaluation_config)):
            raise StudyError("Frozen selection pipeline/corpus/query profile is incompatible")
    selected_weights = _weights(selection["weights"])
    if split == "final_evaluation":
        # Validate the entire frozen protocol before interpreting final labels.
        qrels = _scoped_qrels(qrels_path, set(selected_rows), corpus_ids)
    equal = {branch: 0.25 for branch in BRANCHES}
    prefix = "fixture_" if fixture else ""
    runs = {}
    dense_original_only = all(t["diagnostics"]["branch_diagnostics"]["dense"]["views"] ==
                              [unicodedata.normalize("NFC", selected_rows[t["query_id"]]["text"])] for t in traces)
    dense_method = "B3_dense_original" if dense_original_only else "dense_views"
    for method, branch in (("B1_bm25_original", "bm25_original"), ("B2_protected_roman_view_character", "char_tfidf"), (dense_method, "dense")):
        runs[method] = [RunEntry(t["query_id"], h.doc_id, prefix + method, h.rank, h.score)
                        for t in traces for h in t["branch_results"][branch]]
    dense_original_cost = None
    if dense_original_run is not None:
        original_entries, dense_original_cost = _original_dense(
            dense_original_run, dense_original_manifest, pipeline=pipeline, queries=queries,
            queries_path=queries_path, query_sha=query_sha, split=split, corpus_ids=corpus_ids, fixture=fixture)
        runs["B3_dense_original"] = [replace(entry, branch=prefix + "B3_dense_original") for entry in original_entries]
    for method, weights in (("B5_equal", equal), ("fixed_selected", selected_weights), ("trace_weights", "trace")):
        runs[method] = _fused(traces, weights, evaluation_config, prefix + method)
    remainder = math.fsum(selected_weights[b] for b in BRANCHES if b != "char_tfidf")
    unavailable = {"B4_single_best_normalized": "not recoverable from aggregated trace; no model rerun",
                   "B6_noise_training": "not established by this original-query trace",
                   "P1_consistency_training": "not established by this trace; no claim of consistency-trained model",
                   "independent_human_challenge": "omitted by approved public-data-only scope"}
    if not dense_original_only and dense_original_run is None:
        unavailable["B3_dense_original"] = "stored dense branch averages multiple views; supply a separately manifested original-only dense run"
    if remainder > 0:
        no_char = {branch: (selected_weights[branch] / remainder if branch != "char_tfidf" else 0.0) for branch in BRANCHES}
        runs["no_character_fixed"] = _fused(traces, no_char, evaluation_config, prefix + "no_character_fixed")
    else:
        no_char = None
        unavailable["no_character_fixed"] = "selected fixed weights have no non-character mass to renormalize"
    provenance = {**binding, "trace_sha256": _sha256(trace_path), "selection_sha256": selection["selection_sha256"],
                  "split": split, "models_rerun": False, "serving_final_top_k": config.final_top_k,
                  "evaluation_final_top_k": 100, "bootstrap_unit": "information_need_group_id"}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.study-", dir=output_dir.parent))
    try:
        evaluations = {}
        for method, run in runs.items():
            report = _evaluate(queries, qrels, run, groups, samples=bootstrap_samples, seed=seed,
                               provenance={**provenance, "method": method})
            _write_run(staging / f"{method}.run.jsonl", run)
            _write_json(staging / f"{method}.evaluation.json", report.to_dict())
            evaluations[method] = {"overall": report.summary["overall"], "macro_languages": report.summary["macro_languages"],
                                   "missing_run_query_count": report.summary["missing_run_query_count"]}
        if split == "validation":
            _write_json(staging / "validation-selection.json", selection)
        actual_cost = {"query_count": len(traces), "search_count": manifest["search_count"],
                       "intermediate_candidate_count": manifest["intermediate_candidate_count"],
                       "latency_seconds": manifest["latency_seconds"],
                       "separate_original_dense_run": dense_original_cost,
                       "per_branch": {branch: {
                           "search_count": sum(t["diagnostics"]["branch_diagnostics"][branch]["search_count"] for t in traces),
                           "seconds": math.fsum(t["diagnostics"]["branch_diagnostics"][branch]["seconds"] for t in traces),
                           "masked_query_count": sum(not t["branch_results"][branch] for t in traces),
                           "unavailable_reasons": dict(Counter(t["diagnostics"]["branch_diagnostics"][branch]["unavailable_reason"]
                                                                for t in traces if not t["branch_results"][branch]))
                       } for branch in BRANCHES},
                       "notice": "Measured costs of the supplied shared trace, not separately measured baseline latencies; ranking replay performs no retrieval."}
        summary = {"schema_version": SCHEMA_VERSION, **provenance, "scope": SCOPE,
                   "query_count": len(queries), "group_count": len({groups[q.query_id] for q in queries}),
                   "corpus_document_count": len(corpus_ids), "weights": {"equal": equal, "fixed_selected": selected_weights,
                                                                              "no_character_fixed": no_char},
                   "trace_weight_method": "adaptive_learned_gate" if manifest["weights"] == "learned_gate" else "provided_fixed_or_equal",
                   "dense_branch_mode": "original_only" if dense_original_only else "multi_view_or_unavailable",
                   "character_baseline_notice": "B2 consumes the protected generated Roman query view, not an independently supplied raw-Roman query baseline.",
                   "character_index_coverage": pipeline.get("indexes", {}).get("char_tfidf"),
                   "actual_trace_cost": actual_cost, "evaluations": evaluations, "not_available": unavailable,
                   "no_character_ablation_notice": "Fixed-fusion weight intervention only; no gate retraining or full-system ablation claim.",
                   "quality_scope_notice": "Original-query CPU pilot, not full/noisy/independent-human benchmark evidence. Frozen/finetuned encoder provenance stays in the source pipeline.",
                   "bootstrap_samples": bootstrap_samples, "seed": seed,
                   "source_files_sha256": snapshots,
                   "source_pipeline": pipeline,
                   "artifacts": {path.name: _sha256(path) for path in sorted(staging.iterdir())}}
        summary["study_sha256"] = _digest(summary)
        _write_json(staging / "study_manifest.json", summary)
        if any(_sha256(path) != snapshots[str(path)] for path in inputs):
            raise StudyError("A frozen study input changed during replay/evaluation")
        if output_dir.exists():
            raise StudyError("Output appeared during processing; refusing overwrite")
        os.rename(staging, output_dir)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("trace", "run-manifest", "queries", "qrels", "corpus", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "final_evaluation"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dense-original-run", type=Path)
    parser.add_argument("--dense-original-manifest", type=Path)
    parser.add_argument("--grid-resolution", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    try:
        report = process_study(args.trace, args.run_manifest, args.queries, args.qrels, args.corpus, args.output_dir,
                               split=args.split, selection_path=args.selection, grid_resolution=args.grid_resolution,
                               bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                               dense_original_run=args.dense_original_run, dense_original_manifest=args.dense_original_manifest)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"Study error: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "study_sha256": report["study_sha256"], "split": args.split}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
