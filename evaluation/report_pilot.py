"""Summarize an already-completed, locked public-original-query CPU pilot.

This command never fits, selects, retrieves, or reads relevance labels. It
checks the frozen closure and the sealed final evaluation artifacts before
publishing language-macro and descriptive two-seed summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.study_runs import (
    SCOPE, StudyError, _contains_fixture, _digest, _json, _queries,
    _safe_path, _sha256, _write_json,
)
from evaluation.evaluate import load_run
from retrieval.fusion import BRANCHES
from training.train_gate import LinearSoftmaxGate


LANGUAGES = ("en", "hi", "bn")
SEEDS = (13, 29)
VARIANTS = ("base", "full", "no_uncertainty", "no_agreement")
METRICS = ("ndcg_at_10", "mrr_at_10", "recall_at_100")
REQUIRED_METHODS = {
    "B1_bm25_original", "B2_protected_roman_view_character", "B3_dense_original",
    "B5_equal", "fixed_selected", "trace_weights",
}
LIMITATIONS = [
    "Public MIRACL original queries only; no independently reviewed human challenge and no natural noisy/code-mixed robustness claim.",
    "Original-only retrieval adaptation; no reviewed variant pairs, noise-training result, or consistency-training/consistency-ablation claim.",
    "Reduced CPU pilot: 1,500 fixed passages per language, 128-token right truncation, seeds 13 and 29; not the prepared 50,000-passage or full MIRACL benchmark.",
    "Separate monolingual relevance tracks; pooled results do not establish cross-lingual retrieval.",
    "Groups preserve exact NFKC/casefold/whitespace-normalized duplicates and original split assignments; no independent semantic-paraphrase or closely-related-information-need audit was available.",
    "E5-small has prior MIRACL-training exposure; the frozen model is not zero-shot MIRACL.",
    "Dakshina-training plus algorithmic uroman is not a validated IndicXlit run. Held-out native-to-Roman top-1 was 15.68% Hindi/21.72% Bengali; reverse top-2 was 7.14%/5.45%, not semantic validation.",
    "The auxiliary LID test has no Bengali gold; contextual Bengali LID accuracy is unvalidated. Abstention/masked branches must remain visible.",
    "Protected Roman-view character retrieval is not an independently supplied raw-Roman baseline.",
    "Weights assigned to branches that were always masked on validation are not identifiable evidence of those branches' usefulness; keep unavailable-branch counts visible.",
    "Shared-trace retrieval costs are not standalone baseline latencies; ranking replay performs no retrieval.",
    "No-character fixed fusion is a renormalized weight intervention, not a retrained gate/full-system character ablation. Uncertainty/agreement ablations are separately fitted gates.",
    "Across-seed mean/range is descriptive only: seeds share queries and are not independent query samples. Language macro intervals are not inferred by averaging confidence bounds.",
    "No final-set winner selection or post-final retuning; retain negative/unchanged results and the pre-final deployment/reporting rule.",
]


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise StudyError(message)


def _count(value: Any, label: str, *, minimum: int = 0) -> int:
    _fail(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _number(value: Any, label: str, *, unit: bool = False) -> float:
    _fail(type(value) in (int, float) and math.isfinite(value) and value >= 0
          and (not unit or value <= 1), f"Invalid {label}")
    return float(value)


def _metrics(value: Any) -> dict[str, float]:
    _fail(isinstance(value, dict) and set(value) == set(METRICS), "Incomplete evaluation metrics")
    return {key: _number(value[key], key, unit=True) for key in METRICS}


def _same_metrics(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return all(math.isclose(left[key], right[key], rel_tol=1e-12, abs_tol=1e-12) for key in METRICS)


def _sealed(value: Any, seal: str, label: str) -> dict[str, Any]:
    _fail(isinstance(value, dict) and value.get(seal) == _digest({k: v for k, v in value.items() if k != seal}),
          f"Invalid {label} content seal")
    return value


def _hash_map(value: Any, *, label: str) -> dict[Path, str]:
    _fail(isinstance(value, dict) and bool(value), f"Missing {label} hash closure")
    result = {}
    for name, sha in value.items():
        _fail(isinstance(name, str) and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
              f"Invalid {label} file hash")
        path = _safe_path(Path(name))
        _fail(str(path) == name and path.is_file() and path not in result, f"Noncanonical/missing {label} file: {name}")
        _fail(_sha256(path) == sha, f"Changed {label} file: {path}")
        result[path] = sha
    return result


def _tree_files(path: Path) -> set[Path]:
    _fail(path.is_dir(), f"Missing frozen artifact directory: {path}")
    result = {_safe_path(item) for item in path.rglob("*") if item.is_file() or item.is_symlink()}
    _fail(bool(result), f"Empty frozen artifact directory: {path}")
    return result


def _scope(path: Path) -> dict[str, Any]:
    value = _json(path)
    pilot = value.get("pilot", {})
    _fail(value.get("scope") == "public_data_only_clean_query_study"
          and isinstance(value.get("scope_approved_by_user"), str)
          and value.get("human_challenge_set_available") is False
          and value.get("unreviewed_variants_used_for_supervised_training") is False
          and value.get("unreviewed_variants_used_for_scored_evaluation") is False
          and value.get("prior_e5_miracl_training_exposure_disclosed") is True
          and value.get("consistency_weight") == 0
          and pilot.get("target_passages_per_language") == 1500
          and pilot.get("max_length_tokens") == 128
          and pilot.get("document_encoding_batch_size") == 8
          and pilot.get("encoder_training_seeds") == list(SEEDS), "Incompatible approved public CPU-pilot scope")
    return value


def _evaluation(path: Path, *, method: str, summary: Mapping[str, Any], language: str,
                query_ids: set[str]) -> dict[str, Any]:
    value = _json(path)
    count = summary["query_count"]
    _fail(value.get("schema_version") == 1 and value.get("declared_query_count") == count
          and value.get("evaluated_query_count") == count, "Evaluation query counts disagree")
    provenance = value.get("provenance", {})
    for key in ("pipeline_fingerprint", "corpus_sha256", "query_source_sha256", "selection_sha256",
                "trace_sha256", "split", "scope", "fixture_only", "serving_final_top_k", "evaluation_final_top_k"):
        _fail(provenance.get(key) == summary.get(key), f"Evaluation {key} provenance mismatch")
    _fail(provenance.get("method") == method and provenance.get("models_rerun") is False,
          "Evaluation method/replay provenance mismatch")
    overall = _metrics(value.get("overall"))
    _fail(_same_metrics(overall, _metrics(value.get("macro_languages"))), "Monolingual macro/overall mismatch")
    per_language = value.get("per_language", {})
    _fail(set(per_language) == {language} and per_language[language].get("query_count") == count
          and _same_metrics(overall, _metrics(per_language[language].get("metrics"))), "Evaluation language mismatch")
    rows = value.get("per_query")
    _fail(isinstance(rows, list) and len(rows) == count, "Incomplete per-query results")
    run = load_run(path.with_name(f"{method}.run.jsonl"))
    depths = {qid: 0 for qid in query_ids}
    for entry in run:
        _fail(entry.query_id in query_ids and entry.branch == ("fixture_" if summary["fixture_only"] else "") + method,
              "Ranking run has incompatible query/method provenance")
        depths[entry.query_id] += 1
    _fail(value.get("run_entry_count") == len(run), "Ranking/evaluation result count mismatch")
    seen, missing = set(), 0
    for row in rows:
        qid = row.get("query_id")
        _fail(qid in query_ids and qid not in seen and row.get("benchmark_language") == language
              and row.get("split") == "final_evaluation" and row.get("variant_type") == "original"
              and row.get("status") == "evaluated", "Invalid final original-query result membership")
        seen.add(qid)
        _fail(row.get("run_depth") == depths[qid], "Ranking/evaluation per-query depth mismatch")
        missing += _count(row.get("run_depth"), "run depth") == 0
        for key in METRICS:
            _number(row.get(key), key, unit=True)
    calculated = {key: math.fsum(row[key] for row in rows) / count for key in METRICS}
    _fail(seen == query_ids and _same_metrics(overall, calculated), "Stored metrics disagree with per-query means")
    _fail(value.get("missing_run_query_count") == missing, "Missing-run count disagrees with per-query depths")
    aggregate = summary.get("evaluations", {}).get(method, {})
    _fail(aggregate.get("missing_run_query_count") == missing
          and _same_metrics(overall, _metrics(aggregate.get("overall")))
          and _same_metrics(overall, _metrics(aggregate.get("macro_languages"))), "Study/evaluation summary mismatch")
    bootstrap = value.get("bootstrap", {})
    _fail(bootstrap.get("unit") == "information_need_group_id"
          and bootstrap.get("parent_group_count") == summary["group_count"]
          and bootstrap.get("samples") == summary.get("bootstrap_samples")
          and bootstrap.get("seed") == summary.get("seed"), "Invalid group-bootstrap provenance")
    _number(bootstrap.get("confidence"), "bootstrap confidence", unit=True)
    intervals = bootstrap.get("intervals", {})
    _fail(set(intervals) == set(METRICS), "Incomplete group-bootstrap intervals")
    for key, bounds in intervals.items():
        lower = _number(bounds.get("lower"), "confidence lower", unit=True)
        upper = _number(bounds.get("upper"), "confidence upper", unit=True)
        _fail(lower <= upper, "Inverted group-bootstrap interval")
    _fail(value.get("robustness", {}).get("paired_parent_count") == 0,
          "Original-only pilot cannot report noisy robustness pairs")
    return {"metrics": overall, "missing_run_query_count": missing, "group_bootstrap": bootstrap}


def _cost(summary: Mapping[str, Any]) -> dict[str, Any]:
    cost = summary.get("actual_trace_cost", {})
    count = summary["query_count"]
    _fail(cost.get("query_count") == count and set(cost.get("per_branch", {})) == set(BRANCHES),
          "Incomplete measured trace cost/coverage")
    for key in ("search_count", "intermediate_candidate_count"):
        _count(cost.get(key), key)
    latency = cost.get("latency_seconds", {})
    _fail(set(latency) == {"view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds"},
          "Incomplete measured trace latency")
    for key, value in latency.items():
        _number(value, key)
    for branch, record in cost["per_branch"].items():
        _count(record.get("search_count"), f"{branch} search count")
        _number(record.get("seconds"), f"{branch} seconds")
        masked = _count(record.get("masked_query_count"), f"{branch} masked count")
        reasons = record.get("unavailable_reasons", {})
        _fail(masked <= count and isinstance(reasons, dict) and all(isinstance(k, str) for k in reasons)
              and sum(_count(n, "unavailable reason count") for n in reasons.values()) == masked,
              "Invalid branch masking/unavailability counts")
    return {**cost, "total_seconds_per_query": latency["total_seconds"] / count,
            "notice": "Actual shared-trace retrieval costs, not independent baseline latency measurements."}


def _track(result: Path, *, model_track: Path, language: str, variant: str, adapted: bool,
           locked: Mapping[Path, str], snapshots: dict[Path, str], allow_fixture: bool) -> dict[str, Any]:
    analysis = result / "final-analysis"
    manifest_path = _safe_path(analysis / "study_manifest.json")
    snapshots[manifest_path] = _sha256(manifest_path)
    summary = _sealed(_json(manifest_path), "study_sha256", "final study")
    fixture = _contains_fixture(summary)
    _fail(type(summary.get("fixture_only")) is bool and summary["fixture_only"] == fixture,
          "Inconsistent fixture provenance")
    _fail(not fixture or allow_fixture, "Fixture-only artifacts are not real benchmark evidence; API opt-in required")
    _fail(summary.get("schema_version") == 1 and summary.get("scope") == SCOPE
          and summary.get("split") == "final_evaluation" and summary.get("models_rerun") is False
          and summary.get("serving_final_top_k") == 10 and summary.get("evaluation_final_top_k") == 100
          and summary.get("bootstrap_unit") == "information_need_group_id", "Not a compatible locked final study")
    count = _count(summary.get("query_count"), "final query count", minimum=1)
    groups = _count(summary.get("group_count"), "final group count", minimum=1)
    _fail(groups <= count, "More final groups than queries")
    _count(summary.get("bootstrap_samples"), "bootstrap samples", minimum=1)
    _count(summary.get("seed"), "bootstrap seed")
    _fail(fixture or (summary.get("corpus_document_count") == 1500 and groups == 24),
          "Final study is not the declared 1500-passage/24-final-group pilot")
    pipeline = summary.get("source_pipeline", {})
    _fail(pipeline.get("pipeline_fingerprint") == summary.get("pipeline_fingerprint")
          and pipeline.get("pipeline_fingerprint") == _digest({k: v for k, v in pipeline.items() if k != "pipeline_fingerprint"}),
          "Final pipeline fingerprint mismatch")
    _fail(pipeline.get("config") == {"per_branch_top_k": 100, "per_view_top_k": 100,
                                      "final_top_k": 10, "k_view": 60.0, "k": 60.0},
          "Final fusion candidate/serving settings differ from the declared pilot")
    dense = pipeline.get("indexes", {}).get("dense", {})
    _fail(_json(model_track / "dense" / "dense_manifest.json") == dense,
          "Final pipeline dense index metadata differs from its locked saved index")
    _fail(dense.get("document_count") == summary.get("corpus_document_count")
          and dense.get("config", {}).get("max_length") == 128
          and dense.get("config", {}).get("batch_size") == 8
          and dense.get("config", {}).get("truncation_side") == "right"
          and dense.get("config", {}).get("truncation_policy") == "single_original_passage_head"
          and dense.get("fine_tuned") is adapted, "Frozen/adapted index or 128-token preprocessing mismatch")
    if not fixture:
        encoder = dense.get("encoder", {})
        _fail(dense.get("dimension") == 384 and encoder.get("model_id") == "intfloat/multilingual-e5-small"
              and encoder.get("fixture_only") is False, "Not a real multilingual E5-small index")
        if adapted:
            checkpoint = _json(model_track.parent / "checkpoint" / "encoder_manifest.json")
            training = checkpoint.get("metadata", {}).get("training", {})
            seed = int(model_track.parent.name.rsplit("-", 1)[1])
            _fail(encoder.get("checkpoint_sha256") == checkpoint.get("checkpoint_sha256")
                  and encoder.get("fine_tuned") is True and training.get("seed") == seed
                  and training.get("training_regime") == "original_only_retrieval"
                  and training.get("variant_pair_count") == 0 and training.get("consistency_applicable") is False,
                  "Adapted index is not bound to its declared original-only trained seed checkpoint")
        else:
            _fail(encoder.get("resolved_revision") == "fd1525a9fd15316a2d503bf26ab031a61d056e98",
                  "Frozen E5 revision differs from the predeclared pinned model")
    expected_mode = "provided_fixed_or_equal" if variant == "base" else "adaptive_learned_gate"
    _fail(summary.get("trace_weight_method") == expected_mode, "Wrong base/adaptive track type")
    gate = None
    if variant != "base":
        gate_path = model_track / "gates" / f"{variant}.json"
        gate = LinearSoftmaxGate.load_json(gate_path, expected_pipeline_manifest=pipeline)
        expected_excluded = ([] if variant == "full" else ["mean_token_entropy", "uncertain_token_fraction"]
                             if variant == "no_uncertainty" else [
                                 "overlap_bm25_original_bm25_native", "overlap_bm25_original_char_tfidf",
                                 "overlap_bm25_original_dense", "overlap_bm25_native_char_tfidf",
                                 "overlap_bm25_native_dense", "overlap_char_tfidf_dense"])
        _fail(pipeline.get("weight_provider") == gate.metadata and set(gate.excluded_features) == set(expected_excluded)
              and gate.training_report.get("training_split") == "fusion_train"
              and gate.training_report.get("scaler_fit_split") == "fusion_train"
              and gate.training_report.get("hyperparameter_selection", {}).get("split") == "validation",
              "Gate/feature ablation is not the locked separately fitted validation-selected artifact")
    selection_path = _safe_path(result / "validation-analysis" / "validation-selection.json")
    _fail(selection_path in locked, "Matching validation selection is absent from final lock")
    selection = _sealed(_json(selection_path), "selection_sha256", "validation selection")
    _fail(selection.get("source_split") == "validation"
          and selection.get("selection_sha256") == summary.get("selection_sha256")
          and selection.get("weights") == summary.get("weights", {}).get("fixed_selected")
          and all(selection.get(key) == summary.get(key) for key in ("pipeline_fingerprint", "corpus_sha256", "query_source_sha256", "scope", "fixture_only")),
          "Final result is not bound to its locked validation selection")
    sources = _hash_map(summary.get("source_files_sha256"), label="final source")
    _fail(all(path not in locked or locked[path] == sha for path, sha in sources.items()),
          "Final source disagrees with its pre-final lock digest")
    snapshots.update(sources)
    by_name = {}
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl"):
        matches = [path for path in sources if path.name == name]
        _fail(len(matches) == 1 and matches[0] in locked, f"Final lock lacks unique fixed {name}")
        by_name[name] = matches[0]
    # Selection seals hash canonical content; source_files_sha256 hashes bytes.
    _fail(selection_path in sources and sources[selection_path] == locked[selection_path]
          and sources[by_name["corpus.jsonl"]] == summary["corpus_sha256"]
          and sources[by_name["queries.jsonl"]] == summary["query_source_sha256"], "Final source binding mismatch")
    queries, rows = _queries(by_name["queries.jsonl"], "final_evaluation")
    query_ids = {query.query_id for query in queries}
    _fail(len(query_ids) == count and all(rows[qid]["benchmark_language"] == language for qid in query_ids)
          and len({rows[qid]["group_id"] for qid in query_ids}) == groups,
          "Final query/group/language universe mismatch")
    if gate is not None:
        training_ids = gate.training_report.get("training_query_ids", [])
        validation_ids = gate.training_report.get("validation_query_ids", [])
        _fail(bool(training_ids) and bool(validation_ids)
              and all(qid in rows and rows[qid]["split"] == "fusion_train" for qid in training_ids)
              and all(qid in rows and rows[qid]["split"] == "validation" for qid in validation_ids),
              "Gate fit/selection IDs violate original fixed split boundaries")
    artifacts = summary.get("artifacts")
    _fail(isinstance(artifacts, dict) and bool(artifacts), "Missing sealed final artifacts")
    for name, sha in artifacts.items():
        _fail(isinstance(name, str) and Path(name).name == name and name not in (".", "..")
              and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
              "Invalid final artifact hash/path")
        path = _safe_path(analysis / name)
        _fail(path.is_file() and _sha256(path) == sha, f"Changed final artifact: {path}")
        snapshots[path] = sha
    methods = summary.get("evaluations", {})
    _fail(isinstance(methods, dict) and REQUIRED_METHODS <= methods.keys(), "Missing required final baseline/fusion methods")
    evaluations = {}
    for method in methods:
        _fail(isinstance(method, str) and re.fullmatch(r"[A-Za-z0-9_]+", method) is not None,
              "Invalid evaluation method name")
        _fail(f"{method}.evaluation.json" in artifacts and f"{method}.run.jsonl" in artifacts,
              "Final method is missing its sealed evaluation or ranking run")
        evaluations[method] = _evaluation(analysis / f"{method}.evaluation.json", method=method,
                                          summary=summary, language=language, query_ids=query_ids)
    coverage = summary.get("character_index_coverage", {})
    _fail(coverage == pipeline.get("indexes", {}).get("char_tfidf")
          and coverage.get("source_document_count") == summary["corpus_document_count"]
          and _count(coverage.get("indexed_document_count"), "character indexed count") <= summary["corpus_document_count"],
          "Invalid character index coverage")
    index_bytes = sum(path.stat().st_size for path in _tree_files(model_track / "dense"))
    return {"language": language, "fixture_only": fixture, "analysis_path": str(analysis),
            "study_sha256": summary["study_sha256"], "pipeline_fingerprint": summary["pipeline_fingerprint"],
            "query_ids": sorted(query_ids), "query_count": count, "group_count": groups,
            "corpus_sha256": summary["corpus_sha256"], "query_source_sha256": summary["query_source_sha256"],
            "qrels_sha256": sources[by_name["qrels.jsonl"]], "dense_index_fingerprint": dense.get("index_fingerprint"),
            "dense_index_disk_bytes": index_bytes, "character_index_coverage": coverage,
            "gate_training": ({"excluded_features": list(gate.excluded_features),
                               "config": gate.training_report.get("config"),
                               "hyperparameter_selection": gate.training_report.get("hyperparameter_selection")}
                              if gate is not None else None),
            "actual_trace_cost": _cost(summary), "evaluations": evaluations,
            "not_available": summary.get("not_available", {}), "weights": summary.get("weights")}


def _macro(languages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    methods = set().union(*(track["evaluations"] for track in languages.values()))
    result = {}
    for method in sorted(methods):
        available = [track["evaluations"][method] for track in languages.values() if method in track["evaluations"]]
        result[method] = {"language_count": len(available), "complete_three_language_macro": len(available) == 3,
                          "metrics": {key: math.fsum(item["metrics"][key] for item in available) / len(available) for key in METRICS},
                          "missing_run_query_count": sum(item["missing_run_query_count"] for item in available)}
    return result


def _seed_variation(tracks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for variant in VARIANTS:
        seed_tracks = [tracks[f"adapted-seed-{seed}/{variant}"] for seed in SEEDS]
        methods = set().union(*(track["macro_languages"] for track in seed_tracks))
        result[variant] = {}
        for method in sorted(methods):
            record = {}
            for language in (*LANGUAGES, "macro_languages"):
                available = []
                for track in seed_tracks:
                    collection = track["macro_languages"] if language == "macro_languages" else track["languages"][language]["evaluations"]
                    if method in collection and (language != "macro_languages" or collection[method]["complete_three_language_macro"]):
                        available.append(collection[method]["metrics"])
                record[language] = {"seed_count": len(available), "metrics": {
                    key: {"mean": math.fsum(item[key] for item in available) / len(available),
                          "min": min(item[key] for item in available), "max": max(item[key] for item in available),
                          "range": max(item[key] for item in available) - min(item[key] for item in available)}
                    for key in METRICS} if available else None}
            result[variant][method] = record
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Locked public-data CPU-pilot results", "", "1,500 fixed passages/language; original MIRACL queries; 128-token truncation. Both predeclared seeds are reported without final-set winner selection.", "",
             "## Language-macro quality", "", "| Track | Method | nDCG@10 | MRR@10 | Recall@100 | Missing queries |", "|---|---|---:|---:|---:|---:|"]
    for name, track in report["tracks"].items():
        for method, result in track["macro_languages"].items():
            metrics = result["metrics"]
            label = method + (" (partial languages)" if not result["complete_three_language_macro"] else "")
            lines.append(f"| {name} | {label} | {metrics['ndcg_at_10']:.6f} | {metrics['mrr_at_10']:.6f} | {metrics['recall_at_100']:.6f} | {result['missing_run_query_count']} |")
    lines.extend(["", "## Per-language quality", "", "| Track | Language | Method | nDCG@10 | MRR@10 | Recall@100 | Missing queries |", "|---|---|---|---:|---:|---:|---:|"])
    for name, track in report["tracks"].items():
        for language, row in track["languages"].items():
            for method, result in row["evaluations"].items():
                values = result["metrics"]
                lines.append(f"| {name} | {language} | {method} | {values['ndcg_at_10']:.6f} | {values['mrr_at_10']:.6f} | {values['recall_at_100']:.6f} | {result['missing_run_query_count']} |")
    lines.extend(["", "Per-language information-need-group bootstrap intervals, actual shared-trace latency, masks/unavailable reasons, character document coverage, selected weights, dense-index disk bytes, and descriptive seed mean/min/max/range are retained in `report.json`. No macro confidence interval is invented from language bounds.", "", "## Scope and limitations", ""])
    lines.extend(f"- {notice}" for notice in report["limitations"])
    if report["fixture_only"]:
        lines.extend(["", "**Fixture-only report: not real-data benchmark evidence.**"])
    return "\n".join(lines) + "\n"


def report_pilot(study_root: Path, output_dir: Path, *, final_lock: Path | None = None,
                 scope_config: Path = Path("configs/public-data-study.json"), allow_fixture: bool = False) -> dict[str, Any]:
    """Validate all 27 final tracks and atomically publish a fresh report directory."""
    _fail(type(allow_fixture) is bool, "Fixture API opt-in must be boolean")
    study_root, output_dir, scope_config = map(_safe_path, (study_root, output_dir, scope_config))
    final_lock = _safe_path(final_lock or study_root / "final-lock.json")
    _fail(not output_dir.exists() and output_dir != study_root and output_dir not in study_root.parents,
          "Report output must be fresh and must not overwrite/contain study inputs")
    if study_root in output_dir.parents:
        _fail(output_dir.relative_to(study_root).parts[0] not in ("frozen", *(f"adapted-seed-{seed}" for seed in SEEDS)),
              "Report output must not be nested inside a frozen model/track directory")
    lock_sha = _sha256(final_lock)
    lock = _sealed(_json(final_lock), "lock_sha256", "final lock")
    locked = _hash_map(lock.get("files_sha256"), label="final lock")
    _fail(final_lock not in locked and scope_config in locked, "Final lock must cover the approved scope, not itself")
    _fail(not any(output_dir == path or output_dir in path.parents for path in (*locked, final_lock)),
          "Report output must not contain frozen inputs")
    scope = _scope(scope_config)
    required, jobs = {scope_config}, []
    for model in ("frozen", *(f"adapted-seed-{seed}" for seed in SEEDS)):
        adapted = model != "frozen"
        if adapted:
            required.update(_tree_files(study_root / model / "checkpoint"))
        for language in LANGUAGES:
            model_track = study_root / model / language
            required.update(_tree_files(model_track / "dense"))
            for variant in VARIANTS if adapted else ("base",):
                result = model_track if variant == "base" else model_track / f"{variant}-gate"
                required.update(_tree_files(result / "validation-analysis"))
                if variant != "base":
                    required.add(_safe_path(model_track / "gates" / f"{variant}.json"))
                jobs.append((model, language, variant, result, model_track, adapted))
    _fail(required <= locked.keys(), "Final lock is missing frozen checkpoint/index/gate/validation closure files")
    snapshots = dict(locked)
    tracks, universes, retrievals, lexical = {}, {}, {}, {}
    for model, language, variant, result, model_track, adapted in jobs:
        row = _track(result, model_track=model_track, language=language, variant=variant,
                     adapted=adapted, locked=locked, snapshots=snapshots, allow_fixture=allow_fixture)
        identity = {key: row[key] for key in ("query_ids", "query_count", "group_count", "corpus_sha256", "query_source_sha256", "qrels_sha256")}
        if language in universes:
            _fail(identity == universes[language], "Compared tracks use different fixed corpus/query/judgment universes")
        universes[language] = identity
        lexical_metrics = {method: row["evaluations"][method] for method in ("B1_bm25_original", "B2_protected_roman_view_character")}
        # Bootstrap seed/samples and query universe are identical, so these
        # rankings' full score summaries must not drift across encoder tracks.
        if language in lexical:
            _fail(lexical_metrics == lexical[language], "Lexical baselines changed between compared tracks")
        lexical[language] = lexical_metrics
        model_retrieval = {"dense_index_fingerprint": row["dense_index_fingerprint"],
                           **{method: row["evaluations"][method] for method in ("B3_dense_original", "B5_equal")}}
        key = (model, language)
        if key in retrievals:
            _fail(model_retrieval == retrievals[key], "Gate variants changed the fixed dense index/baseline/equal retrieval")
        retrievals[key] = model_retrieval
        name = "frozen" if not adapted else f"{model}/{variant}"
        tracks.setdefault(name, {"encoder_seed": int(model.rsplit("-", 1)[1]) if adapted else None,
                                 "variant": variant, "languages": {}})["languages"][language] = row
    fixture_flags = {row["fixture_only"] for track in tracks.values() for row in track["languages"].values()}
    _fail(len(fixture_flags) == 1, "Real and fixture final tracks cannot be mixed")
    for track in tracks.values():
        track["macro_languages"] = _macro(track["languages"])
    report = {"schema_version": 1, "scope": SCOPE, "split": "final_evaluation", "fixture_only": fixture_flags.pop(),
              "final_lock_path": str(final_lock), "lock_sha256": lock["lock_sha256"],
              "scope_config_path": str(scope_config), "scope_approval": scope["scope_approved_by_user"],
              "profile": {"passages_per_language": 1500, "max_length_tokens": 128, "encoder_seeds": list(SEEDS)},
              "expected_track_count": 27, "actual_track_count": len(jobs), "tracks": tracks,
              "across_seed_descriptive": _seed_variation(tracks),
              "pre_final_reporting_rule": {key: value for key, value in lock.items()
                                           if key not in ("files_sha256", "lock_sha256")},
              "limitations": LIMITATIONS,
              "source_files_sha256": {str(path): sha for path, sha in sorted(snapshots.items())},
              "final_lock_file_sha256": lock_sha}
    report["report_sha256"] = _digest(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.report-", dir=output_dir.parent))
    try:
        _write_json(staging / "report.json", report)
        with (staging / "report.md").open("xb") as handle:
            handle.write(_markdown(report).encode("utf-8"))
        _fail(_sha256(final_lock) == lock_sha and all(_sha256(path) == sha for path, sha in snapshots.items()),
              "A frozen/report input changed during aggregation")
        _fail(not output_dir.exists(), "Report output appeared during aggregation; refusing overwrite")
        os.rename(staging, output_dir)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path)
    parser.add_argument("--scope-config", type=Path, default=Path("configs/public-data-study.json"))
    args = parser.parse_args(argv)
    try:
        report = report_pilot(args.study_root, args.output_dir, final_lock=args.final_lock, scope_config=args.scope_config)
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        parser.exit(2, f"Pilot report error: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "report_sha256": report["report_sha256"],
                      "actual_track_count": report["actual_track_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
