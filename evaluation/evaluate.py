"""Evaluate ranked retrieval runs with leakage-aware query metadata.

Metrics follow the project specification:

* graded nDCG@10 with gain ``2**relevance - 1``;
* binary MRR@10, where relevance is greater than zero;
* Recall@100 over all positively judged passages.

The declared query manifest is authoritative.  A query with no run entries is
therefore scored as an empty ranking rather than vanishing from the average.
Queries without positive judgments and unverified variants are excluded with
explicit counts.  Unjudged passages receive zero gain for metric computation,
which is standard practice but not evidence that they are truly irrelevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import (
    CorpusPreparationError,
    Qrel,
    _clean_id,
    _first,
    canonical_language,
    iter_jsonl,
    load_qrels,
)


SCHEMA_VERSION = 1
NDCG_CUTOFF = 10
MRR_CUTOFF = 10
RECALL_CUTOFF = 100
METRIC_NAMES = ("ndcg_at_10", "mrr_at_10", "recall_at_100")


class EvaluationError(ValueError):
    """Raised when an evaluation artifact or ranking is invalid."""


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    parent_query_id: str
    judgment_query_id: str | None
    benchmark_language: str
    languages: tuple[str, ...]
    split: str
    variant_type: str
    verified: bool
    review_status: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunEntry:
    query_id: str
    doc_id: str
    branch: str
    rank: int
    score: float


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    parent_query_id: str
    benchmark_language: str
    split: str
    variant_type: str
    status: str
    run_depth: int
    ndcg_at_10: float | None
    mrr_at_10: float | None
    recall_at_100: float | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationReport:
    summary: Mapping[str, Any]
    per_query: tuple[QueryScore, ...]

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.summary)
        value["per_query"] = [asdict(score) for score in self.per_query]
        return value


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationError(f"{field} must be a JSON boolean")
    return value


def _canonical_languages(value: Any, benchmark_language: str) -> tuple[str, ...]:
    if value is None:
        return (benchmark_language,)
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise EvaluationError("languages must be a string or list")
    try:
        languages = tuple(sorted({canonical_language(str(item)) for item in values}))
    except CorpusPreparationError as exc:
        raise EvaluationError(str(exc)) from exc
    if not languages:
        raise EvaluationError("languages must not be empty")
    return languages


def load_evaluation_queries(path: Path, *, split: str | None = None) -> list[EvaluationQuery]:
    """Load the output of :mod:`data.make_splits` as an evaluation universe."""

    if split is not None and split not in ALL_SPLITS:
        raise EvaluationError(
            f"Unknown split {split!r}; expected one of {', '.join(ALL_SPLITS)}"
        )
    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    canonical_fields = {
        "query_id",
        "parent_query_id",
        "judgment_query_id",
        "benchmark_language",
        "languages",
        "split",
        "variant_type",
        "verified",
        "review_status",
        "group_id",
        "source_partition",
        "text",
    }
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            query_id = _clean_id(
                _first(record, ("query_id", "queryid", "qid")), "query_id"
            )
            if query_id in seen:
                raise EvaluationError(f"Duplicate query_id {query_id!r}")
            seen.add(query_id)
            query_split = str(record.get("split", ""))
            if query_split not in ALL_SPLITS:
                raise EvaluationError(
                    f"Query {query_id!r} has invalid split {query_split!r}"
                )
            if split is not None and query_split != split:
                continue
            parent_query_id = _clean_id(
                record.get("parent_query_id", query_id), "parent_query_id"
            )
            verified = _as_bool(record.get("verified", True), field="verified")
            raw_judgment_id = record.get("judgment_query_id")
            judgment_query_id = (
                _clean_id(raw_judgment_id, "judgment_query_id")
                if raw_judgment_id is not None
                else None
            )
            if not verified and judgment_query_id is not None:
                raise EvaluationError(
                    f"Unverified query {query_id!r} must not inherit judgments"
                )
            raw_benchmark = record.get("benchmark_language")
            if raw_benchmark is None:
                raise EvaluationError(
                    f"Query {query_id!r} is missing benchmark_language"
                )
            benchmark_language = canonical_language(str(raw_benchmark))
            languages = _canonical_languages(record.get("languages"), benchmark_language)
            variant_type = str(record.get("variant_type", "original")).strip().casefold()
            if not variant_type:
                raise EvaluationError("variant_type must not be empty")
            review_status = str(
                record.get("review_status", "verified" if verified else "unreviewed")
            ).strip().casefold()
            metadata = {
                str(key): value
                for key, value in record.items()
                if key not in canonical_fields
            }
            queries.append(
                EvaluationQuery(
                    query_id=query_id,
                    parent_query_id=parent_query_id,
                    judgment_query_id=judgment_query_id,
                    benchmark_language=benchmark_language,
                    languages=languages,
                    split=query_split,
                    variant_type=variant_type,
                    verified=verified,
                    review_status=review_status,
                    metadata=metadata,
                )
            )
        except (EvaluationError, CorpusPreparationError) as exc:
            raise EvaluationError(f"{path}:{line_number}: {exc}") from exc
    if not queries:
        suffix = f" for split {split!r}" if split is not None else ""
        raise EvaluationError(f"No evaluation queries found in {path}{suffix}")
    return queries


def _parse_positive_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise EvaluationError(f"{field} must be a positive integer at {context}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid {field} {value!r} at {context}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise EvaluationError(f"{field} must be an integer at {context}")
    if parsed <= 0:
        raise EvaluationError(f"{field} must be positive at {context}")
    return parsed


def load_run(path: Path) -> list[RunEntry]:
    """Load and strictly validate a JSONL retrieval run."""

    entries: list[RunEntry] = []
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        context = f"{path}:{line_number}"
        try:
            query_id = _clean_id(
                _first(record, ("query_id", "queryid", "qid")), "query_id"
            )
            doc_id = _clean_id(
                _first(record, ("doc_id", "docid", "document_id")), "doc_id"
            )
            branch = _clean_id(record.get("branch"), "branch")
            rank = _parse_positive_int(record.get("rank"), field="rank", context=context)
            raw_score = record.get("score")
            if isinstance(raw_score, bool):
                raise EvaluationError(f"score must be numeric at {context}")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise EvaluationError(f"Invalid score {raw_score!r} at {context}") from exc
            if not math.isfinite(score):
                raise EvaluationError(f"score must be finite at {context}")
            entries.append(RunEntry(query_id, doc_id, branch, rank, score))
        except (EvaluationError, CorpusPreparationError) as exc:
            raise EvaluationError(f"{context}: {exc}") from exc
    _group_and_validate_run(entries)
    return entries


def _group_and_validate_run(entries: Sequence[RunEntry]) -> Mapping[str, tuple[RunEntry, ...]]:
    grouped: dict[str, list[RunEntry]] = defaultdict(list)
    for entry in entries:
        if not entry.query_id or not entry.doc_id or not entry.branch:
            raise EvaluationError("Run IDs and branch names must not be empty")
        if entry.rank <= 0:
            raise EvaluationError("Run ranks must be positive and one-based")
        if not math.isfinite(entry.score):
            raise EvaluationError("Run scores must be finite")
        grouped[entry.query_id].append(entry)

    validated: dict[str, tuple[RunEntry, ...]] = {}
    for query_id, query_entries in grouped.items():
        ranked = sorted(query_entries, key=lambda entry: entry.rank)
        doc_ids = [entry.doc_id for entry in ranked]
        ranks = [entry.rank for entry in ranked]
        branches = {entry.branch for entry in ranked}
        if len(doc_ids) != len(set(doc_ids)):
            raise EvaluationError(f"Run contains duplicate documents for query {query_id!r}")
        if len(ranks) != len(set(ranks)):
            raise EvaluationError(f"Run contains duplicate ranks for query {query_id!r}")
        if ranks != list(range(1, len(ranks) + 1)):
            raise EvaluationError(
                f"Run ranks for query {query_id!r} must be contiguous from 1"
            )
        if len(branches) != 1:
            raise EvaluationError(
                f"A system run must use one branch label per query; got {sorted(branches)!r} "
                f"for {query_id!r}"
            )
        validated[query_id] = tuple(ranked)
    return validated


def _qrels_by_query(qrels: Sequence[Qrel]) -> Mapping[str, Mapping[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        if qrel.relevance < 0:
            raise EvaluationError("Qrel relevance must be nonnegative")
        if qrel.doc_id in grouped[qrel.query_id]:
            raise EvaluationError(
                f"Duplicate qrel for {(qrel.query_id, qrel.doc_id)!r}"
            )
        grouped[qrel.query_id][qrel.doc_id] = qrel.relevance
    return grouped


def _gain(relevance: int) -> float:
    try:
        gain = math.pow(2.0, relevance) - 1.0
    except OverflowError as exc:
        raise EvaluationError(f"Relevance grade {relevance} is too large") from exc
    if not math.isfinite(gain):
        raise EvaluationError(f"Relevance grade {relevance} is too large")
    return gain


def _dcg(relevances: Iterable[int]) -> float:
    return sum(
        _gain(relevance) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def score_ranking(ranking: Sequence[RunEntry], judgments: Mapping[str, int]) -> Mapping[str, float]:
    """Score one ranking.  ``judgments`` must contain at least one positive."""

    positive_ids = {doc_id for doc_id, relevance in judgments.items() if relevance > 0}
    if not positive_ids:
        raise EvaluationError("Cannot score a query without positive judgments")
    top_ndcg = [judgments.get(entry.doc_id, 0) for entry in ranking[:NDCG_CUTOFF]]
    ideal = sorted(judgments.values(), reverse=True)[:NDCG_CUTOFF]
    ideal_dcg = _dcg(ideal)
    ndcg = _dcg(top_ndcg) / ideal_dcg

    reciprocal_rank = 0.0
    for entry in ranking[:MRR_CUTOFF]:
        if judgments.get(entry.doc_id, 0) > 0:
            reciprocal_rank = 1.0 / entry.rank
            break
    retrieved_relevant = {
        entry.doc_id
        for entry in ranking[:RECALL_CUTOFF]
        if judgments.get(entry.doc_id, 0) > 0
    }
    recall = len(retrieved_relevant) / len(positive_ids)
    return {
        "ndcg_at_10": ndcg,
        "mrr_at_10": reciprocal_rank,
        "recall_at_100": recall,
    }


def _aggregate(scores: Sequence[QueryScore]) -> Mapping[str, float] | None:
    evaluated = [score for score in scores if score.status == "evaluated"]
    if not evaluated:
        return None
    return {
        metric: statistics.fmean(
            float(getattr(score, metric)) for score in evaluated
        )
        for metric in METRIC_NAMES
    }


def _slice_scores(
    scores: Sequence[QueryScore], queries_by_id: Mapping[str, EvaluationQuery]
) -> Mapping[str, Any]:
    dimensions: dict[str, dict[str, list[QueryScore]]] = {
        "benchmark_language": defaultdict(list),
        "variant_type": defaultdict(list),
        "split": defaultdict(list),
    }
    optional_keys = (
        "variant_provenance",
        "provenance",
        "noise_severity",
        "ambiguity_level",
        "contains_entity",
        "contains_negation",
    )
    for score in scores:
        if score.status != "evaluated":
            continue
        query = queries_by_id[score.query_id]
        dimensions["benchmark_language"][query.benchmark_language].append(score)
        dimensions["variant_type"][query.variant_type].append(score)
        dimensions["split"][query.split].append(score)
        for key in optional_keys:
            if key in query.metadata and query.metadata[key] is not None:
                dimensions.setdefault(key, defaultdict(list))[str(query.metadata[key])].append(score)
    output: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        output[dimension] = {
            label: {
                "query_count": len(items),
                "metrics": _aggregate(items),
            }
            for label, items in sorted(values.items())
        }
    return output


def clean_to_noisy_drop(scores: Sequence[QueryScore]) -> Mapping[str, Any]:
    """Compute clean minus mean-noisy nDCG within each parent group."""

    grouped: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        if score.status == "evaluated":
            grouped[score.parent_query_id].append(score)
    per_parent: dict[str, float] = {}
    for parent_id, members in grouped.items():
        clean = [
            float(member.ndcg_at_10)
            for member in members
            if member.variant_type == "original"
        ]
        noisy = [
            float(member.ndcg_at_10)
            for member in members
            if member.variant_type != "original"
        ]
        if clean and noisy:
            per_parent[parent_id] = statistics.fmean(clean) - statistics.fmean(noisy)
    return {
        "definition": "mean parent-level original nDCG@10 minus mean noisy-variant nDCG@10",
        "paired_parent_count": len(per_parent),
        "mean_drop": statistics.fmean(per_parent.values()) if per_parent else None,
        "per_parent": dict(sorted(per_parent.items())),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise EvaluationError("Cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_parent_groups(
    scores: Sequence[QueryScore],
    *,
    samples: int = 1_000,
    seed: int = 13,
    confidence: float = 0.95,
) -> Mapping[str, Any] | None:
    """Bootstrap CIs after averaging correlated variants within each parent."""

    if samples < 0:
        raise EvaluationError("bootstrap samples must be nonnegative")
    if samples == 0:
        return None
    if not 0 < confidence < 1:
        raise EvaluationError("bootstrap confidence must be between 0 and 1")
    grouped: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        if score.status == "evaluated":
            grouped[score.parent_query_id].append(score)
    if not grouped:
        return None
    parent_ids = sorted(grouped)
    parent_values = {
        parent_id: {
            metric: statistics.fmean(
                float(getattr(score, metric)) for score in grouped[parent_id]
            )
            for metric in METRIC_NAMES
        }
        for parent_id in parent_ids
    }
    generator = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for _sample in range(samples):
        sampled_ids = [generator.choice(parent_ids) for _ in parent_ids]
        for metric in METRIC_NAMES:
            draws[metric].append(
                statistics.fmean(parent_values[parent_id][metric] for parent_id in sampled_ids)
            )
    tail = (1.0 - confidence) / 2.0
    intervals = {}
    for metric in METRIC_NAMES:
        values = sorted(draws[metric])
        intervals[metric] = {
            "lower": _percentile(values, tail),
            "upper": _percentile(values, 1.0 - tail),
        }
    return {
        "unit": "parent_query_id",
        "parent_group_count": len(parent_ids),
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "intervals": intervals,
    }


def evaluate_run(
    queries: Sequence[EvaluationQuery],
    qrels: Sequence[Qrel],
    run: Sequence[RunEntry],
    *,
    bootstrap_samples: int = 1_000,
    seed: int = 13,
    provenance: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate one system run against a declared query universe."""

    if not queries:
        raise EvaluationError("The declared query universe must not be empty")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise EvaluationError("The declared query universe contains duplicate query IDs")
    queries_by_id = {query.query_id: query for query in queries}
    run_by_query = _group_and_validate_run(run)
    unknown_run_queries = sorted(set(run_by_query) - set(queries_by_id))
    if unknown_run_queries:
        raise EvaluationError(
            "Run contains query IDs outside the declared universe: "
            + ", ".join(repr(item) for item in unknown_run_queries[:5])
        )
    judgments_by_query = _qrels_by_query(qrels)

    scored: list[QueryScore] = []
    for query in sorted(queries, key=lambda item: item.query_id):
        ranking = run_by_query.get(query.query_id, ())
        if not query.verified or query.judgment_query_id is None:
            scored.append(
                QueryScore(
                    query.query_id,
                    query.parent_query_id,
                    query.benchmark_language,
                    query.split,
                    query.variant_type,
                    "skipped_unverified",
                    len(ranking),
                    None,
                    None,
                    None,
                    query.metadata,
                )
            )
            continue
        judgments = judgments_by_query.get(query.judgment_query_id, {})
        if not any(relevance > 0 for relevance in judgments.values()):
            scored.append(
                QueryScore(
                    query.query_id,
                    query.parent_query_id,
                    query.benchmark_language,
                    query.split,
                    query.variant_type,
                    "skipped_no_positive",
                    len(ranking),
                    None,
                    None,
                    None,
                    query.metadata,
                )
            )
            continue
        metrics = score_ranking(ranking, judgments)
        scored.append(
            QueryScore(
                query.query_id,
                query.parent_query_id,
                query.benchmark_language,
                query.split,
                query.variant_type,
                "evaluated",
                len(ranking),
                metrics["ndcg_at_10"],
                metrics["mrr_at_10"],
                metrics["recall_at_100"],
                query.metadata,
            )
        )

    evaluated = [score for score in scored if score.status == "evaluated"]
    status_counts: dict[str, int] = defaultdict(int)
    for score in scored:
        status_counts[score.status] += 1
    per_language: dict[str, Any] = {}
    for language in sorted({score.benchmark_language for score in evaluated}):
        language_scores = [
            score for score in evaluated if score.benchmark_language == language
        ]
        per_language[language] = {
            "query_count": len(language_scores),
            "metrics": _aggregate(language_scores),
        }
    macro_languages = None
    if per_language:
        macro_languages = {
            metric: statistics.fmean(
                values["metrics"][metric] for values in per_language.values()
            )
            for metric in METRIC_NAMES
        }
    branches = sorted({entry.branch for entry in run})
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definitions": {
            "ndcg_at_10": "graded gain 2^relevance-1 with log2 discount",
            "mrr_at_10": "reciprocal rank of first relevance>0 passage",
            "recall_at_100": "fraction of relevance>0 judged passages retrieved",
            "unjudged_policy": "zero gain for metric computation; not known nonrelevance",
            "no_positive_policy": "exclude and report",
        },
        "declared_query_count": len(queries),
        "evaluated_query_count": len(evaluated),
        "status_counts": {
            status: status_counts.get(status, 0)
            for status in ("evaluated", "skipped_unverified", "skipped_no_positive")
        },
        "missing_run_query_count": sum(
            1 for score in evaluated if score.run_depth == 0
        ),
        "run_entry_count": len(run),
        "run_branches": branches,
        "overall": _aggregate(evaluated),
        "per_language": per_language,
        "macro_languages": macro_languages,
        "slices": _slice_scores(scored, queries_by_id),
        "robustness": clean_to_noisy_drop(scored),
        "bootstrap": bootstrap_parent_groups(
            scored, samples=bootstrap_samples, seed=seed
        ),
        "provenance": dict(provenance or {}),
    }
    # Catch accidental NaN/Infinity before artifacts are written.
    try:
        json.dumps(summary, allow_nan=False)
    except ValueError as exc:
        raise EvaluationError(f"Evaluation produced a non-finite value: {exc}") from exc
    return EvaluationReport(summary, tuple(scored))


def compare_reports(
    baseline: EvaluationReport,
    comparator: EvaluationReport,
    *,
    metric: str = "ndcg_at_10",
) -> Mapping[str, Any]:
    """Compute the fixed same-query normalization-harm comparison."""

    if metric not in METRIC_NAMES:
        raise EvaluationError(f"Unknown comparison metric {metric!r}")
    baseline_scores = {
        score.query_id: score
        for score in baseline.per_query
        if score.status == "evaluated"
    }
    comparator_scores = {
        score.query_id: score
        for score in comparator.per_query
        if score.status == "evaluated"
    }
    shared = sorted(set(baseline_scores) & set(comparator_scores))
    harmed = improved = tied = 0
    deltas: dict[str, float] = {}
    for query_id in shared:
        baseline_value = float(getattr(baseline_scores[query_id], metric))
        comparator_value = float(getattr(comparator_scores[query_id], metric))
        delta = comparator_value - baseline_value
        deltas[query_id] = delta
        if comparator_value < baseline_value:
            harmed += 1
        elif comparator_value > baseline_value:
            improved += 1
        else:
            tied += 1
    return {
        "definition": f"fraction whose comparator {metric} is lower than baseline",
        "metric": metric,
        "comparable_query_count": len(shared),
        "missing_pair_count": len(set(baseline_scores) | set(comparator_scores)) - len(shared),
        "harmed_count": harmed,
        "improved_count": improved,
        "tied_count": tied,
        "harm_rate": harmed / len(shared) if shared else None,
        "mean_comparator_minus_baseline": (
            statistics.fmean(deltas.values()) if deltas else None
        ),
        "per_query_delta": deltas,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise EvaluationError(f"Refusing to overwrite {path}; pass --force")
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=ALL_SPLITS,
        required=True,
        help="evaluate exactly one stored partition; prevents accidental split mixing",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="optional comparator baseline for normalization-harm reporting",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        queries = load_evaluation_queries(args.queries, split=args.split)
        qrels = load_qrels(args.qrels)
        run = load_run(args.run)
        provenance = {
            "queries_path": str(args.queries),
            "queries_sha256": _file_sha256(args.queries),
            "qrels_path": str(args.qrels),
            "qrels_sha256": _file_sha256(args.qrels),
            "run_path": str(args.run),
            "run_sha256": _file_sha256(args.run),
            "selected_split": args.split,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
        }
        report = evaluate_run(
            queries,
            qrels,
            run,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            provenance=provenance,
        )
        output = report.to_dict()
        if args.baseline_run is not None:
            baseline_entries = load_run(args.baseline_run)
            baseline_report = evaluate_run(
                queries,
                qrels,
                baseline_entries,
                bootstrap_samples=0,
                seed=args.seed,
            )
            output["normalization_harm"] = compare_reports(baseline_report, report)
            output["provenance"]["baseline_run_path"] = str(args.baseline_run)
            output["provenance"]["baseline_run_sha256"] = _file_sha256(args.baseline_run)
        _atomic_write_json(args.output, output, overwrite=args.force)
    except (EvaluationError, CorpusPreparationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"evaluation": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
