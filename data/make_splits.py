"""Create deterministic, language-stratified query-group split manifests.

Only official-training groups are divided into encoder training, fusion
training, and validation.  Official development queries are hard-locked to the
final-evaluation partition.  Variants inherit their parent information need's
assignment; related parents can be joined with an explicit ``group_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data.prepare_corpus import (
    CorpusPreparationError,
    _canonical_json_bytes,
    _clean_id,
    _first,
    _parse_source_spec,
    canonical_language,
    iter_jsonl,
)


SCHEMA_VERSION = 1
OFFICIAL_TRAIN = "official_train"
OFFICIAL_DEV = "official_dev"
ENCODER_TRAIN = "encoder_train"
FUSION_TRAIN = "fusion_train"
VALIDATION = "validation"
FINAL_EVALUATION = "final_evaluation"
TRAIN_SPLITS = (ENCODER_TRAIN, FUSION_TRAIN, VALIDATION)
ALL_SPLITS = (*TRAIN_SPLITS, FINAL_EVALUATION)
REVIEW_STATUSES = {"verified", "unreviewed", "rejected"}


class SplitError(ValueError):
    """Raised when query grouping or split isolation is invalid."""


@dataclass(frozen=True)
class SplitRatios:
    encoder_train: float = 0.70
    fusion_train: float = 0.15
    validation: float = 0.15

    def validate(self) -> None:
        values = (self.encoder_train, self.fusion_train, self.validation)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise SplitError("Split ratios must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SplitError("Split ratios must sum to 1")

    def as_mapping(self) -> Mapping[str, float]:
        return {
            ENCODER_TRAIN: self.encoder_train,
            FUSION_TRAIN: self.fusion_train,
            VALIDATION: self.validation,
        }


@dataclass(frozen=True)
class Query:
    """Canonical query plus immutable benchmark and grouping metadata."""

    query_id: str
    parent_query_id: str
    group_id: str
    text: str
    languages: tuple[str, ...]
    benchmark_language: str
    source_partition: str
    split: str | None
    variant_type: str
    verified: bool
    review_status: str
    judgment_query_id: str | None


@dataclass(frozen=True)
class SplitResult:
    queries: tuple[Query, ...]
    manifest: Mapping[str, Any]


def _canonical_languages(value: Any, *, default_language: str | None) -> tuple[str, ...]:
    if value is None:
        if default_language is None:
            raise SplitError("Query languages are missing and no default language was supplied")
        raw_values = [default_language]
    elif isinstance(value, str):
        raw_values = [item for item in value.replace(",", " ").split() if item]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raise SplitError("languages must be a string or list")
    if not raw_values:
        raise SplitError("languages must not be empty")
    try:
        return tuple(sorted({canonical_language(str(item)) for item in raw_values}))
    except CorpusPreparationError as exc:
        raise SplitError(str(exc)) from exc


def _review_metadata(
    record: Mapping[str, Any], *, variant_type: str, parent_query_id: str
) -> tuple[str, bool, str | None]:
    raw_status = record.get("review_status")
    raw_verified = record.get("verified")
    if raw_status is None:
        if raw_verified is None:
            status = "verified" if variant_type == "original" else "unreviewed"
        elif isinstance(raw_verified, bool):
            status = "verified" if raw_verified else "unreviewed"
        else:
            raise SplitError("verified must be a JSON boolean")
    else:
        status = str(raw_status).strip().casefold()
        if status not in REVIEW_STATUSES:
            raise SplitError(
                f"Invalid review_status {raw_status!r}; expected one of "
                f"{', '.join(sorted(REVIEW_STATUSES))}"
            )
        if raw_verified is not None:
            if not isinstance(raw_verified, bool):
                raise SplitError("verified must be a JSON boolean")
            if raw_verified != (status == "verified"):
                raise SplitError("verified conflicts with review_status")
    if variant_type == "original" and status != "verified":
        raise SplitError("Original benchmark queries must be verified")
    verified = status == "verified"
    explicit_judgment = record.get("judgment_query_id")
    if explicit_judgment is not None:
        judgment_query_id = _clean_id(explicit_judgment, "judgment_query_id")
        if not verified:
            raise SplitError("Only verified queries may have a judgment_query_id")
    elif verified:
        judgment_query_id = parent_query_id
    else:
        judgment_query_id = None
    return status, verified, judgment_query_id


def canonical_query(
    record: Mapping[str, Any],
    *,
    source_partition: str,
    default_language: str | None = None,
) -> Query:
    """Validate and normalize one source query without assigning a split."""

    if source_partition not in {OFFICIAL_TRAIN, OFFICIAL_DEV}:
        raise SplitError(f"Invalid source_partition {source_partition!r}")
    query_id = _clean_id(_first(record, ("query_id", "queryid", "qid", "id")), "query_id")
    parent_query_id = _clean_id(record.get("parent_query_id", query_id), "parent_query_id")
    group_id = _clean_id(record.get("group_id", parent_query_id), "group_id")
    raw_text = _first(record, ("text", "query", "query_text"))
    if raw_text is None:
        raise SplitError(f"Query {query_id!r} is missing text")
    text = unicodedata.normalize("NFC", str(raw_text))
    if not text.strip():
        raise SplitError(f"Query {query_id!r} has empty text")
    if "\x00" in text:
        raise SplitError(f"Query {query_id!r} contains a NUL byte")
    languages = _canonical_languages(record.get("languages", record.get("language")), default_language=default_language)
    raw_benchmark_language = record.get("benchmark_language", default_language)
    if raw_benchmark_language is None:
        if len(languages) != 1:
            raise SplitError(
                f"Query {query_id!r} needs benchmark_language for a mixed-language record"
            )
        raw_benchmark_language = languages[0]
    try:
        benchmark_language = canonical_language(str(raw_benchmark_language))
    except CorpusPreparationError as exc:
        raise SplitError(str(exc)) from exc
    variant_type = str(record.get("variant_type", "original")).strip().casefold()
    if not variant_type:
        raise SplitError("variant_type must not be empty")
    review_status, verified, judgment_query_id = _review_metadata(
        record, variant_type=variant_type, parent_query_id=parent_query_id
    )
    record_partition = record.get("source_partition")
    if record_partition is not None and str(record_partition) != source_partition:
        raise SplitError(
            f"Query {query_id!r} source_partition conflicts with its input collection"
        )
    return Query(
        query_id=query_id,
        parent_query_id=parent_query_id,
        group_id=group_id,
        text=text,
        languages=languages,
        benchmark_language=benchmark_language,
        source_partition=source_partition,
        split=None,
        variant_type=variant_type,
        verified=verified,
        review_status=review_status,
        judgment_query_id=judgment_query_id,
    )


def load_queries(
    path: Path, *, source_partition: str, default_language: str | None = None
) -> list[Query]:
    """Load canonical or MIRACL-like query JSONL."""

    queries: list[Query] = []
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            queries.append(
                canonical_query(
                    record,
                    source_partition=source_partition,
                    default_language=default_language,
                )
            )
        except (SplitError, CorpusPreparationError) as exc:
            raise SplitError(f"{path}:{line_number}: {exc}") from exc
    return queries


def _stable_group_key(seed: int, language: str, group_id: str) -> tuple[bytes, str]:
    payload = f"split-v1\0{seed}\0{language}\0{group_id}".encode("utf-8")
    return hashlib.sha256(payload).digest(), group_id


def _largest_remainder_counts(total: int, ratios: SplitRatios) -> Mapping[str, int]:
    ratio_map = ratios.as_mapping()
    exact = {name: total * value for name, value in ratio_map.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = {name: index for index, name in enumerate(TRAIN_SPLITS)}
    by_remainder = sorted(
        TRAIN_SPLITS,
        key=lambda name: (-(exact[name] - counts[name]), order[name]),
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1
    return counts


def _validate_queries(train_queries: Sequence[Query], final_queries: Sequence[Query]) -> None:
    all_queries = [*train_queries, *final_queries]
    if not all_queries:
        raise SplitError("At least one query is required")
    by_id: dict[str, Query] = {}
    for expected_partition, collection in (
        (OFFICIAL_TRAIN, train_queries),
        (OFFICIAL_DEV, final_queries),
    ):
        for query in collection:
            if query.query_id in by_id:
                raise SplitError(f"Duplicate query_id {query.query_id!r}")
            by_id[query.query_id] = query
            if query.source_partition != expected_partition:
                raise SplitError(
                    f"Query {query.query_id!r} has source_partition "
                    f"{query.source_partition!r}, expected {expected_partition!r}"
                )
            if query.split is not None:
                raise SplitError(f"Query {query.query_id!r} is already assigned to a split")

    for query in all_queries:
        parent = by_id.get(query.parent_query_id)
        if parent is None:
            raise SplitError(
                f"Query {query.query_id!r} references unknown parent "
                f"{query.parent_query_id!r}"
            )
        if parent.source_partition != query.source_partition:
            raise SplitError(
                f"Parent {parent.query_id!r} crosses official source partitions"
            )
        if parent.group_id != query.group_id:
            raise SplitError(
                f"Query {query.query_id!r} and parent {parent.query_id!r} have "
                "different group_id values"
            )
        if parent.benchmark_language != query.benchmark_language:
            raise SplitError(
                f"Query {query.query_id!r} and parent {parent.query_id!r} have "
                "different benchmark_language values"
            )
        if query.variant_type == "original":
            if query.parent_query_id != query.query_id:
                raise SplitError(
                    f"Original query {query.query_id!r} must be its own parent"
                )
        elif parent.variant_type != "original" or parent.parent_query_id != parent.query_id:
            raise SplitError(
                f"Variant {query.query_id!r} must reference an original parent query"
            )
        if query.verified:
            judgment_source = by_id.get(query.judgment_query_id or "")
            if judgment_source is None:
                raise SplitError(
                    f"Query {query.query_id!r} references unknown judgment_query_id "
                    f"{query.judgment_query_id!r}"
                )
            if judgment_source.variant_type != "original" or not judgment_source.verified:
                raise SplitError(
                    f"Query {query.query_id!r} must inherit judgments from a verified original"
                )
            if judgment_source.source_partition != query.source_partition:
                raise SplitError(
                    f"Query {query.query_id!r} judgment source crosses official partitions"
                )
            if judgment_source.group_id != query.group_id:
                raise SplitError(
                    f"Query {query.query_id!r} judgment source crosses information-need groups"
                )
            if judgment_source.benchmark_language != query.benchmark_language:
                raise SplitError(
                    f"Query {query.query_id!r} judgment source crosses language strata"
                )

    groups: dict[str, list[Query]] = defaultdict(list)
    for query in all_queries:
        groups[query.group_id].append(query)
    for group_id, members in groups.items():
        partitions = {member.source_partition for member in members}
        languages = {member.benchmark_language for member in members}
        if len(partitions) != 1:
            raise SplitError(f"Group {group_id!r} crosses official source partitions")
        if len(languages) != 1:
            raise SplitError(f"Group {group_id!r} crosses benchmark-language strata")


def _query_input_record(query: Query) -> Mapping[str, Any]:
    record = asdict(query)
    record["languages"] = list(query.languages)
    record.pop("split")
    return record


def make_splits(
    train_queries: Sequence[Query],
    final_queries: Sequence[Query] = (),
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 13,
    dataset_version: str = "unspecified",
) -> SplitResult:
    """Assign whole query groups and return a reproducible manifest."""

    ratios.validate()
    _validate_queries(train_queries, final_queries)
    train_groups: dict[str, list[Query]] = defaultdict(list)
    final_groups: dict[str, list[Query]] = defaultdict(list)
    for query in train_queries:
        train_groups[query.group_id].append(query)
    for query in final_queries:
        final_groups[query.group_id].append(query)

    by_language: dict[str, list[str]] = defaultdict(list)
    for group_id, members in train_groups.items():
        by_language[members[0].benchmark_language].append(group_id)

    group_to_split: dict[str, str] = {}
    allocation: dict[str, Mapping[str, int]] = {}
    for language in sorted(by_language):
        group_ids = sorted(
            by_language[language],
            key=lambda group_id: _stable_group_key(seed, language, group_id),
        )
        counts = _largest_remainder_counts(len(group_ids), ratios)
        allocation[language] = counts
        offset = 0
        for split in TRAIN_SPLITS:
            next_offset = offset + counts[split]
            for group_id in group_ids[offset:next_offset]:
                group_to_split[group_id] = split
            offset = next_offset
    for group_id in final_groups:
        group_to_split[group_id] = FINAL_EVALUATION

    assigned = [
        replace(query, split=group_to_split[query.group_id])
        for query in [*train_queries, *final_queries]
    ]
    assigned.sort(key=lambda query: query.query_id)

    query_counts = Counter(query.split for query in assigned)
    group_counts = Counter(group_to_split.values())
    language_group_counts: dict[str, dict[str, int]] = {}
    for language in sorted({query.benchmark_language for query in assigned}):
        relevant_groups = {
            query.group_id: query.split
            for query in assigned
            if query.benchmark_language == language
        }
        counter = Counter(relevant_groups.values())
        language_group_counts[language] = {
            split: counter.get(split, 0) for split in ALL_SPLITS
        }

    group_assignments = []
    all_groups = {**train_groups, **final_groups}
    for group_id in sorted(all_groups):
        members = sorted(all_groups[group_id], key=lambda query: query.query_id)
        group_assignments.append(
            {
                "group_id": group_id,
                "benchmark_language": members[0].benchmark_language,
                "source_partition": members[0].source_partition,
                "split": group_to_split[group_id],
                "query_ids": [member.query_id for member in members],
                "parent_query_ids": sorted({member.parent_query_id for member in members}),
            }
        )

    sorted_inputs = sorted(
        (_query_input_record(query) for query in [*train_queries, *final_queries]),
        key=lambda record: str(record["query_id"]),
    )
    input_sha256 = hashlib.sha256(
        b"\n".join(_canonical_json_bytes(record) for record in sorted_inputs) + b"\n"
    ).hexdigest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "seed": seed,
        "ratios": ratios.as_mapping(),
        "split_method": "language_stratified_group_hash_v1",
        "grouping_policy": "explicit_group_id_else_parent_query_id",
        "official_dev_policy": "locked_final_evaluation",
    }
    config_hash = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    manifest: dict[str, Any] = {
        **config,
        "config_hash": config_hash,
        "input_sha256": input_sha256,
        "query_count": len(assigned),
        "group_count": len(group_to_split),
        "query_counts_by_split": {
            split: query_counts.get(split, 0) for split in ALL_SPLITS
        },
        "group_counts_by_split": {
            split: group_counts.get(split, 0) for split in ALL_SPLITS
        },
        "group_counts_by_language_and_split": language_group_counts,
        "training_group_allocation_by_language": allocation,
        "group_assignments": group_assignments,
    }
    return SplitResult(tuple(assigned), manifest)


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


def write_split_artifacts(
    result: SplitResult, output_dir: Path, *, overwrite: bool = False
) -> Mapping[str, Path]:
    paths = {
        "queries": output_dir / "queries.jsonl",
        "manifest": output_dir / "split_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SplitError(
            "Refusing to overwrite existing artifact(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --force"
        )
    records = []
    for query in result.queries:
        record = asdict(query)
        record["languages"] = list(query.languages)
        records.append(record)
    _atomic_write_jsonl(paths["queries"], records)
    _atomic_write_json(paths["manifest"], result.manifest)
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-queries",
        action="append",
        default=[],
        metavar="LANGUAGE=PATH",
        help="official-training query JSONL; repeat for each language",
    )
    parser.add_argument(
        "--dev-queries",
        action="append",
        default=[],
        metavar="LANGUAGE=PATH",
        help="official-development query JSONL, locked to final evaluation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-ratio", type=float, default=0.70)
    parser.add_argument("--fusion-ratio", type=float, default=0.15)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset-version", default="unspecified")
    parser.add_argument("--force", action="store_true")
    return parser


def _load_specs(values: Sequence[str], *, partition: str, option: str) -> list[Query]:
    queries: list[Query] = []
    seen_languages: set[str] = set()
    for value in values:
        language, path = _parse_source_spec(value, option=option)
        if language in seen_languages:
            raise SplitError(f"Duplicate {option} language {language!r}")
        seen_languages.add(language)
        queries.extend(
            load_queries(
                path,
                source_partition=partition,
                default_language=language,
            )
        )
    return queries


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if not args.train_queries and not args.dev_queries:
            raise SplitError("At least one --train-queries or --dev-queries input is required")
        train_queries = _load_specs(
            args.train_queries, partition=OFFICIAL_TRAIN, option="--train-queries"
        )
        final_queries = _load_specs(
            args.dev_queries, partition=OFFICIAL_DEV, option="--dev-queries"
        )
        result = make_splits(
            train_queries,
            final_queries,
            ratios=SplitRatios(
                args.encoder_ratio, args.fusion_ratio, args.validation_ratio
            ),
            seed=args.seed,
            dataset_version=args.dataset_version,
        )
        paths = write_split_artifacts(result, args.output_dir, overwrite=args.force)
    except (SplitError, CorpusPreparationError, argparse.ArgumentTypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
