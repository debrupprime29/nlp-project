"""Generate partition-bound query variants and a content-bound review queue.

Automatic generation is limited to conservative typographic edits.  Full-query
Romanized, code-mixed, mixed-script, and human-written proposals must come from
an explicit source.  Every new candidate stays outside the canonical retrieval
queries until a reviewer verifies unchanged meaning.  No model or download is
needed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.make_splits import (
    ALL_SPLITS,
    FINAL_EVALUATION,
    OFFICIAL_DEV,
    OFFICIAL_TRAIN,
    SplitError,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_languages,
    _validate_queries,
    canonical_query,
)
from data.prepare_corpus import (
    CorpusPreparationError,
    _clean_id,
    iter_jsonl,
)
from nlp.text import (
    TextPreparationError,
    normalize_processing_text,
    protection_spans,
    script_bucket,
    tokenize_with_spans,
)


SCHEMA_VERSION = 1
MAX_VARIANT_QUERY_CHARACTERS = 4096
PROPOSAL_TYPES = frozenset(
    {"romanized", "code_mixed", "mixed_script", "code_mixed_noisy", "human_written"}
)
QUESTION_WORDS = frozenset(
    {
        "what", "which", "where", "when", "whose", "how", "why", "who", "whom",
        "kya", "kyu", "kyun", "kaise", "kab", "kahan", "kahaan", "kaun",
        "koto", "keno", "kothay", "kokhon", "ki", "kise", "kis", "kon",
    }
)


class VariantError(ValueError):
    """Raised when generation would violate split, review, or provenance rules."""


@dataclass(frozen=True)
class VariantResult:
    queries: tuple[Mapping[str, Any], ...]
    candidates: tuple[Mapping[str, Any], ...]
    review_queue: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


def _mapping(record: Any) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        record = asdict(record)
    if not isinstance(record, Mapping):
        raise VariantError("query/proposal/review records must be objects")
    return dict(record)


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except (TypeError, ValueError) as exc:
        raise VariantError("metadata must be JSON serializable") from exc


def _normalized_key(text: str) -> str:
    return " ".join(normalize_processing_text(text).casefold().split())


def _assigned_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    canonical = []
    output: list[dict[str, Any]] = []
    for position, value in enumerate(records, start=1):
        record = _mapping(value)
        partition = record.get("source_partition")
        assigned_split = record.get("split")
        if assigned_split not in ALL_SPLITS:
            raise VariantError(f"query {position} must already have a valid split assignment")
        if partition == OFFICIAL_DEV and assigned_split != FINAL_EVALUATION:
            raise VariantError("official development queries are locked to final_evaluation")
        if partition == OFFICIAL_TRAIN and assigned_split == FINAL_EVALUATION:
            raise VariantError("official training queries cannot enter final_evaluation")
        try:
            query = canonical_query(record, source_partition=str(partition))
            normalize_processing_text(query.text)
            if len(query.text) > MAX_VARIANT_QUERY_CHARACTERS:
                raise VariantError(f"query exceeds the {MAX_VARIANT_QUERY_CHARACTERS}-character variant safety limit")
        except (SplitError, CorpusPreparationError, TextPreparationError) as exc:
            raise VariantError(f"query {position}: {exc}") from exc
        canonical.append(query)
        normalized = {**record, **asdict(query), "split": assigned_split}
        normalized["languages"] = list(query.languages)
        output.append(normalized)
    try:
        _validate_queries(
            [query for query in canonical if query.source_partition == OFFICIAL_TRAIN],
            [query for query in canonical if query.source_partition == OFFICIAL_DEV],
        )
    except SplitError as exc:
        raise VariantError(str(exc)) from exc
    by_id = {record["query_id"]: record for record in output}
    group_splits: dict[str, str] = {}
    for record in output:
        parent = by_id[record["parent_query_id"]]
        if parent["split"] != record["split"]:
            raise VariantError(f"query {record['query_id']!r} crosses its parent split")
        prior_split = group_splits.setdefault(record["group_id"], record["split"])
        if prior_split != record["split"]:
            raise VariantError(f"group {record['group_id']!r} crosses split assignments")
        if record["verified"] and record["judgment_query_id"] != parent["judgment_query_id"]:
            raise VariantError("variants must inherit the original parent's judgment_query_id")
    return sorted(output, key=lambda record: record["query_id"])


def load_assigned_queries(path: Path) -> list[Mapping[str, Any]]:
    """Load a split artifact, validating all parent/group/source assignments."""

    try:
        return _assigned_records(list(iter_jsonl(Path(path))))
    except CorpusPreparationError as exc:
        raise VariantError(str(exc)) from exc


def _provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VariantError("provenance must be an object with source and revision")
    result = dict(value)
    for field in ("source", "revision"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise VariantError(f"provenance.{field} must be a nonempty string")
        result[field] = result[field].strip()
    _digest(result)
    return result


def _explicit_spans(parent: Mapping[str, Any]) -> Sequence[Any]:
    value = parent.get("protected_spans", ())
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise VariantError("protected_spans must be a list")
    return value


def _protected(parent: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return protection_spans(parent["text"], _explicit_spans(parent))
    except TextPreparationError as exc:
        raise VariantError(f"parent {parent['query_id']!r}: {exc}") from exc


def _word_character(character: str) -> bool:
    return unicodedata.category(character).startswith(("L", "M", "N")) or character in {"\u200c", "\u200d"}


def _exact_occurrences(text: str, fragment: str) -> int:
    count = 0
    offset = 0
    while True:
        start = text.find(fragment, offset)
        if start < 0:
            return count
        end = start + len(fragment)
        left_ok = not (_word_character(fragment[0]) and start and _word_character(text[start - 1]))
        right_ok = not (_word_character(fragment[-1]) and end < len(text) and _word_character(text[end]))
        if left_ok and right_ok:
            count += 1
        offset = start + 1


def _safety_reason(parent: Mapping[str, Any], text: str) -> str | None:
    spans = _protected(parent)
    required = Counter(parent["text"][span.start:span.end] for span in spans)
    for fragment, count in sorted(required.items()):
        if _exact_occurrences(text, fragment) != count:
            return f"protected text changed or duplicated: {fragment!r}"
    # Inserting a new number, date, or negation can change meaning even when all
    # original protected fragments survived.  Compare these values in both texts.
    semantic_reasons = {"number", "date", "negation", "url", "email"}
    def values(value: str) -> Counter[str]:
        return Counter(
            value[span.start:span.end]
            for span in protection_spans(value)
            if semantic_reasons.intersection(span.reasons)
        )
    if values(parent["text"]) != values(text):
        return "numbers, dates, negation, URLs, or email values changed"
    return None


def _candidate_protections(parent: Mapping[str, Any], text: str, *, safe: bool) -> list[dict[str, Any]]:
    explicit = []
    if safe:
        used: Counter[str] = Counter()
        for span in _protected(parent):
            fragment = parent["text"][span.start:span.end]
            matches = []
            offset = 0
            while True:
                start = text.find(fragment, offset)
                if start < 0:
                    break
                end = start + len(fragment)
                if (
                    not (_word_character(fragment[0]) and start and _word_character(text[start - 1]))
                    and not (_word_character(fragment[-1]) and end < len(text) and _word_character(text[end]))
                ):
                    matches.append(start)
                offset = start + 1
            start = matches[used[fragment]]
            used[fragment] += 1
            explicit.append({"start": start, "end": start + len(fragment), "reasons": list(span.reasons)})
    return [
        {"start": span.start, "end": span.end, "text": text[span.start:span.end], "reasons": list(span.reasons)}
        for span in protection_spans(text, explicit)
    ]


def _candidate(
    parent: Mapping[str, Any],
    *,
    text: str,
    variant_type: str,
    languages: Sequence[str],
    provenance: Mapping[str, Any],
    author_id: str | None = None,
    noise_severity: Any = None,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text = normalize_processing_text(text)
    if not text.strip():
        raise VariantError("candidate text must not be empty")
    if len(text) > MAX_VARIANT_QUERY_CHARACTERS:
        raise VariantError(f"candidate exceeds the {MAX_VARIANT_QUERY_CHARACTERS}-character variant safety limit")
    declared_languages = tuple(languages)
    if variant_type == "romanized":
        if set(declared_languages) != set(parent["languages"]):
            raise VariantError("romanization must preserve the parent query languages")
        languages = parent["languages"]
    if noise_severity is not None:
        _digest(noise_severity)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "parent": {key: parent[key] for key in ("query_id", "text", "group_id", "split", "source_partition", "benchmark_language")},
        "variant_type": variant_type,
        "text": text,
        "languages": list(languages),
        "declared_languages": list(declared_languages),
        "provenance": dict(provenance),
        "author_id": author_id,
    }
    digest = _digest(identity)
    candidate_id = existing["query_id"] if existing else f"{parent['query_id']}::v-{digest[:24]}"
    if existing and any(key in existing for key in ("candidate_id", "candidate_sha256", "review")):
        if existing.get("candidate_id") != candidate_id:
            raise VariantError("stored candidate_id conflicts with query_id")
        if existing.get("candidate_sha256") != digest:
            raise VariantError(f"stale or altered stored candidate identity for {candidate_id!r}")
        stored_review = existing.get("review")
        if stored_review is None:
            if existing["verified"]:
                raise VariantError("verified bound candidates must retain their content-bound review")
        elif not isinstance(stored_review, Mapping):
            raise VariantError("stored candidate review must be an object")
        else:
            if stored_review.get("candidate_id") != candidate_id or stored_review.get("candidate_sha256") != digest:
                raise VariantError(f"stale or altered stored review for {candidate_id!r}")
            if stored_review.get("decision") != existing["review_status"]:
                raise VariantError("stored review decision conflicts with candidate review_status")
            if stored_review.get("reviewer_id") != existing.get("reviewer_id"):
                raise VariantError("stored reviewer_id conflicts with bound review")
    reason = _safety_reason(parent, text)
    if variant_type == "romanized":
        if any(character.isalpha() and script_bucket(character) != "latin" for character in text):
            reason = "romanized proposals must contain only Latin-script letters"
    if _normalized_key(text) == _normalized_key(parent["text"]):
        reason = "candidate duplicates the original query"
    record = {
        **(dict(existing) if existing else {}),
        "query_id": candidate_id,
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "parent_query_id": parent["query_id"],
        "group_id": parent["group_id"],
        "source_partition": parent["source_partition"],
        "split": parent["split"],
        "benchmark_language": parent["benchmark_language"],
        "text": text,
        "languages": list(languages),
        "variant_type": variant_type,
        "verified": bool(existing and existing["verified"] and reason is None),
        "review_status": "rejected" if reason else (existing["review_status"] if existing else "unreviewed"),
        "judgment_query_id": parent["judgment_query_id"] if existing and existing["verified"] and reason is None else None,
        "provenance": dict(provenance),
        "generation_kind": "human_written" if variant_type == "human_written" else "synthetic",
        "variant_provenance": "human_written" if variant_type == "human_written" else "synthetic",
        "contains_entity": any(
            {"name_heuristic", "acronym", "explicit"}.intersection(span.reasons)
            for span in _protected(parent)
        ),
        "contains_negation": any("negation" in span.reasons for span in _protected(parent)),
        "noise_severity": noise_severity if noise_severity is not None else {"label": "unspecified"},
        "parent_protected_spans": [
            {"start": span.start, "end": span.end, "text": parent["text"][span.start:span.end], "reasons": list(span.reasons)}
            for span in _protected(parent)
        ],
        "parent_protected_span_coordinate_space": "parent_nfc_text",
        "protected_spans": _candidate_protections(parent, text, safe=reason is None),
        "protected_span_coordinate_space": "nfc_text",
        "safety_status": "rejected" if reason else "safe",
    }
    if author_id is not None:
        record["author_id"] = author_id
    if tuple(languages) != declared_languages:
        record["declared_languages"] = list(declared_languages)
    if reason:
        record["rejection_reason"] = reason
    return record


def _typo_candidates(parent: Mapping[str, Any], *, seed: int) -> list[dict[str, Any]]:
    protected = _protected(parent)
    options: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for token in tokenize_with_spans(parent["text"]):
        if (
            len(token.text) < 5
            or not token.text.isascii()
            or token.script != "latin"
            or not token.text.islower()
            or not all(script_bucket(character) == "latin" for character in token.text)
            or token.text.casefold() in QUESTION_WORDS
            or any(token.start < span.end and token.end > span.start for span in protected)
        ):
            continue
        for operation in ("adjacent_swap", "deletion"):
            stop = len(token.text) - (2 if operation == "adjacent_swap" else 1)
            for position in range(1, stop):
                if operation == "adjacent_swap":
                    changed = token.text[:position] + token.text[position + 1] + token.text[position] + token.text[position + 2:]
                else:
                    changed = token.text[:position] + token.text[position + 1:]
                if changed == token.text:
                    continue
                text = parent["text"][:token.start] + changed + parent["text"][token.end:]
                if text in seen:
                    continue
                seen.add(text)
                provenance = {
                    "source": "conservative_typographic_generator",
                    "revision": "1",
                    "seed": seed,
                    "operation": operation,
                    "token_start": token.start,
                    "token_end": token.end,
                    "character_offset": position,
                }
                record = _candidate(
                    parent,
                    text=text,
                    variant_type="typo_only",
                    languages=parent["languages"],
                    provenance=provenance,
                    noise_severity={"character_edits": 1, "token_edits": 1, "character_edit_fraction": 1 / len(parent["text"])},
                )
                priority = _digest({"seed": seed, "parent": parent["query_id"], "candidate": record["candidate_sha256"]})
                options.append((priority, record))
    return [record for _, record in sorted(options, key=lambda item: (item[0], item[1]["candidate_id"]))]


def _review_candidates(
    candidates: list[dict[str, Any]], reviews: Sequence[Mapping[str, Any]], parents: Mapping[str, Mapping[str, Any]]
) -> None:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    seen: dict[str, dict[str, Any]] = {}
    for value in reviews:
        review = _mapping(value)
        try:
            candidate_id = _clean_id(review.get("candidate_id"), "candidate_id")
            reviewer = _clean_id(review.get("reviewer_id"), "reviewer_id")
            parent_id = _clean_id(review.get("parent_query_id"), "parent_query_id")
        except CorpusPreparationError as exc:
            raise VariantError(str(exc)) from exc
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise VariantError(f"review references unknown candidate {candidate_id!r}")
        if review.get("candidate_sha256") != candidate["candidate_sha256"]:
            raise VariantError(f"stale or missing candidate_sha256 for {candidate_id!r}")
        if parent_id != candidate["parent_query_id"]:
            raise VariantError(f"review parent_query_id mismatch for {candidate_id!r}")
        decision = review.get("decision")
        if decision not in {"verified", "rejected"}:
            raise VariantError("review decision must be verified or rejected")
        if decision == "verified" and review.get("meaning_preserved") is not True:
            raise VariantError("verified reviews require meaning_preserved=true")
        if "meaning_preserved" in review and not isinstance(review["meaning_preserved"], bool):
            raise VariantError("review meaning_preserved must be a JSON boolean")
        if candidate.get("author_id") == reviewer:
            raise VariantError("human-written variants require an independent reviewer")
        if "fluent_languages" in review:
            try:
                fluent = _canonical_languages(review["fluent_languages"], default_language=None)
            except SplitError as exc:
                raise VariantError(str(exc)) from exc
            if not set(candidate["languages"]).issubset(fluent):
                raise VariantError("reviewer fluent_languages must cover the candidate languages")
        if decision == "verified" and candidate["safety_status"] != "safe":
            raise VariantError(f"cannot verify unsafe candidate {candidate_id!r}")
        previous = seen.get(candidate_id)
        if previous is not None:
            if previous != review:
                raise VariantError(f"contradictory or duplicate reviews for {candidate_id!r}")
            continue
        seen[candidate_id] = review
        candidate["review"] = dict(review)
        candidate["reviewer_id"] = reviewer
        candidate["review_status"] = decision
        candidate["verified"] = decision == "verified"
        candidate["judgment_query_id"] = parents[parent_id]["judgment_query_id"] if decision == "verified" else None


def generate_variants(
    records: Sequence[Mapping[str, Any]],
    *,
    proposals: Sequence[Mapping[str, Any]] = (),
    reviews: Sequence[Mapping[str, Any]] = (),
    seed: int = 13,
    typo_variants: int = 1,
    max_variants_per_parent: int = 6,
    split: str | None = None,
) -> VariantResult:
    """Return verified retrieval queries plus a separate, auditable review queue.

    Call once to obtain candidate IDs and hashes, then call with those bound
    reviews to approve candidates.  Input must already have split assignments.
    Split selection affects output only after all input assignments validate.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise VariantError("seed must be an integer")
    for name, value in (("typo_variants", typo_variants), ("max_variants_per_parent", max_variants_per_parent)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VariantError(f"{name} must be a nonnegative integer")
    if split is not None and split not in ALL_SPLITS:
        raise VariantError(f"invalid split {split!r}")
    assigned = _assigned_records(records)
    selected = [record for record in assigned if split is None or record["split"] == split]
    parents = {record["query_id"]: record for record in selected if record["variant_type"] == "original"}
    all_parents = {record["query_id"]: record for record in assigned if record["variant_type"] == "original"}
    candidates: list[dict[str, Any]] = []
    for record in selected:
        if record["variant_type"] == "original":
            _protected(record)
            continue
        provenance = _provenance(record.get("provenance", {"source": "existing_query_artifact", "revision": "1"}))
        candidates.append(_candidate(
            parents[record["parent_query_id"]], text=record["text"], variant_type=record["variant_type"],
            languages=record["languages"], provenance=provenance, author_id=record.get("author_id"),
            noise_severity=record.get("noise_severity"), existing=record,
        ))
    new_candidates = []
    parents_without_eligible_typo = 0
    for parent in parents.values():
        if typo_variants:
            typos = _typo_candidates(parent, seed=seed)
            if not typos:
                parents_without_eligible_typo += 1
            new_candidates.extend(typos[:typo_variants])
    for value in proposals:
        proposal = _mapping(value)
        parent_id = proposal.get("parent_query_id")
        if parent_id not in all_parents:
            raise VariantError(f"proposal references unknown original parent {parent_id!r}")
        if parent_id not in parents:
            continue
        variant_type = proposal.get("variant_type")
        if variant_type not in PROPOSAL_TYPES:
            raise VariantError(f"unsupported proposal variant_type {variant_type!r}")
        if not isinstance(proposal.get("text"), str):
            raise VariantError("proposal text must be a string")
        try:
            languages = _canonical_languages(proposal.get("languages"), default_language=None)
            provenance = _provenance(proposal.get("provenance"))
            author_id = proposal.get("author_id", provenance.get("author_id"))
            if author_id is not None:
                author_id = _clean_id(author_id, "author_id")
            if variant_type == "human_written" and author_id is None:
                raise VariantError("human_written proposals require author_id")
            new_candidates.append(_candidate(
                parents[parent_id], text=proposal["text"], variant_type=variant_type, languages=languages,
                provenance=provenance, author_id=author_id, noise_severity=proposal.get("noise_severity"),
            ))
        except (SplitError, CorpusPreparationError, TextPreparationError) as exc:
            raise VariantError(str(exc)) from exc
    # Existing IDs remain stable.  New candidate ordering does not depend on
    # proposal order; a duplicate query text can never gain a second qrel copy.
    seen_ids = {record["query_id"] for record in selected if record["variant_type"] == "original"}
    seen_text: dict[str, set[str]] = {parent_id: {_normalized_key(parent["text"])} for parent_id, parent in parents.items()}
    counts: Counter[str] = Counter()
    unique: list[dict[str, Any]] = []
    for candidate in [*candidates, *sorted(new_candidates, key=lambda item: item["candidate_id"])]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen_ids:
            prior = next((item for item in unique if item["candidate_id"] == candidate_id), None)
            if prior is not None and prior["candidate_sha256"] == candidate["candidate_sha256"]:
                continue
            raise VariantError(f"candidate ID collision {candidate_id!r}")
        seen_ids.add(candidate_id)
        parent_id = candidate["parent_query_id"]
        if candidate["safety_status"] != "safe":
            unique.append(candidate)
            continue
        key = _normalized_key(candidate["text"])
        reason = None
        if key in seen_text[parent_id]:
            reason = "duplicate normalized query text for parent"
        elif counts[parent_id] >= max_variants_per_parent:
            reason = "maximum variants per parent exceeded"
        else:
            seen_text[parent_id].add(key)
            counts[parent_id] += 1
        if reason:
            candidate.update(verified=False, review_status="rejected", judgment_query_id=None, safety_status="rejected", rejection_reason=reason)
        unique.append(candidate)
    stored_reviews = [candidate["review"] for candidate in unique if "review" in candidate]
    _review_candidates(unique, [*stored_reviews, *reviews], parents)
    unique.sort(key=lambda record: record["candidate_id"])
    queries = sorted([*parents.values(), *(candidate for candidate in unique if candidate["verified"])], key=lambda record: record["query_id"])
    review_queue = [candidate for candidate in unique if candidate["review_status"] == "unreviewed"]
    config = {
        "schema_version": SCHEMA_VERSION, "generator": "partition_bound_variants_v1", "seed": seed,
        "typo_variants": typo_variants, "max_variants_per_parent": max_variants_per_parent,
        "split": split, "automatic_operations": ["internal_adjacent_swap", "internal_deletion"],
        "protection_policy": "nlp.text.protection_spans_exact_values_v1",
        "qrel_inheritance_policy": "verified_meaning_preserving_only",
        "canonical_queries_policy": "originals_and_verified_variants_only",
    }
    manifest = {
        **config, "config_hash": _digest(config), "input_sha256": _digest(assigned),
        "output_sha256": _digest(queries), "candidate_sha256": _digest(unique),
        "original_query_count": len(parents), "query_count": len(queries), "candidate_count": len(unique),
        "review_queue_count": len(review_queue), "verified_variant_count": sum(candidate["verified"] for candidate in unique),
        "candidate_counts_by_review_status": dict(sorted(Counter(candidate["review_status"] for candidate in unique).items())),
        "candidate_counts_by_generation_kind": dict(sorted(Counter(candidate["generation_kind"] for candidate in unique).items())),
        "generated_from_originals_only": True, "split_assignments_preserved": True,
        "human_review_required_for_new_candidates": True,
        "parents_without_eligible_typo": parents_without_eligible_typo,
        "max_query_characters": MAX_VARIANT_QUERY_CHARACTERS,
    }
    return VariantResult(tuple(queries), tuple(unique), tuple(review_queue), manifest)


def write_variant_artifacts(result: VariantResult, output_dir: Path, *, overwrite: bool = False) -> Mapping[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "queries": output_dir / "queries.jsonl", "candidates": output_dir / "candidates.jsonl",
        "review_queue": output_dir / "review_queue.jsonl", "manifest": output_dir / "manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise VariantError("Refusing to overwrite existing artifact(s): " + ", ".join(existing) + "; pass --force")
    _atomic_write_jsonl(paths["queries"], result.queries)
    _atomic_write_jsonl(paths["candidates"], result.candidates)
    _atomic_write_jsonl(paths["review_queue"], result.review_queue)
    _atomic_write_json(paths["manifest"], result.manifest)
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=ALL_SPLITS, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--typo-variants", type=int, default=1)
    parser.add_argument("--max-variants", "--max-variants-per-parent", dest="max_variants", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = generate_variants(
            load_assigned_queries(args.queries),
            proposals=list(iter_jsonl(args.proposals)) if args.proposals else (),
            reviews=list(iter_jsonl(args.reviews)) if args.reviews else (),
            seed=args.seed, typo_variants=args.typo_variants, max_variants_per_parent=args.max_variants, split=args.split,
        )
        paths = write_variant_artifacts(result, args.output_dir, overwrite=args.force)
    except (VariantError, CorpusPreparationError, TextPreparationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - covered by CLI smoke tests
    raise SystemExit(main())
