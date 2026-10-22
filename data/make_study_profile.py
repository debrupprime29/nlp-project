"""Derive a small, explicitly reduced CPU study from already assigned groups.

Group selection uses only language, existing split, group ID, and a seed. Qrels
are then used solely to construct one fixed corpus, preserving every judged
passage, including grade-zero passages. This is not a full MIRACL experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from data.make_splits import ALL_SPLITS, FINAL_EVALUATION, OFFICIAL_DEV, OFFICIAL_TRAIN, canonical_query
from data.prepare_corpus import iter_jsonl, load_qrels


SCHEMA_VERSION = 1
LANGUAGES = ("en", "hi", "bn")
PROFILE = "public_data_only_cpu_pilot"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
REQUIRED_ARTIFACTS = ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "corpus_manifest.json", "split_manifest.json")


class StudyProfileError(ValueError):
    """An input seal, preserved split, corpus, or output contract is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(path: str | Path) -> Path:
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise StudyProfileError("Symlinks are forbidden in profile inputs/outputs")
    return path


def _member(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or "\\" in name or ":" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise StudyProfileError("Unsafe relative manifest artifact path")
    return _safe(root / name)


def _json(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    path = _safe(path)
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise StudyProfileError("Expected a bounded regular manifest file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise StudyProfileError("Manifest must be a JSON object")
    if sealed and document.get("manifest_sha256") != _digest({key: value for key, value in document.items() if key != "manifest_sha256"}):
        raise StudyProfileError(f"Manifest seal mismatch: {path.name}")
    return document


def _verify_artifact(path: Path, metadata: Mapping[str, Any]) -> None:
    if (not isinstance(metadata, Mapping) or not isinstance(metadata.get("sha256"), str)
            or not path.is_file() or ("size" in metadata and path.stat().st_size != metadata["size"])
            or _sha256(path) != metadata["sha256"]):
        raise StudyProfileError(f"Artifact size/checksum mismatch: {path.name}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value) + b"\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for record in records:
            handle.write(_canonical(record) + b"\n")


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    document["manifest_sha256"] = _digest({key: value for key, value in document.items() if key != "manifest_sha256"})
    return document


def _existing_groups(queries: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any], language: str):
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in queries:
        split, source = row.get("split"), row.get("source_partition")
        if split not in ALL_SPLITS or source not in (OFFICIAL_TRAIN, OFFICIAL_DEV):
            raise StudyProfileError("Invalid original split/source partition")
        if (split == FINAL_EVALUATION) != (source == OFFICIAL_DEV):
            raise StudyProfileError("Official development must remain locked to final_evaluation")
        query = canonical_query(row, source_partition=source, default_language=language)
        if query.benchmark_language != language or query.query_id in by_id:
            raise StudyProfileError("Cross-language query or duplicate query ID")
        if row.get("group_id") != query.group_id:
            raise StudyProfileError("Explicit original group IDs are required")
        by_id[query.query_id] = row
        groups[query.group_id].append(row)
    assignments = split_manifest.get("group_assignments")
    if not isinstance(assignments, list):
        raise StudyProfileError("Source split manifest lacks group assignments")
    assigned = {}
    for assignment in assignments:
        if (not isinstance(assignment, dict) or not isinstance(assignment.get("group_id"), str)
                or not assignment["group_id"] or assignment["group_id"] in assigned):
            raise StudyProfileError("Malformed or duplicate original group assignment")
        for key in ("query_ids", "parent_query_ids"):
            values = assignment.get(key)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(set(values)) != len(values)):
                raise StudyProfileError("Malformed original group membership IDs")
        group = assignment["group_id"]
        members = groups.get(group, [])
        if (not members or assignment.get("benchmark_language") != language
                or set(assignment.get("query_ids", [])) != {row["query_id"] for row in members}
                or set(assignment.get("parent_query_ids", [])) != {row["parent_query_id"] for row in members}
                or any(row["split"] != assignment.get("split")
                       or row["source_partition"] != assignment.get("source_partition") for row in members)):
            raise StudyProfileError("Queries disagree with immutable source group assignments")
        assigned[group] = assignment
    if set(groups) != set(assigned):
        raise StudyProfileError("Source split manifest does not cover all groups")
    for row in queries:
        parent = by_id.get(row["parent_query_id"])
        if parent is None or parent["group_id"] != row["group_id"] or parent["split"] != row["split"]:
            raise StudyProfileError("Query parent is missing or outside its original split/group")
        judgment = row.get("judgment_query_id")
        if judgment is not None and (judgment not in by_id or by_id[judgment]["group_id"] != row["group_id"]):
            raise StudyProfileError("Judgment query is outside its source group")
    return groups, assigned


def _track(prepared_root: Path, romanized_root: Path, destination: Path, language: str,
           parent_manifest: Mapping[str, Any], roman_manifest: Mapping[str, Any],
           group_limits: Mapping[str, int], target_size: int, seed: int,
           config_sha256: str) -> dict[str, Any]:
    reference = parent_manifest["tracks"][language]
    source = _member(prepared_root, reference["path"])
    source_manifest_path = source / "import_manifest.json"
    source_manifest = _json(source_manifest_path, sealed=True)
    if source_manifest["manifest_sha256"] != reference.get("manifest_sha256") or source_manifest.get("language") != language:
        raise StudyProfileError("Language import does not match its parent manifest")
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not set(REQUIRED_ARTIFACTS).issubset(artifacts):
        raise StudyProfileError("Source import lacks required artifact checksums")
    for name, metadata in artifacts.items():
        _verify_artifact(_member(source, name), metadata)
    split_path = source / "split_manifest.json"
    split_manifest = _json(split_path)
    if split_manifest.get("official_dev_policy") != "locked_final_evaluation":
        raise StudyProfileError("Source split does not lock official development")
    for import_key, split_key in (("split_config_hash", "config_hash"), ("split_input_sha256", "input_sha256")):
        if import_key in source_manifest and source_manifest[import_key] != split_manifest.get(split_key):
            raise StudyProfileError("Source import and full split lineage disagree")
    queries = list(iter_jsonl(source / "queries.jsonl"))
    groups, assignments = _existing_groups(queries, split_manifest, language)
    selected_groups = set()
    selection = {}
    # Query/group sampling must be complete before any qrels are parsed.
    for split, limit in group_limits.items():
        candidates = [group for group, assignment in assignments.items() if assignment["split"] == split]
        ordered = sorted(candidates, key=lambda group: (_digest([PROFILE, seed, language, split, group]), group))
        chosen = ordered[:limit]
        if not chosen:
            raise StudyProfileError(f"No groups available for {language}/{split}")
        selected_groups.update(chosen)
        selection[split] = {"requested_groups": limit, "available_groups": len(candidates),
                            "selected_group_ids": sorted(chosen), "selected_groups": len(chosen),
                            "excluded_groups": len(candidates) - len(chosen),
                            "group_shortfall": max(0, limit - len(chosen))}
    selected_queries = sorted((row for group in selected_groups for row in groups[group]), key=lambda row: row["query_id"])
    selected_ids = {row["query_id"] for row in selected_queries}
    source_ids = {row["query_id"] for row in queries}
    all_qrels = load_qrels(source / "qrels.jsonl")
    if any(row.query_id not in source_ids for row in all_qrels):
        raise StudyProfileError("Source qrel refers to a missing source query")
    selected_qrels = [asdict(row) for row in all_qrels if row.query_id in selected_ids]
    selected_qrels.sort(key=lambda row: (row["query_id"], row["doc_id"]))
    required_ids = {row["doc_id"] for row in selected_qrels}
    if not selected_qrels:
        raise StudyProfileError("Selected profile has no supplied relevance judgments")
    original = {}
    original_digest = hashlib.sha256()
    for row in iter_jsonl(source / "corpus.jsonl"):
        identifier = row.get("doc_id")
        fields = ("doc_id", "language", "title", "text", "source", "source_doc_id", "romanized_text", "romanization_status")
        if (not isinstance(identifier, str) or not identifier or identifier in original
                or row.get("language") != language
                or any(not isinstance(row.get(key), str) for key in fields)):
            raise StudyProfileError("Invalid original corpus document ID/language")
        core = {key: value for key, value in row.items() if key not in {"romanized_text", "romanization_status"}}
        original[identifier] = (tuple(core), _digest(core))
        original_digest.update(_canonical(row) + b"\n")
    corpus_manifest = _json(source / "corpus_manifest.json")
    if original_digest.hexdigest() != corpus_manifest.get("corpus_sha256"):
        raise StudyProfileError("Original canonical corpus digest disagrees with its manifest")
    if "qrels_sha256" in corpus_manifest:
        qrel_digest = hashlib.sha256()
        for qrel in all_qrels:
            qrel_digest.update(_canonical(asdict(qrel)) + b"\n")
        if qrel_digest.hexdigest() != corpus_manifest["qrels_sha256"]:
            raise StudyProfileError("Original canonical qrels digest disagrees with its manifest")
    if "corpus_config_hash" in source_manifest and source_manifest["corpus_config_hash"] != corpus_manifest.get("config_hash"):
        raise StudyProfileError("Source import and corpus configuration lineage disagree")
    if any(row.doc_id not in original for row in all_qrels):
        raise StudyProfileError("Source judged passage is absent from the fixed original corpus")
    roman_metadata = roman_manifest.get("corpora", {}).get(language)
    if not isinstance(roman_metadata, dict) or roman_metadata.get("input_sha256") != artifacts["corpus.jsonl"]["sha256"]:
        raise StudyProfileError("Romanized corpus does not match original input lineage")
    roman_path = _member(romanized_root, roman_metadata["output"])
    _verify_artifact(roman_path, {"sha256": roman_metadata.get("output_sha256")})
    romanized = {}
    for row in iter_jsonl(roman_path):
        identifier = row.get("doc_id")
        if identifier not in original or identifier in romanized:
            raise StudyProfileError("Romanized corpus has a duplicate/unknown document ID")
        keys, fingerprint = original[identifier]
        if (any(key not in row for key in keys)
                or _digest({key: row[key] for key in keys}) != fingerprint
                or not isinstance(row.get("romanized_text"), str)
                or not isinstance(row.get("romanization_status"), str)):
            raise StudyProfileError("Romanized corpus changed original passage fields or lacks explicit views")
        romanized[identifier] = row
    if set(romanized) != set(original) or len(romanized) != roman_metadata.get("document_count"):
        raise StudyProfileError("Romanized corpus must cover exactly the original fixed corpus")
    distractors = sorted(set(romanized) - required_ids,
                         key=lambda identifier: (_digest([PROFILE, "distractor", seed, language, identifier]), identifier))
    effective_target = max(target_size, len(required_ids))
    chosen_ids = required_ids | set(distractors[:max(0, effective_target - len(required_ids))])
    passages = [romanized[identifier] for identifier in sorted(chosen_ids)]
    destination.mkdir()
    _write_jsonl(destination / "corpus.jsonl", passages)
    _write_jsonl(destination / "queries.jsonl", selected_queries)
    _write_jsonl(destination / "qrels.jsonl", selected_qrels)
    selected_assignments = [assignments[group] for group in sorted(selected_groups)]
    split_counts = dict(Counter(row["split"] for row in selected_queries))
    derived_split = {
        "schema_version": SCHEMA_VERSION, "dataset_version": source_manifest.get("dataset_version"),
        "profile": PROFILE, "split_method": "subset_existing_whole_groups_no_reassignment_v1",
        "official_dev_policy": "locked_final_evaluation", "seed": seed,
        "config_hash": config_sha256, "source_split_config_hash": split_manifest.get("config_hash"),
        "source_fullsplit_manifest": {"path": str(split_path), "sha256": artifacts["split_manifest.json"]["sha256"],
                                      "input_sha256": split_manifest.get("input_sha256")},
        "group_assignments": selected_assignments, "query_count": len(selected_queries),
        "group_count": len(selected_groups), "query_counts_by_split": split_counts,
        "group_counts_by_split": dict(Counter(row["split"] for row in selected_assignments)),
        "assignments_changed": False,
    }
    _write_json(destination / "split_manifest.json", _seal(derived_split))
    output_corpus_manifest = {
        "schema_version": SCHEMA_VERSION, "profile": PROFILE,
        "dataset_version": source_manifest.get("dataset_version"), "config_hash": config_sha256,
        "languages": [language], "passage_count": len(passages), "qrel_count": len(selected_qrels),
        "corpus_sha256": _sha256(destination / "corpus.jsonl"), "qrels_sha256": _sha256(destination / "qrels.jsonl"),
        "target_size_per_language": target_size, "source_profile_passage_count": len(original),
        "target_expanded_for_judgments": len(required_ids) > target_size,
        "sampling": "all_selected_query_judged_passages_including_zero_then_sha256_distractors",
        "same_fixed_corpus_for_all_methods": True,
    }
    _write_json(destination / "corpus_manifest.json", _seal(output_corpus_manifest))
    report = {
        "schema_version": SCHEMA_VERSION, "profile": PROFILE, "language": language,
        "config_sha256": config_sha256, "selection": selection,
        "selected_query_ids": sorted(selected_ids),
        "selected_parent_query_ids": sorted({row["parent_query_id"] for row in selected_queries}),
        "selected_group_ids": sorted(selected_groups), "source_query_count": len(queries),
        "selected_query_count": len(selected_queries), "excluded_query_count": len(queries) - len(selected_queries),
        "selected_qrel_count": len(selected_qrels), "excluded_qrel_count": len(all_qrels) - len(selected_qrels),
        "source_passage_count": len(original), "selected_passage_count": len(passages),
        "excluded_passage_count": len(original) - len(passages),
        "requested_target_size": target_size, "effective_target_size": effective_target,
        "target_shortfall": max(0, effective_target - len(passages)),
        "judged_passage_count_including_zero": len(required_ids),
        "target_expanded_for_judgments": len(required_ids) > target_size,
        "source_import_manifest": {"path": str(source_manifest_path), "sha256": _sha256(source_manifest_path),
                                   "manifest_sha256": source_manifest["manifest_sha256"]},
        "source_fullsplit_manifest": derived_split["source_fullsplit_manifest"],
        "source_romanization_manifest_sha256": roman_manifest["manifest_sha256"],
        "source_romanized_corpus": {"path": str(roman_path), "sha256": roman_metadata["output_sha256"]},
        "query_counts_by_split": split_counts,
        "judgment_use": "fixed_corpus_construction_only_no_quality_sampling_or_model_selection",
        "official_dev_policy": "locked_final_evaluation", "same_fixed_corpus_for_all_methods": True,
        "artifacts": {path.name: _file_metadata(path) for path in sorted(destination.iterdir())},
    }
    _write_json(destination / "profile_manifest.json", _seal(report))
    return report


def make_study_profile(prepared_root: str | Path, romanized_root: str | Path, output_dir: str | Path, *,
                       languages: Sequence[str] = LANGUAGES, encoder_groups: int = 24,
                       fusion_groups: int = 16, validation_groups: int = 16,
                       final_groups: int = 24, target_size: int = 1500, seed: int = 13) -> Mapping[str, Any]:
    """Validate sources, select existing groups, and atomically publish a pilot."""
    limits = dict(zip(ALL_SPLITS, (encoder_groups, fusion_groups, validation_groups, final_groups)))
    if (any(type(value) is not int or not 0 < value <= 100000 for value in (*limits.values(), target_size))
            or type(seed) is not int):
        raise StudyProfileError("Group limits/target must be positive bounded integers; seed must be an integer")
    languages = tuple(sorted(languages))
    if not languages or len(set(languages)) != len(languages) or any(language not in LANGUAGES for language in languages):
        raise StudyProfileError("Choose distinct supported language tracks")
    prepared_root, romanized_root, output = map(_safe, (prepared_root, romanized_root, output_dir))
    if output.exists() or prepared_root in output.parents or romanized_root in output.parents:
        raise StudyProfileError("Choose a new output directory outside the source roots")
    parent = _json(prepared_root / "import_manifest.json", sealed=True)
    roman = _json(romanized_root / "romanization_manifest.json", sealed=True)
    if (parent.get("official_dev_policy") != "locked_final_evaluation"
            or not isinstance(parent.get("tracks"), dict)
            or not set(languages).issubset(parent["tracks"])):
        raise StudyProfileError("Parent import must contain requested locked language tracks")
    lexicon = roman.get("lexicon")
    if not isinstance(lexicon, dict):
        raise StudyProfileError("Romanization manifest must identify the shared transliteration resource")
    lexicon_path = _member(romanized_root, lexicon["path"])
    _verify_artifact(lexicon_path, {"sha256": lexicon.get("sha256")})
    config = {
        "schema_version": SCHEMA_VERSION, "profile": PROFILE, "languages": list(languages),
        "seed": seed, "groups_per_language_by_existing_split": limits,
        "target_size_per_language": target_size,
        "group_selection": "sha256_seed_language_existing_split_group_id_whole_groups_v1",
        "resplit_existing_groups": False, "official_dev_policy": "locked_final_evaluation",
        "query_sampling_uses_qrels_or_relevance": False,
        "qrels_used_only_for_fixed_corpus_construction": True,
        "preserve_all_selected_query_judged_passages_including_zero": True,
        "same_fixed_corpus_for_all_methods": True,
        "scope": "public_data_only_reduced_cpu_pilot_not_full_MIRACL_or_full_50k_experiment",
        "omitted_requirements": ["unreviewed_generated_variants", "independently_human_verified_bilingual_challenge_set"],
    }
    config_sha256 = _digest(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".study-profile-", dir=output.parent))
    try:
        _write_json(stage / "profile_config.json", config)
        tracks = {language: _track(prepared_root, romanized_root, stage / language, language,
                                   parent, roman, limits, target_size, seed, config_sha256)
                  for language in languages}
        report = {
            "schema_version": SCHEMA_VERSION, "profile": PROFILE,
            "scope": config["scope"], "config_sha256": config_sha256,
            "languages": list(languages), "official_dev_policy": "locked_final_evaluation",
            "source_import_manifest": {"path": str(prepared_root / "import_manifest.json"),
                                       "sha256": _sha256(prepared_root / "import_manifest.json"),
                                       "manifest_sha256": parent["manifest_sha256"]},
            "source_romanization_manifest": {"path": str(romanized_root / "romanization_manifest.json"),
                                             "sha256": _sha256(romanized_root / "romanization_manifest.json"),
                                             "manifest_sha256": roman["manifest_sha256"]},
            "shared_transliteration_resource": {"path": str(lexicon_path), "sha256": lexicon["sha256"]},
            "tracks": {language: {"path": language, "manifest_sha256": track["manifest_sha256"],
                                   "selected_passage_count": track["selected_passage_count"],
                                   "query_counts_by_split": track["query_counts_by_split"]}
                       for language, track in tracks.items()},
            "profile_config": {"path": "profile_config.json", **_file_metadata(stage / "profile_config.json")},
            "limitations": ["Pilot subset, not full-query or full-corpus MIRACL benchmark scores",
                            "Small final group sample implies wide uncertainty intervals",
                            "E5 pretrained training data includes MIRACL training queries",
                            "No natural noisy-query challenge or human semantic invariance evidence"],
        }
        _write_json(stage / "study_manifest.json", _seal(report))
        if output.exists():
            raise StudyProfileError("Output appeared during construction; not replacing it")
        os.rename(stage, output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--romanized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--encoder-groups", type=int, default=24)
    parser.add_argument("--fusion-groups", type=int, default=16)
    parser.add_argument("--validation-groups", type=int, default=16)
    parser.add_argument("--final-groups", type=int, default=24)
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    try:
        report = make_study_profile(args.prepared_root, args.romanized_root, args.output_dir,
                                    languages=args.languages, encoder_groups=args.encoder_groups,
                                    fusion_groups=args.fusion_groups, validation_groups=args.validation_groups,
                                    final_groups=args.final_groups, target_size=args.target_size, seed=args.seed)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Study profile error: {exc}\n")
    print(json.dumps({"output": str(args.output_dir), "manifest_sha256": report["manifest_sha256"],
                      "tracks": report["tracks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
