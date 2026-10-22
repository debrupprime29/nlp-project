"""Offline, bounded-memory import of pinned official MIRACL TSV/gzip files.

Download/permission handling belongs to ``data.acquire_resources``.  This
module does no networking.  It scans *every* corpus shard, audits duplicate
document IDs in a temporary SQLite database, and preserves all selected-query
judgments (including grade zero) before hash-priority distractor sampling.
Each language is a separate evaluation track; official development queries
are sealed into ``final_evaluation``, never training or model selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from data.make_splits import (
    OFFICIAL_DEV,
    OFFICIAL_TRAIN,
    canonical_query,
    make_splits,
    write_split_artifacts,
)
from data.prepare_corpus import (
    MAX_JSONL_LINE_BYTES,
    CorpusPreparationError,
    CorpusSource,
    Qrel,
    _atomic_write_json,
    _atomic_write_jsonl,
    _canonical_json_bytes,
    _canonical_passage,
    _clean_id,
    _sample_priority,
    canonical_language,
    load_qrels,
    prepare_corpus,
    write_prepared_corpus,
)


SCHEMA_VERSION = 1
IMPORTER = "offline_miracl_import_v1"
Progress = Callable[[str], None]


class MiraclImportError(ValueError):
    """Invalid official input or unsafe destination."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalized_information_need(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_topics(path: Path, language: str, partition: str) -> list[dict[str, Any]]:
    """Read official two-column topic TSV and namespace source query IDs."""

    language = canonical_language(language)
    if partition not in {"train", "dev"}:
        raise MiraclImportError("Topic partition must be train or dev")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2:
                raise MiraclImportError(f"Expected topic ID<TAB>text at {path}:{number}")
            raw_id = _clean_id(fields[0], "query_id")
            if raw_id in seen:
                raise MiraclImportError(f"Duplicate topic ID {raw_id!r} at {path}:{number}")
            seen.add(raw_id)
            text = unicodedata.normalize("NFC", fields[1])
            if not text.strip() or "\x00" in text:
                raise MiraclImportError(f"Empty/NUL topic text at {path}:{number}")
            query_id = f"{language}:{partition}:{raw_id}"
            group_id = f"{language}:need:{_digest(_normalized_information_need(text))}"
            records.append({
                "query_id": query_id,
                "parent_query_id": query_id,
                "group_id": group_id,
                "text": text,
                "languages": [language],
                "benchmark_language": language,
                "variant_type": "original",
                "verified": True,
            })
    if not records:
        raise MiraclImportError(f"No topics in {path}")
    return records


def _limit_topics(
    records: Sequence[dict[str, Any]], limit: int | None, seed: int,
) -> list[dict[str, Any]]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise MiraclImportError("Query limits must be positive, or omitted for all queries")
    ordered = sorted(records, key=lambda row: (
        _digest(["miracl-query-limit-v1", seed, row["query_id"]]), row["query_id"]
    ))
    return sorted(ordered if limit is None else ordered[:limit], key=lambda row: row["query_id"])


def _find_directory(source_dir: Path, name: str) -> Path:
    direct = source_dir / name
    if direct.is_dir():
        return direct
    matches = sorted(path for path in source_dir.rglob(name) if path.is_dir())
    if len(matches) != 1:
        raise MiraclImportError(f"Expected exactly one {name} directory under {source_dir}")
    return matches[0]


def discover_sources(source_dir: Path, language: str) -> dict[str, Any]:
    """Find the unmodified official folder/file layout, not an arbitrary shard subset."""

    topics_root = _find_directory(source_dir, f"miracl-v1.0-{language}")
    corpus_root = _find_directory(source_dir, f"miracl-corpus-v1.0-{language}")
    result: dict[str, Any] = {}
    for partition in ("train", "dev"):
        for kind in ("topics", "qrels"):
            path = topics_root / kind / f"{kind}.miracl-v1.0-{language}-{partition}.tsv"
            if not path.is_file():
                raise MiraclImportError(f"Missing official source file: {path}")
            result[f"{kind}_{partition}"] = path
    shards = sorted(corpus_root.rglob("docs-*.jsonl.gz"))
    if not shards:
        raise MiraclImportError(f"No docs-*.jsonl.gz shards in {corpus_root}")
    result["shards"] = shards
    return result


def _source_inventory(
    paths: Sequence[Path], source_dir: Path, lock_path: Path | None,
) -> list[dict[str, Any]]:
    lock_assets: dict[str, Mapping[str, Any]] = {}
    if lock_path is not None:
        with lock_path.open("r", encoding="utf-8") as handle:
            lock = json.load(handle)
        if not isinstance(lock, dict) or not isinstance(lock.get("assets"), list):
            raise MiraclImportError("Resource lock must contain an assets list")
        for asset in lock["assets"]:
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
                raise MiraclImportError("Invalid resource-lock asset")
            relative = Path(asset["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise MiraclImportError("Resource-lock paths must be safe relative paths")
            key = relative.as_posix()
            if key in lock_assets:
                raise MiraclImportError(f"Duplicate resource-lock path {key}")
            lock_assets[key] = asset
    inventory = []
    for path in sorted(set(paths)):
        relative = path.relative_to(source_dir).as_posix()
        asset = lock_assets.get(relative)
        if lock_path is not None and asset is None:
            raise MiraclImportError(f"Required source is missing from lock: {relative}")
        size = path.stat().st_size
        if asset is not None and asset.get("size") != size:
            raise MiraclImportError(f"Locked size mismatch for {relative}")
        sha256 = hashlib.sha256()
        git_hash = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(block)
                git_hash.update(block)
        if asset is not None:
            if not asset.get("sha256") and not asset.get("git_blob_sha1"):
                raise MiraclImportError(f"No content checksum in lock for {relative}")
            if asset.get("sha256") and asset["sha256"] != sha256.hexdigest():
                raise MiraclImportError(f"Locked SHA-256 mismatch for {relative}")
            if asset.get("git_blob_sha1") and asset["git_blob_sha1"] != git_hash.hexdigest():
                raise MiraclImportError(f"Locked Git blob checksum mismatch for {relative}")
        inventory.append({
            "path": relative, "size": size, "sha256": sha256.hexdigest(),
            **({"revision": asset.get("revision"), "license": asset.get("license")}
               if asset is not None else {}),
        })
    # A lock listing more shards than are present must not silently produce a
    # sample over a partial source corpus.  Unrequested languages are harmless.
    required = {item["path"] for item in inventory}
    selected_corpora = {part for name in required for part in Path(name).parts
                        if part.startswith("miracl-corpus-v1.0-")}
    for name in lock_assets:
        if (any(part in selected_corpora for part in Path(name).parts)
                and name.endswith(".jsonl.gz") and name not in required):
            raise MiraclImportError(f"Locked corpus shard is missing from source discovery: {name}")
    return inventory


def _iter_gzip_records(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with gzip.open(path, "rb") as handle:
            number = 0
            while raw := handle.readline(MAX_JSONL_LINE_BYTES + 1):
                number += 1
                if len(raw) > MAX_JSONL_LINE_BYTES:
                    raise MiraclImportError(f"Oversized JSON line at {path}:{number}")
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise MiraclImportError(f"Invalid JSON at {path}:{number}") from exc
                if not isinstance(record, dict):
                    raise MiraclImportError(f"Expected JSON object at {path}:{number}")
                yield record
    except (gzip.BadGzipFile, EOFError) as exc:
        raise MiraclImportError(f"Invalid gzip shard {path}: {exc}") from exc


def _sample_corpus(
    shards: Sequence[Path], *, language: str, judged_ids: set[str],
    target_size: int, seed: int, working_dir: Path, progress: Progress | None,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    capacity = max(0, target_size - len(judged_ids))
    mandatory: dict[str, Any] = {}
    heap: list[tuple[int, str, Any]] = []
    shard_counts = []
    total = 0
    database = working_dir / ".document_id_audit.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
        for shard in shards:
            count = 0
            for record in _iter_gzip_records(shard):
                passage = _canonical_passage(record, language=language, source="MIRACL", prefix_ids=False)
                raw_id = passage.source_doc_id
                try:
                    connection.execute("INSERT INTO ids(id) VALUES (?)", (raw_id,))
                except sqlite3.IntegrityError as exc:
                    raise MiraclImportError(f"Duplicate passage ID {raw_id!r} in {shard}") from exc
                count += 1
                total += 1
                if total % 10_000 == 0:
                    connection.commit()
                if progress is not None and total % 250_000 == 0:
                    progress(f"{language}: scanned {total:,} passages across all shards")
                if raw_id in judged_ids:
                    mandatory[raw_id] = passage
                elif capacity:
                    priority = _sample_priority(seed, language, raw_id)
                    entry = (-priority, raw_id, passage)
                    if len(heap) < capacity:
                        heapq.heappush(heap, entry)
                    elif priority < -heap[0][0]:
                        heapq.heapreplace(heap, entry)
            shard_counts.append({"name": shard.name, "passage_count": count})
        connection.commit()
    finally:
        connection.close()
        database.unlink(missing_ok=True)
    missing = sorted(judged_ids - mandatory.keys())
    if missing:
        raise MiraclImportError(f"{language}: {len(missing)} judged documents missing: {missing[:5]}")
    selected = sorted([*mandatory.values(), *(entry[2] for entry in heap)], key=lambda row: row.doc_id)
    pool = working_dir / "candidate_pool.jsonl"
    _atomic_write_jsonl(pool, (asdict(row) for row in selected))
    if progress is not None:
        progress(f"{language}: completed {total:,} source passages; selected {len(selected):,}")
    return pool, shard_counts, [row.source_doc_id for row in selected]


def _import_language(
    source_dir: Path, destination: Path, language: str, sources: Mapping[str, Any],
    *, target_size: int, seed: int, dataset_version: str,
    train_query_limit: int | None, dev_query_limit: int | None,
    inventory: Sequence[Mapping[str, Any]], progress: Progress | None,
) -> Mapping[str, Any]:
    destination.mkdir()
    train_all = load_topics(sources["topics_train"], language, "train")
    dev_all = load_topics(sources["topics_dev"], language, "dev")
    dev_groups = {record["group_id"] for record in dev_all}
    excluded = [record["query_id"] for record in train_all if record["group_id"] in dev_groups]
    train = _limit_topics([row for row in train_all if row["group_id"] not in dev_groups], train_query_limit, seed)
    dev = _limit_topics(dev_all, dev_query_limit, seed)
    if not train:
        raise MiraclImportError(f"{language}: no training queries remain after development precedence")
    qrels: list[Qrel] = []
    for partition, selected, all_topics in (("train", train, train_all), ("dev", dev, dev_all)):
        source_qrels = load_qrels(sources[f"qrels_{partition}"])
        known = {row["query_id"] for row in all_topics}
        wanted = {row["query_id"] for row in selected}
        positive: set[str] = set()
        for qrel in source_qrels:
            query_id = f"{language}:{partition}:{qrel.query_id}"
            if query_id not in known:
                raise MiraclImportError(f"Qrel references unknown topic {query_id}")
            if query_id in wanted:
                qrels.append(Qrel(query_id, qrel.doc_id, qrel.relevance))
                if qrel.relevance > 0:
                    positive.add(query_id)
        if missing := sorted(wanted - positive):
            raise MiraclImportError(f"Selected queries lack positive judgments: {missing[:5]}")
    pool, counts, selected_ids = _sample_corpus(
        sources["shards"], language=language, judged_ids={row.doc_id for row in qrels},
        target_size=target_size, seed=seed, working_dir=destination, progress=progress,
    )
    raw_qrels = destination / "selected_source_qrels.jsonl"
    _atomic_write_jsonl(raw_qrels, (asdict(row) for row in sorted(qrels, key=lambda row: (row.query_id, row.doc_id))))
    prepared = prepare_corpus(
        [CorpusSource(language, pool, raw_qrels)], target_size=target_size,
        seed=seed, dataset_version=dataset_version, prefix_doc_ids=True,
    )
    corpus_manifest = dict(prepared.manifest)
    source_summary = dict(corpus_manifest["sources"][0])
    source_summary.update({
        "passages_path": "candidate_pool.jsonl", "qrels_path": "selected_source_qrels.jsonl",
        "source_passage_count": sum(row["passage_count"] for row in counts),
        "candidate_pool_passage_count": len(selected_ids),
        "full_source_scanned": True,
    })
    corpus_manifest["sources"] = [source_summary]
    prepared = replace(prepared, manifest=corpus_manifest)
    corpus_paths = write_prepared_corpus(prepared, destination)
    splits = make_splits(
        [canonical_query(row, source_partition=OFFICIAL_TRAIN, default_language=language) for row in train],
        [canonical_query(row, source_partition=OFFICIAL_DEV, default_language=language) for row in dev],
        seed=seed, dataset_version=dataset_version,
    )
    split_paths = write_split_artifacts(splits, destination)
    all_paths = [pool, raw_qrels, *corpus_paths.values(), *split_paths.values()]
    artifacts = {path.name: {"sha256": _sha256(path), "size": path.stat().st_size} for path in all_paths}
    needed = {path.relative_to(source_dir).as_posix() for path in [
        sources["topics_train"], sources["topics_dev"], sources["qrels_train"],
        sources["qrels_dev"], *sources["shards"],
    ]}
    manifest = {
        "schema_version": SCHEMA_VERSION, "importer": IMPORTER,
        "dataset_version": dataset_version, "language": language,
        "evaluation_track": "monolingual_reduced_corpus",
        "seed": seed, "requested_target_size": target_size,
        "source_files": [row for row in inventory if row["path"] in needed],
        "source_shards": counts,
        "source_passage_count": sum(row["passage_count"] for row in counts),
        "selected_passage_count": len(selected_ids), "selected_source_doc_ids": selected_ids,
        "query_policy": {
            "namespacing": "language:official_partition:source_query_id",
            "duplicate_normalization": "NFKC_casefold_whitespace_collapse",
            "duplicate_scope": "exact_normalized_text_within_language_not_semantic_paraphrases",
            "cross_partition_duplicates": "development_precedence_exclude_training",
            "excluded_training_query_ids": sorted(excluded),
            "official_dev_policy": "locked_final_evaluation",
            "query_sampling": "sha256_query_priority_v1",
            "train_query_limit": train_query_limit, "dev_query_limit": dev_query_limit,
            "source_train_query_count": len(train_all), "source_dev_query_count": len(dev_all),
            "selected_train_query_ids": [row["query_id"] for row in train],
            "selected_dev_query_ids": [row["query_id"] for row in dev],
        },
        "sampling": "sha256_priority_v1_over_all_source_shards",
        "all_selected_query_judgments_preserved_including_zero": True,
        "judgment_use": "fixed_corpus_construction_and_partition_scoped_supervision_or_scoring_only",
        "corpus_config_hash": prepared.manifest["config_hash"],
        "corpus_sha256": prepared.manifest["corpus_sha256"],
        "split_config_hash": splits.manifest["config_hash"],
        "split_input_sha256": splits.manifest["input_sha256"],
        "query_counts_by_split": splits.manifest["query_counts_by_split"],
        "artifacts": artifacts,
        "limitations": [
            "reduced_corpus_not_full_MIRACL_benchmark",
            "hi_bn_passage_romanization_pending",
            "exact_duplicate_grouping_does_not_detect_related_paraphrases",
            "E5_pretraining_includes_MIRACL_training_data",
        ],
    }
    manifest["manifest_sha256"] = _digest(manifest)
    _atomic_write_json(destination / "import_manifest.json", manifest)
    return manifest


def import_miracl(
    source_dir: Path, output_dir: Path, *, languages: Sequence[str] = ("en", "hi", "bn"),
    target_size: int = 50_000, seed: int = 13, dataset_version: str = "miracl-v1.0",
    train_query_limit: int | None = None, dev_query_limit: int | None = None,
    lock_path: Path | None = None, force: bool = False, progress: Progress | None = None,
) -> Mapping[str, Any]:
    """Validate, stage, then publish independent language-specific benchmark tracks.

    ``force`` replaces only a previous importer-owned output directory, retaining
    that directory as a recoverable sibling backup. Inputs and outputs must be
    disjoint. Query limits create an explicitly labeled pilot profile, not a
    substitute for the untouched complete final-development evaluation.
    """

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise MiraclImportError(f"Source directory does not exist: {source_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise MiraclImportError("Source and output directories must be disjoint (no nesting)")
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise MiraclImportError("target_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MiraclImportError("seed must be an integer")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise MiraclImportError("dataset_version must be a nonempty string")
    for limit in (train_query_limit, dev_query_limit):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise MiraclImportError("Query limits must be positive integers or omitted")
    canonical_languages = sorted(canonical_language(value) for value in languages)
    if not canonical_languages or len(canonical_languages) != len(set(canonical_languages)):
        raise MiraclImportError("Provide at least one language and no duplicates")
    if lock_path is not None:
        lock_path = Path(lock_path).resolve()
        if output_dir == lock_path or output_dir in lock_path.parents:
            raise MiraclImportError("Output must not contain or replace the resource lock")
    if output_dir.exists():
        marker = output_dir / "import_manifest.json"
        if not force:
            raise MiraclImportError(f"Refusing to overwrite {output_dir}; pass --force")
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MiraclImportError("--force only replaces a previous importer-owned directory") from exc
        if not isinstance(previous, dict) or previous.get("importer") != IMPORTER:
            raise MiraclImportError("--force only replaces a previous importer-owned directory")
    discovered = {language: discover_sources(source_dir, language) for language in canonical_languages}
    paths = [path for sources in discovered.values() for key, value in sources.items()
             for path in (value if key == "shards" else [value])]
    inventory = _source_inventory(paths, source_dir, lock_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.import-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        manifests = {}
        for language in canonical_languages:
            manifests[language] = _import_language(
                source_dir, staging / language, language, discovered[language],
                target_size=target_size, seed=seed, dataset_version=dataset_version,
                train_query_limit=train_query_limit, dev_query_limit=dev_query_limit,
                inventory=inventory, progress=progress,
            )
        config = {
            "schema_version": SCHEMA_VERSION, "importer": IMPORTER,
            "dataset_version": dataset_version, "languages": canonical_languages,
            "target_size_per_language": target_size, "seed": seed,
            "train_query_limit": train_query_limit, "dev_query_limit": dev_query_limit,
            "profile": "pilot" if train_query_limit is not None or dev_query_limit is not None else "full_query_reduced_corpus",
            "source_inventory_sha256": _digest(inventory),
            "resource_lock_sha256": _sha256(lock_path) if lock_path is not None else None,
            "official_dev_policy": "locked_final_evaluation",
            "evaluation_tracks": "separate_monolingual_not_pooled",
        }
        report = {**config, "config_sha256": _digest(config), "tracks": {
            language: {
                "path": language, "manifest_sha256": value["manifest_sha256"],
                "source_passage_count": value["source_passage_count"],
                "selected_passage_count": value["selected_passage_count"],
                "query_counts_by_split": value["query_counts_by_split"],
            } for language, value in manifests.items()
        }}
        report["manifest_sha256"] = _digest(report)
        _atomic_write_json(staging / "import_manifest.json", report)
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
            backup.rmdir()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None:
                os.replace(backup, output_dir)
            raise
        return {"output_dir": str(output_dir), "manifest": report,
                "previous_output_backup": str(backup) if backup else None}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "bn"])
    parser.add_argument("--target-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset-version", default="miracl-v1.0")
    parser.add_argument("--train-query-limit", type=int)
    parser.add_argument("--dev-query-limit", type=int)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = import_miracl(
            args.source_dir, args.output_dir, languages=args.languages, target_size=args.target_size,
            seed=args.seed, dataset_version=args.dataset_version, train_query_limit=args.train_query_limit,
            dev_query_limit=args.dev_query_limit, lock_path=args.lock, force=args.force,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (MiraclImportError, CorpusPreparationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
