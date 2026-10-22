"""Batch the public CPU pilot without changing its retrieval or study protocol.

The CLI uses local, already-built models/indexes only. Final evaluation requires
a content-sealed closure of *every* frozen input, checked before and after work.
Testing injection is deliberately available only through the Python API.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from data.prepare_corpus import iter_jsonl
from evaluation.study_runs import SCOPE, _queries, process_study
from nlp.language_id import TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator
from retrieval.bm25 import BM25Index, RetrievalResult, load_corpus as bm25_corpus
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as char_corpus
from retrieval.dense import (
    DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DenseConfig, DenseIndex, TransformersE5Encoder, TrainedCheckpointEncoder,
    load_corpus as dense_corpus, run_queries,
)
from retrieval.fusion import FusionConfig, RetrievalPipeline, load_queries
from training.checkpoints import verify_checkpoint
from training.train_gate import FEATURE_NAMES, GateConfig, LinearSoftmaxGate, collect_examples, train_gate


ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "full": (),
    "no_uncertainty": ("mean_token_entropy", "uncertain_token_fraction"),
    "no_agreement": tuple(name for name in FEATURE_NAMES if name.startswith("overlap_")),
}
PROFILE_FILES = ("profile_manifest.json", "corpus.jsonl", "queries.jsonl", "qrels.jsonl",
                 "corpus_manifest.json", "split_manifest.json")


class WorkflowError(ValueError):
    """Unsafe, incomplete, changed, or incompatible pilot workflow inputs."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str | Path, *, file: bool = False) -> Path:
    path = Path(value).absolute()
    if any(member.is_symlink() for member in (path, *path.parents)):
        raise WorkflowError(f"Symlink paths are prohibited: {path}")
    path = path.resolve()
    if file and not path.is_file():
        raise WorkflowError(f"Required regular input file is missing: {path}")
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(path: Path, *, seal: str | None = None) -> dict[str, Any]:
    _path(path, file=True)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise WorkflowError(f"Manifest is too large: {path}")
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if not isinstance(document, dict):
        raise WorkflowError(f"Expected a JSON object: {path}")
    if seal and document.get(seal) != _digest({key: value for key, value in document.items() if key != seal}):
        raise WorkflowError(f"Invalid {seal} content seal: {path}")
    return document


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical(document) + b"\n")


def _write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical(row) + b"\n")


def _tree(directory: Path) -> list[Path]:
    directory = _path(directory)
    if not directory.is_dir():
        raise WorkflowError(f"Required input directory is missing: {directory}")
    members = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise WorkflowError(f"Symlink artifacts are prohibited: {directory}")
    return [_path(path, file=True) for path in members if not path.is_dir()]


def _languages(profile: Path) -> tuple[str, ...]:
    document = _json(profile / "study_manifest.json", seal="manifest_sha256")
    languages = document.get("languages")
    if (not isinstance(languages, list) or not languages or len(languages) != len(set(languages))
            or any(language not in ("en", "hi", "bn") for language in languages)):
        raise WorkflowError("Profile languages must be unique en/hi/bn tracks")
    if document.get("official_dev_policy") != "locked_final_evaluation":
        raise WorkflowError("Official dev must remain locked final evaluation")
    return tuple(languages)


def _model_tracks(study_root: Path, seeds: Sequence[int], mode: str):
    if mode == "final":
        yield "frozen", None, study_root / "frozen", None
    for seed in seeds:
        root = study_root / f"adapted-seed-{seed}"
        yield "adapted", seed, root, root / "checkpoint"


def _frozen_cache() -> Path:
    """Resolve the cache both libraries will actually use, rejecting overrides."""
    home = Path(os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"))
    cache = _path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", home / "hub")))
    for name in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        if name in os.environ and _path(os.environ[name]) != cache:
            raise WorkflowError(f"Conflicting {name}: frozen cache would escape the input lock")
    for module_name, attribute in (("huggingface_hub.constants", "HF_HUB_CACHE"),
                                   ("transformers.utils.hub", "TRANSFORMERS_CACHE")):
        module = sys.modules.get(module_name)
        if module is not None and _path(getattr(module, attribute)) != cache:
            raise WorkflowError("Already-loaded model library cache differs from the sealed cache")
    return cache


def required_input_paths(profile: str | Path, study_root: str | Path, lid_model: str | Path,
                         lexicon: str | Path, *, mode: str = "final", seeds: Sequence[int] = (13, 29),
                         allow_fixture: bool = False) -> tuple[Path, ...]:
    """Return the canonical existing closure required by this helper, without models.

    Lock builders may union additional upstream provenance and cached-model files.
    All files in the resulting final lock (not just this subset) are verified.
    This function never reads relevance labels or loads an encoder.
    """
    if mode not in ("gates", "final"):
        raise WorkflowError("mode must be gates or final")
    profile, study_root = _path(profile), _path(study_root)
    languages = _languages(profile)
    paths = [profile / "study_manifest.json", profile / "profile_config.json", Path(lid_model), Path(lexicon)]
    for language in languages:
        paths.extend(profile / language / name for name in PROFILE_FILES)
    for _, _, model_root, checkpoint in _model_tracks(study_root, seeds, mode):
        if checkpoint is not None:
            paths.extend(_tree(checkpoint))
        for language in languages:
            track = model_root / language
            paths.extend((track / "dense" / "dense_manifest.json", track / "dense" / "vectors.jsonl"))
            if mode == "final":
                variants = [track]
                if checkpoint is not None:
                    for name in GATES:
                        paths.extend((track / "gates" / f"{name}.json", track / "gates" / f"{name}.report.json"))
                        variants.append(track / f"{name}-gate")
                for variant in variants:
                    paths.extend(variant / f"validation.{suffix}" for suffix in ("run.jsonl", "run.manifest.json", "trace.jsonl"))
                    paths.extend(_tree(variant / "validation-analysis"))
                for suffix in ("run.jsonl", "run.manifest.json"):
                    member = track / f"original-validation.{suffix}"
                    if member.exists():
                        paths.append(member)
    # Pin the implementation and checked-in protocol as part of the executable closure.
    for directory in ("data", "nlp", "retrieval", "training", "evaluation"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "configs").glob("*.json"))
    if (ROOT / "pyproject.toml").is_file():
        paths.append(ROOT / "pyproject.toml")
    if mode == "final" and not allow_fixture:
        # The frozen loader is cache-only. Seal its full pinned safe snapshot,
        # including tokenizer/config files, not only its safetensors weights.
        cache = _frozen_cache()
        sources = _json(ROOT / "configs" / "e5-small.sources.json")
        paths.extend(cache / asset["path"] for asset in sources["assets"])
    return tuple(sorted({_path(path, file=True) for path in paths}))


def verify_final_lock(path: str | Path, required: Sequence[Path] = ()) -> dict[str, Any]:
    """Verify the self-seal and every frozen file, not only known required files."""
    path = _path(path, file=True)
    document = _json(path, seal="lock_sha256")
    files = document.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise WorkflowError("Final lock requires a nonempty files_sha256 mapping")
    for name, expected in files.items():
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise WorkflowError("Lock file paths must be absolute and hashes full lowercase SHA256")
        member = _path(name, file=True)
        if str(member) != name or member == path:
            raise WorkflowError("Lock paths must be canonical, and cannot include the lock itself")
        if _sha(member) != expected:
            raise WorkflowError(f"Frozen lock input checksum mismatch: {member}")
    missing = {str(_path(member, file=True)) for member in required} - set(files)
    if missing:
        raise WorkflowError(f"Final lock is missing required inputs: {sorted(missing)}")
    return document


class _MetadataEncoder:
    """Validate saved index vectors/metadata without initializing a real model."""

    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.dimension = metadata["dimension"]

    def encode(self, *args, **kwargs):
        raise WorkflowError("Preflight must never encode text")


def _profile_preflight(profile: Path, languages: Sequence[str]) -> None:
    parent = _json(profile / "study_manifest.json", seal="manifest_sha256")
    config = _json(profile / "profile_config.json")
    if parent.get("config_sha256") != _digest(config):
        raise WorkflowError("Profile configuration content mismatch")
    reference = parent.get("profile_config", {})
    if reference.get("sha256") != _sha(profile / "profile_config.json"):
        raise WorkflowError("Profile configuration file checksum mismatch")
    for language in languages:
        track = profile / language
        manifest = _json(track / "profile_manifest.json", seal="manifest_sha256")
        if parent.get("tracks", {}).get(language, {}).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise WorkflowError("Profile track is not sealed by its parent")
        for name in PROFILE_FILES[1:]:
            metadata = manifest.get("artifacts", {}).get(name, {})
            path = track / name
            if metadata.get("sha256") != _sha(path) or metadata.get("size") != path.stat().st_size:
                raise WorkflowError(f"Profile artifact checksum mismatch: {path}")
        split = _json(track / "split_manifest.json", seal="manifest_sha256")
        if split.get("official_dev_policy") != "locked_final_evaluation":
            raise WorkflowError("Profile split must lock official dev")
        # Validates all records, partition/group boundaries and original-only scope;
        # no qrels are opened here, including in final mode.
        _, records = _queries(track / "queries.jsonl", "validation")
        _queries(track / "queries.jsonl", "final_evaluation")
        if (manifest.get("language") != language or manifest.get("config_sha256") != parent["config_sha256"]
                or manifest.get("selected_query_ids") != sorted(records)
                or manifest.get("selected_group_ids") != sorted({row["group_id"] for row in records.values()})):
            raise WorkflowError("Profile selected query/group membership mismatch")
        assignments = split.get("group_assignments", [])
        assigned = {}
        for group in assignments:
            for qid in group.get("query_ids", []):
                if qid in assigned:
                    raise WorkflowError("Duplicate split-manifest query assignment")
                assigned[qid] = group
        if set(assigned) != set(records) or any(
                assigned[qid].get("group_id") != row["group_id"] or assigned[qid].get("split") != row["split"]
                for qid, row in records.items()):
            raise WorkflowError("Split manifest disagrees with query assignments")
        corpus_manifest = _json(track / "corpus_manifest.json", seal="manifest_sha256")
        if (corpus_manifest.get("corpus_sha256") != _sha(track / "corpus.jsonl")
                or corpus_manifest.get("qrels_sha256") != _sha(track / "qrels.jsonl")):
            raise WorkflowError("Corpus manifest file checksum mismatch")


def _index_preflight(track: Path, corpus: Path, config: DenseConfig, checkpoint, allow_fixture: bool, seed=None):
    path = track / "dense"
    manifest = _json(path / "dense_manifest.json")
    signature = manifest.get("encoder", {})
    if signature.get("fixture_only") and not allow_fixture:
        raise WorkflowError("Fixture indexes are prohibited in the real workflow")
    if checkpoint is not None:
        verified = verify_checkpoint(checkpoint)
        source, fingerprint = verified["metadata"], verified["checkpoint_sha256"]
        training = source["training"]
        if training.get("seed", training.get("config", {}).get("seed")) != seed:
            raise WorkflowError("Checkpoint training seed does not match its study track")
        if "seed" in training.get("config", {}) and training["config"]["seed"] != seed:
            raise WorkflowError("Checkpoint training configuration seed disagrees with its study track")
        if not allow_fixture:
            training_data = training.get("training_data", {})
            # The shared encoder is trained on all language encoder_train groups.
            expected_queries, expected_groups = set(), set()
            for language in _languages(corpus.parent.parent):
                _, source_rows = _queries(corpus.parent.parent / language / "queries.jsonl", "validation")
                expected_queries.update(qid for qid, row in source_rows.items() if row["split"] == "encoder_train")
                expected_groups.update(row["group_id"] for row in source_rows.values() if row["split"] == "encoder_train")
            if (training_data.get("selected_split") != "encoder_train"
                    or set(training_data.get("training_query_ids", [])) != expected_queries
                    or set(training_data.get("group_ids", [])) != expected_groups):
                raise WorkflowError("Checkpoint training membership differs from this profile encoder_train")
        metadata = {"backend": "fixture_torch_trained" if source["fixture_only"] else "transformers_e5_finetuned",
                    "model_id": source["model_id"], "resolved_revision": fingerprint, "tokenizer_revision": fingerprint,
                    "dimension": source["dimension"], "fixture_only": source["fixture_only"], "fine_tuned": True,
                    "checkpoint_sha256": fingerprint, "preprocessing": source["preprocessing"]}
        if signature.get("checkpoint_sha256") != fingerprint:
            raise WorkflowError("Dense index does not match its trained checkpoint")
    elif signature.get("fine_tuned"):
        raise WorkflowError("Frozen track cannot use a fine-tuned index")
    else:
        metadata = signature if allow_fixture else {
            "backend": "transformers_auto_model", "model_id": DEFAULT_MODEL_ID,
            "resolved_revision": DEFAULT_MODEL_REVISION, "tokenizer_revision": DEFAULT_MODEL_REVISION,
            "dimension": 384, "fixture_only": False}
    index = DenseIndex.load(path, _MetadataEncoder(metadata), expected_config=config)
    documents = dense_corpus(corpus)
    fingerprint = hashlib.sha256()
    for document in sorted(documents, key=lambda row: row.doc_id):
        fingerprint.update(_canonical({key: getattr(document, key) for key in ("doc_id", "title", "text", "language", "source")}) + b"\n")
    if (manifest.get("corpus_fingerprint") != fingerprint.hexdigest()
            or index.document_count != len(documents)
            or tuple(index._doc_ids) != tuple(sorted(document.doc_id for document in documents))):
        raise WorkflowError("Dense index does not cover exactly this profile corpus")
    if manifest.get("corpus_file_sha256") not in (None, _sha(corpus)):
        raise WorkflowError("Dense index corpus file checksum mismatch")
    return index


def _final_track_preflight(track, profile_track, checkpoint, dense, identifier, transliterator, locked_paths):
    """Validate every frozen gate/selection identity before ANY final retrieval."""
    pipeline = RetrievalPipeline(BM25Index(bm25_corpus(profile_track / "corpus.jsonl")),
                                 CharTfidfIndex(char_corpus(profile_track / "corpus.jsonl")),
                                 dense, identifier, transliterator, FusionConfig())
    _, rows = _queries(profile_track / "queries.jsonl", "validation")
    validation_ids = sorted(qid for qid, row in rows.items() if row["split"] == "validation")
    validation_groups = sorted({rows[qid]["group_id"] for qid in validation_ids})
    variants = [(None, track)] + ([(name, track / f"{name}-gate") for name in GATES] if checkpoint else [])
    for name, destination in variants:
        try:
            if name:
                gate = LinearSoftmaxGate.load_json(track / "gates" / f"{name}.json", expected_pipeline_manifest=pipeline.manifest)
                report = _json(track / "gates" / f"{name}.report.json")
                if tuple(gate.excluded_features) != GATES[name] or report != gate.training_report:
                    raise WorkflowError("Gate ablation identity or its separate training report disagrees")
                training_ids = set(report.get("training_query_ids", []))
                fusion_ids = {qid for qid, row in rows.items() if row["split"] == "fusion_train"}
                if (report.get("training_split") != "fusion_train" or report.get("scaler_fit_split") != "fusion_train"
                        or not training_ids or not training_ids <= fusion_ids
                        or set(report.get("training_group_ids", [])) != {rows[qid]["group_id"] for qid in training_ids}
                        or set(report.get("validation_query_ids", [])) != set(validation_ids)
                        or report.get("hyperparameter_selection", {}).get("split") != "validation"):
                    raise WorkflowError("Gate training/validation split membership is incompatible")
                for path, sha in report.get("input_sha256", {}).items():
                    member = _path(path, file=True)
                    if str(member) not in locked_paths or _sha(member) != sha:
                        raise WorkflowError("Gate training report input checksum mismatch")
                pipeline.weight_provider = gate
            expected = pipeline.manifest
            manifest = _json(destination / "validation.run.manifest.json")
            summary = _json(destination / "validation-analysis" / "study_manifest.json", seal="study_sha256")
            selection = _json(destination / "validation-analysis" / "validation-selection.json", seal="selection_sha256")
            if manifest.get("pipeline") != expected or summary.get("source_pipeline") != expected:
                raise WorkflowError("Validation pipeline differs from the frozen final pipeline")
            retrieval_hashes = manifest.get("input_sha256", {})
            for path in (profile_track / "corpus.jsonl", profile_track / "queries.jsonl"):
                matches = [sha for name, sha in retrieval_hashes.items() if Path(name).resolve() == path]
                if matches != [_sha(path)]:
                    raise WorkflowError("Validation retrieval input checksum mismatch")
            if any(Path(path).resolve() == profile_track / "qrels.jsonl" for path in retrieval_hashes):
                raise WorkflowError("Validation retrieval must not consume relevance judgments")
            if (selection.get("pipeline_fingerprint") != expected["pipeline_fingerprint"]
                    or selection.get("source_split") != "validation"
                    or selection.get("corpus_sha256") != _sha(profile_track / "corpus.jsonl")
                    or selection.get("query_source_sha256") != _sha(profile_track / "queries.jsonl")
                    or selection.get("validation_query_ids") != validation_ids
                    or selection.get("validation_group_ids") != validation_groups
                    or selection.get("schema_version") != 1
                    or selection.get("scope") != SCOPE
                    or selection.get("fixture_only") != expected["fixture_only"]
                    or selection.get("serving_fusion_config") != asdict(FusionConfig())
                    or selection.get("evaluation_fusion_config") != asdict(replace(FusionConfig(), final_top_k=100))
                    or summary.get("selection_sha256") != selection["selection_sha256"]
                    or summary.get("split") != "validation"):
                raise WorkflowError("Frozen validation selection/profile identity is incompatible")
            for member, sha in summary.get("artifacts", {}).items():
                path = _path(destination / "validation-analysis" / member, file=True)
                if path.parent != destination / "validation-analysis" or _sha(path) != sha:
                    raise WorkflowError("Validation analysis artifact checksum mismatch")
            trace = destination / "validation.trace.jsonl"
            if (manifest.get("split") != "validation" or Path(manifest.get("trace", "")).resolve() != trace
                    or summary.get("trace_sha256") != _sha(trace)):
                raise WorkflowError("Validation trace provenance/checksum mismatch")
        finally:
            pipeline.weight_provider = None


def _targets(study_root: Path, languages: Sequence[str], seeds: Sequence[int], mode: str):
    prefix = "validation" if mode == "gates" else "final"
    paths = [study_root / f"{mode}-workflow-report.json"]
    for _, _, model_root, checkpoint in _model_tracks(study_root, seeds, mode):
        for language in languages:
            track = model_root / language
            paths.extend((track / f"original-{prefix}.run.jsonl", track / f"original-{prefix}.run.manifest.json"))
            variants = [track]
            if checkpoint is not None:
                if mode == "gates":
                    paths.append(track / "gates")
                for name in GATES:
                    variant = track / f"{name}-gate"
                    if mode == "gates":
                        paths.append(variant)
                    variants.append(variant)
            for variant in variants:
                paths.extend(variant / f"{prefix}.{suffix}" for suffix in ("run.jsonl", "run.manifest.json", "trace.jsonl"))
                paths.append(variant / f"{prefix}-analysis")
    return paths


def _default_encoder(*, kind, seed, checkpoint, device, allow_fixture):
    if kind == "frozen":
        _frozen_cache()
        return TransformersE5Encoder(device=device, allow_download=False)
    return TrainedCheckpointEncoder(checkpoint, device=device, allow_fixture=allow_fixture)


def _retrieval_inputs(profile_track: Path, track: Path, lid: Path, lexicon: Path, checkpoint, gate=None):
    inputs = [profile_track / "corpus.jsonl", profile_track / "queries.jsonl", lid, lexicon,
              track / "dense" / "dense_manifest.json", track / "dense" / "vectors.jsonl"]
    if checkpoint is not None:
        inputs.extend(_tree(checkpoint))
    if gate is not None:
        inputs.append(gate)
    return inputs


def _original_run(index, encoder, track: Path, queries_path: Path, split: str, prefix: str):
    queries = load_queries(queries_path, split=split)
    start = time.perf_counter()
    results = run_queries(index, queries, top_k=100)
    elapsed = time.perf_counter() - start
    signature = index.manifest["encoder"]
    branch = (results[0].branch if results else
              "dense_fixture_trained" if signature.get("fixture_only") and signature.get("fine_tuned") else
              "dense_fixture_hash" if signature.get("fixture_only") else
              "dense_e5_trained" if signature.get("fine_tuned") else "dense_e5_frozen")
    manifest = {
        "schema_version": 1, "index": index.manifest, "index_dir": str(track / "dense"),
        "index_manifest_file_sha256": _sha(track / "dense" / "dense_manifest.json"),
        "index_vector_file_sha256": _sha(track / "dense" / "vectors.jsonl"),
        "queries_path": str(queries_path), "queries_file_sha256": _sha(queries_path),
        "selected_split": split, "branch": branch, "top_k": 100, "query_count": len(queries),
        "search_count": sum(bool(query.text.strip()) for query in queries) if index.document_count else 0,
        "intermediate_candidate_count": len(results), "result_count": len(results),
        "search_seconds": elapsed, "latency_includes_view_generation": False,
        "fixture_only": signature["fixture_only"], "query_encoder_operational_metadata": dict(encoder.metadata),
        "query_batch_size": index.config.batch_size,
    }
    run, metadata = track / f"original-{prefix}.run.jsonl", track / f"original-{prefix}.run.manifest.json"
    _write_rows(run, (asdict(result) for result in results))
    _write_json(metadata, manifest)
    return run, metadata


def _search(pipeline, queries_path: Path, split: str, destination: Path, prefix: str, inputs, *, adaptive):
    queries = load_queries(queries_path, split=split)
    fixture = pipeline.manifest["fixture_only"]
    branch = ("fusion_adaptive_fixture" if adaptive else "fusion_fixture") if fixture else ("fusion_adaptive" if adaptive else "fusion_fixed")
    results, traces = [], []
    for query in queries:
        result = pipeline.search(query.text)
        results.extend(RetrievalResult(query.query_id, hit.doc_id, branch, hit.rank, hit.score) for hit in result.hits)
        traces.append({"query_id": query.query_id, "active_weights": dict(result.active_weights),
                       "hits": [asdict(hit) for hit in result.hits],
                       "branch_results": {name: [asdict(hit) for hit in hits] for name, hits in result.branch_results.items()},
                       "diagnostics": dict(result.diagnostics)})
    run, manifest_path, trace = (destination / f"{prefix}.{suffix}" for suffix in ("run.jsonl", "run.manifest.json", "trace.jsonl"))
    manifest = {"schema_version": 1, "pipeline": pipeline.manifest, "branch": branch, "split": split,
                "weights": "learned_gate" if adaptive else "equal", "query_count": len(queries),
                "result_count": len(results), "fixture_only": fixture,
                "input_sha256": {str(path): _sha(path) for path in inputs},
                "search_count": sum(row["diagnostics"]["search_count"] for row in traces),
                "intermediate_candidate_count": sum(row["diagnostics"]["intermediate_candidate_count"] for row in traces),
                "latency_seconds": {key: sum(row["diagnostics"][key] for row in traces)
                                    for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds")},
                "trace": str(trace)}
    _write_rows(run, (asdict(result) for result in results))
    _write_json(manifest_path, manifest)
    _write_rows(trace, traces)
    return trace, manifest_path


def _fit_gates(pipeline, profile_track, track, inputs, *, epochs, temperatures, l2s):
    if pipeline.weight_provider is not None:
        raise WorkflowError("Gate collection requires the untouched equal-weight pipeline")
    records = list(iter_jsonl(profile_track / "queries.jsonl"))
    # collect_examples scopes query IDs BEFORE parsing relevance. In particular,
    # no official-dev relevance value is interpreted during gate fitting.
    qrels = list(iter_jsonl(profile_track / "qrels.jsonl"))
    start = time.perf_counter()
    train = collect_examples(pipeline, records, qrels, split="fusion_train")
    validation = collect_examples(pipeline, records, qrels, split="validation")
    collection_seconds = time.perf_counter() - start
    outcomes = {}
    for name, excluded in GATES.items():
        print(f"pilot_workflow: fitting {track} {name}", flush=True)
        candidates = [train_gate(train, pipeline_manifest=pipeline.manifest,
                                 config=GateConfig(temperature, epochs, 0.05, l2),
                                 validation_examples=validation, excluded_features=excluded)
                      for temperature, l2 in itertools.product(temperatures, l2s)]
        if not candidates or not all(result.report["validation"]["gate"]["query_count"] for result in candidates):
            raise WorkflowError("Gate selection requires positive validation judgments")
        best = max(candidates, key=lambda result: result.report["validation"]["gate"]["language_macro_ndcg_at_10"])
        report = dict(best.report)
        report["hyperparameter_selection"] = {
            "split": "validation", "metric": "language_macro_actual_fused_ndcg_at_10",
            "candidates": [{"config": result.report["config"], "excluded_features": result.report["excluded_features"],
                            "validation_ndcg_at_10": result.report["validation"]["gate"]["ndcg_at_10"],
                            "validation_language_macro_ndcg_at_10": result.report["validation"]["gate"]["language_macro_ndcg_at_10"]}
                           for result in candidates],
        }
        report["input_sha256"] = {str(path): _sha(path) for path in (*inputs, profile_track / "qrels.jsonl")}
        gate = LinearSoftmaxGate(best.gate.weights, best.gate.bias, best.gate.feature_mean, best.gate.feature_scale,
                                 pipeline_manifest=pipeline.manifest, training_report=report)
        path = track / "gates" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        gate.save_json(path, overwrite=False)
        _write_json(path.with_suffix(".report.json"), report)
        outcomes[name] = {"path": str(path), "sha256": _sha(path), "config": report["config"]}
    return {"train_query_count": len(train), "validation_query_count": len(validation),
            "collection_seconds": collection_seconds, "gates": outcomes}


def run_pilot_workflow(profile: str | Path, study_root: str | Path, lid_model: str | Path, lexicon: str | Path,
                       *, mode: str, seeds: Sequence[int] = (13, 29), max_length: int = 128,
                       batch_size: int = 8, device: str = "cpu", final_lock: str | Path | None = None,
                       encoder_factory: Callable | None = None, allow_fixture: bool = False,
                       gate_epochs: int = 300, temperatures: Sequence[float] = (0.25, 0.5),
                       l2s: Sequence[float] = (0.001, 0.01), bootstrap_samples: int = 1000) -> dict[str, Any]:
    """Run fresh gates or a locked final batch. Never train/reindex an encoder."""
    if mode not in ("gates", "final") or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise WorkflowError("Use gates/final and unique nonnegative integer seeds")
    if not allow_fixture and (encoder_factory is not None or gate_epochs != 300 or tuple(temperatures) != (0.25, 0.5)
                              or tuple(l2s) != (0.001, 0.01) or bootstrap_samples != 1000):
        raise WorkflowError("Testing injection/settings require API-only allow_fixture=True")
    config = DenseConfig(max_length=max_length, batch_size=batch_size)
    config.validate()
    # Verify all advertised files before even reading profile/query/label content.
    lock_path = _path(final_lock, file=True) if final_lock is not None else None
    if mode == "final" and lock_path is None:
        raise WorkflowError("Final mode requires --final-lock")
    lock = verify_final_lock(lock_path) if mode == "final" else None
    lock_file_sha256 = _sha(lock_path) if lock_path is not None else None
    profile, study_root = _path(profile), _path(study_root)
    lid_model, lexicon = _path(lid_model, file=True), _path(lexicon, file=True)
    required = required_input_paths(profile, study_root, lid_model, lexicon, mode=mode, seeds=seeds, allow_fixture=allow_fixture)
    if lock is not None:
        missing = set(map(str, required)) - set(lock["files_sha256"])
        if missing:
            raise WorkflowError(f"Final lock is missing required inputs: {sorted(missing)}")
    snapshots = {str(path): _sha(path) for path in required}
    languages = _languages(profile)
    targets = [_path(path) for path in _targets(study_root, languages, seeds, mode)]
    if any(path.exists() for path in targets):
        raise WorkflowError("Refusing to overwrite existing workflow outputs")
    inputs = set(required) | ({lock_path} if lock_path else set())
    if any(path in inputs or any(member.is_relative_to(path) for member in inputs) for path in targets):
        raise WorkflowError("Workflow outputs must not contain/overwrite frozen inputs")
    if lock and any(str(path) in lock["files_sha256"] for path in targets):
        raise WorkflowError("Final outputs must not be locked inputs")
    _profile_preflight(profile, languages)
    identifier = TokenLanguageIdentifier.load_json(lid_model)
    transliterator = LexiconTransliterator.from_jsonl(lexicon)
    for _, seed, model_root, checkpoint in _model_tracks(study_root, seeds, mode):
        for language in languages:
            index = _index_preflight(model_root / language, profile / language / "corpus.jsonl", config, checkpoint, allow_fixture, seed)
            if mode == "final":
                _final_track_preflight(model_root / language, profile / language, checkpoint, index, identifier, transliterator,
                                       lock["files_sha256"])
            del index
    factory = encoder_factory or _default_encoder
    prefix, split = ("validation", "validation") if mode == "gates" else ("final", "final_evaluation")
    results = []
    start = time.perf_counter()
    for kind, seed, model_root, checkpoint in _model_tracks(study_root, seeds, mode):
        print(f"pilot_workflow: loading {kind} seed={seed}", flush=True)
        encoder = factory(kind=kind, seed=seed, checkpoint=checkpoint, device=device, allow_fixture=allow_fixture)
        try:
            for language in languages:
                track, profile_track = model_root / language, profile / language
                print(f"pilot_workflow: {mode} {track}", flush=True)
                # All per-language references live in a separate frame so no previous
                # pipeline/model references survive into the next track or encoder.
                results.append(_run_track(encoder, identifier, transliterator, track, profile_track, checkpoint,
                                          lid_model, lexicon, config, mode, split, prefix, allow_fixture,
                                          gate_epochs, temperatures, l2s, bootstrap_samples))
                gc.collect()
        finally:
            del encoder
            gc.collect()
    if any(_sha(Path(path)) != sha for path, sha in snapshots.items()):
        raise WorkflowError("A frozen workflow input changed during processing")
    if lock is not None:
        if _sha(lock_path) != lock_file_sha256:
            raise WorkflowError("Final lock changed during processing")
        verify_final_lock(lock_path, required)
    generated = []
    for target in targets:
        if target.is_file():
            generated.append(target)
        elif target.is_dir():
            generated.extend(_tree(target))
    report = {"schema_version": 1, "mode": mode, "fixture_only": allow_fixture,
              "scope": "public_data_only_reduced_cpu_pilot_not_full_noisy_or_human_challenge",
              "profile": str(profile), "study_root": str(study_root), "seeds": list(seeds),
              "languages": list(languages), "max_length": max_length, "batch_size": batch_size,
              "elapsed_seconds": time.perf_counter() - start, "input_sha256": snapshots,
              "final_lock_sha256": lock_file_sha256, "results": results,
              "files_sha256": {str(path): _sha(path) for path in sorted(set(generated))}}
    report["report_sha256"] = _digest(report)
    _write_json(study_root / f"{mode}-workflow-report.json", report)
    return report


def _run_track(encoder, identifier, transliterator, track, profile_track, checkpoint,
               lid_model, lexicon, config, mode, split, prefix, allow_fixture,
               epochs, temperatures, l2s, bootstrap_samples):
    dense = DenseIndex.load(track / "dense", encoder, expected_config=config)
    pipeline = RetrievalPipeline(BM25Index(bm25_corpus(profile_track / "corpus.jsonl")),
                                 CharTfidfIndex(char_corpus(profile_track / "corpus.jsonl")),
                                 dense, identifier, transliterator, FusionConfig())
    if pipeline.manifest["fixture_only"] and not allow_fixture:
        raise WorkflowError("Fixture pipelines are prohibited")
    inputs = _retrieval_inputs(profile_track, track, lid_model, lexicon, checkpoint)
    fitted = _fit_gates(pipeline, profile_track, track, inputs, epochs=epochs,
                        temperatures=temperatures, l2s=l2s) if mode == "gates" else None
    variants = [(None, track)] + ([(name, track / f"{name}-gate") for name in GATES] if checkpoint is not None else [])
    original_run, original_manifest = _original_run(dense, encoder, track, profile_track / "queries.jsonl", split, prefix)
    analyses = {}
    for name, destination in variants:
        gate_path = track / "gates" / f"{name}.json" if name else None
        try:
            if gate_path:
                pipeline.weight_provider = LinearSoftmaxGate.load_json(gate_path, expected_pipeline_manifest=pipeline.manifest)
            retrieval_inputs = _retrieval_inputs(profile_track, track, lid_model, lexicon, checkpoint, gate_path)
            trace, manifest = _search(pipeline, profile_track / "queries.jsonl", split, destination, prefix,
                                      retrieval_inputs, adaptive=name is not None)
            summary = process_study(trace, manifest, profile_track / "queries.jsonl", profile_track / "qrels.jsonl",
                                    profile_track / "corpus.jsonl", destination / f"{prefix}-analysis", split=split,
                                    selection_path=destination / "validation-analysis" / "validation-selection.json" if mode == "final" else None,
                                    bootstrap_samples=bootstrap_samples, allow_fixture=allow_fixture,
                                    dense_original_run=original_run, dense_original_manifest=original_manifest)
            analyses[name or "base"] = {"output": str(destination / f"{prefix}-analysis"), "study_sha256": summary["study_sha256"]}
        finally:
            pipeline.weight_provider = None
    return {"track": str(track), "gate_training": fitted, "analyses": analyses}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gates", "final"), required=True)
    for name in ("profile", "study-root", "lid-model", "lexicon"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--final-lock", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_pilot_workflow(args.profile, args.study_root, args.lid_model, args.lexicon,
                                    mode=args.mode, seeds=args.seeds, max_length=args.max_length,
                                    batch_size=args.batch_size, device=args.device, final_lock=args.final_lock)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"Pilot workflow error: {exc}\n")
    print(json.dumps({"mode": args.mode, "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
