"""Content-addressed, local encoder checkpoints using JSON and safetensors.

The manifest seals every artifact in a checkpoint directory.  Loading verifies
both the exact file membership and each SHA256 before model code opens weights.
No pickle-based model artifacts or symbolic links are permitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from retrieval.bm25 import _atomic_write_json


CHECKPOINT_SCHEMA_VERSION = 1
SCHEMA_VERSION = CHECKPOINT_SCHEMA_VERSION
CHECKPOINT_MANIFEST_NAME = "encoder_manifest.json"
MANIFEST_NAME = CHECKPOINT_MANIFEST_NAME
_UNSAFE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}


class CheckpointError(ValueError):
    """Invalid, unsafe, changed, or incomplete encoder checkpoint."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError("checkpoint metadata must contain finite JSON data") from exc


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CheckpointError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError("checkpoint metadata must be an object")
    metadata = json.loads(_canonical(dict(value)))
    required = {"backend", "model_id", "init_revision", "dimension", "fine_tuned", "fixture_only", "preprocessing", "training"}
    if not required <= set(metadata):
        raise CheckpointError("checkpoint metadata is missing required model or training fields")
    if metadata["backend"] not in {"transformers_e5", "fixture_torch"}:
        raise CheckpointError("unsupported trained checkpoint backend")
    if metadata["fine_tuned"] is not True or not isinstance(metadata["fixture_only"], bool):
        raise CheckpointError("trained checkpoint must declare fine_tuned=True and boolean fixture_only")
    fixture = metadata["backend"] == "fixture_torch"
    if metadata["fixture_only"] != fixture:
        raise CheckpointError("checkpoint backend and fixture flag disagree")
    for field in ("model_id", "init_revision"):
        if not isinstance(metadata[field], str) or not metadata[field].strip() or metadata[field] != metadata[field].strip():
            raise CheckpointError(f"checkpoint {field} must be a nonempty string")
    dimension = metadata["dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise CheckpointError("checkpoint dimension must be a positive integer")
    if not fixture and (metadata["model_id"] != "intfloat/multilingual-e5-small" or not re.fullmatch(r"[0-9a-f]{40}", metadata["init_revision"]) or dimension != 384):
        raise CheckpointError("trained E5 requires the multilingual-e5-small model, pinned revision, and 384 dimensions")
    # A local import avoids a dependency cycle while keeping one source of truth
    # for prefixes, pooling, dtype, normalization, and tokenizer truncation.
    from retrieval.dense import DenseConfig, DenseError
    try:
        config = DenseConfig.from_mapping(metadata["preprocessing"])
    except (DenseError, TypeError) as exc:
        raise CheckpointError(f"invalid checkpoint preprocessing: {exc}") from exc
    if metadata["preprocessing"] != config.compatibility_mapping():
        raise CheckpointError("checkpoint preprocessing must contain exactly the supported semantic settings")
    if not isinstance(metadata["training"], dict) or not metadata["training"]:
        raise CheckpointError("checkpoint must contain training provenance")
    if "schema_version" in metadata and metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint metadata schema")
    return metadata


def _files(directory: Path) -> dict[str, str]:
    if any(path.is_symlink() for path in (directory.absolute(), *directory.absolute().parents)):
        raise CheckpointError("checkpoint directory paths must not traverse symlinks")
    if directory.is_symlink() or not directory.is_dir():
        raise CheckpointError("checkpoint directory must be an existing directory, not a symlink")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CheckpointError("checkpoint artifacts must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointError("checkpoint artifacts must be regular files")
        relative = path.relative_to(directory).as_posix()
        if relative == CHECKPOINT_MANIFEST_NAME:
            continue
        if path.suffix.lower() in _UNSAFE_SUFFIXES:
            raise CheckpointError("pickle-based checkpoint artifacts are prohibited; use safetensors")
        files[relative] = _sha_file(path)
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise CheckpointError("checkpoint contains no safetensors weights")
    return files


def _validate_filemap(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise CheckpointError("checkpoint manifest needs an artifact file map")
    for name, digest in value.items():
        if not isinstance(name, str) or not name or "\\" in name:
            raise CheckpointError("invalid checkpoint artifact path")
        path = PurePosixPath(name)
        if path.is_absolute() or path.as_posix() != name or any(part in {".", ".."} for part in path.parts) or name == CHECKPOINT_MANIFEST_NAME:
            raise CheckpointError("checkpoint artifact paths must be canonical relative paths without traversal")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CheckpointError("checkpoint artifact has an invalid SHA256")
    return dict(value)


def write_checkpoint_manifest(directory: Path, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Seal a newly saved checkpoint, refusing to replace an existing seal."""
    directory = Path(directory)
    manifest_path = directory / CHECKPOINT_MANIFEST_NAME
    try:
        if manifest_path.exists() or manifest_path.is_symlink():
            raise CheckpointError("checkpoint already has a manifest; write a new checkpoint directory")
        checked = _metadata(metadata)
        files = _validate_filemap(_files(directory))
        payload = {"metadata": checked, "files": files}
        manifest = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **payload, "checkpoint_sha256": _hash(payload)}
        _atomic_write_json(manifest_path, manifest)
        return manifest
    except OSError as exc:
        raise CheckpointError(f"could not seal encoder checkpoint: {exc}") from exc


def verify_checkpoint(directory: Path) -> Mapping[str, Any]:
    """Verify exact membership and content before opening an encoder model."""
    directory = Path(directory)
    manifest_path = directory / CHECKPOINT_MANIFEST_NAME
    try:
        if directory.is_symlink() or manifest_path.is_symlink():
            raise CheckpointError("checkpoint paths must not be symlinks")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "metadata", "files", "checkpoint_sha256"} or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("invalid encoder checkpoint manifest schema")
        checked = _metadata(manifest["metadata"])
        expected = _validate_filemap(manifest["files"])
        actual = _files(directory)
        if set(actual) != set(expected):
            raise CheckpointError("checkpoint artifact file membership changed")
        if actual != expected:
            raise CheckpointError("checkpoint artifact SHA256 mismatch")
        payload = {"metadata": checked, "files": expected}
        if manifest["checkpoint_sha256"] != _hash(payload):
            raise CheckpointError("checkpoint content fingerprint mismatch")
        return manifest
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"could not verify encoder checkpoint: {exc}") from exc
