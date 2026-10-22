"""Build, safely unpack and verify the minimal real CPU-pilot app runtime.

This module uses only the standard library. It never edits frozen artifacts,
installs dependencies, downloads files, or publishes anything.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping
from urllib.parse import quote

MAX_FILES = 128
MAX_BYTES = 2 * 1024**3
MAX_MANIFEST_BYTES = 2 * 1024**2
MANIFEST_NAME = "runtime_manifest.json"
LANGUAGES = ("en", "hi", "bn")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BundleError(ValueError):
    """Unsafe, incomplete, incompatible or corrupted distribution artifact."""


def safe_relative_path(name: str) -> str:
    if not isinstance(name, str) or not name or any(c in name for c in ("\\", ":", "\x00")):
        raise BundleError("Bundle paths must be nonempty portable relative paths")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or path.as_posix() != name
            or any(part in ("", ".", "..") for part in path.parts)):
        raise BundleError(f"Unsafe bundle member: {name!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise BundleError("Control characters are forbidden in paths")
    return name


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    try:
        result = json.loads(data, object_pairs_hook=_no_duplicates,
                            parse_constant=lambda value: (_ for _ in ()).throw(BundleError("Nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"Invalid bundle JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise BundleError("Bundle JSON must be an object")
    return result


def _regular(path: Path) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise BundleError(f"Expected a regular file without symlink parents: {path}")
    return path


def _manifest(value: dict) -> dict:
    if (set(value) != {"schema_version", "metadata", "files"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise BundleError("Unsupported bundle manifest schema")
    if not isinstance(value["metadata"], dict) or not isinstance(value["files"], dict):
        raise BundleError("Invalid manifest metadata or file map")
    files = value["files"]
    if not files or len(files) + 1 > MAX_FILES:
        raise BundleError("Bundle has an invalid file count")
    total = 0
    for name, record in files.items():
        safe_relative_path(name)
        if name == MANIFEST_NAME or not isinstance(record, dict) or set(record) != {"sha256", "bytes"}:
            raise BundleError("Invalid manifest file record")
        if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise BundleError("Invalid file SHA256")
        size = record["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError("Invalid file byte count")
        total += size
        if total > MAX_BYTES:
            raise BundleError("Expanded bundle exceeds byte limit")
    names = set(files)
    for name in names:
        if any(parent.as_posix() in names for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise BundleError("Bundle contains conflicting file/directory paths")
    return value


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_bundle(output: Path, sources: Mapping[str, Path], metadata: dict | None = None) -> dict:
    """Create an exact-whitelist deterministic tar.gz and companion SHA256."""
    output = Path(output).absolute()
    checksum = Path(str(output) + ".sha256")
    if output.exists() or checksum.exists() or output.is_symlink() or checksum.is_symlink():
        raise BundleError("Refusing to overwrite a bundle or checksum")
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (output.parent, *output.parent.parents)):
        raise BundleError("Bundle output must not traverse symlinks")
    checked = {safe_relative_path(name): _regular(path) for name, path in sources.items()}
    manifest = _manifest({"schema_version": 1, "metadata": metadata or {}, "files": {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in sorted(checked.items())
    }})
    data = _canonical(manifest)
    if len(data) > MAX_MANIFEST_BYTES:
        raise BundleError("Manifest exceeds byte limit")
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=output.parent) as temporary:
        temporary_archive = Path(temporary) / "archive.tar.gz"
        with temporary_archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for name in (MANIFEST_NAME, *sorted(checked)):
                        info = tarfile.TarInfo(name)
                        info.mode, info.mtime, info.uid, info.gid = 0o644, 0, 0, 0
                        info.size = len(data) if name == MANIFEST_NAME else manifest["files"][name]["bytes"]
                        if name == MANIFEST_NAME:
                            archive.addfile(info, io.BytesIO(data))
                        else:
                            with checked[name].open("rb") as source:
                                archive.addfile(info, source)
        # Detect source changes during export; never publish an inconsistent payload.
        for name, path in checked.items():
            record = manifest["files"][name]
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise BundleError(f"Source changed during export: {name}")
        archive_sha = sha256_file(temporary_archive)
        os.link(temporary_archive, output)  # Atomic refusal to overwrite.
        with checksum.open("x", encoding="ascii") as stream:
            stream.write(f"{archive_sha}  {output.name}\n")
    return {"archive": str(output), "archive_sha256": archive_sha,
            "archive_bytes": output.stat().st_size,
            "payload_bytes": sum(v["bytes"] for v in manifest["files"].values()),
            "file_count": len(manifest["files"]), "manifest": manifest}


def verify_runtime(root: Path) -> dict:
    root = Path(root).absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise BundleError("Runtime must be a regular directory without symlink parents")
    path = _regular(root / MANIFEST_NAME)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise BundleError("Manifest exceeds byte limit")
    manifest = _manifest(_json(path.read_bytes()))
    actual = set()
    for member in root.rglob("*"):
        if member.is_symlink() or (not member.is_dir() and not member.is_file()):
            raise BundleError("Runtime contains a symlink or nonregular member")
        if member.is_file():
            actual.add(member.relative_to(root).as_posix())
    if actual != set(manifest["files"]) | {MANIFEST_NAME}:
        raise BundleError("Runtime file membership does not match manifest")
    for name, record in manifest["files"].items():
        member = root / name
        if member.stat().st_size != record["bytes"] or sha256_file(member) != record["sha256"]:
            raise BundleError(f"Runtime checksum/size mismatch: {name}")
    return manifest


def install_bundle(archive: Path, destination: Path, expected_sha256: str) -> dict:
    """Authenticate bytes against the caller's trusted hash, then safely stage."""
    archive = _regular(archive)
    destination = Path(destination).absolute()
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise BundleError("Supply a full lowercase SHA256 from a trusted release")
    if archive.stat().st_size > MAX_BYTES or sha256_file(archive) != expected_sha256:
        raise BundleError("Archive checksum/size mismatch")
    if destination.exists() or destination.is_symlink():
        raise BundleError("Refusing to overwrite an existing runtime destination")
    if any(p.is_symlink() for p in (destination.parent, *destination.parent.parents)):
        raise BundleError("Runtime destination must not traverse symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = {}
            expanded = 0
            for member in bundle:
                safe_relative_path(member.name)
                if not member.isfile() or member.name in members or member.size < 0:
                    raise BundleError("Archive has duplicate, linked or nonregular members")
                expanded += member.size
                members[member.name] = member
                if len(members) > MAX_FILES or expanded > MAX_BYTES + MAX_MANIFEST_BYTES:
                    raise BundleError("Archive exceeds safe expansion limits")
            header = members.get(MANIFEST_NAME)
            if header is None or header.size > MAX_MANIFEST_BYTES:
                raise BundleError("Archive lacks a bounded runtime manifest")
            manifest_stream = bundle.extractfile(header)
            if manifest_stream is None:
                raise BundleError("Cannot read runtime manifest")
            with manifest_stream:
                manifest = _manifest(_json(manifest_stream.read(MAX_MANIFEST_BYTES + 1)))
            if set(members) != set(manifest["files"]) | {MANIFEST_NAME}:
                raise BundleError("Archive member list differs from manifest")
            for name, record in manifest["files"].items():
                if members[name].size != record["bytes"]:
                    raise BundleError("Archive member size differs from manifest")
            with tempfile.TemporaryDirectory(prefix=".runtime-", dir=destination.parent) as temporary:
                staging = Path(temporary) / "payload"
                staging.mkdir()
                for name, member in members.items():
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise BundleError("Cannot read regular archive member")
                    with source, target.open("xb") as output:
                        remaining = member.size
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise BundleError("Truncated archive member")
                            output.write(chunk)
                            remaining -= len(chunk)
                checked = verify_runtime(staging)
                if destination.exists() or destination.is_symlink():
                    raise BundleError("Runtime destination appeared during install")
                staging.rename(destination)
                return checked
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise BundleError(f"Could not safely install archive: {exc}") from exc


def runtime_sources(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    model = root / "artifacts/study/cpu-pilot/adapted-seed-13"
    from training.checkpoints import verify_checkpoint
    checkpoint = verify_checkpoint(model / "checkpoint")
    files = {f"checkpoint/{name}": model / "checkpoint" / name
             for name in (*checkpoint["files"], "encoder_manifest.json")}
    files.update({
        "models/lid.json": root / "artifacts/models/lid-public-v1.json",
        "resources/transliteration.jsonl": root / "artifacts/real/miracl-v1-romanized/transliteration.jsonl",
    })
    for lang in LANGUAGES:
        files[f"languages/{lang}/corpus.jsonl"] = root / f"artifacts/real/miracl-v1-cpu-pilot/{lang}/corpus.jsonl"
        files[f"languages/{lang}/gate.json"] = model / lang / "gates/full.json"
        for name in ("dense_manifest.json", "vectors.jsonl"):
            files[f"languages/{lang}/dense/{name}"] = model / lang / "dense" / name
    cards = {
        "miracl-queries-card.md": "artifacts/raw/miracl-v1/provenance/miracl-topics-README.md",
        "miracl-corpus-card.md": "artifacts/raw/miracl-v1/provenance/miracl-corpus-README.md",
        "comi-lingua-card.md": "artifacts/raw/comi-lingua/README.md",
        "e5-model-card.md": "artifacts/hf-cache/models--intfloat--multilingual-e5-small/snapshots/fd1525a9fd15316a2d503bf26ab031a61d056e98/README.md",
        "dakshina-acquisition.json": "artifacts/raw/dakshina-v1/dakshina_manifest.json",
        "romanization-provenance.json": "artifacts/real/miracl-v1-romanized/romanization_manifest.json",
    }
    for name, relative in cards.items():
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        files[f"notices/{name}"] = path
    for name in ("e5-small", "miracl-v1", "comi-lingua", "uroman"):
        files[f"notices/{name}.sources.json"] = root / f"configs/{name}.sources.json"
    for lang in LANGUAGES:
        files[f"notices/{lang}-corpus-manifest.json"] = root / f"artifacts/real/miracl-v1-cpu-pilot/{lang}/corpus_manifest.json"
    for path in sorted((root / "sharing/notices").glob("*")):
        if path.is_file():
            files[f"notices/{path.name}"] = path
    if not any(name.endswith("THIRD_PARTY_NOTICES.md") for name in files):
        raise BundleError("Third-party redistribution notices are required")
    return files


def build_release(root: Path, output: Path) -> dict:
    root = Path(root).resolve()
    files = runtime_sources(root)
    # Attribution is additional sidecar data; never rewrite sealed corpora.
    with tempfile.TemporaryDirectory(prefix="nlp-attribution-") as temporary:
        attribution = Path(temporary) / "passage-attribution.jsonl"
        with attribution.open("w", encoding="utf-8") as stream:
            for lang in LANGUAGES:
                with files[f"languages/{lang}/corpus.jsonl"].open(encoding="utf-8") as corpus:
                    for line in corpus:
                        doc = json.loads(line)
                        title = doc["title"]
                        record = {"doc_id": doc["doc_id"], "source_doc_id": doc.get("source_doc_id"),
                                  "language": lang, "title": title, "source": doc.get("source"),
                                  "source_url": f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}",
                                  "history_url": f"https://{lang}.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'), safe='')}&action=history",
                                  "note": "Title-derived article/history links for attribution, not a claim of an exact archived revision. Original dataset version retained in notices."}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        files["notices/passage-attribution.jsonl"] = attribution
        code = {}
        for name in ("app.py", "pyproject.toml"):
            code[name] = sha256_file(root / name)
        for directory in ("data", "nlp", "retrieval", "training", "evaluation"):
            for path in sorted((root / directory).glob("*.py")):
                code[path.relative_to(root).as_posix()] = sha256_file(path)
        metadata = {"name": "cpu-pilot-runtime-v1", "seed": 13, "gate": "full",
                    "languages": list(LANGUAGES), "passages_per_language": 1500,
                    "max_length": 128, "batch_size": 8, "device": "cpu",
                    "source_code_sha256": code,
                    "scope": "Original-query monolingual CPU pilot; no robustness/superiority claim; not full MIRACL."}
        return write_bundle(output, files, metadata)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    build.add_argument("--output", type=Path, default=Path("artifacts/releases/cpu-pilot-runtime-v1.tar.gz"))
    install = commands.add_parser("unpack")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--destination", type=Path, default=Path("artifacts/shared-runtime"))
    install.add_argument("--sha256-file", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--runtime-dir", type=Path, default=Path("artifacts/shared-runtime"))
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_release(args.project_root, args.output)
            result = {key: value for key, value in result.items() if key != "manifest"}
        elif args.command == "unpack":
            line = args.sha256_file.read_text(encoding="ascii").strip().split()
            if len(line) != 2 or line[1] != args.archive.name:
                raise BundleError("Checksum file must name the requested archive exactly")
            manifest = install_bundle(args.archive, args.destination, line[0])
            result = {"installed": str(args.destination), "files": len(manifest["files"])}
        else:
            manifest = verify_runtime(args.runtime_dir)
            result = {"verified": str(args.runtime_dir), "files": len(manifest["files"])}
        print(json.dumps(result, indent=2))
        return 0
    except (BundleError, OSError, ValueError) as exc:
        parser.exit(2, f"Runtime bundle error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
