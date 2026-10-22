"""Explicitly acquire only six Dakshina lexicons using pinned HTTP byte ranges.

The official archive is an uncompressed 2 GB tar.  We walk its checked 512-byte
headers, skip unrelated bodies, and download only Hindi/Bengali lexicon splits.
No archive path is ever extracted.  Per-member SHA-256 values are computed from
the generation-pinned HTTPS responses; they are NOT independently published
official checksums.  The complete archive's published MD5 is not verified by
this range-only acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


ARCHIVE_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
ARCHIVE_GENERATION = "1590618303711280"
ARCHIVE_SIZE = 2_008_340_480
ARCHIVE_MD5_BASE64 = "pWCGix1nAmAsTqToN5T6Jg=="
PINNED_URL = f"{ARCHIVE_URL}?generation={ARCHIVE_GENERATION}"
LICENSE = "CC-BY-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SOURCE_REPOSITORY = "https://github.com/google-research-datasets/dakshina"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_HEADERS = 1000
MAX_EXTENSION_BYTES = 1024 * 1024
MANIFEST_NAME = "dakshina_manifest.json"
SCHEMA_VERSION = 1
WANTED_MEMBERS = tuple(
    f"dakshina_dataset_v1.0/{language}/lexicons/{language}.translit.sampled.{split}.tsv"
    for language in ("hi", "bn") for split in ("train", "dev", "test")
)


class DakshinaAcquisitionError(ValueError):
    """An unsafe cache, malformed archive, or invalid bounded response."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DakshinaAcquisitionError("Redirects are not allowed for the pinned GCS object")


def _digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe_root(output_dir: Path) -> Path:
    root = Path(output_dir).absolute()
    for path in [*reversed(root.parents), root]:
        if path.is_symlink():
            raise DakshinaAcquisitionError(f"Symlinks are not allowed in the output path: {path}")
        if path.exists() and not path.is_dir():
            raise DakshinaAcquisitionError(f"Output root/parent must be a directory: {path}")
    return root


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or "\x00" in name or ":" in name):
        raise DakshinaAcquisitionError(f"Unsafe tar member name: {name!r}")


def _sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify_cache(output_dir: Path) -> Mapping[str, Any] | None:
    """Verify every cached member offline; never repair or overwrite a bad cache."""

    root = _safe_root(output_dir)
    if not root.exists():
        return None
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DakshinaAcquisitionError("Existing output is not a sealed Dakshina cache; choose a new output directory")
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise DakshinaAcquisitionError("Cached manifest exceeds its size limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise DakshinaAcquisitionError("Invalid cached manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise DakshinaAcquisitionError("Invalid cached manifest object")
    seal = manifest.get("manifest_sha256")
    if not isinstance(seal, str) or seal != _digest({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise DakshinaAcquisitionError("Cached manifest seal mismatch")
    expected_archive = {
        "url": ARCHIVE_URL, "pinned_url": PINNED_URL,
        "generation": ARCHIVE_GENERATION, "size": ARCHIVE_SIZE,
        "md5_base64": ARCHIVE_MD5_BASE64, "full_archive_md5_verified": False,
    }
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("archive") != expected_archive
            or manifest.get("license") != LICENSE):
        raise DakshinaAcquisitionError("Cached provenance does not match the pinned Dakshina source")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(WANTED_MEMBERS):
        raise DakshinaAcquisitionError("Cached manifest must contain exactly six lexicons")
    seen = set()
    for member in members:
        if not isinstance(member, dict) or member.get("path") not in WANTED_MEMBERS or member["path"] in seen:
            raise DakshinaAcquisitionError("Invalid/duplicate cached member path")
        name = member["path"]
        seen.add(name)
        size = member.get("size")
        header_offset, body_offset = member.get("header_offset"), member.get("body_offset")
        if (type(size) is not int or size <= 0 or type(header_offset) is not int
                or header_offset < 0 or header_offset % 512 or body_offset != header_offset + 512
                or body_offset + size > ARCHIVE_SIZE or member.get("source_url") != PINNED_URL):
            raise DakshinaAcquisitionError("Invalid cached member offsets/size/provenance")
        for key in ("sha256", "header_sha256"):
            if not isinstance(member.get(key), str) or not re.fullmatch("[0-9a-f]{64}", member[key]):
                raise DakshinaAcquisitionError(f"Invalid cached {key}")
        target = root / name
        for path in [target, *target.parents]:
            if path.is_symlink():
                raise DakshinaAcquisitionError("Symlinks are not allowed in the cache")
            if path == root:
                break
        if not target.is_file() or target.stat().st_size != size or _sha256_file(target) != member["sha256"]:
            raise DakshinaAcquisitionError(f"Cached lexicon size/checksum mismatch: {name}")
    return manifest


class _RangeReader:
    def __init__(self, opener: Callable[..., Any], max_bytes: int, timeout: float):
        self.opener, self.max_bytes, self.timeout = opener, max_bytes, timeout
        self.received = 0
        self.requests = 0

    def read(self, offset: int, length: int) -> bytes:
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > ARCHIVE_SIZE:
            raise DakshinaAcquisitionError("Invalid archive byte range")
        if self.received + length + 1 > self.max_bytes:
            raise DakshinaAcquisitionError("Range acquisition exceeds the total byte budget")
        end = offset + length - 1
        request = Request(PINNED_URL, headers={
            "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
            "User-Agent": "NLP-project-Dakshina-range-acquisition/1",
        })
        self.requests += 1
        try:
            with self.opener(request, timeout=self.timeout) as response:
                if response.geturl() != PINNED_URL:
                    raise DakshinaAcquisitionError("Response URL differs from the generation-pinned source")
                if getattr(response, "status", None) != 206:
                    raise DakshinaAcquisitionError("Expected HTTP 206; server must honor Range requests")
                headers = response.headers
                expected = f"bytes {offset}-{end}/{ARCHIVE_SIZE}"
                if headers.get("Content-Range") != expected:
                    raise DakshinaAcquisitionError("Content-Range does not match the requested pinned archive range")
                if headers.get("Content-Encoding", "identity").casefold() != "identity":
                    raise DakshinaAcquisitionError("Encoded range responses are not allowed")
                if headers.get("Content-Length") != str(length):
                    raise DakshinaAcquisitionError("Content-Length does not match the requested range")
                if headers.get("x-goog-generation", ARCHIVE_GENERATION) != ARCHIVE_GENERATION:
                    raise DakshinaAcquisitionError("GCS archive generation mismatch")
                result: list[bytes] = []
                count = 0
                while chunk := response.read(min(1024 * 1024, length - count + 1)):
                    count += len(chunk)
                    self.received += len(chunk)
                    if count > length:
                        raise DakshinaAcquisitionError("Range response exceeds the declared size")
                    result.append(chunk)
                if count != length:
                    raise DakshinaAcquisitionError("Truncated range response")
                return b"".join(result)
        except (URLError, TimeoutError, OSError) as exc:
            raise DakshinaAcquisitionError(f"Pinned range request failed at offset {offset}: {exc}") from exc


def _pax_attributes(payload: bytes) -> dict[str, str]:
    attributes: dict[str, str] = {}
    offset = 0
    while offset < len(payload):
        space = payload.find(b" ", offset)
        if space < 0 or not payload[offset:space].isdigit():
            raise DakshinaAcquisitionError("Invalid PAX record length")
        length = int(payload[offset:space])
        if length <= space - offset + 2 or offset + length > len(payload):
            raise DakshinaAcquisitionError("Invalid/truncated PAX record")
        record = payload[space + 1:offset + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise DakshinaAcquisitionError("Invalid PAX key/value record")
        key, value = record[:-1].split(b"=", 1)
        try:
            attributes[key.decode("utf-8")] = value.decode("utf-8")
        except UnicodeError as exc:
            raise DakshinaAcquisitionError("PAX attributes must be UTF-8") from exc
        offset += length
    return attributes


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _walk_archive(reader: _RangeReader, staging: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    offset = 0
    found: dict[str, dict[str, Any]] = {}
    headers = []
    global_pax: dict[str, str] = {}
    local_pax: dict[str, str] = {}
    long_name: str | None = None
    for _ in range(MAX_HEADERS):
        raw = reader.read(offset, 512)
        if raw == b"\0" * 512:
            break
        try:
            info = tarfile.TarInfo.frombuf(raw, "utf-8", "strict")
        except (tarfile.HeaderError, UnicodeError, ValueError) as exc:
            raise DakshinaAcquisitionError(f"Invalid tar header/checksum at offset {offset}") from exc
        if info.size < 0:
            raise DakshinaAcquisitionError("Negative tar member size")
        headers.append({"offset": offset, "name": info.name, "type": info.type.decode("ascii", "replace"),
                        "size": info.size, "sha256": hashlib.sha256(raw).hexdigest()})
        body_offset = offset + 512
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            if info.size > MAX_EXTENSION_BYTES:
                raise DakshinaAcquisitionError("Tar extension metadata exceeds its bounded size")
            payload = reader.read(body_offset, info.size) if info.size else b""
            if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                attributes = _pax_attributes(payload)
                if any(key.startswith("GNU.sparse") for key in attributes):
                    raise DakshinaAcquisitionError("Sparse tar metadata is unsupported")
                if info.type == tarfile.XGLTYPE:
                    global_pax.update(attributes)
                else:
                    local_pax.update(attributes)
            elif info.type == tarfile.GNUTYPE_LONGNAME:
                try:
                    long_name = payload.rstrip(b"\0").decode("utf-8")
                except UnicodeError as exc:
                    raise DakshinaAcquisitionError("GNU long names must be UTF-8") from exc
            size = info.size
        else:
            attributes = {**global_pax, **local_pax}
            name = attributes.get("path", long_name if long_name is not None else info.name)
            _safe_member_name(name)
            try:
                raw_size = attributes.get("size", str(info.size))
                if not raw_size.isdecimal():
                    raise ValueError
                size = int(raw_size)
            except (TypeError, ValueError) as exc:
                raise DakshinaAcquisitionError("Invalid PAX member size") from exc
            if info.type == tarfile.GNUTYPE_SPARSE:
                raise DakshinaAcquisitionError("Sparse tar members are unsupported")
            if name in WANTED_MEMBERS:
                if name in found or not info.isreg() or size <= 0:
                    raise DakshinaAcquisitionError("Wanted lexicon is duplicated, empty, or not a regular file")
                data = reader.read(body_offset, size)
                # Minimal format validation before sealing public input. Each
                # official lexicon row is native<TAB>roman<TAB>attestation count.
                try:
                    rows = data.decode("utf-8").splitlines()
                except UnicodeError as exc:
                    raise DakshinaAcquisitionError("Lexicon member is not UTF-8") from exc
                if not rows or any(len(row.split("\t")) != 3 for row in rows if row.strip()):
                    raise DakshinaAcquisitionError("Lexicon member must contain three-column TSV rows")
                _write_bytes(staging / name, data)
                found[name] = {"path": name, "size": size, "sha256": hashlib.sha256(data).hexdigest(),
                               "header_offset": offset, "body_offset": body_offset,
                               "header_sha256": hashlib.sha256(raw).hexdigest(), "source_url": PINNED_URL}
            local_pax = {}
            long_name = None
        offset = body_offset + ((size + 511) // 512) * 512
        if offset > ARCHIVE_SIZE:
            raise DakshinaAcquisitionError("Tar member extends beyond the pinned archive size")
        if len(found) == len(WANTED_MEMBERS):
            return [found[name] for name in WANTED_MEMBERS], headers
    missing = sorted(set(WANTED_MEMBERS) - found.keys())
    raise DakshinaAcquisitionError(f"Archive ended/header limit reached before all six lexicons: {missing}")


def fetch_dakshina(
    output_dir: Path, *, allow_download: bool = False, max_bytes: int = DEFAULT_MAX_BYTES,
    opener: Callable[..., Any] | None = None, timeout: float = 30.0,
) -> Mapping[str, Any]:
    """Verify an existing cache, or explicitly acquire into a new directory."""

    if type(max_bytes) is not int or max_bytes < 0:
        raise DakshinaAcquisitionError("max_bytes must be a nonnegative integer")
    if (isinstance(timeout, bool) or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout) or not 0 < timeout <= 300):
        raise DakshinaAcquisitionError("timeout must be finite, positive, and at most 300 seconds")
    root = _safe_root(output_dir)
    if cached := verify_cache(root):
        return cached
    if not allow_download:
        raise DakshinaAcquisitionError("Missing Dakshina cache; downloads require --allow-download")
    if max_bytes < 513:
        raise DakshinaAcquisitionError("Byte budget cannot cover even one checked tar header")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.dakshina-", dir=root.parent))
    try:
        reader = _RangeReader(opener or build_opener(_NoRedirect()).open, max_bytes, timeout)
        members, headers = _walk_archive(reader, staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "archive": {"url": ARCHIVE_URL, "pinned_url": PINNED_URL,
                        "generation": ARCHIVE_GENERATION, "size": ARCHIVE_SIZE,
                        "md5_base64": ARCHIVE_MD5_BASE64, "full_archive_md5_verified": False},
            "license": LICENSE, "license_url": LICENSE_URL, "source_repository": SOURCE_REPOSITORY,
            "acquisition_method": "generation_pinned_https_ranges_checked_tar_headers_no_extractall",
            "checksum_provenance": "member_sha256_computed_from_pinned_TLS_source_not_independent_official_checksums",
            "members": members, "headers": headers, "headers_sha256": _digest(headers),
            "downloaded_response_body_bytes": reader.received, "range_requests": reader.requests,
            "byte_budget": max_bytes, "archive_walk_stopped_after_all_six_lexicons": True,
            "split_policy": "preserve_official_train_dev_test_separation",
        }
        manifest["manifest_sha256"] = _digest(manifest)
        _write_bytes(staging / MANIFEST_NAME, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        # Validate exactly what will become the offline cache before publishing.
        verify_cache(staging)
        if root.exists() or root.is_symlink():
            raise DakshinaAcquisitionError("Output appeared during acquisition; refusing replacement")
        os.rename(staging, root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        manifest = fetch_dakshina(args.output_dir, allow_download=args.allow_download,
                                  max_bytes=args.max_bytes, timeout=args.timeout)
    except (DakshinaAcquisitionError, HTTPError, OSError) as exc:
        parser.exit(2, f"Dakshina acquisition error: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "verified_members": len(manifest["members"]),
                      "downloaded_response_body_bytes": manifest["downloaded_response_body_bytes"],
                      "archive_generation": ARCHIVE_GENERATION}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
