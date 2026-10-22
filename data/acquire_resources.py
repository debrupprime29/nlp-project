"""Acquire explicitly approved, content-pinned public HTTPS assets.

No dataset code is executed and archives are never extracted. The byte budget
limits the total expected size of missing assets (not retry traffic). Existing
files are verified, never replaced. A lock is a list, or ``{"assets": [...]}``.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


CHUNK_BYTES = 1024 * 1024
MAX_GIT_BLOB_BYTES = 16 * 1024 * 1024
MAX_LOCK_BYTES = 4 * 1024 * 1024


class AcquisitionError(ValueError):
    """The lock, local cache, download, or acquisition policy is invalid."""


def _https_url(url: str, *, redirect: bool = False) -> None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        invalid_port = parsed.port is not None and parsed.port != 443
        if (parsed.scheme != "https" or not host or parsed.username is not None
                or parsed.password is not None or parsed.fragment or invalid_port
                or (parsed.query and not redirect)
                or any(ord(ch) <= 32 or ord(ch) == 127 for ch in url)):
            raise ValueError
        if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError
    except (ValueError, TypeError) as exc:
        raise AcquisitionError("Expected a public HTTPS URL without credentials or query parameters") from exc


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Request | None:
        # CDN-generated signed queries are allowed only after a public request.
        _https_url(newurl, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_asset(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata and an unambiguous POSIX-relative target path."""
    if not isinstance(record, Mapping):
        raise AcquisitionError("Each asset must be a JSON object")
    asset = dict(record)
    for key in ("name", "url", "path", "license", "revision"):
        value = asset.get(key)
        if not isinstance(value, str) or not value.strip() or any(ord(ch) < 32 for ch in value):
            raise AcquisitionError(f"Asset {key} must be a nonempty string")
    _https_url(asset["url"])
    path = PurePosixPath(asset["path"])
    if (path.is_absolute() or "\\" in asset["path"] or ":" in asset["path"]
            or any(part in ("", ".", "..") for part in asset["path"].split("/"))):
        raise AcquisitionError("Asset path must be a safe relative path")
    size = asset.get("size")
    if type(size) is not int or size < 0:
        raise AcquisitionError("Asset size must be a nonnegative integer")
    kinds = [key for key in ("sha256", "git_blob_sha1") if key in asset]
    if len(kinds) != 1:
        raise AcquisitionError("Provide exactly one sha256 or git_blob_sha1 digest")
    kind = kinds[0]
    length = 64 if kind == "sha256" else 40
    if not isinstance(asset[kind], str) or not re.fullmatch(f"[0-9a-fA-F]{{{length}}}", asset[kind]):
        raise AcquisitionError(f"Invalid {kind} digest")
    if kind == "git_blob_sha1" and size > MAX_GIT_BLOB_BYTES:
        raise AcquisitionError("Git blob SHA1 is permitted only for small files; use SHA256")
    asset[kind] = asset[kind].lower()
    return asset


def load_lock(path: str | Path) -> list[dict[str, Any]]:
    """Read a bounded JSON lock; all records are checked before returning."""
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_LOCK_BYTES + 1)
    if len(raw) > MAX_LOCK_BYTES:
        raise AcquisitionError("Lock file exceeds size limit")
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise AcquisitionError("Invalid lock JSON") from exc
    records = document.get("assets") if isinstance(document, dict) else document
    if not isinstance(records, list) or not records:
        raise AcquisitionError("Lock must contain a nonempty assets list")
    assets = [validate_asset(record) for record in records]
    _unique_assets(assets)
    return assets


def _unique_assets(assets: Sequence[Mapping[str, Any]]) -> None:
    names: set[str] = set()
    targets: set[str] = set()
    for asset in assets:
        if asset["name"] in names or asset["path"] in targets:
            raise AcquisitionError("Asset names and target paths must be unique")
        names.add(asset["name"])
        targets.add(asset["path"])
    for target in targets:
        if any(str(parent) in targets for parent in PurePosixPath(target).parents):
            raise AcquisitionError("An asset cannot be the parent directory of another asset")


def _target(root: str | Path, relative: str) -> Path:
    target = Path(root).absolute() / relative
    for item in [*reversed(target.parents), target]:
        if item.is_symlink():
            raise AcquisitionError(f"Symlinks are not allowed in asset targets: {item}")
        if item.exists() and item != target and not item.is_dir():
            raise AcquisitionError(f"Asset parent is not a directory: {item}")
    return target


def _digest(asset: Mapping[str, Any]) -> tuple[Any, str]:
    if "sha256" in asset:
        return hashlib.sha256(), asset["sha256"]
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {asset['size']}\0".encode("ascii"))
    return digest, asset["git_blob_sha1"]


def verify_asset(record: Mapping[str, Any], output_root: str | Path) -> bool:
    """Return False for a missing asset; reject mismatched or unsafe cache files."""
    asset = validate_asset(record)
    target = _target(output_root, asset["path"])
    if not target.exists():
        return False
    if not target.is_file() or target.stat().st_size != asset["size"]:
        raise AcquisitionError(f"Cached asset has incorrect type or size: {asset['name']}")
    digest, expected = _digest(asset)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(target, flags), "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise AcquisitionError(f"Cached asset checksum mismatch: {asset['name']}")
    return True


def _limits(max_bytes: int, timeout: float, retries: int) -> None:
    if type(max_bytes) is not int or max_bytes < 0:
        raise AcquisitionError("max_bytes must be a nonnegative integer")
    if not isinstance(timeout, (float, int)) or not 0 < timeout <= 300:
        raise AcquisitionError("timeout must be positive and at most 300 seconds")
    if type(retries) is not int or not 0 <= retries <= 5:
        raise AcquisitionError("retries must be between 0 and 5")


def download_asset(record: Mapping[str, Any], output_root: str | Path, *,
                   allow_download: bool = False, max_bytes: int,
                   opener: Callable[..., Any] | None = None,
                   timeout: float = 30.0, retries: int = 2) -> Path:
    """Verify a cached asset or explicitly download it without overwriting files."""
    asset = validate_asset(record)
    _limits(max_bytes, timeout, retries)
    target = _target(output_root, asset["path"])
    if verify_asset(asset, output_root):
        return target
    if asset["size"] > max_bytes:
        raise AcquisitionError("Asset exceeds the remaining byte budget")
    if not allow_download:
        raise AcquisitionError("Missing asset; downloads require --allow-download")
    target.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener or build_opener(_SafeRedirect()).open
    request = Request(asset["url"], headers={"User-Agent": "NLP-project-resource-acquisition/1", "Accept-Encoding": "identity"})
    for attempt in range(retries + 1):
        partial: Path | None = None
        try:
            _target(output_root, asset["path"])
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".acquire-", dir=target.parent, delete=False) as handle:
                partial = Path(handle.name)
                digest, expected = _digest(asset)
                received = 0
                with open_url(request, timeout=timeout) as response:
                    final_url = response.geturl() if hasattr(response, "geturl") else asset["url"]
                    _https_url(final_url, redirect=True)
                    if getattr(response, "status", 200) != 200:
                        raise AcquisitionError("Expected HTTP 200 for the complete asset")
                    while chunk := response.read(min(CHUNK_BYTES, asset["size"] - received + 1)):
                        received += len(chunk)
                        if received > asset["size"]:
                            raise AcquisitionError(f"Downloaded asset exceeds declared size: {asset['name']}")
                        digest.update(chunk)
                        handle.write(chunk)
                if received != asset["size"] or digest.hexdigest() != expected:
                    raise AcquisitionError(f"Downloaded asset size/checksum mismatch: {asset['name']}")
                handle.flush()
                os.fsync(handle.fileno())
            _target(output_root, asset["path"])
            try:
                # Hard-link publication is atomic and cannot replace a user file.
                os.link(partial, target, follow_symlinks=False)
            except FileExistsError:
                if not verify_asset(asset, output_root):
                    raise AcquisitionError("Asset target appeared during publication")
            return target
        except (URLError, TimeoutError, OSError) as exc:
            retryable = not isinstance(exc, HTTPError) or exc.code in (408, 429, 500, 502, 503, 504)
            if attempt == retries or not retryable:
                raise AcquisitionError(f"Download failed for {asset['name']} after {attempt + 1} attempt(s)") from exc
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def acquire_resources(records: Sequence[Mapping[str, Any]], output_root: str | Path,
                      *, max_bytes: int, allow_download: bool = False,
                      verify_only: bool = False, opener: Callable[..., Any] | None = None,
                      timeout: float = 30.0, retries: int = 2) -> list[Path]:
    """Preflight every asset and the total missing-byte budget before any network."""
    _limits(max_bytes, timeout, retries)
    assets = [validate_asset(record) for record in records]
    if not assets:
        raise AcquisitionError("No assets specified")
    _unique_assets(assets)
    missing = [asset for asset in assets if not verify_asset(asset, output_root)]
    if verify_only and missing:
        raise AcquisitionError(f"Missing cached assets: {', '.join(asset['name'] for asset in missing)}")
    if sum(asset["size"] for asset in missing) > max_bytes:
        raise AcquisitionError("Missing assets exceed the total byte budget")
    if missing and not allow_download:
        raise AcquisitionError("Missing assets; downloads require --allow-download")
    return [download_asset(asset, output_root, allow_download=allow_download and not verify_only,
                           max_bytes=max_bytes, opener=opener, timeout=timeout,
                           retries=retries) for asset in assets]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        paths = acquire_resources(load_lock(args.lock), args.output_dir,
                                  max_bytes=args.max_bytes, allow_download=args.allow_download,
                                  verify_only=args.verify_only, timeout=args.timeout, retries=args.retries)
    except (AcquisitionError, OSError) as exc:
        parser.exit(2, f"Acquisition error: {exc}\n")
    print(json.dumps({"verified_assets": len(paths), "paths": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
