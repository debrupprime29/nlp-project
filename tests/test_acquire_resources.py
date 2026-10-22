from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

from data.acquire_resources import (
    MAX_GIT_BLOB_BYTES,
    AcquisitionError,
    _SafeRedirect,
    acquire_resources,
    download_asset,
    load_lock,
    main,
    validate_asset,
    verify_asset,
)


class Response(io.BytesIO):
    status = 200

    def __init__(self, content: bytes, url: str = "https://cdn.example.org/file") -> None:
        super().__init__(content)
        self.url = url

    def geturl(self) -> str:
        return self.url


class AcquireResourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "raw"
        self.content = b"public resource\n"
        self.asset = self.record()

    def record(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "name": "fixture", "url": "https://example.org/version-1/resource",
            "path": "language/resource.jsonl.gz", "size": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "license": "CC-BY-4.0", "revision": "version-1",
        }
        record.update(overrides)
        return record

    def opener(self, content: bytes | None = None, **kwargs: object) -> Mock:
        return Mock(return_value=Response(self.content if content is None else content, **kwargs))

    def assert_no_partials(self) -> None:
        self.assertEqual(list(self.output.rglob(".acquire-*")), [])

    def test_download_streams_verifies_and_is_cached_offline(self) -> None:
        opener = self.opener()
        target = download_asset(self.asset, self.output, allow_download=True,
                                max_bytes=len(self.content), opener=opener)
        self.assertEqual(target.read_bytes(), self.content)
        self.assertEqual(opener.call_args.kwargs, {"timeout": 30.0})
        self.assertEqual(opener.call_args.args[0].get_header("Accept-encoding"), "identity")
        self.assertTrue(verify_asset(self.asset, self.output))
        opener.reset_mock()
        self.assertEqual(download_asset(self.asset, self.output, max_bytes=0, opener=opener), target)
        opener.assert_not_called()
        self.assert_no_partials()

    def test_missing_asset_requires_explicit_download_permission(self) -> None:
        opener = self.opener()
        self.assertFalse(verify_asset(self.asset, self.output))
        with self.assertRaisesRegex(AcquisitionError, "allow-download"):
            download_asset(self.asset, self.output, max_bytes=100, opener=opener)
        opener.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_total_budget_checked_before_network_or_directory_creation(self) -> None:
        opener = self.opener()
        second = self.record(name="second", path="second")
        with self.assertRaisesRegex(AcquisitionError, "total byte budget"):
            acquire_resources([self.asset, second], self.output,
                              allow_download=True, max_bytes=len(self.content), opener=opener)
        opener.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_single_asset_budget_prevents_network(self) -> None:
        opener = self.opener()
        with self.assertRaisesRegex(AcquisitionError, "budget"):
            download_asset(self.asset, self.output, allow_download=True,
                           max_bytes=len(self.content) - 1, opener=opener)
        opener.assert_not_called()

    def test_verify_only_never_downloads_even_when_permission_is_set(self) -> None:
        opener = self.opener()
        with self.assertRaisesRegex(AcquisitionError, "Missing cached"):
            acquire_resources([self.asset], self.output, max_bytes=100,
                              allow_download=True, verify_only=True, opener=opener)
        opener.assert_not_called()

    def test_valid_cache_does_not_count_toward_download_budget(self) -> None:
        opener = self.opener()
        target = download_asset(self.asset, self.output, allow_download=True,
                                max_bytes=100, opener=opener)
        opener.reset_mock()
        self.assertEqual(acquire_resources([self.asset], self.output, max_bytes=0,
                                          verify_only=True, opener=opener), [target])
        opener.assert_not_called()

    def test_wrong_cached_file_is_preserved_and_all_assets_preflighted(self) -> None:
        target = self.output / str(self.asset["path"])
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user data")
        opener = self.opener()
        with self.assertRaisesRegex(AcquisitionError, "Cached asset"):
            acquire_resources([self.record(name="new", path="new"), self.asset], self.output,
                              max_bytes=1000, allow_download=True, opener=opener)
        self.assertEqual(target.read_bytes(), b"user data")
        opener.assert_not_called()
        self.assertFalse((self.output / "new").exists())

    def test_equal_size_corrupt_cache_is_rejected(self) -> None:
        target = self.output / str(self.asset["path"])
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * len(self.content))
        with self.assertRaisesRegex(AcquisitionError, "checksum"):
            verify_asset(self.asset, self.output)

    def test_download_size_and_checksum_failures_leave_no_published_file(self) -> None:
        for content in (b"short", self.content + b"extra", b"x" * len(self.content)):
            with self.subTest(content=content), self.assertRaises(AcquisitionError):
                download_asset(self.asset, self.output, allow_download=True,
                               max_bytes=100, opener=self.opener(content))
            self.assertFalse((self.output / str(self.asset["path"])).exists())
            self.assert_no_partials()

    def test_git_blob_digest_has_git_header_and_supports_small_hf_files(self) -> None:
        asset = dict(self.asset)
        del asset["sha256"]
        asset["git_blob_sha1"] = hashlib.sha1(
            f"blob {len(self.content)}\0".encode() + self.content,
            usedforsecurity=False).hexdigest()
        download_asset(asset, self.output, allow_download=True,
                       max_bytes=100, opener=self.opener())
        self.assertTrue(verify_asset(asset, self.output))
        asset["git_blob_sha1"] = hashlib.sha1(self.content, usedforsecurity=False).hexdigest()
        with self.assertRaisesRegex(AcquisitionError, "checksum"):
            verify_asset(asset, self.output)

    def test_git_digest_large_asset_is_rejected(self) -> None:
        asset = dict(self.asset)
        del asset["sha256"]
        asset.update(git_blob_sha1="a" * 40, size=MAX_GIT_BLOB_BYTES + 1)
        with self.assertRaisesRegex(AcquisitionError, "small files"):
            validate_asset(asset)

    def test_empty_asset(self) -> None:
        asset = self.record(size=0, sha256=hashlib.sha256(b"").hexdigest())
        target = download_asset(asset, self.output, max_bytes=0,
                                allow_download=True, opener=self.opener(b""))
        self.assertEqual(target.read_bytes(), b"")

    def test_unsafe_relative_paths_rejected(self) -> None:
        for path in ("/tmp/x", "../x", "a/../x", "./x", "a//x", "a/./x",
                     "a/", "C:/x", "a\\x", "", ".", "..", "a\x00x"):
            with self.subTest(path=path), self.assertRaises(AcquisitionError):
                validate_asset(self.record(path=path))

    def test_nonpublic_or_credential_urls_are_rejected(self) -> None:
        for url in ("http://example.org/x", "file:///tmp/x", "https://user@example.org/x",
                    "https://user:secret@example.org/x", "https://example.org/x?token=secret",
                    "https://example.org/x#fragment", "https://localhost/x",
                    "https://127.0.0.1/x", "https://10.1.2.3/x", "https://[::1]/x",
                    "https://example.org:8080/x", "https://example.org/\nx"):
            with self.subTest(url=url), self.assertRaises(AcquisitionError):
                validate_asset(self.record(url=url))

    def test_missing_or_bad_metadata_is_rejected(self) -> None:
        for key in ("name", "url", "path", "license", "revision", "size", "sha256"):
            asset = dict(self.asset)
            del asset[key]
            with self.subTest(key=key), self.assertRaises(AcquisitionError):
                validate_asset(asset)
        for changes in ({"size": True}, {"size": -1}, {"size": 3.2}, {"sha256": "xyz"},
                        {"git_blob_sha1": "a" * 40}, {"revision": ""}):
            with self.subTest(changes=changes), self.assertRaises(AcquisitionError):
                validate_asset(self.record(**changes))

    def test_duplicate_and_nested_targets_rejected_before_network(self) -> None:
        opener = self.opener()
        for second in (self.record(name="another"),
                       self.record(path="different"),
                       self.record(name="child", path="language/resource.jsonl.gz/child")):
            with self.subTest(second=second), self.assertRaises(AcquisitionError):
                acquire_resources([self.asset, second], self.output, max_bytes=1000,
                                  allow_download=True, opener=opener)
        opener.assert_not_called()

    def test_rejects_symlinked_output_root_parent_and_target(self) -> None:
        real = self.root / "real"
        real.mkdir()
        self.output.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(AcquisitionError, "Symlinks"):
            verify_asset(self.asset, self.output)
        self.output.unlink()
        self.output.mkdir()
        (self.output / "language").symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(AcquisitionError, "Symlinks"):
            verify_asset(self.asset, self.output)
        (self.output / "language").unlink()
        (self.output / "language").mkdir()
        target = self.output / str(self.asset["path"])
        target.symlink_to(real / "missing")
        with self.assertRaisesRegex(AcquisitionError, "Symlinks"):
            verify_asset(self.asset, self.output)
        self.assertEqual(list(real.iterdir()), [])

    def test_directory_instead_of_asset_is_rejected(self) -> None:
        target = self.output / str(self.asset["path"])
        target.mkdir(parents=True)
        with self.assertRaisesRegex(AcquisitionError, "incorrect type"):
            verify_asset(self.asset, self.output)

    def test_retry_is_limited_and_transient_errors_can_recover(self) -> None:
        opener = Mock(side_effect=[URLError("temporary"), Response(self.content)])
        download_asset(self.asset, self.output, max_bytes=100,
                       allow_download=True, opener=opener, retries=1)
        self.assertEqual(opener.call_count, 2)
        self.assert_no_partials()

    def test_network_failure_cleans_only_own_partials(self) -> None:
        self.output.mkdir()
        user_partial = self.output / ".acquire-user-file"
        user_partial.write_bytes(b"keep")
        opener = Mock(side_effect=TimeoutError("timeout"))
        with self.assertRaisesRegex(AcquisitionError, "3 attempt"):
            download_asset(self.asset, self.output, max_bytes=100,
                           allow_download=True, opener=opener)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(user_partial.read_bytes(), b"keep")
        self.assertEqual(list((self.output / "language").iterdir()), [])

    def test_nonretryable_http_failure_is_not_retried(self) -> None:
        opener = Mock(side_effect=HTTPError(str(self.asset["url"]), 404, "missing", {}, None))
        with self.assertRaisesRegex(AcquisitionError, "1 attempt"):
            download_asset(self.asset, self.output, max_bytes=100,
                           allow_download=True, opener=opener)
        self.assertEqual(opener.call_count, 1)
        self.assert_no_partials()

    def test_https_redirect_enforced_and_signed_cdn_query_allowed(self) -> None:
        handler = _SafeRedirect()
        request = Request(str(self.asset["url"]))
        for url in ("http://cdn.example.org/x", "https://localhost/x", "https://user@example.org/x"):
            with self.subTest(url=url), self.assertRaises(AcquisitionError):
                handler.redirect_request(request, None, 302, "found", {}, url)
        signed = "https://cdn.example.org/x?Signature=cdn-generated"
        redirected = handler.redirect_request(request, None, 302, "found", {}, signed)
        self.assertEqual(redirected.full_url, signed)
        with self.assertRaises(AcquisitionError):
            download_asset(self.asset, self.output, max_bytes=100, allow_download=True,
                           opener=self.opener(url="http://example.org/file"))
        self.assert_no_partials()

    def test_partial_http_response_rejected(self) -> None:
        response = Response(self.content)
        response.status = 206
        with self.assertRaisesRegex(AcquisitionError, "HTTP 200"):
            download_asset(self.asset, self.output, max_bytes=100, allow_download=True,
                           opener=Mock(return_value=response))
        self.assert_no_partials()

    def test_concurrently_created_user_file_is_never_overwritten(self) -> None:
        target = self.output / str(self.asset["path"])

        def opening(request: Request, **kwargs: object) -> Response:
            target.write_bytes(b"concurrent user data")
            return Response(self.content)

        with self.assertRaisesRegex(AcquisitionError, "Cached asset"):
            download_asset(self.asset, self.output, max_bytes=100,
                           allow_download=True, opener=opening)
        self.assertEqual(target.read_bytes(), b"concurrent user data")
        self.assert_no_partials()

    def test_invalid_runtime_limits_prevent_network(self) -> None:
        for changes in ({"max_bytes": -1}, {"max_bytes": True}, {"timeout": 0},
                        {"timeout": float("nan")}, {"timeout": 301},
                        {"retries": 6}, {"retries": -1}, {"retries": True}):
            options = {"max_bytes": 100, **changes}
            with self.subTest(changes=changes), self.assertRaises(AcquisitionError):
                download_asset(self.asset, self.output, **options)

    def test_load_lock_accepts_envelope_and_list_and_checks_all_records(self) -> None:
        lock = self.root / "lock.json"
        for document in ([self.asset], {"schema_version": 1, "assets": [self.asset]}):
            lock.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_lock(lock), [self.asset])
        for document in ([], {}, {"assets": "wrong"}, [self.asset, self.asset], [None]):
            lock.write_text(json.dumps(document), encoding="utf-8")
            with self.subTest(document=document), self.assertRaises(AcquisitionError):
                load_lock(lock)
        lock.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(AcquisitionError, "Invalid lock JSON"):
            load_lock(lock)

    def test_cli_offline_verification_and_missing_assets_exit(self) -> None:
        lock = self.root / "lock.json"
        lock.write_text(json.dumps({"assets": [self.asset]}), encoding="utf-8")
        args = ["--lock", str(lock), "--output-dir", str(self.output), "--max-bytes", "0", "--verify-only"]
        with self.assertRaises(SystemExit) as caught:
            main(args)
        self.assertEqual(caught.exception.code, 2)
        download_asset(self.asset, self.output, max_bytes=100,
                       allow_download=True, opener=self.opener())
        result = subprocess.run([sys.executable, "-m", "data.acquire_resources", *args],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["verified_assets"], 1)


if __name__ == "__main__":
    unittest.main()
