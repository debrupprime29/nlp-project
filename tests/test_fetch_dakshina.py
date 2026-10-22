"""Pinned-range acquisition tests: no external network or full tar download."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from data.fetch_dakshina import (
    ARCHIVE_GENERATION, ARCHIVE_SIZE, MANIFEST_NAME, PINNED_URL, WANTED_MEMBERS,
    DakshinaAcquisitionError, _digest, fetch_dakshina, main, verify_cache,
)


def make_archive(*, members=WANTED_MEMBERS, extra=None, format=tarfile.USTAR_FORMAT):
    buffer = io.BytesIO()
    payloads = {}
    with tarfile.open(fileobj=buffer, mode="w", format=format) as archive:
        if extra:
            info, data = extra
            archive.addfile(info, io.BytesIO(data) if data is not None else None)
        unrelated = tarfile.TarInfo("dakshina_dataset_v1.0/en/unrelated.txt")
        unrelated.size = 200_000
        archive.addfile(unrelated, io.BytesIO(b"x" * unrelated.size))
        for name in members:
            data = ("शब्द\tshabd\t12\n" if "/hi/" in name else "শব্দ\tshobdo\t9\n").encode("utf-8")
            payloads[name] = data
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue(), payloads


class Response(io.BytesIO):
    def __init__(self, data, headers, *, status=206, url=PINNED_URL):
        super().__init__(data)
        self.headers, self.status, self.url = headers, status, url

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, data, modifier=None):
        self.data, self.modifier = data, modifier
        self.requests = []
        self.response_bytes = 0

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        assert request.full_url == PINNED_URL
        assert request.get_header("Accept-encoding") == "identity"
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        data = self.data[start:end + 1]
        headers = Message()
        headers["Content-Range"] = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        headers["Content-Length"] = str(end - start + 1)
        headers["x-goog-generation"] = ARCHIVE_GENERATION
        response = Response(data, headers)
        if self.modifier:
            self.modifier(response, start, end)
        self.response_bytes += len(response.getvalue())
        return response


class FetchDakshinaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "dakshina"

    def acquire(self, **kwargs):
        data, payloads = make_archive()
        opener = FakeOpener(data)
        manifest = fetch_dakshina(self.output, allow_download=True, opener=opener, **kwargs)
        return manifest, opener, payloads

    def test_fetches_only_six_members_with_checked_offsets_and_honest_provenance(self):
        manifest, opener, payloads = self.acquire()
        self.assertEqual(len(manifest["members"]), 6)
        self.assertEqual(manifest["archive"]["generation"], ARCHIVE_GENERATION)
        self.assertFalse(manifest["archive"]["full_archive_md5_verified"])
        self.assertIn("not_independent_official", manifest["checksum_provenance"])
        self.assertEqual(manifest["license"], "CC-BY-SA-4.0")
        self.assertLess(manifest["downloaded_response_body_bytes"], 5000)
        self.assertEqual(manifest["downloaded_response_body_bytes"], opener.response_bytes)
        self.assertEqual(manifest["range_requests"], 13)
        files = {path.relative_to(self.output).as_posix() for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(files, {*WANTED_MEMBERS, MANIFEST_NAME})
        for member in manifest["members"]:
            self.assertEqual((self.output / member["path"]).read_bytes(), payloads[member["path"]])
            self.assertEqual(member["sha256"], hashlib.sha256(payloads[member["path"]]).hexdigest())
            self.assertEqual(member["body_offset"], member["header_offset"] + 512)
            self.assertEqual(member["source_url"], PINNED_URL)

    def test_cache_verification_precedes_any_network_and_permission(self):
        manifest, _, _ = self.acquire()
        def never_network(*args, **kwargs):
            self.fail("Cache verification must not use network")
        self.assertEqual(fetch_dakshina(self.output, opener=never_network, max_bytes=0), manifest)
        self.assertEqual(verify_cache(self.output), manifest)

    def test_missing_cache_requires_explicit_download_permission(self):
        with self.assertRaisesRegex(DakshinaAcquisitionError, "allow-download"):
            fetch_dakshina(self.output, opener=lambda *args, **kwargs: self.fail("network used"))
        self.assertFalse(self.output.exists())

    def test_bad_or_unsealed_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(DakshinaAcquisitionError, "not a sealed"):
            fetch_dakshina(self.output, allow_download=True, opener=lambda *args, **kwargs: self.fail("network used"))
        self.assertEqual(sentinel.read_text(), "keep")

    def test_corrupt_cache_fails_offline_without_repair(self):
        self.acquire()
        member = self.output / WANTED_MEMBERS[0]
        data = member.read_bytes()
        member.write_bytes(data.replace(b"shabd", b"SHABD"))
        with self.assertRaisesRegex(DakshinaAcquisitionError, "checksum mismatch"):
            fetch_dakshina(self.output, allow_download=True, opener=lambda *args, **kwargs: self.fail("network used"))

    def test_manifest_corruption_and_wrong_source_rejected(self):
        self.acquire()
        path = self.output / MANIFEST_NAME
        original = json.loads(path.read_text())
        changed = dict(original)
        changed["license"] = "wrong"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(DakshinaAcquisitionError, "seal mismatch"):
            verify_cache(self.output)
        changed["manifest_sha256"] = _digest({key: value for key, value in changed.items() if key != "manifest_sha256"})
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(DakshinaAcquisitionError, "provenance"):
            verify_cache(self.output)

    def test_symlink_member_is_not_followed(self):
        self.acquire()
        target = self.output / WANTED_MEMBERS[0]
        alternative = self.root / "outside.tsv"
        alternative.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(alternative)
        with self.assertRaisesRegex(DakshinaAcquisitionError, "Symlinks"):
            verify_cache(self.output)

    def test_symlink_root_is_not_followed(self):
        other = self.root / "outside"
        other.mkdir()
        self.output.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(DakshinaAcquisitionError, "Symlinks"):
            fetch_dakshina(self.output, allow_download=True)

    def assert_response_rejected(self, modifier, message):
        archive, _ = make_archive()
        opener = FakeOpener(archive, modifier)
        with self.assertRaisesRegex(DakshinaAcquisitionError, message):
            fetch_dakshina(self.output, allow_download=True, opener=opener)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".dakshina.dakshina-*")))

    def test_server_ignoring_ranges_rejected_before_body(self):
        self.assert_response_rejected(lambda response, *_: setattr(response, "status", 200), "HTTP 206")

    def test_wrong_content_range_rejected(self):
        def modify(response, *_):
            response.headers.replace_header("Content-Range", "bytes 1-512/999")
        self.assert_response_rejected(modify, "Content-Range")

    def test_wrong_content_length_rejected(self):
        def modify(response, *_):
            response.headers.replace_header("Content-Length", "99999999")
        self.assert_response_rejected(modify, "Content-Length")

    def test_compressed_range_response_rejected(self):
        def modify(response, *_):
            response.headers["Content-Encoding"] = "gzip"
        self.assert_response_rejected(modify, "Encoded")

    def test_redirected_response_rejected(self):
        self.assert_response_rejected(lambda response, *_: setattr(response, "url", "https://example.invalid/"), "URL differs")

    def test_wrong_generation_rejected(self):
        def modify(response, *_):
            response.headers.replace_header("x-goog-generation", "different")
        self.assert_response_rejected(modify, "generation mismatch")

    def test_short_or_oversized_response_rejected(self):
        for extra in (False, True):
            def modify(response, *_, extra=extra):
                data = response.getvalue()
                response.seek(0)
                response.truncate()
                response.write(data + b"X" if extra else data[:-1])
                response.seek(0)
            with self.subTest(extra=extra):
                self.assert_response_rejected(modify, "exceeds|Truncated")

    def test_byte_budget_stops_before_requesting_oversized_member(self):
        archive, _ = make_archive()
        opener = FakeOpener(archive)
        with self.assertRaisesRegex(DakshinaAcquisitionError, "budget"):
            fetch_dakshina(self.output, allow_download=True, opener=opener, max_bytes=1025)
        self.assertLessEqual(opener.response_bytes, 1025)
        self.assertFalse(self.output.exists())

    def test_bad_tar_checksum_rejected(self):
        archive, _ = make_archive()
        malformed = b"X" + archive[1:]
        with self.assertRaisesRegex(DakshinaAcquisitionError, "header/checksum"):
            fetch_dakshina(self.output, allow_download=True, opener=FakeOpener(malformed))

    def test_missing_member_fails_without_partial_publication(self):
        archive, _ = make_archive(members=WANTED_MEMBERS[:-1])
        with self.assertRaisesRegex(DakshinaAcquisitionError, "all six"):
            fetch_dakshina(self.output, allow_download=True, opener=FakeOpener(archive))
        self.assertFalse(self.output.exists())

    def test_wanted_symlink_is_not_extracted_or_followed(self):
        info = tarfile.TarInfo(WANTED_MEMBERS[0])
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive, _ = make_archive(extra=(info, None))
        with self.assertRaisesRegex(DakshinaAcquisitionError, "not a regular"):
            fetch_dakshina(self.output, allow_download=True, opener=FakeOpener(archive))

    def test_path_traversal_member_rejected(self):
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        archive, _ = make_archive(extra=(info, b"x"))
        with self.assertRaisesRegex(DakshinaAcquisitionError, "Unsafe tar"):
            fetch_dakshina(self.output, allow_download=True, opener=FakeOpener(archive))
        self.assertFalse((self.root / "escape.txt").exists())

    def test_skips_safe_pax_and_gnu_longname_extensions_without_extracting(self):
        for tar_format in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT):
            info = tarfile.TarInfo("dakshina_dataset_v1.0/unneeded/" + "a" * 200)
            info.size = 4
            archive, _ = make_archive(extra=(info, b"skip"), format=tar_format)
            destination = self.root / str(tar_format)
            manifest = fetch_dakshina(destination, allow_download=True, opener=FakeOpener(archive))
            self.assertEqual(len(manifest["members"]), 6)
            self.assertGreater(len(manifest["headers"]), 7)

    def test_maximum_header_count_is_bounded(self):
        archive, _ = make_archive()
        with patch("data.fetch_dakshina.MAX_HEADERS", 2), self.assertRaisesRegex(DakshinaAcquisitionError, "header limit"):
            fetch_dakshina(self.output, allow_download=True, opener=FakeOpener(archive))

    def test_invalid_limits_fail_before_network(self):
        for kwargs in ({"max_bytes": -1}, {"max_bytes": True}, {"timeout": 0}, {"timeout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DakshinaAcquisitionError):
                fetch_dakshina(self.output, allow_download=True, opener=lambda *args, **kw: self.fail("network used"), **kwargs)

    def test_offline_cli_verifies_cached_members(self):
        self.acquire()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--output-dir", str(self.output)]), 0)


if __name__ == "__main__":
    unittest.main()
