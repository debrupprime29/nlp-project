"""Independent security and launcher tests for the portable runtime bundle.

These tests use tiny synthetic files; they neither download neural models nor
modify the closed research experiment.  Run from the repository root with
``python3 -m unittest sharing.test_runtime_bundle -v``.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from sharing import runtime_bundle as bundle
from sharing.run_app import build_arguments


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rejection_sequence = 0
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "a.txt").write_bytes(b"a small original file\n")
        (self.source / "b.json").write_bytes(b'{"language":"hi"}\n')
        self.sources = {
            "checkpoint/a.txt": self.source / "a.txt",
            "languages/hi/b.json": self.source / "b.json",
        }

    def make_bundle(self, name: str = "runtime.tar.gz") -> Path:
        archive = self.root / name
        bundle.write_bundle(archive, self.sources, metadata={"kind": "test"})
        return archive

    @staticmethod
    def manifest(files: dict[str, bytes]) -> dict:
        return {
            "schema_version": 1,
            "metadata": {"kind": "security fixture"},
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in files.items()
            },
        }

    def crafted_archive(
        self,
        files: dict[str, bytes] | None = None,
        *,
        manifest: dict | bytes | None = None,
        extra: list[tuple[tarfile.TarInfo | str, bytes]] | None = None,
        name: str = "crafted.tar.gz",
    ) -> Path:
        payload = files if files is not None else {"safe.txt": b"safe"}
        description = self.manifest(payload) if manifest is None else manifest
        raw = description if isinstance(description, bytes) else json.dumps(description).encode()
        archive = self.root / name
        with tarfile.open(archive, "w:gz") as handle:
            entries: list[tuple[tarfile.TarInfo | str, bytes]] = [
                ("runtime_manifest.json", raw), *payload.items(), *(extra or [])
            ]
            for entry, data in entries:
                info = tarfile.TarInfo(entry) if isinstance(entry, str) else entry
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
        return archive

    def assert_install_rejected(self, archive: Path) -> None:
        self.rejection_sequence += 1
        destination = self.root / f"rejected-install-{self.rejection_sequence}"
        before = set(self.root.iterdir())
        with self.assertRaises(bundle.BundleError):
            bundle.install_bundle(archive, destination, digest(archive))
        self.assertFalse(destination.exists(), "Failed installs must not publish partial data")
        self.assertEqual(before, set(self.root.iterdir()), "Failed installs must clean staging directories")

    def test_canonical_relative_paths(self) -> None:
        for name in ("model.safetensors", "checkpoint/tokenizer.json", "languages/bn/corpus.jsonl"):
            with self.subTest(name=name):
                self.assertEqual(bundle.safe_relative_path(name), name)

    def test_unsafe_relative_paths_rejected(self) -> None:
        for name in (
            "", ".", "..", "../escape", "/absolute", "./safe", "a/../b",
            "a/./b", "a//b", "a/", "C:/escape", "C:\\escape", "a\\escape", "nul\x00file",
        ):
            with self.subTest(name=name):
                with self.assertRaises(bundle.BundleError):
                    bundle.safe_relative_path(name)

    def test_round_trip_manifest_membership_and_checksums(self) -> None:
        originals = {path: digest(path) for path in self.sources.values()}
        archive = self.make_bundle()
        checksum = Path(str(archive) + ".sha256")
        self.assertEqual(checksum.read_text().strip(), f"{digest(archive)}  {archive.name}")
        with tarfile.open(archive, "r:gz") as handle:
            self.assertEqual(set(handle.getnames()), {"runtime_manifest.json", *self.sources})
        destination = self.root / "installed"
        manifest = bundle.install_bundle(archive, destination, digest(archive))
        self.assertEqual(manifest["metadata"], {"kind": "test"})
        self.assertEqual(set(manifest["files"]), set(self.sources))
        self.assertEqual(bundle.verify_runtime(destination), manifest)
        for relative, path in self.sources.items():
            self.assertEqual((destination / relative).read_bytes(), path.read_bytes())
        self.assertEqual(originals, {path: digest(path) for path in self.sources.values()})

    def test_wrong_archive_hash_rejected_before_install(self) -> None:
        archive = self.make_bundle()
        with self.assertRaises(bundle.BundleError):
            bundle.install_bundle(archive, self.root / "installed", "0" * 64)
        self.assertFalse((self.root / "installed").exists())

    def test_identical_sources_produce_identical_archive_bytes(self) -> None:
        first = self.make_bundle("first.tar.gz")
        second = self.make_bundle("second.tar.gz")
        self.assertEqual(digest(first), digest(second))

    def test_existing_destination_is_never_overwritten(self) -> None:
        archive = self.make_bundle()
        destination = self.root / "installed"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("user content")
        with self.assertRaises(bundle.BundleError):
            bundle.install_bundle(archive, destination, digest(archive))
        self.assertEqual(marker.read_text(), "user content")
        self.assertEqual(list(destination.iterdir()), [marker])

    def test_existing_archive_is_never_overwritten(self) -> None:
        archive = self.make_bundle()
        before = digest(archive)
        with self.assertRaises(bundle.BundleError):
            bundle.write_bundle(archive, self.sources)
        self.assertEqual(digest(archive), before)

    def test_export_refuses_symlink_sources(self) -> None:
        link = self.source / "linked.txt"
        link.symlink_to(self.source / "a.txt")
        with self.assertRaises(bundle.BundleError):
            bundle.write_bundle(self.root / "bad.tar.gz", {"linked.txt": link})
        self.assertFalse((self.root / "bad.tar.gz").exists())

    def test_export_refuses_sources_under_symlink_parents(self) -> None:
        link = self.root / "source-link"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(bundle.BundleError):
            bundle.write_bundle(self.root / "bad.tar.gz", {"a.txt": link / "a.txt"})

    def test_install_refuses_destination_symlink_parents(self) -> None:
        archive = self.make_bundle()
        physical = self.root / "physical"
        physical.mkdir()
        link = self.root / "linked-parent"
        link.symlink_to(physical, target_is_directory=True)
        with self.assertRaises(bundle.BundleError):
            bundle.install_bundle(archive, link / "installed", digest(archive))
        self.assertEqual(list(physical.iterdir()), [])

    def test_install_refuses_broken_destination_symlink(self) -> None:
        archive = self.make_bundle()
        destination = self.root / "broken-destination"
        destination.symlink_to(self.root / "does-not-exist")
        with self.assertRaises(bundle.BundleError):
            bundle.install_bundle(archive, destination, digest(archive))
        self.assertTrue(destination.is_symlink())

    def test_export_refuses_unsafe_names_and_reserved_manifest(self) -> None:
        for name in ("../escape", "runtime_manifest.json"):
            with self.subTest(name=name):
                with self.assertRaises(bundle.BundleError):
                    bundle.write_bundle(self.root / "bad.tar.gz", {name: self.source / "a.txt"})

    def test_export_enforces_payload_file_and_byte_limits(self) -> None:
        with mock.patch.object(bundle, "MAX_FILES", 1):
            with self.assertRaises(bundle.BundleError):
                bundle.write_bundle(self.root / "too-many.tar.gz", self.sources)
        with mock.patch.object(bundle, "MAX_BYTES", 1):
            with self.assertRaises(bundle.BundleError):
                bundle.write_bundle(self.root / "too-large.tar.gz", self.sources)

    def test_archive_path_traversal_and_absolute_members_rejected(self) -> None:
        for name in ("../escape", "/absolute", "nested/../../escape", "C:/escape", "a\\escape"):
            with self.subTest(name=name):
                self.assert_install_rejected(self.crafted_archive({name: b"evil"}))
        self.assertFalse((self.root.parent / "escape").exists())

    def test_archive_rejects_links_and_special_members(self) -> None:
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("untrusted")
                info.type = kind
                info.linkname = "../../escape"
                self.assert_install_rejected(self.crafted_archive(extra=[(info, b"")]))

    def test_duplicate_file_and_manifest_members_rejected(self) -> None:
        for name in ("safe.txt", "runtime_manifest.json"):
            with self.subTest(name=name):
                self.assert_install_rejected(self.crafted_archive(extra=[(name, b"duplicate")]))

    def test_unlisted_payload_rejected(self) -> None:
        self.assert_install_rejected(self.crafted_archive(extra=[("unlisted.txt", b"unexpected")]))

    def test_missing_payload_rejected(self) -> None:
        manifest = self.manifest({"safe.txt": b"safe", "missing.txt": b"missing"})
        self.assert_install_rejected(self.crafted_archive(manifest=manifest))

    def test_corrupted_payload_rejected(self) -> None:
        manifest = self.manifest({"safe.txt": b"original"})
        self.assert_install_rejected(self.crafted_archive({"safe.txt": b"modified"}, manifest=manifest))

    def test_wrong_declared_size_rejected(self) -> None:
        manifest = self.manifest({"safe.txt": b"safe"})
        manifest["files"]["safe.txt"]["bytes"] += 1
        self.assert_install_rejected(self.crafted_archive(manifest=manifest))

    def test_manifest_schema_and_byte_counts_are_strict_integers(self) -> None:
        for version in (True, 1.0, "1", 2):
            with self.subTest(schema_version=version):
                manifest = self.manifest({"safe.txt": b"safe"})
                manifest["schema_version"] = version
                self.assert_install_rejected(self.crafted_archive(manifest=manifest))
        for size in (True, -1, 4.0, "4"):
            with self.subTest(bytes=size):
                manifest = self.manifest({"safe.txt": b"safe"})
                manifest["files"]["safe.txt"]["bytes"] = size
                self.assert_install_rejected(self.crafted_archive(manifest=manifest))

    def test_duplicate_json_keys_rejected(self) -> None:
        encoded = json.dumps(self.manifest({"safe.txt": b"safe"}), separators=(",", ":"))
        raw = encoded.replace('"schema_version":1,', '"schema_version":1,"schema_version":1,', 1).encode()
        self.assert_install_rejected(self.crafted_archive(manifest=raw))

    def test_file_directory_conflict_rejected_atomically(self) -> None:
        self.assert_install_rejected(self.crafted_archive({"conflict": b"file", "conflict/child": b"child"}))

    def test_install_enforces_expanded_size_limit(self) -> None:
        archive = self.crafted_archive({"safe.txt": b"a" * 1000})
        with mock.patch.object(bundle, "MAX_BYTES", 999):
            self.assert_install_rejected(archive)

    def test_install_enforces_file_limit(self) -> None:
        archive = self.crafted_archive({"one": b"1", "two": b"2"})
        with mock.patch.object(bundle, "MAX_FILES", 1):
            self.assert_install_rejected(archive)

    def test_install_enforces_manifest_limit(self) -> None:
        archive = self.crafted_archive()
        with mock.patch.object(bundle, "MAX_MANIFEST_BYTES", 10):
            self.assert_install_rejected(archive)

    def test_unreadable_archive_rejected(self) -> None:
        archive = self.root / "not-an-archive.tar.gz"
        archive.write_bytes(b"not an archive")
        self.assert_install_rejected(archive)

    def test_runtime_tampering_extra_missing_and_linked_files_rejected(self) -> None:
        for change in ("corrupt", "extra", "missing", "symlink"):
            with self.subTest(change=change):
                archive = self.make_bundle(f"{change}.tar.gz")
                destination = self.root / f"installed-{change}"
                bundle.install_bundle(archive, destination, digest(archive))
                target = destination / "checkpoint/a.txt"
                if change == "corrupt":
                    target.write_bytes(b"corruption")
                elif change == "extra":
                    (destination / "extra.txt").write_text("unexpected")
                elif change == "missing":
                    target.unlink()
                else:
                    target.unlink()
                    target.symlink_to(self.source / "a.txt")
                with self.assertRaises(bundle.BundleError):
                    bundle.verify_runtime(destination)


class RuntimeLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime path with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def test_build_arguments_for_every_supported_language(self) -> None:
        for language in ("en", "hi", "bn"):
            with self.subTest(language=language):
                arguments = build_arguments(self.runtime, language)
                option = lambda name: arguments[arguments.index(name) + 1]
                self.assertEqual(option("--corpus"), str(self.runtime / "languages" / language / "corpus.jsonl"))
                self.assertEqual(option("--dense-index"), str(self.runtime / "languages" / language / "dense"))
                self.assertEqual(option("--gate-model"), str(self.runtime / "languages" / language / "gate.json"))
                self.assertEqual(option("--checkpoint"), str(self.runtime / "checkpoint"))
                self.assertEqual(option("--lid-model"), str(self.runtime / "models/lid.json"))
                self.assertEqual(option("--lexicon"), str(self.runtime / "resources/transliteration.jsonl"))
                self.assertEqual(option("--backend"), "e5-trained")
                self.assertEqual(option("--device"), "cpu")
                self.assertEqual(option("--max-length"), "128")
                self.assertEqual(option("--batch-size"), "8")
                self.assertEqual(option("--host"), "127.0.0.1")
                self.assertEqual(option("--port"), "8000")
                self.assertNotIn("--allow-download", arguments)
                self.assertNotIn("--demo", arguments)

    def test_unicode_query_and_custom_port_are_preserved(self) -> None:
        query = "শ্যাম্পু শব্দটির উৎস কোন ভাষা?"
        arguments = build_arguments(self.runtime, "bn", query=query, port=8123)
        self.assertEqual(arguments[arguments.index("--query") + 1], query)
        self.assertEqual(arguments[arguments.index("--port") + 1], "8123")

    def test_unsupported_language_and_invalid_ports_rejected(self) -> None:
        for language in ("bengali", "hindi", "", "../en"):
            with self.subTest(language=language):
                with self.assertRaises(bundle.BundleError):
                    build_arguments(self.runtime, language)
        for port in (-1, 65536, True, "8000"):
            with self.subTest(port=port):
                with self.assertRaises(bundle.BundleError):
                    build_arguments(self.runtime, "en", port=port)


if __name__ == "__main__":
    unittest.main()
