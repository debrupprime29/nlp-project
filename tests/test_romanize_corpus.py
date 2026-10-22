"""Offline passage/resource preparation; injected Romanizers are fixtures only."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data.fetch_dakshina import (
    ARCHIVE_GENERATION, ARCHIVE_MD5_BASE64, ARCHIVE_SIZE, ARCHIVE_URL,
    MANIFEST_NAME as DAKSHINA_MANIFEST_NAME, PINNED_URL, SCHEMA_VERSION,
    WANTED_MEMBERS, _digest as dakshina_digest,
)
from data.prepare_corpus import iter_jsonl
from data.romanize_corpus import (
    LEXICON_NAME, MANIFEST_NAME, POLICY_VERSION, UROMAN_VERSION, UROMAN_LICENSE,
    RomanizationError, _digest, _load_uroman, main, prepare_romanized_corpora,
)
from nlp.query_views import LexiconTransliterator, build_query_views
from retrieval.char_tfidf import CharTfidfIndex, load_corpus
from tests.test_query_views import FixedIdentifier


def write_records(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def passage(doc_id, text, *, language="hi", title=""):
    return {"doc_id": doc_id, "title": title, "text": text, "language": language, "source": "explicit-test-fixture",
            "source_doc_id": doc_id, "romanized_text": "", "romanization_status": "pending", "custom_provenance": {"keep": True}}


def sealed_cache(root, *, overrides=None):
    """Local six-member fixture with real cache seals; no mocked verifier."""
    overrides = overrides or {}
    defaults = {
        ("hi", "train"): "भूकंप\tbhukamp\t9\nभूकंप\tbhookamp\t3\nपानी\tpani\t8\n",
        ("bn", "train"): "ভূমিকম্প\tbhumikompo\t8\nজল\tjol\t4\n",
        ("hi", "dev"): "नदियाँ\tnadiyan\t3\n",
        ("hi", "test"): "किनारा\tkinara\t2\n",
        ("bn", "dev"): "নদী\tnodi\t4\n",
        ("bn", "test"): "তীর\ttir\t2\n",
    }
    members = []
    for position, name in enumerate(WANTED_MEMBERS):
        language = Path(name).name.split(".")[0]
        split = Path(name).name.split(".")[-2]
        data = overrides.get((language, split), defaults[(language, split)]).encode("utf-8")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        members.append({"path": name, "size": len(data), "header_offset": position * 1024,
                        "body_offset": position * 1024 + 512, "source_url": PINNED_URL,
                        "sha256": hashlib.sha256(data).hexdigest(), "header_sha256": "0" * 64})
    manifest = {"schema_version": SCHEMA_VERSION, "archive": {
        "url": ARCHIVE_URL, "pinned_url": PINNED_URL, "generation": ARCHIVE_GENERATION,
        "size": ARCHIVE_SIZE, "md5_base64": ARCHIVE_MD5_BASE64, "full_archive_md5_verified": False},
        "license": "CC-BY-SA-4.0", "members": members}
    manifest["manifest_sha256"] = dakshina_digest(manifest)
    (root / DAKSHINA_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class FakeRomanizer:
    def __init__(self, values=None):
        self.values = values or {("नदियाँ", "hin"): "nadiyan", ("नदी", "hin"): "nadi", ("नहीं", "hin"): "nahi", ("নদী", "ben"): "nodi"}
        self.calls = []

    def romanize_string(self, text, *, lcode):
        self.calls.append((text, lcode))
        return self.values.get((text, lcode), "not a single token")


class RomanizeCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "dakshina"
        self.dakshina = sealed_cache(self.cache)
        self.source = self.root / "input.jsonl"
        self.output = self.root / "romanized"

    def run_preparation(self, records, **kwargs):
        write_records(self.source, records)
        return prepare_romanized_corpora({"pilot": self.source}, self.cache, self.output, **kwargs)

    def test_attested_only_preserves_original_fields_english_and_exact_latin_numeric_tokens(self):
        records = [passage("hi:d", "भूकंप पानी Delhi café २०२६ ২০২৬ 2026", title="भूकंप"),
                   passage("en:d", "Earthquake 2026 café", language="en", title="Earthquakes")]
        report = self.run_preparation(records)
        rows = list(iter_jsonl(self.output / "pilot.jsonl"))
        self.assertEqual(rows[0]["romanized_text"], "bhukamp pani Delhi café २०२६ ২০২৬ 2026")
        self.assertEqual(rows[0]["romanized_title"], "bhukamp")
        self.assertEqual(rows[0]["romanization_status"], "attested")
        self.assertEqual(rows[1]["romanized_text"], records[1]["text"])
        self.assertEqual(rows[1]["romanization_status"], "identity")
        for output, original in zip(rows, records):
            for key in set(original) - {"romanized_text", "romanization_status"}:
                self.assertEqual(output[key], original[key])
        self.assertEqual(report["corpora"]["pilot"]["body_counts"]["attested_token_count"], 2)
        self.assertEqual(report["algorithmic_resource_record_count"], 0)
        self.assertFalse(report["fallback"]["enabled"])
        self.assertFalse(report["fixture_only"])

    def test_global_resource_caches_oov_across_corpora_and_labels_algorithmic_provenance(self):
        write_records(self.source, [passage("hi:d", "भूकंप नदियाँ नदियाँ")])
        other = self.root / "other.jsonl"
        write_records(other, [passage("bn:d", "ভূমিকম্প নদী", language="bn"), passage("hi:e", "नदियाँ")])
        engine = FakeRomanizer()
        report = prepare_romanized_corpora({"pilot": self.source, "reduced": other}, self.cache, self.output,
                                         allow_uroman=True, romanizer=engine)
        self.assertEqual(engine.calls.count(("नदियाँ", "hin")), 1)
        self.assertEqual(engine.calls.count(("নদী", "ben")), 1)
        self.assertEqual(report["uroman_unique_corpus_word_calls"], 2)
        self.assertEqual(report["algorithmic_resource_record_count"], 2)
        self.assertTrue(report["fixture_only"])
        self.assertEqual(report["fallback"]["package"], "injected_test_fixture")
        self.assertEqual(report["licenses"][1]["license"], UROMAN_LICENSE)
        self.assertNotIn("Apache", report["licenses"][1]["license"])
        resource = list(iter_jsonl(self.output / LEXICON_NAME))
        fallback = [row for row in resource if row["source"] == "uroman_algorithmic"]
        self.assertEqual(len(fallback), 2)
        self.assertTrue(all(row["count"] == 1 and row["revision"] == f"{UROMAN_VERSION}:{POLICY_VERSION}" for row in fallback))
        source_hashes = {member["sha256"] for member in self.dakshina["members"] if member["path"].endswith(".train.tsv")}
        self.assertTrue(all(row["revision"] in source_hashes for row in resource if row["source"] == "Dakshina-v1-train"))
        self.assertEqual(list(iter_jsonl(self.output / "reduced.jsonl"))[0]["romanized_text"], "bhumikompo nodi")
        self.assertFalse(report["contract"]["algorithmic_pairs_are_human_attested"])

    def test_persisted_generated_pairs_drive_identical_query_views_and_character_search(self):
        self.run_preparation([passage("hi:d", "नदियाँ भूकंप")], allow_uroman=True, romanizer=FakeRomanizer())
        backend = LexiconTransliterator.from_jsonl(self.output / LEXICON_NAME)
        native = build_query_views("नदियाँ भूकंप", FixedIdentifier({"नदियाँ": "hi", "भूकंप": "hi"}), backend)
        row = next(iter_jsonl(self.output / "pilot.jsonl"))
        self.assertEqual(native.roman_view.text, row["romanized_text"])
        roman = build_query_views("nadiyan bhukamp", FixedIdentifier({"nadiyan": "hi", "bhukamp": "hi"}), backend)
        self.assertEqual(roman.native_views[0].text, row["text"])
        index = CharTfidfIndex(load_corpus(self.output / "pilot.jsonl"))
        self.assertEqual(index.search(native.roman_view.text, top_k=1)[0].doc_id, "hi:d")

    def test_incomplete_unsupported_and_multitoken_results_are_unavailable_not_native_fallbacks(self):
        rows = [passage("hi:oov", "भूकंप किनारा"), passage("hi:other", "भूकंप Москва"), passage("hi:multi", "नदियाँ")]
        report = self.run_preparation(rows, allow_uroman=True, romanizer=FakeRomanizer({("नदियाँ", "hin"): "two words"}))
        for row in iter_jsonl(self.output / "pilot.jsonl"):
            self.assertEqual(row["romanized_text"], "")
            self.assertEqual(row["romanization_status"], "unavailable")
            self.assertEqual(row["text"], next(original["text"] for original in rows if original["doc_id"] == row["doc_id"]))
        corpus = report["corpora"]["pilot"]
        self.assertEqual(corpus["unavailable_document_count"], 3)
        self.assertEqual(corpus["body_counts"]["unavailable_token_count"], 3)
        self.assertEqual(corpus["unavailable_reasons"]["unsupported_native_script"], 1)
        self.assertEqual(corpus["unavailable_reasons"]["uroman_not_single_latin_token"], 2)
        self.assertEqual(report["algorithmic_resource_record_count"], 0)

    def test_oov_without_optin_never_initializes_or_imports_optional_engine(self):
        with patch("data.romanize_corpus._load_uroman", side_effect=AssertionError("must not initialize")):
            report = self.run_preparation([passage("hi:d", "नदियाँ")])
        row = next(iter_jsonl(self.output / "pilot.jsonl"))
        self.assertEqual(row["romanized_text"], "")
        self.assertEqual(report["corpora"]["pilot"]["unavailable_reasons"], {"oov_no_fallback": 1})

    def test_query_negation_protection_remains_unchanged_with_corpus_pair_available(self):
        self.run_preparation([passage("hi:d", "नहीं भूकंप")], allow_uroman=True, romanizer=FakeRomanizer())
        row = next(iter_jsonl(self.output / "pilot.jsonl"))
        self.assertEqual(row["romanized_text"], "nahi bhukamp")
        backend = LexiconTransliterator.from_jsonl(self.output / LEXICON_NAME)
        views = build_query_views("नहीं भूकंप", FixedIdentifier({"नहीं": "hi", "भूकंप": "hi"}), backend)
        self.assertIsNone(views.roman_view)
        self.assertEqual(views.diagnostics["roman_unavailable_reason"], "protected_native_token")
        self.assertEqual(views.original_text, row["text"])

    def test_heldout_labels_are_only_evaluated_after_freezing_and_never_become_pairs(self):
        engine = FakeRomanizer()
        report = self.run_preparation([passage("hi:d", "भूकंप")], allow_uroman=True, romanizer=engine, evaluate_heldout=True)
        resource = list(iter_jsonl(self.output / LEXICON_NAME))
        self.assertTrue(all(row["source"] == "Dakshina-v1-train" for row in resource))
        self.assertNotIn("नदियाँ", {row["native"] for row in resource})
        self.assertNotIn("নদী", {row["native"] for row in resource})
        heldout = json.loads((self.output / report["heldout_evaluation"]["path"]).read_text())
        self.assertFalse(heldout["heldout_labels_used_for_fit"])
        self.assertTrue(heldout["resource_frozen_before_evaluation"])
        self.assertEqual(heldout["frozen_resource_sha256"], report["lexicon"]["sha256"])
        self.assertEqual(heldout["splits"]["hi_dev"]["native_to_roman"]["top1_attested_variant_match_rate"], 1.0)
        self.assertEqual(heldout["splits"]["hi_dev"]["roman_to_native"]["top1_attested_variant_match_rate"], 0.0)
        self.assertEqual(report["algorithmic_resource_record_count"], 0)
        self.assertEqual(report["uroman_unique_corpus_word_calls"], 0)

    def test_heldout_unknown_without_fallback_reports_zero_coverage_not_success(self):
        report = self.run_preparation([passage("hi:d", "भूकंप")], evaluate_heldout=True)
        heldout = json.loads((self.output / report["heldout_evaluation"]["path"]).read_text())
        for split in heldout["splits"].values():
            for direction in ("native_to_roman", "roman_to_native"):
                self.assertEqual(split[direction]["covered_word_count"], 0)
                self.assertEqual(split[direction]["top1_attested_variant_match_rate"], 0.0)
                self.assertEqual(split[direction]["top2_attested_variant_match_rate"], 0.0)

    def test_manifests_seal_inputs_outputs_and_package_policy_without_touching_qrels(self):
        qrels = self.root / "qrels.jsonl"
        qrels.write_bytes(b"original qrels bytes\n")
        records = [passage("hi:d", "भूकंप")]
        report = self.run_preparation(records)
        saved = json.loads((self.output / MANIFEST_NAME).read_text())
        self.assertEqual(saved, report)
        self.assertEqual(saved["manifest_sha256"], _digest({key: value for key, value in saved.items() if key != "manifest_sha256"}))
        for path, checksum in saved["input_files_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), checksum)
        self.assertEqual(hashlib.sha256((self.output / LEXICON_NAME).read_bytes()).hexdigest(), saved["lexicon"]["sha256"])
        self.assertEqual(qrels.read_bytes(), b"original qrels bytes\n")
        self.assertEqual(list(iter_jsonl(self.source)), records)

    def test_existing_outputs_and_sealed_cache_destinations_are_not_overwritten(self):
        write_records(self.source, [passage("hi:d", "भूकंप")])
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(RomanizationError, "overwrite"):
            prepare_romanized_corpora({"pilot": self.source}, self.cache, self.output)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(RomanizationError, "sealed Dakshina"):
            prepare_romanized_corpora({"pilot": self.source}, self.cache, self.cache / "new-output")

    def test_noncanonical_duplicate_and_invalid_input_roll_back_outputs(self):
        for records in ([passage("hi:d", "cafe\u0301 भूकंप")],
                        [passage("hi:d", "भूकंप"), passage("hi:d", "पानी")],
                        [passage("hi:d", "भूकंप", language="mr")], []):
            with self.subTest(records=records):
                with self.assertRaises((RomanizationError, ValueError)):
                    self.run_preparation(records)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".romanized.romanize-*")), [])

    def test_rejects_symlink_inputs_outputs_and_bad_corpus_labels(self):
        write_records(self.source, [passage("hi:d", "भूकंप")])
        linked = self.root / "linked.jsonl"
        linked.symlink_to(self.source)
        with self.assertRaisesRegex(RomanizationError, "Symbolic"):
            prepare_romanized_corpora({"pilot": linked}, self.cache, self.output)
        for name in ("../escape", "bad/name", "", "transliteration"):
            with self.subTest(name=name), self.assertRaises(RomanizationError):
                prepare_romanized_corpora({name: self.source}, self.cache, self.output)

    def test_invalid_train_pair_is_counted_and_not_allowed_to_break_the_resource(self):
        self.dakshina = sealed_cache(self.cache, overrides={("hi", "train"): "भूकंप\tbhukamp\t9\nदोशब्द\ttwo words\t1\n"})
        report = self.run_preparation([passage("hi:d", "भूकंप")])
        self.assertEqual(report["training_lexicon_row_counts"]["hi"]["excluded_non_single_script_token_rows"], 1)
        resource = LexiconTransliterator.from_jsonl(self.output / LEXICON_NAME)
        self.assertEqual(resource.roman_candidates("भूकंप", "hi", 1)[0].text, "bhukamp")

    def test_injected_fallback_requires_optin_and_real_package_version_is_enforced(self):
        write_records(self.source, [passage("hi:d", "भूकंप")])
        with self.assertRaisesRegex(RomanizationError, "allow_uroman"):
            prepare_romanized_corpora({"pilot": self.source}, self.cache, self.output, romanizer=FakeRomanizer())
        class WrongVersion:
            version = "unexpected-version"
        with patch("data.romanize_corpus.importlib.metadata.distribution", return_value=WrongVersion()):
            with self.assertRaisesRegex(RomanizationError, "must be pinned"):
                prepare_romanized_corpora({"pilot": self.source}, self.cache, self.output, allow_uroman=True)
        self.assertFalse(self.output.exists())

    def test_cli_is_offline_and_duplicate_names_fail_without_outputs(self):
        write_records(self.source, [passage("hi:d", "भूकंप")])
        arguments = ["--corpus", f"pilot={self.source}", "--dakshina-dir", str(self.cache), "--output-dir", str(self.output), "--evaluate-heldout"]
        with patch("socket.socket", side_effect=AssertionError("network prohibited")), redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        other_output = self.root / "duplicate"
        arguments[arguments.index(str(self.output))] = str(other_output)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main([*arguments, "--corpus", f"pilot={self.source}"]), 2)
        self.assertFalse(other_output.exists())

    def test_fallback_exceptions_invalid_types_and_nonlatin_text_are_counted(self):
        class FailingRomanizer:
            def romanize_string(self, text, *, lcode):
                if text == "नदियाँ":
                    raise RuntimeError("explicit injected failure")
                return 17 if text == "किनारा" else "ভুল"
        report = self.run_preparation([passage("hi:d", "नदियाँ किनारा अज्ञात")],
                                      allow_uroman=True, romanizer=FailingRomanizer())
        self.assertEqual(report["corpora"]["pilot"]["unavailable_reasons"], {
            "uroman_error": 1, "uroman_non_string": 1, "uroman_not_single_latin_token": 1})
        self.assertEqual(report["unavailable_unique_supported_native_word_count"], 3)
        self.assertEqual(next(iter_jsonl(self.output / "pilot.jsonl"))["romanized_text"], "")

    def test_input_change_during_processing_is_detected_and_staged_outputs_removed(self):
        class MutatingRomanizer:
            def romanize_string(inner_self, text, *, lcode):
                self.source.write_text("changed sealed source", encoding="utf-8")
                return "nadiyan"
        with self.assertRaisesRegex(RomanizationError, "changed during"):
            self.run_preparation([passage("hi:d", "नदियाँ")], allow_uroman=True, romanizer=MutatingRomanizer())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".romanized.romanize-*")), [])

    def test_bad_attestation_counts_and_missing_cache_fail_without_outputs(self):
        sealed_cache(self.cache, overrides={("hi", "train"): "भूकंप\tbhukamp\t0\n"})
        with self.assertRaisesRegex(RomanizationError, "positive integers"):
            self.run_preparation([passage("hi:d", "भूकंप")])
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(RomanizationError, "sealed six-file"):
            prepare_romanized_corpora({"pilot": self.source}, self.root / "absent-cache", self.output)

    def test_real_loader_seals_actual_installed_resources_and_custom_license_without_downloads(self):
        package_root = self.root / "installed-test-fixture"
        implementation = Path("uroman/uroman.py")
        license_path = Path(f"uroman-{UROMAN_VERSION}.dist-info/licenses/LICENSE.txt")
        for relative, text in ((implementation, "explicit implementation fixture"), (license_path, "custom USC license fixture")):
            (package_root / relative).parent.mkdir(parents=True, exist_ok=True)
            (package_root / relative).write_text(text, encoding="utf-8")
        distribution = types.SimpleNamespace(version=UROMAN_VERSION, files=[implementation, license_path], locate_file=lambda relative: package_root / relative)
        fake_module = types.ModuleType("uroman")
        fake_module.Uroman = FakeRomanizer
        with patch("data.romanize_corpus.importlib.metadata.distribution", return_value=distribution), patch.dict("sys.modules", {"uroman": fake_module}), patch("socket.socket", side_effect=AssertionError("network prohibited")):
            engine, metadata = _load_uroman()
        self.assertIsInstance(engine, FakeRomanizer)
        self.assertEqual(metadata["version"], UROMAN_VERSION)
        self.assertEqual(metadata["license"], UROMAN_LICENSE)
        self.assertEqual(metadata["license_files_sha256"], {str(license_path): hashlib.sha256((package_root / license_path).read_bytes()).hexdigest()})
        self.assertIn(str(implementation), metadata["installed_files_sha256"])
        self.assertFalse(metadata["wheel_digest_verified_by_this_command"])


if __name__ == "__main__":
    unittest.main()
