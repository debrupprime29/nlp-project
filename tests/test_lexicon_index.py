"""Indexed lexicon lookups must preserve the original scan's exact contract."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from nlp.query_views import (
    LexiconTransliterator,
    IndicXlitTransliterator,
    LEXICON_CANDIDATE_POLICY,
    QueryViewError,
    TARGET_SCRIPTS,
    TransliterationCandidate,
    _validated_candidates,
    _metadata,
    build_query_views,
)
from nlp.text import normalize_processing_text
from tests.test_query_views import FixedIdentifier
from nlp.language_id import LanguageIdConfig, TokenLanguageIdentifier
from tests.test_language_id import training_examples, write_modified_artifact


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record(language, native, romanized, *, count=1, source="large-test-resource", revision="v1"):
    return {"language": language, "native": native, "romanized": romanized, "count": count, "source": source, "revision": revision}


def word(number, alphabet):
    result = []
    for _ in range(4):
        result.append(alphabet[number % len(alphabet)])
        number //= len(alphabet)
    return "".join(result)


def large_records():
    records = []
    for language, letters in (("hi", "कखगघचछजझटठडढतथदधनपफबभमयरलवशषसह"), ("bn", "কখগঘচছজঝটঠডঢতথদধনপফবভমযরলশষসহ")):
        for number in range(1000):
            records.append(record(language, word(number, letters), "word" + word(number, "abcdefghijklmnopqrstuvwxyz")))
    records.extend((
        record("hi", "भूकंप", "bhukamp", count=7),
        record("hi", "भुकम्प", "bhukamp", count=7),
        record("hi", "भूकंप", "bhukamp", count=3),
        record("hi", "भूकंप", "bhukamp", count=7, source="a-resource"),
        record("hi", "भूकंप", "bhukamp", count=7, source="a-resource", revision="v0"),
        record("hi", "भूकंप", "bhookamp", count=9),
        record("hi", "भूकंप", "bhookamp", count=9),  # Exact resource duplicate.
        record("bn", "ভূমিকম্প", "bhukamp", count=11),
        record("hi", "का", "café", count=2),
    ))
    return records


def scan_reference(records, token, language, limit, *, native):
    source, target = ("romanized", "native") if native else ("native", "romanized")
    key = normalize_processing_text(token).casefold()
    candidates = [
        TransliterationCandidate(item[target], math.log(item["count"]), "lexicon:" + item["source"], item["revision"])
        for item in records if item["language"] == language and item[source].casefold() == key
    ]
    return _validated_candidates(candidates, script=TARGET_SCRIPTS[language] if native else "latin", limit=limit)


class UnreadableRecords:
    def __iter__(self):
        raise AssertionError("lookup scanned the full lexicon")

    def __getitem__(self, key):
        raise AssertionError("lookup indexed the full lexicon")

    def __len__(self):
        raise AssertionError("lookup inspected the full lexicon")


class LexiconIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = large_records()
        cls.canonical_records = tuple(json.loads(item) for item in sorted({canonical(item).decode("utf-8") for item in cls.records}))

    def test_large_resource_matches_scan_order_dedup_scores_limits_and_languages(self):
        backend = LexiconTransliterator(self.records)
        for native, token, language in (
            (True, "BHUKAMP", "hi"), (True, "bhukamp", "bn"),
            (False, "भूकंप", "hi"), (False, "ভূমিকম্প", "bn"),
            (True, "CAFE\u0301", "hi"), (False, "का", "hi"),
            (True, "missingword", "hi"), (False, "অজানা", "bn"),
            (True, "word" + word(999, "abcdefghijklmnopqrstuvwxyz"), "hi"),
        ):
            for limit in (1, 2, 8):
                with self.subTest(native=native, token=token, language=language, limit=limit):
                    method = backend.native_candidates if native else backend.roman_candidates
                    self.assertEqual(method(token, language, limit), scan_reference(self.canonical_records, token, language, limit, native=native))

    def test_manifest_and_fingerprints_remain_identical_to_canonical_resource(self):
        backend = LexiconTransliterator(self.records)
        expected = {
            "backend": "supplied_lexicon", "schema_version": 1,
            "record_count": len(self.canonical_records),
            "resource_sha256": hashlib.sha256(canonical(self.canonical_records)).hexdigest(),
            "sources": sorted({item["source"] for item in self.canonical_records}),
            "revisions": sorted({item["revision"] for item in self.canonical_records}),
            "score_type": "log_attestation_count_uncalibrated",
            "candidate_policy": LEXICON_CANDIDATE_POLICY,
            "max_candidate_response": 64,
        }
        self.assertEqual(backend.manifest, expected)
        self.assertEqual(backend.fingerprint, hashlib.sha256(canonical(expected)).hexdigest())
        reordered = LexiconTransliterator(list(reversed(self.records)))
        self.assertEqual(reordered.manifest, expected)
        self.assertEqual(reordered.native_candidates("bhukamp", "hi", 8), backend.native_candidates("bhukamp", "hi", 8))

    def test_repeated_hit_and_oov_lookups_do_not_read_the_full_resource(self):
        backend = LexiconTransliterator(self.records)
        expected_native = backend.native_candidates("bhukamp", "hi", 8)
        expected_roman = backend.roman_candidates("भूकंप", "hi", 8)
        # A structural test, not an elapsed-time assertion: even inspecting the
        # complete resource after construction must now fail.
        backend._records = UnreadableRecords()
        for _ in range(30):
            self.assertEqual(backend.native_candidates("BHUKAMP", "hi", 8), expected_native)
            self.assertEqual(backend.roman_candidates("भूकंप", "hi", 8), expected_roman)
            self.assertEqual(backend.native_candidates("missingword", "hi", 8), ())
            self.assertEqual(backend.roman_candidates("অজানা", "bn", 8), ())
            self.assertEqual(backend.manifest["record_count"], len(self.canonical_records))

    def test_empty_resources_and_invalid_language_or_limit_keep_the_contract(self):
        backend = LexiconTransliterator([])
        self.assertEqual(backend.native_candidates("word", "hi", 1), ())
        self.assertEqual(backend.roman_candidates("শব্দ", "bn", 1), ())
        for method in (backend.native_candidates, backend.roman_candidates):
            with self.assertRaises(QueryViewError):
                method("word", "en", 1)
            for limit in (0, 9, True, 1.5):
                with self.subTest(limit=limit), self.assertRaises(QueryViewError):
                    method("word", "hi", limit)

    def test_large_alternative_bucket_is_ranked_deduplicated_and_bounded_before_validation(self):
        records = [record("hi", "शब्द", "word" + word(number, "abcdefghijklmnopqrstuvwxyz"), count=number + 1) for number in range(100)]
        # Repeated high-score spellings with distinct provenance must not use
        # up the response budget before the alternatives are deduplicated.
        records.extend(record("hi", "शब्द", "highest", count=1000, source=f"resource-{number}") for number in range(80))
        backend = LexiconTransliterator(records)
        with patch("nlp.query_views._validated_candidates", wraps=_validated_candidates) as validate:
            candidates = backend.roman_candidates("शब्द", "hi", 8)
        supplied = validate.call_args.args[0]
        self.assertEqual(len(supplied), 64)
        self.assertEqual(len({candidate.text for candidate in supplied}), 64)
        self.assertEqual([candidate.text for candidate in candidates], ["highest", *["word" + word(number, "abcdefghijklmnopqrstuvwxyz") for number in range(99, 92, -1)]])
        self.assertEqual(candidates[0].backend, "lexicon:resource-0")
        self.assertTrue(all(left.raw_score >= right.raw_score for left, right in zip(supplied, supplied[1:])))

    def test_prohibited_negation_targets_are_filtered_at_top_and_below_requested_limit(self):
        for prohibited_count, safe_count in ((20, 1), (1, 20)):
            with self.subTest(prohibited_count=prohibited_count):
                backend = LexiconTransliterator([
                    record("hi", "नहीं", "naya", count=prohibited_count),
                    record("hi", "नया", "naya", count=safe_count),
                    record("hi", "नोट", "not", count=prohibited_count),
                    record("hi", "नोट", "note", count=safe_count),
                ])
                self.assertEqual([candidate.text for candidate in backend.native_candidates("naya", "hi", 1)], ["नया"])
                self.assertEqual([candidate.text for candidate in backend.roman_candidates("नोट", "hi", 1)], ["note"])

    def test_all_prohibited_targets_mean_unavailable_and_original_protections_survive(self):
        backend = LexiconTransliterator([
            record("hi", "वर्जित", "not", count=20),
            record("hi", "वर्जित", "never", count=1),
            record("hi", "नहीं", "naa", count=9),
            record("hi", "नोट", "not", count=3),
        ])
        self.assertEqual(backend.roman_candidates("वर्जित", "hi", 2), ())
        self.assertEqual(backend.native_candidates("naa", "hi", 2), ())
        native = build_query_views("वर्जित", FixedIdentifier({"वर्जित": "hi"}), backend)
        self.assertIsNone(native.roman_view)
        self.assertEqual(native.diagnostics["roman_unavailable_reason"], "no_romanization_candidate")
        self.assertEqual(native.original_text, "वर्जित")
        protected = build_query_views("not Delhi 2026", FixedIdentifier({"not": "hi", "Delhi": "hi"}), backend)
        self.assertEqual(protected.original_text, "not Delhi 2026")
        self.assertEqual(protected.native_views, ())
        self.assertEqual(protected.roman_view.text, "not Delhi 2026")

    def test_generic_and_direct_indicxlit_responses_remain_fail_closed(self):
        prohibited = TransliterationCandidate("नहीं", 1.0, "upstream", "v1")
        with self.assertRaisesRegex(QueryViewError, "negation"):
            _validated_candidates([prohibited], script="devanagari", limit=1)
        backend = IndicXlitTransliterator(allow_model_download=True)
        backend._engines["hi"] = types.SimpleNamespace(translit_word=lambda *args, **kwargs: ["नहीं"])
        with self.assertRaisesRegex(QueryViewError, "negation"):
            backend.native_candidates("naya", "hi", 1)
        backend._engines["hi"] = types.SimpleNamespace(translit_word=lambda *args, **kwargs: ["शब्द"] * 66)
        with self.assertRaisesRegex(QueryViewError, "bounded"):
            backend.native_candidates("shabd", "hi", 1)

    def test_rewriting_policy_is_versioned_without_changing_resource_content_hash(self):
        backend = LexiconTransliterator(self.records)
        self.assertEqual(backend.manifest["candidate_policy"], "safe_target_score_ordered_top64_v1")
        self.assertEqual(backend.manifest["max_candidate_response"], 64)
        self.assertEqual(backend.manifest["resource_sha256"], hashlib.sha256(canonical(self.canonical_records)).hexdigest())
        old_manifest = {key: value for key, value in backend.manifest.items() if key not in ("candidate_policy", "max_candidate_response")}
        self.assertNotEqual(backend.fingerprint, hashlib.sha256(canonical(old_manifest)).hexdigest(), "Changed query behavior must invalidate older frozen gate fingerprints")

    def test_compact_identifier_metadata_is_preferred_without_eager_full_manifest_access(self):
        model = TokenLanguageIdentifier.fit(training_examples(), LanguageIdConfig(epochs=5))
        expected = dict(model.inference_metadata)
        with patch.object(TokenLanguageIdentifier, "manifest", new_callable=PropertyMock, side_effect=AssertionError("full manifest must not be accessed")):
            metadata = _metadata(model, identifier=True)
            views = build_query_views("bhukamp", model)
        self.assertEqual(metadata["model_sha256"], expected["model_sha256"])
        self.assertEqual(views.diagnostics["identifier"], metadata)
        self.assertLess(len(canonical(views.diagnostics)), 8192)

    def test_changed_numeric_identifier_and_older_verbose_metadata_invalidate_frozen_gate(self):
        from tests.test_train_gate import fixture_pipeline
        from training.train_gate import FEATURE_NAMES, GateError, LinearSoftmaxGate
        model = TokenLanguageIdentifier.fit(training_examples(), LanguageIdConfig(epochs=5))
        pipeline = fixture_pipeline()
        pipeline.language_identifier = model
        current = pipeline.manifest
        legacy = copy.deepcopy(current)
        legacy["resources"]["language_identifier"] = {**dict(model.manifest), "probability_calibration": "uncalibrated_or_unknown"}
        def gate(manifest):
            return LinearSoftmaxGate([[0.0] * len(FEATURE_NAMES) for _ in range(4)], [0.0] * 4, [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES), pipeline_manifest=manifest, training_report={})
        with self.assertRaises(GateError):
            gate(legacy).assert_compatible(current)
        fitted = gate(current)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            model.save_json(path)
            artifact = json.loads(path.read_text())
            artifact["weights"][0][0] += 0.5
            write_modified_artifact(path, artifact)
            pipeline.language_identifier = TokenLanguageIdentifier.load_json(path)
        self.assertNotEqual(pipeline.manifest["pipeline_fingerprint"], current["pipeline_fingerprint"])
        with self.assertRaises(GateError):
            fitted.assert_compatible(pipeline.manifest)


if __name__ == "__main__":
    unittest.main()
