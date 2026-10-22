from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from nlp.language_id import (
    AUXILIARY_TRAIN,
    LABELS,
    LabeledTokenExample,
    LanguageIdConfig,
    TokenLanguageIdentifier,
    TokenPrediction,
)
from nlp.query_views import (
    IndicXlitTransliterator,
    LexiconTransliterator,
    QueryViewError,
    TransliterationCandidate,
    ViewConfig,
    build_query_views,
)
from nlp.text import normalize_processing_text, tokenize_with_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedIdentifier:
    def __init__(self, labels: dict[str, str], uncertain: tuple[str, ...] = ()):
        self.labels = labels
        self.uncertain = set(uncertain)

    @property
    def manifest(self) -> dict:
        return {"model_type": "controlled-test-fixture", "revision": "1"}

    def predict(self, text: str) -> tuple[TokenPrediction, ...]:
        predictions = []
        for span in tokenize_with_spans(text):
            label = self.labels.get(span.text, "other")
            uncertain = span.text in self.uncertain or label == "other"
            probabilities = {name: 0.02 for name in LABELS}
            probabilities[label] = 0.94
            if uncertain:
                probabilities = {name: 0.25 for name in LABELS}
                label = "en"
            entropy = -sum(value * math.log(value) for value in probabilities.values())
            predictions.append(TokenPrediction(span, probabilities, label, "other" if uncertain else label, uncertain, entropy))
        return tuple(predictions)


class RecordingTransliterator:
    def __init__(self, native: dict | None = None, roman: dict | None = None):
        self.native = native or {}
        self.roman = roman or {}
        self.calls: list[tuple[str, str, str, int]] = []

    def native_candidates(self, token: str, target_language: str, limit: int):
        self.calls.append(("native", token, target_language, limit))
        return [TransliterationCandidate(text, score, "test-fixture", "v1") for text, score in self.native.get((token, target_language), ())]

    def roman_candidates(self, token: str, source_language: str, limit: int):
        self.calls.append(("roman", token, source_language, limit))
        return [TransliterationCandidate(text, score, "test-fixture", "v1") for text, score in self.roman.get((token, source_language), ())]


def lexicon_records() -> list[dict]:
    return [
        {"language": "hi", "native": "भूकंप", "romanized": "bhukamp", "source": "controlled-fixture", "revision": "1", "count": 20},
        {"language": "hi", "native": "भुकम्प", "romanized": "bhukamp", "source": "controlled-fixture", "revision": "1", "count": 5},
        {"language": "bn", "native": "ভূমিকম্প", "romanized": "bhumikompo", "source": "controlled-fixture", "revision": "1", "count": 15},
    ]


class QueryViewTests(unittest.TestCase):
    def test_unicode_digits_stay_exact_and_do_not_disable_roman_view(self) -> None:
        result = build_query_views("Delhi २०२६ ২০২৬")
        self.assertEqual(result.roman_view.text, "Delhi २०२६ ২০২৬")
        self.assertEqual(result.native_views, ())
        self.assertEqual(result.diagnostics["backend_calls"], 0)

    def test_script_composition_uses_character_proportions(self) -> None:
        text = "earthquake क्यों"
        result = build_query_views(text)
        length = len("earthquake") + len("क्यों")
        self.assertAlmostEqual(result.diagnostics["script_fractions"]["latin"], len("earthquake") / length)
        self.assertAlmostEqual(result.diagnostics["token_script_fractions"]["latin"], 0.5)

    def test_foreign_script_marks_and_inconsistent_entropy_are_rejected(self) -> None:
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंपা", 1.0)]})
        with self.assertRaises(QueryViewError):
            build_query_views("bhukamp", FixedIdentifier({"bhukamp": "hi"}), backend)
        prediction = FixedIdentifier({"bhukamp": "hi"}).predict("bhukamp")[0]
        class WrongEntropyIdentifier:
            def predict(self, text):
                return (replace(prediction, entropy=100.0),)
        with self.assertRaises(QueryViewError):
            build_query_views("bhukamp", WrongEntropyIdentifier())

    def test_original_is_exact_raw_unicode_while_edits_use_nfc_coordinates(self) -> None:
        raw = "cafe\u0301 bhukamp"
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views(raw, FixedIdentifier({"café": "en", "bhukamp": "hi"}), backend)
        self.assertEqual(result.original_text, raw)
        self.assertEqual(result.original.text, raw)
        self.assertEqual(result.processing_text, "café bhukamp")
        self.assertEqual(len(result.native_views), 1)
        view = result.native_views[0]
        self.assertEqual(view.text, "café भूकंप")
        reconstructed = result.processing_text
        for edit in sorted(view.edits, key=lambda item: item.start, reverse=True):
            self.assertEqual(result.processing_text[edit.start:edit.end], edit.original)
            reconstructed = reconstructed[:edit.start] + edit.replacement + reconstructed[edit.end:]
        self.assertEqual(reconstructed, view.text)
        for prediction in result.tokens:
            self.assertEqual(result.processing_text[prediction.span.start:prediction.span.end], prediction.span.text)

    def test_protected_tokens_and_contacts_never_reach_transliterator(self) -> None:
        text = "Delhi not 2026 https://example.org bhukamp"
        labels = {span.text: "hi" for span in tokenize_with_spans(text)}
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views(text, FixedIdentifier(labels), backend)
        self.assertEqual([call[1] for call in backend.calls], ["bhukamp"])
        self.assertEqual(result.native_views[0].text, "Delhi not 2026 https://example.org भूकंप")
        for span in result.protections:
            original = result.processing_text[span.start:span.end]
            self.assertIn(original, result.native_views[0].text)

    def test_uncertain_tokens_remain_unchanged_without_backend_calls(self) -> None:
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views(
            "kal bhukamp", FixedIdentifier({"kal": "hi", "bhukamp": "hi"}, uncertain=("kal",)), backend,
        )
        self.assertEqual([call[1] for call in backend.calls], ["bhukamp"])
        self.assertEqual(result.native_views[0].text, "kal भूकंप")
        self.assertEqual(result.roman_view.text, "kal bhukamp")

    def test_english_words_are_retained_in_a_native_hypothesis(self) -> None:
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views("bhukamp causes", FixedIdentifier({"bhukamp": "hi", "causes": "en"}), backend)
        self.assertEqual(result.native_views[0].text, "भूकंप causes")
        self.assertEqual([call[1] for call in backend.calls], ["bhukamp"])

    def test_native_hypotheses_have_a_global_two_view_limit_and_stable_order(self) -> None:
        native = {
            ("bhukamp", "hi"): [("भूकंप", 2.0), ("भुकम्प", 1.0)],
            ("keno", "bn"): [("কেন", 2.0), ("কেনো", 1.0)],
        }
        lid = FixedIdentifier({"bhukamp": "hi", "keno": "bn"})
        result = build_query_views("bhukamp keno", lid, RecordingTransliterator(native=native))
        self.assertEqual(len(result.native_views), 2)
        self.assertEqual(len({view.text for view in result.native_views}), 2)
        self.assertEqual(result.native_views[0].text, "भूकंप কেন")
        reversed_candidates = {key: list(reversed(values)) for key, values in native.items()}
        repeated = build_query_views("bhukamp keno", lid, RecordingTransliterator(native=reversed_candidates))
        self.assertEqual(result.native_views, repeated.native_views)
        with self.assertRaises(QueryViewError):
            build_query_views("bhukamp", lid, RecordingTransliterator(native=native), config=ViewConfig(max_native_views=3))

    def test_duplicate_hypotheses_are_collapsed_and_native_identity_is_unavailable(self) -> None:
        duplicate = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 2.0), ("भूकंप", 1.0)]})
        result = build_query_views("bhukamp", FixedIdentifier({"bhukamp": "hi"}), duplicate)
        self.assertEqual([view.text for view in result.native_views], ["भूकंप"])
        native = build_query_views(
            "भूकंप", FixedIdentifier({"भूकंप": "hi"}),
            RecordingTransliterator(roman={("भूकंप", "hi"): [("bhukamp", 1.0)]}),
        )
        self.assertEqual(native.native_views, ())
        self.assertFalse(native.diagnostics["native_available"])
        self.assertEqual(native.roman_view.text, "bhukamp")

    def test_backend_call_budget_prevents_combinatorial_work(self) -> None:
        backend = RecordingTransliterator(native={
            ("bhukamp", "hi"): [("भूकंप", 1.0)], ("keno", "bn"): [("কেন", 1.0)],
        })
        result = build_query_views(
            "bhukamp keno", FixedIdentifier({"bhukamp": "hi", "keno": "bn"}), backend,
            config=ViewConfig(max_backend_calls=1),
        )
        self.assertLessEqual(len(backend.calls), 1)
        self.assertLessEqual(len(result.native_views), 2)
        self.assertEqual(result.diagnostics["backend_calls"], len(backend.calls))

    def test_roman_view_requires_complete_conversion_of_native_letters(self) -> None:
        complete = build_query_views(
            "भूकंप rainfall", FixedIdentifier({"भूकंप": "hi", "rainfall": "en"}),
            RecordingTransliterator(roman={("भूकंप", "hi"): [("bhukamp", 1.0)]}),
        )
        self.assertEqual(complete.roman_view.text, "bhukamp rainfall")
        incomplete = build_query_views(
            "भूकंप অজানা", FixedIdentifier({"भूकंप": "hi", "অজানা": "bn"}),
            RecordingTransliterator(roman={("भूकंप", "hi"): [("bhukamp", 1.0)]}),
        )
        self.assertIsNone(incomplete.roman_view)
        self.assertFalse(incomplete.diagnostics["roman_available"])
        self.assertTrue(incomplete.diagnostics["roman_unavailable_reason"])

    def test_protected_native_negation_prevents_incomplete_roman_branch(self) -> None:
        backend = RecordingTransliterator(roman={("भूकंप", "hi"): [("bhukamp", 1.0)], ("नहीं", "hi"): [("nahin", 1.0)]})
        result = build_query_views("भूकंप नहीं", FixedIdentifier({"भूकंप": "hi", "नहीं": "hi"}), backend)
        self.assertIsNone(result.roman_view)
        self.assertNotIn("नहीं", [call[1] for call in backend.calls])
        self.assertEqual(result.original.text, "भूकंप नहीं")

    def test_absent_resources_preserve_original_and_report_branch_availability(self) -> None:
        latin = build_query_views("bhukamp causes")
        self.assertEqual(latin.original.text, "bhukamp causes")
        self.assertEqual(latin.native_views, ())
        self.assertEqual(latin.roman_view.text, "bhukamp causes")
        native = build_query_views("भूकंप क्यों")
        self.assertEqual(native.original.text, "भूकंप क्यों")
        self.assertEqual(native.native_views, ())
        self.assertIsNone(native.roman_view)

    def test_explicit_protection_uses_nfc_offsets_and_prevents_rewriting(self) -> None:
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views(
            "bhukamp", FixedIdentifier({"bhukamp": "hi"}), backend,
            explicit_protections=[{"start": 0, "end": 7, "text": "bhukamp", "reason": "entity"}],
        )
        self.assertEqual(result.native_views, ())
        self.assertEqual(backend.calls, [])
        with self.assertRaises(QueryViewError):
            build_query_views("cafe\u0301", explicit_protections=[{"start": 0, "end": 5}])

    def test_malformed_identifier_probabilities_and_offsets_are_rejected(self) -> None:
        prediction = FixedIdentifier({"bhukamp": "hi"}).predict("bhukamp")[0]
        invalid_predictions = [
            (),
            (replace(prediction, span=replace(prediction.span, start=1)),),
            (replace(prediction, probabilities={"en": 0.0, "hi": 1.0}),),
            (replace(prediction, probabilities={"en": -0.1, "hi": 1.0, "bn": 0.1, "other": 0.0}),),
            (replace(prediction, probabilities={"en": 0.0, "hi": float("nan"), "bn": 0.0, "other": 0.0}),),
            (replace(prediction, probabilities={"en": 0.0, "hi": 0.7, "bn": 0.0, "other": 0.0}),),
            (replace(prediction, top_label="bn"),),
            (replace(prediction, decision_label="en"),),
        ]
        for predictions in invalid_predictions:
            class BrokenIdentifier:
                def predict(self, text):
                    return predictions
            with self.subTest(predictions=predictions), self.assertRaises(QueryViewError):
                build_query_views("bhukamp", BrokenIdentifier(), RecordingTransliterator())

    def test_low_confidence_cannot_rewrite_even_if_identifier_disables_abstention(self) -> None:
        base = FixedIdentifier({"bhukamp": "hi"}).predict("bhukamp")[0]
        probabilities = {"en": 0.25, "hi": 0.26, "bn": 0.25, "other": 0.24}
        invalid = replace(base, probabilities=probabilities, entropy=-sum(value * math.log(value) for value in probabilities.values()))
        class OverconfidentIdentifier:
            def predict(self, text):
                return (invalid,)
        backend = RecordingTransliterator(native={("bhukamp", "hi"): [("भूकंप", 1.0)]})
        result = build_query_views("bhukamp", OverconfidentIdentifier(), backend)
        self.assertEqual(result.native_views, ())
        self.assertEqual(backend.calls, [])

    def test_malformed_backend_candidates_fail_closed(self) -> None:
        malformed = [
            "भूकंप",
            ["भूकंप"],
            [TransliterationCandidate("earthquake", 1.0, "fixture", "1")],
            [TransliterationCandidate("भूकंप causes", 1.0, "fixture", "1")],
            [TransliterationCandidate("भूकंप 2027", 1.0, "fixture", "1")],
            [TransliterationCandidate("नहीं", 1.0, "fixture", "1")],
            [TransliterationCandidate("भूकंप\x00", 1.0, "fixture", "1")],
            [TransliterationCandidate("भूकंप", float("inf"), "fixture", "1")],
            [TransliterationCandidate("भूकंप", 1.0, "", "1")],
            [TransliterationCandidate("भूकंप", 1.0, "fixture", "")],
        ]
        for candidates in malformed:
            class BrokenBackend:
                def native_candidates(self, token, target_language, limit):
                    return candidates
                def roman_candidates(self, token, source_language, limit):
                    return ()
            with self.subTest(candidates=candidates), self.assertRaises(QueryViewError):
                build_query_views("bhukamp", FixedIdentifier({"bhukamp": "hi"}), BrokenBackend())


class LexiconAndOptionalBackendTests(unittest.TestCase):
    def test_attested_lexicon_jsonl_roundtrip_ranking_and_reverse_lookup(self) -> None:
        records = lexicon_records()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lexicon.jsonl"
            path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
            backend = LexiconTransliterator.from_jsonl(path)
            reversed_backend = LexiconTransliterator(list(reversed(records)))
            self.assertEqual(backend.fingerprint, reversed_backend.fingerprint)
            self.assertEqual([item.text for item in backend.native_candidates("bhukamp", "hi", 2)], ["भूकंप", "भुकम्प"])
            self.assertEqual([item.text for item in backend.roman_candidates("ভূমিকম্প", "bn", 2)], ["bhumikompo"])
            self.assertEqual(backend.native_candidates("unseenword", "hi", 2), ())

    def test_lexicon_rejects_wrong_scripts_and_missing_provenance(self) -> None:
        valid = lexicon_records()[0]
        for changed in (
            {"language": "en"}, {"native": "earthquake"}, {"romanized": "भूकंप"},
            {"source": ""}, {"revision": ""}, {"count": -1},
        ):
            with self.subTest(changed=changed), self.assertRaises(QueryViewError):
                LexiconTransliterator([{**valid, **changed}])

    def test_indicxlit_requires_explicit_model_download_permission(self) -> None:
        with patch.dict(sys.modules, {"ai4bharat.transliteration": None}):
            with self.assertRaises(QueryViewError) as caught:
                IndicXlitTransliterator(allow_model_download=False)
        self.assertIn("allow", str(caught.exception).lower())

    def test_indicxlit_adapter_passes_direction_language_and_topk_without_network(self) -> None:
        constructions = []
        inferences = []
        class MockEngine:
            def __init__(self, *args, **kwargs):
                constructions.append((args, kwargs))
                self.direction = kwargs["src_script_type"]
            def translit_word(self, token, *, lang_code, topk):
                inferences.append((self.direction, token, lang_code, topk))
                return {lang_code: ["भूकंप"]} if self.direction == "roman" else ["bhukamp"]
        package = types.ModuleType("ai4bharat")
        package.__path__ = []
        module = types.ModuleType("ai4bharat.transliteration")
        module.XlitEngine = MockEngine
        with patch.dict(sys.modules, {"ai4bharat": package, "ai4bharat.transliteration": module}), patch("nlp.query_views.importlib.metadata.version", return_value="fixture-version"):
            backend = IndicXlitTransliterator(allow_model_download=True)
            self.assertEqual(constructions, [])
            native = backend.native_candidates("bhukamp", "hi", 2)
            roman = backend.roman_candidates("भूकंप", "hi", 2)
            backend.native_candidates("bhukamp", "hi", 1)
        self.assertEqual([item.text for item in native], ["भूकंप"])
        self.assertEqual([item.text for item in roman], ["bhukamp"])
        self.assertEqual(constructions, [
            (("hi",), {"src_script_type": "roman", "beam_width": 4, "rescore": False}),
            ((), {"src_script_type": "indic", "beam_width": 4, "rescore": False}),
        ])
        self.assertEqual(inferences, [
            ("roman", "bhukamp", "hi", 2),
            ("indic", "भूकंप", "hi", 2),
            ("roman", "bhukamp", "hi", 1),
        ])
        self.assertEqual(native[0].backend, "IndicXlit")
        self.assertIn("fixture-version", native[0].revision)

    def test_trained_language_identifier_drives_lexicon_views(self) -> None:
        words = {"hi": "bhukamp", "bn": "bhumikompo", "en": "causes", "other": "zzqqx"}
        examples = [LabeledTokenExample(word, 0, len(word), label, AUXILIARY_TRAIN, label, "controlled-fixture") for label, word in words.items()]
        model = TokenLanguageIdentifier.fit(examples, LanguageIdConfig(epochs=220))
        backend = LexiconTransliterator(lexicon_records())
        result = build_query_views("bhukamp causes", model, backend)
        self.assertTrue(result.native_views)
        self.assertEqual(result.native_views[0].text, "भूकंप causes")
        self.assertEqual(result.roman_view.text, "bhukamp causes")

    def test_cli_uses_saved_language_model_and_explicit_lexicon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            word_labels = {"bhukamp": "hi", "causes": "en", "bhumikompo": "bn", "zzqqx": "other"}
            model = TokenLanguageIdentifier.fit(
                [LabeledTokenExample(word, 0, len(word), label, AUXILIARY_TRAIN, label, "fixture") for word, label in word_labels.items()],
                LanguageIdConfig(epochs=220),
            )
            model_path = root / "lid.json"
            model.save_json(model_path)
            lexicon_path = root / "lexicon.jsonl"
            lexicon_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in lexicon_records()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "nlp.query_views", "--text", "bhukamp causes", "--lid-model", str(model_path), "--backend", "lexicon", "--lexicon", str(lexicon_path)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["original_text"], "bhukamp causes")
            self.assertEqual(output["native_views"][0]["text"], "भूकंप causes")
            self.assertEqual(output["roman_view"]["text"], "bhukamp causes")


if __name__ == "__main__":
    unittest.main()
