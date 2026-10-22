from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

from nlp.language_id import (
    AUXILIARY_TRAIN,
    AUXILIARY_VALIDATION,
    LABELS,
    LabeledTokenExample,
    LanguageIdConfig,
    LanguageIdError,
    ScriptEvidenceBaseline,
    TokenLanguageIdentifier,
)
from nlp.text import normalize_processing_text, tokenize_with_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROMAN_WORDS = {
    "en": ("earthquake", "rainfall", "tomorrow", "computer"),
    "hi": ("bhukamp", "mausam", "kyunki", "aayega"),
    "bn": ("bhumikompo", "brishti", "kemon", "hochhe"),
    "other": ("zxqvv", "qvvzx", "vzzqx", "xqqvz"),
}


def example(
    text: str,
    label: str,
    *,
    index: int = 0,
    partition: str = AUXILIARY_TRAIN,
    group: str | None = None,
) -> LabeledTokenExample:
    token = tokenize_with_spans(text)[index]
    return LabeledTokenExample(
        text,
        token.start,
        token.end,
        label,
        partition,
        group or f"{partition}:{label}:{text}:{index}",
        "controlled-token-fixture-v1",
    )


def training_examples() -> list[LabeledTokenExample]:
    return [example(word, label) for label, words in ROMAN_WORDS.items() for word in words]


def validation_examples() -> list[LabeledTokenExample]:
    return [
        example(f"check {words[0]}", label, index=1, partition=AUXILIARY_VALIDATION)
        for label, words in ROMAN_WORDS.items()
    ]


def nll(model: TokenLanguageIdentifier, examples: list[LabeledTokenExample]) -> float:
    losses = []
    for item in examples:
        prediction = next(token for token in model.predict(item.text) if token.span.start == item.start)
        losses.append(-math.log(max(prediction.probabilities[item.label], 1e-300)))
    return sum(losses) / len(losses)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def write_modified_artifact(path: Path, artifact: dict) -> None:
    artifact["checksum_sha256"] = canonical_hash(
        {key: artifact[key] for key in ("manifest", "features", "weights", "bias")}
    )
    path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


class TokenLanguageIdentifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = training_examples()
        cls.config = LanguageIdConfig(epochs=220, seed=29)
        cls.model = TokenLanguageIdentifier.fit(cls.records, cls.config)

    def test_supervised_model_distinguishes_all_four_labels_in_roman_script(self) -> None:
        for label, words in ROMAN_WORDS.items():
            for word in words:
                with self.subTest(label=label, word=word):
                    prediction = self.model.predict(word)[0]
                    self.assertEqual(prediction.top_label, label)
                    self.assertEqual(prediction.decision_label, label)
                    self.assertFalse(prediction.uncertain)
        self.assertEqual(self.model.manifest["training_partition"], AUXILIARY_TRAIN)
        self.assertEqual(self.model.manifest["training_example_count"], len(self.records))

    def test_probabilities_are_finite_normalized_and_offsets_reconstruct_nfc_text(self) -> None:
        raw = "cafe\u0301 bhukamp কেন 2026"
        processed = normalize_processing_text(raw)
        predictions = self.model.predict(raw)
        self.assertEqual(len(predictions), len(tokenize_with_spans(raw)))
        for prediction in predictions:
            self.assertEqual(set(prediction.probabilities), set(LABELS))
            self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in prediction.probabilities.values()))
            self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0)
            self.assertTrue(math.isfinite(prediction.entropy))
            self.assertGreaterEqual(prediction.entropy, 0)
            self.assertLessEqual(prediction.entropy, math.log(len(LABELS)) + 1e-12)
            self.assertEqual(processed[prediction.span.start:prediction.span.end], prediction.span.text)
        self.assertEqual(self.model.predict("  !!! "), ())

    def test_unknown_tokens_abstain_and_retain_the_probability_distribution(self) -> None:
        prediction = self.model.predict("jjjqqq")[0]
        self.assertTrue(prediction.uncertain)
        self.assertEqual(prediction.decision_label, "other")
        self.assertEqual(set(prediction.probabilities), set(LABELS))
        self.assertLess(max(prediction.probabilities.values()), self.config.min_confidence)

    def test_high_confidence_requirement_causes_explicit_abstention(self) -> None:
        strict = TokenLanguageIdentifier.fit(
            self.records,
            replace(self.config, epochs=10, min_confidence=1.0, min_margin=1.0),
        )
        prediction = strict.predict("rainfall")[0]
        self.assertTrue(prediction.uncertain)
        self.assertEqual(prediction.decision_label, "other")
        self.assertIn(prediction.top_label, LABELS)

    def test_context_can_disambiguate_identical_roman_token(self) -> None:
        examples = [
            example("kal meeting", "en"),
            example("kal conference", "en"),
            example("kal hoga", "hi"),
            example("kal aayega", "hi"),
            example("kal chilo", "bn"),
            example("kal hoyechilo", "bn"),
        ]
        model = TokenLanguageIdentifier.fit(examples, LanguageIdConfig(epochs=900, seed=5))
        predictions = [model.predict(text)[0] for text in ("kal meeting", "kal hoga", "kal chilo")]
        self.assertEqual([item.span.text for item in predictions], ["kal"] * 3)
        self.assertEqual([item.top_label for item in predictions], ["en", "hi", "bn"])
        self.assertNotEqual(predictions[0].probabilities, predictions[1].probabilities)

    def test_training_and_saved_model_are_independent_of_example_order(self) -> None:
        reverse = TokenLanguageIdentifier.fit(list(reversed(self.records)), self.config)
        self.assertEqual(self.model.manifest, reverse.manifest)
        self.assertEqual(self.model.predict("rainfall bhukamp brishti"), reverse.predict("rainfall bhukamp brishti"))
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first.json", Path(temporary) / "second.json"
            self.model.save_json(first)
            reverse.save_json(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            restored = TokenLanguageIdentifier.load_json(first)
            self.assertEqual(restored.manifest, self.model.manifest)
            self.assertEqual(restored.predict("rainfall bhukamp brishti"), self.model.predict("rainfall bhukamp brishti"))
            with self.assertRaises(LanguageIdError):
                self.model.save_json(first)

    def test_manifest_is_a_copy_and_cannot_mutate_predictions_or_provenance(self) -> None:
        manifest = self.model.manifest
        manifest["training_groups"].append("foreign-group")
        manifest["config"]["min_confidence"] = 0.0
        self.assertNotIn("foreign-group", self.model.manifest["training_groups"])
        self.assertEqual(self.model.manifest["config"]["min_confidence"], self.config.min_confidence)

    def test_training_rejects_nontraining_partitions_and_invalid_token_labels(self) -> None:
        base = self.records[0]
        for invalid in (
            replace(base, partition=AUXILIARY_VALIDATION),
            replace(base, partition="final_evaluation"),
            replace(base, label="latin"),
            replace(base, start=base.start + 1),
            replace(base, end=999),
            replace(base, weight=float("nan")),
            replace(base, weight=0),
            replace(base, group_id=""),
            replace(base, text="cafe\u0301", start=0, end=5),
        ):
            with self.subTest(example=invalid), self.assertRaises(LanguageIdError):
                TokenLanguageIdentifier.fit([invalid])
        with self.assertRaises(LanguageIdError):
            TokenLanguageIdentifier.fit([base, replace(base, label="hi")])

    def test_script_baseline_keeps_roman_tokens_uncertain(self) -> None:
        predictions = ScriptEvidenceBaseline().predict("kal bhukamp rainfall हिंदी বাংলা")
        for prediction in predictions[:3]:
            self.assertEqual(prediction.decision_label, "other")
            self.assertTrue(prediction.uncertain)
            self.assertAlmostEqual(prediction.probabilities["en"], prediction.probabilities["hi"])
            self.assertAlmostEqual(prediction.probabilities["en"], prediction.probabilities["bn"])
        self.assertEqual([item.top_label for item in predictions[3:]], ["hi", "bn"])
        self.assertTrue(all(item.uncertain for item in predictions))
        self.assertTrue(all(item.decision_label == "other" for item in predictions))


class LanguageIdCalibrationTests(unittest.TestCase):
    def test_large_finite_weights_keep_losses_and_calibration_artifact_finite(self) -> None:
        training = replace(example("hello", "en"), weight=1e308)
        model = TokenLanguageIdentifier.fit([training], LanguageIdConfig(epochs=30))
        self.assertTrue(all(math.isfinite(value) for value in model.manifest["loss_history"]))
        validation = replace(
            example("hello context", "hi", partition=AUXILIARY_VALIDATION),
            weight=1e308,
        )
        before = nll(model, [validation])
        report = model.calibrate([validation])
        self.assertTrue(math.isfinite(report["nll_before"]))
        self.assertTrue(math.isfinite(report["nll_after"]))
        self.assertAlmostEqual(report["nll_before"], before)
        self.assertLessEqual(report["nll_after"], before)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large-weights.json"
            model.save_json(path)
            restored = TokenLanguageIdentifier.load_json(path)
            self.assertEqual(restored.manifest, model.manifest)
            self.assertEqual(restored.predict(validation.text), model.predict(validation.text))

    def test_heldout_calibration_never_increases_external_nll_or_fits_features(self) -> None:
        model = TokenLanguageIdentifier.fit(training_examples(), LanguageIdConfig(epochs=120))
        validation = validation_examples()
        features = model.feature_names
        with tempfile.TemporaryDirectory() as temporary:
            before_path = Path(temporary) / "before.json"
            model.save_json(before_path)
            parameters_before = json.loads(before_path.read_text(encoding="utf-8"))
        before = nll(model, validation)
        report = model.calibrate(validation)
        after = nll(model, validation)
        self.assertEqual(report["partition"], AUXILIARY_VALIDATION)
        self.assertAlmostEqual(report["nll_before"], before)
        self.assertAlmostEqual(report["nll_after"], after)
        self.assertLessEqual(after, before + 1e-12)
        self.assertEqual(model.feature_names, features)
        self.assertEqual(report, model.calibrate(list(reversed(validation))))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibrated.json"
            model.save_json(path)
            parameters_after = json.loads(path.read_text(encoding="utf-8"))
            for field in ("features", "weights", "bias"):
                self.assertEqual(parameters_after[field], parameters_before[field])
            restored = TokenLanguageIdentifier.load_json(path)
            self.assertEqual(restored.manifest, model.manifest)
            self.assertEqual(restored.predict("check bhukamp"), model.predict("check bhukamp"))
        with self.assertRaises(LanguageIdError):
            model.calibrate([
                example("separate rainfall", "en", index=1, partition=AUXILIARY_VALIDATION),
            ])

    def test_calibration_rejects_partitions_group_overlap_and_exact_span_overlap(self) -> None:
        train = training_examples()
        model = TokenLanguageIdentifier.fit(train, LanguageIdConfig(epochs=10))
        valid = validation_examples()[0]
        for invalid in (
            replace(valid, partition=AUXILIARY_TRAIN),
            replace(valid, partition="final_evaluation"),
            replace(valid, group_id=train[0].group_id),
            replace(train[0], partition=AUXILIARY_VALIDATION, group_id="independent-but-duplicate"),
        ):
            with self.subTest(example=invalid), self.assertRaises(LanguageIdError):
                model.calibrate([invalid])
        self.assertEqual(model.temperature, 1.0)
        self.assertIsNone(model.manifest["calibration"])


class LanguageIdArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = TokenLanguageIdentifier.fit(training_examples(), LanguageIdConfig(epochs=20))

    def test_loader_rejects_checksum_shape_label_and_nonfinite_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            self.model.save_json(path)
            original = json.loads(path.read_text(encoding="utf-8"))
            checksum_bad = json.loads(json.dumps(original))
            checksum_bad["weights"][0][0] += 1.0
            path.write_text(json.dumps(checksum_bad), encoding="utf-8")
            with self.assertRaises(LanguageIdError):
                TokenLanguageIdentifier.load_json(path)
            for kind in ("shape", "labels", "nonfinite", "duplicate_features", "bad_temperature"):
                artifact = json.loads(json.dumps(original))
                if kind == "shape":
                    artifact["weights"][0].pop()
                elif kind == "labels":
                    artifact["manifest"]["labels"] = ["en", "hi", "bn", "invalid"]
                elif kind == "nonfinite":
                    artifact["weights"][0][0] = "NaN"
                elif kind == "duplicate_features":
                    artifact["features"][1] = artifact["features"][0]
                else:
                    artifact["manifest"]["temperature"] = 0
                write_modified_artifact(path, artifact)
                with self.subTest(kind=kind), self.assertRaises(LanguageIdError):
                    TokenLanguageIdentifier.load_json(path)
            path.write_text('{"weights": [NaN]}', encoding="utf-8")
            with self.assertRaises(LanguageIdError):
                TokenLanguageIdentifier.load_json(path)


class LanguageIdInferenceMetadataTests(unittest.TestCase):
    def model(self):
        return TokenLanguageIdentifier.fit(training_examples(), LanguageIdConfig(epochs=8))

    def test_compact_identity_matches_the_unchanged_full_saved_payload_and_reload(self):
        model = self.model()
        before = model.manifest
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save_json(path)
            original_bytes = path.read_bytes()
            artifact = json.loads(original_bytes)
            self.assertEqual(model.inference_metadata["model_sha256"], artifact["checksum_sha256"])
            self.assertEqual(model.inference_metadata["full_manifest_sha256"], canonical_hash(artifact["manifest"]))
            restored = TokenLanguageIdentifier.load_json(path)
            self.assertEqual(restored.inference_metadata, model.inference_metadata)
            self.assertEqual(restored.manifest, before)
            other = Path(temporary) / "restored.json"
            restored.save_json(other)
            self.assertEqual(other.read_bytes(), original_bytes)

    def test_actual_weights_bias_and_features_are_bound_even_when_full_manifest_is_identical(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save_json(path)
            original = json.loads(path.read_text())
            for change in ("weights", "bias", "features"):
                artifact = json.loads(json.dumps(original))
                if change == "weights":
                    artifact["weights"][0][0] += 0.25
                elif change == "bias":
                    artifact["bias"][0] += 0.25
                else:
                    artifact["features"][-1] += "\uffff"
                write_modified_artifact(path, artifact)
                changed = TokenLanguageIdentifier.load_json(path)
                self.assertEqual(changed.manifest, model.manifest)
                self.assertNotEqual(changed.inference_metadata["model_sha256"], model.inference_metadata["model_sha256"], change)

    def test_calibration_reseals_identity_and_repeated_calibration_is_stable(self):
        model = self.model()
        before = model.inference_metadata
        model.calibrate(validation_examples())
        after = model.inference_metadata
        self.assertNotEqual(before["model_sha256"], after["model_sha256"])
        self.assertEqual(after["calibration"]["method"], "heldout_temperature_grid")
        self.assertNotIn("validation_groups", after["calibration"])
        model.calibrate(list(reversed(validation_examples())))
        self.assertEqual(model.inference_metadata, after)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibrated.json"
            model.save_json(path)
            self.assertEqual(TokenLanguageIdentifier.load_json(path).inference_metadata, after)

    def test_fitted_state_is_immutable_and_returned_summaries_are_detached(self):
        model = self.model()
        before = model.inference_metadata
        with self.assertRaises(TypeError):
            model._weights[0][0] = 10.0
        with self.assertRaises(TypeError):
            model._bias[0] = 10.0
        with self.assertRaises(TypeError):
            model._metadata["training_groups"][0] = "changed"
        with self.assertRaises(TypeError):
            model._feature_index[model.feature_names[0]] = 1
        for name in ("_weights", "_bias", "_metadata", "feature_names", "config", "_temperature"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(model, name, getattr(model, name))
        with self.assertRaises(AttributeError):
            del model.feature_names
        summary = model.inference_metadata
        summary["config"]["min_confidence"] = 0.0
        summary["labels"].append("invented")
        self.assertEqual(model.inference_metadata, before)

    def test_summary_stays_bounded_with_large_provenance_and_does_not_access_manifest_property(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.json"
            model.save_json(path)
            artifact = json.loads(path.read_text())
            artifact["manifest"]["training_sources"] = [f"source-{number:05d}" for number in range(10000)]
            write_modified_artifact(path, artifact)
            large = TokenLanguageIdentifier.load_json(path)
            with patch.object(TokenLanguageIdentifier, "manifest", new_callable=PropertyMock, side_effect=AssertionError("full manifest must not enter inference metadata")):
                summary = large.inference_metadata
            self.assertLess(len(json.dumps(summary).encode()), 4096)
            for field in ("training_groups", "training_sources", "training_span_fingerprints", "loss_history"):
                self.assertNotIn(field, summary)


class LanguageIdCliTests(unittest.TestCase):
    def test_train_predict_and_partitioned_batch_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = root / "examples.jsonl"
            examples.write_text(
                "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in training_examples()),
                encoding="utf-8",
            )
            model_path = root / "model.json"
            command = [sys.executable, "-m", "nlp.language_id"]
            trained = subprocess.run(
                [*command, "train", "--examples", str(examples), "--output", str(model_path), "--epochs", "150", "--seed", "19"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertTrue(model_path.is_file())
            predicted = subprocess.run(
                [*command, "predict", "--model", str(model_path), "--text", "rainfall bhukamp brishti"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(predicted.returncode, 0, predicted.stderr)
            tokens = json.loads(predicted.stdout)["tokens"]
            self.assertEqual([item["top_label"] for item in tokens], ["en", "hi", "bn"])
            queries = root / "queries.jsonl"
            queries.write_text(
                json.dumps({"query_id": "train", "text": "rainfall", "split": "encoder_train"}) + "\n"
                + json.dumps({"query_id": "final", "text": "brishti", "split": "final_evaluation"}) + "\n",
                encoding="utf-8",
            )
            output = root / "predictions.jsonl"
            denied = subprocess.run(
                [*command, "predict", "--model", str(model_path), "--queries", str(queries), "--output", str(output)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("--split", denied.stderr)
            completed = subprocess.run(
                [*command, "predict", "--model", str(model_path), "--queries", str(queries), "--split", "encoder_train", "--output", str(output)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            batch = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["query_id"] for item in batch], ["train"])
            self.assertEqual(batch[0]["split"], "encoder_train")


if __name__ == "__main__":
    unittest.main()
