from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluate_lid import (
    AUXILIARY_TEST, LidEvaluationError, _hash, evaluate_lid,
    evaluate_lid_files, write_lid_report,
)
from nlp.language_id import (
    AUXILIARY_TRAIN, AUXILIARY_VALIDATION, LABELS,
    LabeledTokenExample, LanguageIdConfig, ScriptEvidenceBaseline,
    TokenLanguageIdentifier, TokenPrediction, _identity,
)
from nlp.text import tokenize_with_spans


def example(text: str, label: str, index: int = 0, **overrides: object) -> LabeledTokenExample:
    token = tokenize_with_spans(text)[index]
    values = {"text": text, "start": token.start, "end": token.end,
              "label": label, "partition": AUXILIARY_TEST,
              "group_id": "test:" + text, "source": "controlled-human-label-fixture"}
    values.update(overrides)
    return LabeledTokenExample(**values)


class FrozenFixture(TokenLanguageIdentifier):
    """Controlled predictions without model fitting, calibration, or saving."""

    def __init__(self, choices: dict[str, tuple[str, bool]] | None = None):
        self.choices = choices or {}
        self.calls: Counter[str] = Counter()
        self.provenance = {
            "labels": list(LABELS), "training_partition": AUXILIARY_TRAIN,
            "training_groups": ["fixture-training-only"],
            "training_span_fingerprints": [_hash(["training-only", 0, 8])],
            "calibration": None, "temperature": 1.0,
        }

    @property
    def manifest(self):
        return copy.deepcopy(self.provenance)

    def predict(self, text):
        self.calls[text] += 1
        result = []
        for span in tokenize_with_spans(text):
            label, uncertain = self.choices.get(span.text, ("en", False))
            probabilities = {value: 0.85 if value == label else 0.05 for value in LABELS}
            entropy = -sum(value * math.log(value) for value in probabilities.values())
            result.append(TokenPrediction(span, probabilities, label,
                                          "other" if uncertain else label, uncertain, entropy))
        return tuple(result)

    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation must never fit")

    def calibrate(self, *args, **kwargs):
        raise AssertionError("Evaluation must never calibrate")

    def save_json(self, *args, **kwargs):
        raise AssertionError("Evaluation must never save a model")


def write_manual_model(path: Path) -> None:
    """Write a structurally valid fixed logistic model; no optimization occurs."""
    config = LanguageIdConfig(epochs=1, max_features=1)
    manifest = {
        "schema_version": 1, "model_type": "multinomial_logistic_regression",
        "labels": list(LABELS), "feature_count": 1,
        "feature_policy": "char_2_5_neighbor_script_shape_v1", "config": asdict(config),
        "temperature": 1.0, "training_partition": AUXILIARY_TRAIN,
        "training_example_count": 1, "training_groups": ["training-group"],
        "training_span_fingerprints": [_hash(["trainingword", 0, 12])],
        "training_data_fingerprint": _hash("training-fixture"),
        "training_sources": ["fixed-no-fit-model-fixture"],
        "loss_history": [1.0], "calibration": None,
    }
    payload = {"manifest": manifest, "features": ["word:hello"],
               "weights": [[0.0] for _ in LABELS], "bias": [4.0, 0.0, 0.0, 0.0]}
    path.write_text(json.dumps({**payload, "checksum_sha256": _hash(payload)}), encoding="utf-8")


class EvaluateLidTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = FrozenFixture({"hello": ("en", False), "नमस्ते": ("en", True),
                                    "123": ("other", False), "বাংলা": ("bn", False)})
        self.examples = [example("hello नमस्ते 123 বাংলা", label, index)
                         for index, label in enumerate(("en", "hi", "other", "bn"))]

    def test_exact_accuracy_confusion_f1_and_selective_metrics(self):
        report = evaluate_lid(self.examples, self.model)
        self.assertEqual(report["partition"], "auxiliary_test")
        self.assertEqual(report["evaluation_kind"], "auxiliary_token_language_identification")
        self.assertEqual(report["metrics"]["top_label_accuracy"], 0.75)
        self.assertEqual(report["confusion_matrix"]["hi"]["en"], 1)
        self.assertEqual(report["confusion_matrix"]["en"]["en"], 1)
        self.assertAlmostEqual(report["per_label"]["en"]["precision"], 0.5)
        self.assertAlmostEqual(report["per_label"]["en"]["f1"], 2 / 3)
        self.assertAlmostEqual(report["metrics"]["macro_f1"], (2 / 3 + 0 + 1 + 1) / 4)
        self.assertEqual(report["metrics"]["coverage"], 0.75)
        self.assertEqual(report["metrics"]["uncertainty_rate"], 0.25)
        self.assertEqual(report["metrics"]["decision_accuracy_nonabstained"], 1.0)
        self.assertEqual(report["per_label"]["hi"]["support"], 1)
        self.assertEqual(report["per_label"]["hi"]["nonabstained_support"], 0)
        self.assertIsNone(report["per_label"]["hi"]["decision_accuracy_nonabstained"])
        self.assertEqual(report["distinct_sentences"], 1)
        self.assertEqual(self.model.calls[self.examples[0].text], 1)

    def test_confident_other_is_covered_not_automatically_abstention(self):
        report = evaluate_lid([example("123", "other")], self.model)
        self.assertEqual(report["metrics"]["coverage"], 1.0)
        self.assertEqual(report["metrics"]["decision_accuracy_nonabstained"], 1.0)

    def test_all_abstentions_have_null_decision_accuracy_and_finite_json(self):
        model = FrozenFixture({"hello": ("en", True)})
        report = evaluate_lid([example("hello", "en")], model)
        self.assertEqual(report["metrics"]["top_label_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["coverage"], 0.0)
        self.assertIsNone(report["metrics"]["decision_accuracy_nonabstained"])
        self.assertNotIn("NaN", json.dumps(report, allow_nan=False))

    def test_missing_bengali_support_is_explicit_and_macro_denominators_documented(self):
        report = evaluate_lid([example("hello", "en")], self.model)
        self.assertEqual(report["metrics"]["macro_f1"], 0.25)
        self.assertEqual(report["metrics"]["macro_f1_supported_labels"], 1.0)
        self.assertEqual(report["metrics"]["supported_labels"], ["en"])
        self.assertEqual(report["per_label"]["bn"]["support"], 0)
        self.assertIsNone(report["per_label"]["bn"]["top_label_accuracy"])
        self.assertTrue(any("no Bengali gold" in message for message in report["limitations"]))

    def test_no_model_mutation_or_repeated_prediction_and_order_invariance(self):
        before = self.model.manifest
        first = evaluate_lid(self.examples, self.model)
        self.assertEqual(self.model.manifest, before)
        second_model = FrozenFixture(self.model.choices)
        second = evaluate_lid(reversed(self.examples), second_model)
        self.assertEqual(first, second)
        self.assertEqual(sum(second_model.calls.values()), 1)

    def test_auxiliary_test_is_required_before_shared_validation_adapter(self):
        for partition in (AUXILIARY_TRAIN, AUXILIARY_VALIDATION, "final_evaluation", "test"):
            with self.subTest(partition=partition), self.assertRaisesRegex(LidEvaluationError, "original auxiliary_test"):
                evaluate_lid([replace(self.examples[0], partition=partition)], self.model)
        self.assertFalse(self.model.calls)

    def test_empty_nonexample_and_duplicate_records_rejected(self):
        for records in ([], [asdict(self.examples[0])], [self.examples[0], self.examples[0]]):
            with self.subTest(records=records), self.assertRaises(LidEvaluationError):
                evaluate_lid(records, self.model)

    def test_nfc_complete_spans_labels_sources_groups_and_weights_validated(self):
        invalid = (replace(self.examples[0], text="cafe\u0301"),
                   replace(self.examples[0], start=1),
                   replace(self.examples[0], end=3),
                   replace(self.examples[0], label="unknown"),
                   replace(self.examples[0], group_id=" "),
                   replace(self.examples[0], source=""),
                   replace(self.examples[0], weight=float("nan")),
                   replace(self.examples[0], weight=0))
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(LidEvaluationError):
                evaluate_lid([item], self.model)

    def test_training_group_and_exact_span_overlap_rejected_before_predict(self):
        for key, value in (("training_groups", [self.examples[0].group_id]),
                           ("training_span_fingerprints", [_identity(self.examples[0])])):
            model = FrozenFixture()
            model.provenance[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(LidEvaluationError, "overlaps model training"):
                evaluate_lid([self.examples[0]], model)
            self.assertFalse(model.calls)

    def test_calibration_group_and_span_overlap_rejected(self):
        for key, value in (("validation_groups", [self.examples[0].group_id]),
                           ("validation_span_fingerprints", [_identity(self.examples[0])])):
            model = FrozenFixture()
            calibration = {"partition": AUXILIARY_VALIDATION, "validation_groups": ["validation-only"],
                           "validation_span_fingerprints": [_hash(["validation-only", 0, 10])],
                           "method": "heldout_temperature_grid"}
            calibration[key] = value
            model.provenance["calibration"] = calibration
            with self.subTest(key=key), self.assertRaisesRegex(LidEvaluationError, "overlaps model calibration"):
                evaluate_lid([self.examples[0]], model)
            self.assertFalse(model.calls)

    def test_calibration_scope_reported_without_calibrating(self):
        self.model.provenance["calibration"] = {
            "partition": AUXILIARY_VALIDATION, "validation_groups": ["validation-only"],
            "validation_span_fingerprints": [_hash(["validation-only", 0, 10])],
            "method": "heldout_temperature_grid"}
        report = evaluate_lid(self.examples, self.model)
        self.assertTrue(report["calibration"]["applied"])
        self.assertFalse(report["calibration"]["changed_during_evaluation"])
        self.assertEqual(report["calibration"]["partition"], AUXILIARY_VALIDATION)

    def test_malformed_model_provenance_and_baseline_rejected(self):
        for key, value in (("training_partition", "final_evaluation"), ("labels", ["en"]),
                           ("training_groups", []), ("training_groups", ["x", "x"]),
                           ("training_span_fingerprints", ["bad"]),
                           ("calibration", {"partition": AUXILIARY_TEST})):
            model = FrozenFixture()
            model.provenance[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(LidEvaluationError):
                evaluate_lid(self.examples, model)
        with self.assertRaisesRegex(LidEvaluationError, "supervised"):
            evaluate_lid(self.examples, ScriptEvidenceBaseline())

    def test_malformed_probabilities_decisions_and_span_predictions_rejected(self):
        base = self.model.predict(self.examples[0].text)
        mutations = (
            replace(base[0], probabilities={"en": float("nan"), "hi": 0.05, "bn": 0.05, "other": 0.05}),
            replace(base[0], probabilities={"en": 0.2, "hi": 0.2, "bn": 0.2, "other": 0.2}),
            replace(base[0], probabilities={"en": 1.0}),
            replace(base[0], top_label="hi"),
            replace(base[0], uncertain=1),
            replace(base[0], decision_label="other"),
            replace(base[0], entropy=float("inf")),
            replace(base[0], entropy=4.0),
            replace(base[0], span=replace(base[0].span, start=1)),
        )
        for prediction in mutations:
            with self.subTest(prediction=prediction), patch.object(self.model, "predict", return_value=(prediction, *base[1:])):
                with self.assertRaises(LidEvaluationError):
                    evaluate_lid(self.examples, self.model)
        for predictions in (base[:-1], (*base, base[0]), [None]):
            with self.subTest(predictions=predictions), patch.object(self.model, "predict", return_value=predictions):
                with self.assertRaises(LidEvaluationError):
                    evaluate_lid(self.examples, self.model)

    def test_source_file_hashes_and_model_bytes_unchanged_without_fitting(self):
        model_path = self.root / "model.json"
        write_manual_model(model_path)
        examples_path = self.root / "test.jsonl"
        examples_path.write_text(json.dumps(asdict(example("hello", "en"))) + "\n", encoding="utf-8")
        before = model_path.read_bytes()
        with patch.object(TokenLanguageIdentifier, "fit", side_effect=AssertionError("fit forbidden")), \
                patch.object(TokenLanguageIdentifier, "calibrate", side_effect=AssertionError("calibration forbidden")), \
                patch.object(TokenLanguageIdentifier, "save_json", side_effect=AssertionError("save forbidden")):
            report = evaluate_lid_files(model_path, examples_path)
        self.assertEqual(report["source_files"]["model"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(report["source_files"]["examples"]["sha256"], hashlib.sha256(examples_path.read_bytes()).hexdigest())
        self.assertEqual(model_path.read_bytes(), before)
        self.assertEqual(report["metrics"]["top_label_accuracy"], 1.0)

    def test_changed_inputs_or_invalid_jsonl_structure_rejected(self):
        model_path = self.root / "model.json"
        write_manual_model(model_path)
        examples_path = self.root / "test.jsonl"
        examples_path.write_text(json.dumps(asdict(example("hello", "en"))) + "\n", encoding="utf-8")
        from evaluation import evaluate_lid as module
        original_hash = module._file_sha256
        calls = 0

        def changed_hash(path):
            nonlocal calls
            calls += 1
            return "0" * 64 if calls == 3 else original_hash(path)

        with patch.object(module, "_file_sha256", side_effect=changed_hash):
            with self.assertRaisesRegex(LidEvaluationError, "changed during"):
                evaluate_lid_files(model_path, examples_path)
        examples_path.write_text(json.dumps({**asdict(example("hello", "en")), "extra": "bad"}) + "\n")
        with self.assertRaisesRegex(LidEvaluationError, "example fields"):
            evaluate_lid_files(model_path, examples_path)

    def test_example_and_artifact_safety_limits(self):
        from evaluation import evaluate_lid as module
        with patch.object(module, "MAX_EXAMPLES", 1):
            with self.assertRaisesRegex(LidEvaluationError, "safety limit"):
                evaluate_lid(self.examples, self.model)
        model_path = self.root / "model.json"
        write_manual_model(model_path)
        examples_path = self.root / "test.jsonl"
        examples_path.write_text(json.dumps(asdict(example("hello", "en"))) + "\n")
        with patch.object(module, "MAX_ARTIFACT_BYTES", 1):
            with self.assertRaisesRegex(LidEvaluationError, "bounded regular"):
                evaluate_lid_files(model_path, examples_path)

    def test_report_is_sealed_finite_and_published_without_overwrite(self):
        report = evaluate_lid(self.examples, self.model)
        output = self.root / "nested" / "report.json"
        self.assertEqual(write_lid_report(report, output), output)
        saved = json.loads(output.read_text())
        self.assertEqual(saved, report)
        self.assertEqual(saved["report_sha256"], _hash({key: value for key, value in saved.items()
                                                     if key != "report_sha256"}))
        with self.assertRaisesRegex(LidEvaluationError, "already exists"):
            write_lid_report(report, output)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertFalse(list(output.parent.glob(".lid-report-*")))

    def test_writer_rejects_corrupt_nonfinite_or_wrong_scope_report(self):
        report = evaluate_lid(self.examples, self.model)
        for changes in ({"partition": "final_evaluation"}, {"examples": 999},
                        {"bad": float("nan")}, {"bad": object()}):
            invalid = {**report, **changes}
            output = self.root / "report.json"
            with self.subTest(changes=changes), self.assertRaises(LidEvaluationError):
                write_lid_report(invalid, output)
            self.assertFalse(output.exists())

    def test_writer_symlink_and_atomic_publication_failure_preserve_user_files(self):
        report = evaluate_lid(self.examples, self.model)
        real = self.root / "real"
        real.mkdir()
        symlink = self.root / "link"
        symlink.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(LidEvaluationError, "Symlinks"):
            write_lid_report(report, symlink / "report.json")
        output = self.root / "report.json"

        def concurrent_file(*args, **kwargs):
            output.write_text("user content")
            raise FileExistsError("appeared")

        with patch("evaluation.evaluate_lid.os.link", side_effect=concurrent_file):
            with self.assertRaises(FileExistsError):
                write_lid_report(report, output)
        self.assertEqual(output.read_text(), "user content")
        self.assertFalse(list(self.root.glob(".lid-report-*")))

    def test_cli_uses_only_auxiliary_test_and_never_overwrites_report(self):
        model_path = self.root / "model.json"
        write_manual_model(model_path)
        examples_path = self.root / "test.jsonl"
        examples_path.write_text(json.dumps(asdict(example("hello", "en"))) + "\n", encoding="utf-8")
        output = self.root / "cli-report.json"
        command = [sys.executable, "-m", "evaluation.evaluate_lid", "--model", str(model_path),
                   "--examples", str(examples_path), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["partition"], AUXILIARY_TEST)
        before = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), before)
        examples_path.write_text(json.dumps(asdict(example("hello", "en", partition=AUXILIARY_TRAIN))) + "\n")
        command[-1] = str(self.root / "forbidden.json")
        third = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(third.returncode, 2)
        self.assertFalse((self.root / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
