"""Evaluate frozen token LID on auxiliary_test without fitting or calibration.

This is an auxiliary language-identification check, NOT the locked retrieval
evaluation. Metrics count tokens equally. Abstention is the explicit uncertain
flag, not the ``other`` class: a confident other prediction remains covered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from data.prepare_corpus import iter_jsonl
from nlp.language_id import (
    AUXILIARY_TRAIN, AUXILIARY_VALIDATION, LABELS, LabeledTokenExample,
    LanguageIdError, TokenLanguageIdentifier, _identity, _validated_examples,
)
from nlp.text import tokenize_with_spans


AUXILIARY_TEST = "auxiliary_test"
SCHEMA_VERSION = 1
MAX_EXAMPLES = 100000
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


class LidEvaluationError(ValueError):
    """Invalid auxiliary-test labels, model provenance, or unsafe output."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seal(report: dict[str, Any]) -> dict[str, Any]:
    report["report_sha256"] = _hash({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def _test_examples(examples: Iterable[LabeledTokenExample]) -> tuple[LabeledTokenExample, ...]:
    records: list[LabeledTokenExample] = []
    for item in examples:
        if not isinstance(item, LabeledTokenExample) or item.partition != AUXILIARY_TEST:
            raise LidEvaluationError("Evaluation requires original auxiliary_test examples only")
        if len(records) >= MAX_EXAMPLES:
            raise LidEvaluationError("Auxiliary test exceeds the example safety limit")
        records.append(item)
    if not records:
        raise LidEvaluationError("Auxiliary test must not be empty")
    # Reuse span/NFC/label/metadata validation only AFTER checking the original
    # partition. This adapter never passes records into fit() or calibrate().
    try:
        checked = _validated_examples((replace(item, partition=AUXILIARY_VALIDATION)
                                       for item in records), AUXILIARY_VALIDATION)
    except LanguageIdError as exc:
        raise LidEvaluationError(str(exc)) from exc
    result = tuple(replace(item, partition=AUXILIARY_TEST) for item in checked)
    if len({_identity(item) for item in result}) != len(result):
        raise LidEvaluationError("Duplicate evaluation text/span identities are forbidden")
    return result


def _provenance_list(manifest: Mapping[str, Any], key: str, *, fingerprints: bool = False) -> set[str]:
    values = manifest.get(key)
    if (not isinstance(values, list) or not values
            or any(not isinstance(item, str) or not item.strip() for item in values)
            or values != sorted(set(values))):
        raise LidEvaluationError(f"Missing or invalid model {key} provenance")
    if fingerprints and any(not re.fullmatch("[0-9a-f]{64}", item) for item in values):
        raise LidEvaluationError(f"Invalid model {key} fingerprints")
    return set(values)


def _check_isolation(model: TokenLanguageIdentifier, records: Sequence[LabeledTokenExample]) -> Mapping[str, Any]:
    if not isinstance(model, TokenLanguageIdentifier):
        raise LidEvaluationError("Expected a fitted supervised TokenLanguageIdentifier")
    manifest = model.manifest
    if manifest.get("training_partition") != AUXILIARY_TRAIN or manifest.get("labels") != list(LABELS):
        raise LidEvaluationError("Model label schema or auxiliary training partition is invalid")
    training_groups = _provenance_list(manifest, "training_groups")
    training_spans = _provenance_list(manifest, "training_span_fingerprints", fingerprints=True)
    calibration = manifest.get("calibration")
    calibration_groups: set[str] = set()
    calibration_spans: set[str] = set()
    if calibration is not None:
        if not isinstance(calibration, Mapping) or calibration.get("partition") != AUXILIARY_VALIDATION:
            raise LidEvaluationError("Calibration provenance must declare auxiliary_validation")
        calibration_groups = _provenance_list(calibration, "validation_groups")
        calibration_spans = _provenance_list(calibration, "validation_span_fingerprints", fingerprints=True)
    groups = {item.group_id for item in records}
    spans = {_identity(item) for item in records}
    for kind, fit_groups, fit_spans in (("training", training_groups, training_spans),
                                       ("calibration", calibration_groups, calibration_spans)):
        if groups.intersection(fit_groups):
            raise LidEvaluationError(f"Held-out test group overlaps model {kind}")
        if spans.intersection(fit_spans):
            raise LidEvaluationError(f"Held-out test span overlaps model {kind}")
    return manifest


def _predictions(model: TokenLanguageIdentifier, text: str) -> Mapping[tuple[int, int], Any]:
    expected = {(span.start, span.end): span for span in tokenize_with_spans(text)}
    predictions: dict[tuple[int, int], Any] = {}
    try:
        for prediction in model.predict(text):
            identity = (prediction.span.start, prediction.span.end)
            if identity not in expected or prediction.span != expected[identity] or identity in predictions:
                raise LidEvaluationError("Model prediction spans do not match complete shared tokens")
            probabilities = prediction.probabilities
            if (set(probabilities) != set(LABELS)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1
                           for value in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8)):
                raise LidEvaluationError("Model probabilities must be finite, normalized, and use the four labels")
            top = max(LABELS, key=lambda label: probabilities[label])
            if (prediction.top_label != top or type(prediction.uncertain) is not bool
                    or prediction.decision_label != ("other" if prediction.uncertain else top)
                    or isinstance(prediction.entropy, bool) or not isinstance(prediction.entropy, (int, float))
                    or not math.isfinite(prediction.entropy)
                    or not 0 <= prediction.entropy <= math.log(len(LABELS)) + 1e-8):
                raise LidEvaluationError("Inconsistent top-label, decision, uncertainty, or entropy prediction")
            predictions[identity] = prediction
    except (TypeError, AttributeError, KeyError) as exc:
        raise LidEvaluationError("Malformed token prediction") from exc
    if set(predictions) != set(expected):
        raise LidEvaluationError("Model omitted shared-token predictions")
    return predictions


def evaluate_lid(examples: Iterable[LabeledTokenExample], model: TokenLanguageIdentifier) -> dict[str, Any]:
    """Score a frozen model; never fit, calibrate, tune, or save it."""
    records = _test_examples(examples)
    manifest = _check_isolation(model, records)
    by_text: dict[str, Mapping[tuple[int, int], Any]] = {}
    confusion = {gold: {predicted: 0 for predicted in LABELS} for gold in LABELS}
    support: Counter[str] = Counter()
    covered_support: Counter[str] = Counter()
    covered_correct: Counter[str] = Counter()
    correct = covered = uncertain = 0
    entropy = 0.0
    for item in records:
        if item.text not in by_text:
            by_text[item.text] = _predictions(model, item.text)
        prediction = by_text[item.text][(item.start, item.end)]
        support[item.label] += 1
        confusion[item.label][prediction.top_label] += 1
        correct += prediction.top_label == item.label
        entropy += prediction.entropy
        if prediction.uncertain:
            uncertain += 1
        else:
            covered += 1
            covered_support[item.label] += 1
            covered_correct[item.label] += prediction.decision_label == item.label
    per_label: dict[str, Any] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        predicted_support = sum(confusion[gold][label] for gold in LABELS)
        precision = true_positive / predicted_support if predicted_support else 0.0
        recall = true_positive / support[label] if support[label] else 0.0
        per_label[label] = {
            "support": support[label], "predicted_support": predicted_support,
            "correct": true_positive,
            "top_label_accuracy": recall if support[label] else None,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "nonabstained_support": covered_support[label],
            "nonabstained_correct": covered_correct[label],
            "coverage": covered_support[label] / support[label] if support[label] else None,
            "decision_accuracy_nonabstained": covered_correct[label] / covered_support[label] if covered_support[label] else None,
        }
    supported_labels = [label for label in LABELS if support[label]]
    calibration = manifest.get("calibration")
    limitations = [
        "Auxiliary token LID evaluation, not final retrieval evaluation or relevance quality",
        "Held-out sentence labels do not establish calibration or accuracy on code-mixed retrieval queries",
        "Any temperature calibration was learned on auxiliary_validation only; this test never changes it",
        "Prepared public samples are label-stratified; metrics do not estimate natural query class priors",
    ]
    if not support["bn"]:
        limitations.append("This auxiliary test has no Bengali gold labels; Bengali contextual accuracy is not established")
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "auxiliary_token_language_identification",
        "partition": AUXILIARY_TEST, "labels": list(LABELS),
        "examples": len(records), "groups": len({item.group_id for item in records}),
        "distinct_sentences": len(by_text), "source_count": len({item.source for item in records}),
        "sources_fingerprint": _hash(sorted({item.source for item in records})),
        "examples_fingerprint": _hash([asdict(item) for item in records]),
        "model_manifest_sha256": _hash(manifest),
        "metric_policy": {"unit": "token", "example_weights_applied": False,
                          "macro_f1": "all_four_labels_zero_for_undefined_f1",
                          "supported_macro_f1": "gold_supported_labels_only",
                          "abstention": "prediction.uncertain, not the other label"},
        "metrics": {"top_label_accuracy": correct / len(records),
                    "macro_f1": sum(per_label[label]["f1"] for label in LABELS) / len(LABELS),
                    "macro_f1_supported_labels": sum(per_label[label]["f1"] for label in supported_labels) / len(supported_labels),
                    "supported_labels": supported_labels,
                    "uncertain_tokens": uncertain, "uncertainty_rate": uncertain / len(records),
                    "nonabstained_tokens": covered, "coverage": covered / len(records),
                    "decision_accuracy_nonabstained": sum(covered_correct.values()) / covered if covered else None,
                    "mean_entropy": entropy / len(records)},
        "confusion_matrix": confusion, "per_label": per_label,
        "calibration": {"applied": calibration is not None,
                        "partition": calibration.get("partition") if calibration else None,
                        "method": calibration.get("method") if calibration else None,
                        "temperature": manifest.get("temperature"),
                        "changed_during_evaluation": False},
        "leakage_checks": {"training_group_overlap": 0, "training_span_overlap": 0,
                           "calibration_group_overlap": 0, "calibration_span_overlap": 0},
        "limitations": limitations,
    }
    return _seal(report)


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise LidEvaluationError("Input must be a bounded regular artifact file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_lid_files(model_path: str | Path, examples_path: str | Path) -> dict[str, Any]:
    """Load immutable inputs, score, and attach model/test source-file hashes."""
    model_path, examples_path = Path(model_path), Path(examples_path)
    model_hash, examples_hash = _file_sha256(model_path), _file_sha256(examples_path)
    model = TokenLanguageIdentifier.load_json(model_path)

    def examples() -> Iterator[LabeledTokenExample]:
        for number, record in enumerate(iter_jsonl(examples_path), start=1):
            if number > MAX_EXAMPLES:
                raise LidEvaluationError("Auxiliary test exceeds the example safety limit")
            try:
                yield LabeledTokenExample(**record)
            except TypeError as exc:
                raise LidEvaluationError("Invalid labeled token example fields") from exc

    report = evaluate_lid(examples(), model)
    if _file_sha256(model_path) != model_hash or _file_sha256(examples_path) != examples_hash:
        raise LidEvaluationError("Input artifact changed during evaluation")
    report["source_files"] = {"model": {"path": str(model_path), "sha256": model_hash},
                              "examples": {"path": str(examples_path), "sha256": examples_hash}}
    return _seal(report)


def write_lid_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically publish a finite JSON report without replacing existing files."""
    if report.get("partition") != AUXILIARY_TEST or report.get("evaluation_kind") != "auxiliary_token_language_identification":
        raise LidEvaluationError("Expected an auxiliary_test LID report")
    try:
        if report.get("report_sha256") != _hash({key: value for key, value in report.items() if key != "report_sha256"}):
            raise LidEvaluationError("LID report checksum mismatch")
        serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError) as exc:
        raise LidEvaluationError("LID report must be finite JSON with a valid checksum") from exc
    target = Path(output).absolute()
    for path in [*reversed(target.parents), target]:
        if path.is_symlink():
            raise LidEvaluationError("Symlinks are forbidden in the report path")
    if target.exists():
        raise LidEvaluationError("Report output already exists; choose a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".lid-report-",
                                         dir=target.parent, delete=False) as handle:
            partial = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, target, follow_symlinks=False)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_lid_files(args.model, args.examples)
        path = write_lid_report(report, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Auxiliary LID evaluation error: {exc}\n")
    print(json.dumps({"output": str(path), "partition": report["partition"],
                      "metrics": report["metrics"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
