"""Offline token language identification with explicit uncertainty and isolation.

The supervised model is multinomial logistic regression over character 2–5
grams, neighboring tokens, script, and token shape.  Vocabulary fitting uses
auxiliary training examples only.  ScriptEvidenceBaseline is an explicitly
uncalibrated fallback: Latin text remains uncertain rather than implying English.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from nlp.text import TextPreparationError, TokenSpan, normalize_processing_text, tokenize_with_spans


LABELS = ("en", "hi", "bn", "other")
AUXILIARY_TRAIN = "auxiliary_train"
AUXILIARY_VALIDATION = "auxiliary_validation"
SCHEMA_VERSION = 1
TEMPERATURE_GRID = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


class LanguageIdError(ValueError):
    """Invalid language labels, training partitions, or model artifacts."""


@dataclass(frozen=True)
class LanguageIdConfig:
    epochs: int = 180
    learning_rate: float = 1.0
    l2: float = 0.0001
    max_features: int = 12000
    min_feature_count: int = 1
    seed: int = 13
    min_confidence: float = 0.60
    min_margin: float = 0.15

    def validate(self) -> None:
        for name in ("epochs", "max_features", "min_feature_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LanguageIdError(f"{name} must be a positive integer")
        if self.max_features > 100000 or self.epochs > 100000:
            raise LanguageIdError("model complexity exceeds the safety limit")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise LanguageIdError("seed must be an integer")
        for name in ("learning_rate", "l2", "min_confidence", "min_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise LanguageIdError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise LanguageIdError("learning_rate must be positive and l2 nonnegative")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.min_margin <= 1:
            raise LanguageIdError("confidence and margin thresholds must be in [0, 1]")


@dataclass(frozen=True)
class LabeledTokenExample:
    text: str
    start: int
    end: int
    label: str
    partition: str
    group_id: str
    source: str
    weight: float = 1.0


@dataclass(frozen=True)
class TokenPrediction:
    span: TokenSpan
    probabilities: Mapping[str, float]
    top_label: str
    decision_label: str
    uncertain: bool
    entropy: float


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _identity(example: LabeledTokenExample) -> str:
    return _fingerprint([example.text, example.start, example.end])


def _validated_examples(examples: Iterable[LabeledTokenExample], partition: str) -> tuple[LabeledTokenExample, ...]:
    result = tuple(examples)
    if not result:
        raise LanguageIdError("at least one labeled token example is required")
    labels_by_span: dict[str, str] = {}
    for example in result:
        if not isinstance(example, LabeledTokenExample):
            raise LanguageIdError("examples must be LabeledTokenExample records")
        if example.partition != partition:
            raise LanguageIdError(f"expected {partition} examples; final_evaluation and other partitions are forbidden")
        try:
            text = normalize_processing_text(example.text)
        except TextPreparationError as exc:
            raise LanguageIdError(str(exc)) from exc
        if text != example.text:
            raise LanguageIdError("labeled text must already be NFC for span offsets")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (example.start, example.end)):
            raise LanguageIdError("token offsets must be integers")
        if not any(span.start == example.start and span.end == example.end for span in tokenize_with_spans(text)):
            raise LanguageIdError("labeled span must match one complete shared token")
        if example.label not in LABELS:
            raise LanguageIdError(f"label must be one of {LABELS}")
        if any(not isinstance(value, str) or not value.strip() for value in (example.group_id, example.source)):
            raise LanguageIdError("group_id and source must be nonempty strings")
        if isinstance(example.weight, bool) or not isinstance(example.weight, (int, float)) or not math.isfinite(example.weight) or example.weight <= 0:
            raise LanguageIdError("example weight must be finite and positive")
        identity = _identity(example)
        if identity in labels_by_span and labels_by_span[identity] != example.label:
            raise LanguageIdError("the same text/span has conflicting labels")
        labels_by_span[identity] = example.label
    if not math.isfinite(sum(example.weight for example in result)):
        raise LanguageIdError("total example weight must remain finite")
    return tuple(sorted(result, key=lambda example: _canonical(asdict(example))))


def _features(tokens: Sequence[TokenSpan], position: int) -> dict[str, float]:
    token = tokens[position]
    word = token.text.casefold()
    padded = "<" + word + ">"
    values: Counter[str] = Counter()
    for size in range(2, 6):
        values.update("char:" + padded[start:start + size] for start in range(len(padded) - size + 1))
    values["word:" + word] = 1
    values["script:" + token.script] = 1
    values["shape:" + ("number" if token.text.isnumeric() else "upper" if token.text.isupper() else "title" if token.text.istitle() else "lower" if token.text.islower() else "other")] = 1
    for direction, neighbor in (("previous", position - 1), ("next", position + 1)):
        if 0 <= neighbor < len(tokens):
            values[direction + ":word:" + tokens[neighbor].text.casefold()] = 1
            values[direction + ":script:" + tokens[neighbor].script] = 1
        else:
            values[direction + ":boundary"] = 1
    return dict(values)


def _example_features(example: LabeledTokenExample) -> dict[str, float]:
    tokens = tokenize_with_spans(example.text)
    position = next(index for index, token in enumerate(tokens) if token.start == example.start and token.end == example.end)
    return _features(tokens, position)


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    if not logits or any(not math.isfinite(value) for value in logits) or not math.isfinite(temperature) or temperature <= 0:
        raise LanguageIdError("softmax requires finite logits and a positive finite temperature")
    maximum = max(logits)
    values = [math.exp((value - maximum) / temperature) for value in logits]
    total = sum(values)
    return tuple(value / total for value in values)


def _prediction(span: TokenSpan, probabilities: Sequence[float], config: LanguageIdConfig) -> TokenPrediction:
    order = sorted(range(len(LABELS)), key=lambda index: (-probabilities[index], index))
    top, runner_up = order[:2]
    uncertain = probabilities[top] < config.min_confidence or probabilities[top] - probabilities[runner_up] < config.min_margin
    return TokenPrediction(span, dict(zip(LABELS, probabilities)), LABELS[top], "other" if uncertain else LABELS[top], uncertain, -sum(value * math.log(value) for value in probabilities if value > 0))


class ScriptEvidenceBaseline:
    """Conservative heuristic evidence; its probabilities are not calibrated."""

    def __init__(self, config: LanguageIdConfig | None = None) -> None:
        self.config = config or LanguageIdConfig()
        self.config.validate()

    def predict(self, text: str) -> tuple[TokenPrediction, ...]:
        result = []
        for span in tokenize_with_spans(text):
            preferred = {"devanagari": "hi", "bengali": "bn", "numeric": "other", "other_letter": "other"}.get(span.script)
            probabilities = [0.25] * len(LABELS) if preferred is None else [0.05] * len(LABELS)
            if preferred is not None:
                probabilities[LABELS.index(preferred)] = 0.85
            prediction = _prediction(span, probabilities, self.config)
            if preferred in {None, "hi", "bn"}:
                prediction = TokenPrediction(span, prediction.probabilities, prediction.top_label, "other", True, prediction.entropy)
            result.append(prediction)
        return tuple(result)


class TokenLanguageIdentifier:
    """A fitted sparse-feature regressor with immutable inference state.

    Fit/load create sealed numeric/provenance state. Calibration is the only
    supported update and reseals it. This makes a cached, content-bound compact
    inference identity safe without hashing megabytes of training provenance
    on every query or trusting a manually maintained mutation version counter.
    """

    _SEALED_FIELDS = frozenset({"config", "feature_names", "_feature_index", "_weights", "_bias", "_temperature", "_metadata", "_inference_summary", "_inference_sealed", "_SEALED_FIELDS"})

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_inference_sealed", False) and name in self._SEALED_FIELDS:
            raise AttributeError("Fitted language-model state is immutable; fit/load a new model or use calibrate()")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_inference_sealed", False) and name in self._SEALED_FIELDS:
            raise AttributeError("Fitted language-model state is immutable")
        object.__delattr__(self, name)

    def _seal_inference_state(self) -> None:
        object.__setattr__(self, "_weights", tuple(tuple(row) for row in self._weights))
        object.__setattr__(self, "_bias", tuple(self._bias))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "_feature_index", MappingProxyType(dict(self._feature_index)))
        object.__setattr__(self, "_metadata", _freeze_json(self._metadata))
        manifest = self._manifest_value()
        payload = {"manifest": manifest, "features": list(self.feature_names), "weights": self._weights, "bias": self._bias}
        calibration = manifest["calibration"]
        summary = {
            "schema_version": SCHEMA_VERSION, "inference_metadata_schema_version": 1,
            "model_type": manifest["model_type"], "labels": list(LABELS),
            "feature_count": len(self.feature_names), "feature_policy": manifest["feature_policy"],
            "config": manifest["config"], "temperature": self.temperature,
            "model_sha256": _fingerprint(payload), "full_manifest_sha256": _fingerprint(manifest),
            "training_partition": manifest["training_partition"],
            "training_example_count": manifest["training_example_count"],
            "training_data_fingerprint": manifest["training_data_fingerprint"],
            "calibration": {key: calibration[key] for key in ("method", "partition", "validation_example_count", "validation_data_fingerprint", "temperature")} if calibration else None,
        }
        object.__setattr__(self, "_inference_summary", _freeze_json(summary))
        object.__setattr__(self, "_inference_sealed", True)

    def _manifest_value(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "model_type": "multinomial_logistic_regression", "labels": list(LABELS), "feature_count": len(self.feature_names), "feature_policy": "char_2_5_neighbor_script_shape_v1", "config": asdict(self.config), "temperature": self.temperature, **_thaw_json(self._metadata)}

    @classmethod
    def fit(cls, examples: Iterable[LabeledTokenExample], config: LanguageIdConfig | None = None) -> TokenLanguageIdentifier:
        config = config or LanguageIdConfig()
        config.validate()
        records = _validated_examples(examples, AUXILIARY_TRAIN)
        raw_features = [_example_features(example) for example in records]
        counts: Counter[str] = Counter()
        for values in raw_features:
            counts.update(values.keys())
        selected = sorted((feature for feature, count in counts.items() if count >= config.min_feature_count), key=lambda feature: (-counts[feature], feature))[:config.max_features]
        if not selected:
            raise LanguageIdError("training examples produced no features")
        model = cls()
        model.config = config
        model.feature_names = tuple(sorted(selected))
        model._feature_index = {feature: index for index, feature in enumerate(model.feature_names)}
        generator = random.Random(config.seed)
        model._weights = [[generator.uniform(-0.001, 0.001) for _ in selected] for _ in LABELS]
        model._bias = [0.0] * len(LABELS)
        model._temperature = 1.0
        vectors = [model._vector(values) for values in raw_features]
        total_weight = sum(example.weight for example in records)
        losses = []
        for _ in range(config.epochs):
            gradient = [[0.0] * len(selected) for _ in LABELS]
            bias_gradient = [0.0] * len(LABELS)
            loss = 0.0
            for example, vector in zip(records, vectors):
                probabilities = _softmax(model._logits(vector))
                target = LABELS.index(example.label)
                loss -= (example.weight / total_weight) * math.log(max(probabilities[target], 1e-300))
                for label in range(len(LABELS)):
                    error = example.weight * (probabilities[label] - (label == target)) / total_weight
                    bias_gradient[label] += error
                    for index, value in vector:
                        gradient[label][index] += error * value
            losses.append(loss)
            for label in range(len(LABELS)):
                model._bias[label] -= config.learning_rate * bias_gradient[label]
                for index in range(len(selected)):
                    model._weights[label][index] -= config.learning_rate * (gradient[label][index] + config.l2 * model._weights[label][index])
        model._metadata = {
            "training_partition": AUXILIARY_TRAIN,
            "training_example_count": len(records),
            "training_groups": sorted({example.group_id for example in records}),
            "training_span_fingerprints": sorted({_identity(example) for example in records}),
            "training_data_fingerprint": _fingerprint([asdict(example) for example in records]),
            "training_sources": sorted({example.source for example in records}),
            "loss_history": losses,
            "calibration": None,
        }
        if any(not math.isfinite(value) for row in model._weights for value in row) or any(not math.isfinite(value) for value in model._bias + losses):
            raise LanguageIdError("training diverged; reduce learning_rate or example weights")
        model._seal_inference_state()
        return model

    def _vector(self, values: Mapping[str, float]) -> tuple[tuple[int, float], ...]:
        selected = [(self._feature_index[feature], value) for feature, value in values.items() if feature in self._feature_index]
        norm = math.sqrt(sum(value * value for _, value in selected))
        return tuple(sorted((index, value / norm) for index, value in selected)) if norm else ()

    def _logits(self, vector: Sequence[tuple[int, float]]) -> tuple[float, ...]:
        return tuple(self._bias[label] + sum(self._weights[label][index] * value for index, value in vector) for label in range(len(LABELS)))

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest_value()

    @property
    def inference_metadata(self) -> Mapping[str, Any]:
        """Detached, bounded-size identity; full provenance remains in artifacts."""
        if not getattr(self, "_inference_sealed", False):
            raise LanguageIdError("Language model must be fitted or loaded before inference")
        return _thaw_json(self._inference_summary)

    def predict(self, text: str) -> tuple[TokenPrediction, ...]:
        tokens = tokenize_with_spans(text)
        return tuple(_prediction(span, _softmax(self._logits(self._vector(_features(tokens, position))), self.temperature), self.config) for position, span in enumerate(tokens))

    def calibrate(self, validation_examples: Iterable[LabeledTokenExample]) -> Mapping[str, Any]:
        records = _validated_examples(validation_examples, AUXILIARY_VALIDATION)
        groups = {example.group_id for example in records}
        if groups.intersection(self._metadata["training_groups"]):
            raise LanguageIdError("training/validation group leakage")
        if {_identity(example) for example in records}.intersection(self._metadata["training_span_fingerprints"]):
            raise LanguageIdError("training/validation duplicate text/span leakage")
        fingerprint = _fingerprint([asdict(example) for example in records])
        previous = self._metadata["calibration"]
        if previous is not None:
            if previous["validation_data_fingerprint"] == fingerprint:
                return _thaw_json(previous)
            raise LanguageIdError("model already calibrated; use a fresh training model for a different validation set")
        logits = [self._logits(self._vector(_example_features(example))) for example in records]
        total_weight = sum(example.weight for example in records)
        def nll(temperature: float) -> float:
            return -sum((example.weight / total_weight) * math.log(max(_softmax(values, temperature)[LABELS.index(example.label)], 1e-300)) for example, values in zip(records, logits))
        object.__setattr__(self, "_temperature", min(TEMPERATURE_GRID, key=lambda value: (nll(value), abs(value - 1.0))))
        report = {"method": "heldout_temperature_grid", "partition": AUXILIARY_VALIDATION, "validation_example_count": len(records), "validation_groups": sorted(groups), "validation_data_fingerprint": fingerprint, "validation_span_fingerprints": sorted({_identity(example) for example in records}), "temperature": self.temperature, "nll_before": nll(1.0), "nll_after": nll(self.temperature)}
        metadata = _thaw_json(self._metadata)
        metadata["calibration"] = report
        object.__setattr__(self, "_metadata", metadata)
        self._seal_inference_state()
        return copy.deepcopy(report)

    def save_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise LanguageIdError(f"refusing to overwrite {destination}")
        payload = {"manifest": self.manifest, "features": list(self.feature_names), "weights": self._weights, "bias": self._bias}
        artifact = {**payload, "checksum_sha256": _fingerprint(payload)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".language-id-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(_canonical(artifact).decode("utf-8") + "\n")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load_json(cls, path: str | Path) -> TokenLanguageIdentifier:
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
            if set(artifact) != {"manifest", "features", "weights", "bias", "checksum_sha256"}:
                raise LanguageIdError("invalid language model artifact fields")
            payload = {key: artifact[key] for key in ("manifest", "features", "weights", "bias")}
            if artifact["checksum_sha256"] != _fingerprint(payload):
                raise LanguageIdError("language model checksum mismatch")
            manifest = artifact["manifest"]
            expected_manifest = {"schema_version", "model_type", "labels", "feature_count", "feature_policy", "config", "temperature", "training_partition", "training_example_count", "training_groups", "training_span_fingerprints", "training_data_fingerprint", "training_sources", "loss_history", "calibration"}
            if not isinstance(manifest, dict) or set(manifest) != expected_manifest:
                raise LanguageIdError("invalid model manifest fields")
            if manifest["schema_version"] != SCHEMA_VERSION or manifest["labels"] != list(LABELS) or manifest["model_type"] != "multinomial_logistic_regression" or manifest["feature_policy"] != "char_2_5_neighbor_script_shape_v1":
                raise LanguageIdError("incompatible language model schema or labels")
            config = LanguageIdConfig(**manifest["config"])
            config.validate()
            features, weights, bias = artifact["features"], artifact["weights"], artifact["bias"]
            if not isinstance(features, list) or not features or any(not isinstance(value, str) or not value for value in features) or features != sorted(set(features)) or len(features) > config.max_features or isinstance(manifest["feature_count"], bool) or manifest["feature_count"] != len(features):
                raise LanguageIdError("invalid feature vocabulary")
            if not isinstance(weights, list) or len(weights) != len(LABELS) or any(not isinstance(row, list) or len(row) != len(features) for row in weights) or not isinstance(bias, list) or len(bias) != len(LABELS):
                raise LanguageIdError("invalid language model weight shapes")
            def finite(value: Any) -> bool:
                return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            if any(not finite(value) for row in weights for value in row) or any(not finite(value) for value in bias) or not finite(manifest["temperature"]) or manifest["temperature"] not in TEMPERATURE_GRID:
                raise LanguageIdError("invalid finite weights or temperature")
            groups = manifest["training_groups"]
            fingerprints = manifest["training_span_fingerprints"]
            def valid_hash(value: Any) -> bool:
                return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
            def string_list(value: Any) -> bool:
                return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value) and value == sorted(set(value))
            if manifest["training_partition"] != AUXILIARY_TRAIN or not string_list(groups) or not string_list(fingerprints) or not all(valid_hash(value) for value in fingerprints) or not valid_hash(manifest["training_data_fingerprint"]) or not string_list(manifest["training_sources"]):
                raise LanguageIdError("invalid training provenance")
            if isinstance(manifest["training_example_count"], bool) or not isinstance(manifest["training_example_count"], int) or manifest["training_example_count"] < len(fingerprints) or not isinstance(manifest["loss_history"], list) or len(manifest["loss_history"]) != config.epochs or any(not finite(value) or value < 0 for value in manifest["loss_history"]):
                raise LanguageIdError("invalid training count or loss history")
            calibration = manifest["calibration"]
            if calibration is None:
                if manifest["temperature"] != 1.0:
                    raise LanguageIdError("uncalibrated model must have temperature 1")
            elif not isinstance(calibration, dict) or calibration["partition"] != AUXILIARY_VALIDATION or calibration["method"] != "heldout_temperature_grid" or calibration["temperature"] != manifest["temperature"] or not valid_hash(calibration["validation_data_fingerprint"]) or not string_list(calibration["validation_groups"]) or not string_list(calibration["validation_span_fingerprints"]) or isinstance(calibration["validation_example_count"], bool) or not isinstance(calibration["validation_example_count"], int) or calibration["validation_example_count"] < len(calibration["validation_span_fingerprints"]) or set(groups).intersection(calibration["validation_groups"]) or set(fingerprints).intersection(calibration["validation_span_fingerprints"]) or any(not valid_hash(value) for value in calibration["validation_span_fingerprints"]) or any(not finite(calibration[key]) or calibration[key] < 0 for key in ("nll_before", "nll_after")) or calibration["nll_after"] > calibration["nll_before"] + 1e-12:
                raise LanguageIdError("invalid calibration provenance or leakage")
            model = cls()
            model.config, model.feature_names = config, tuple(features)
            model._feature_index = {feature: index for index, feature in enumerate(features)}
            model._weights, model._bias, model._temperature = weights, bias, manifest["temperature"]
            model._metadata = {key: manifest[key] for key in ("training_partition", "training_example_count", "training_groups", "training_span_fingerprints", "training_data_fingerprint", "training_sources", "loss_history", "calibration")}
            model._seal_inference_state()
            return model
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            if isinstance(exc, LanguageIdError):
                raise
            raise LanguageIdError(f"cannot load language model: {exc}") from exc


def _read_examples(path: str | Path) -> tuple[LabeledTokenExample, ...]:
    from data.prepare_corpus import iter_jsonl
    try:
        return tuple(LabeledTokenExample(**record) for record in iter_jsonl(Path(path)))
    except (TypeError, ValueError) as exc:
        raise LanguageIdError(f"invalid labeled token JSONL: {exc}") from exc


def _prediction_record(text: str, predictions: Sequence[TokenPrediction], model: Any) -> Mapping[str, Any]:
    supervised = isinstance(model, TokenLanguageIdentifier)
    return {"text": text, "processing_text": normalize_processing_text(text), "model_type": "multinomial_logistic_regression" if supervised else "script_evidence_baseline", "calibrated": supervised and model.manifest["calibration"] is not None, "tokens": [asdict(prediction) for prediction in predictions]}


def _batch_predictions(path: str, split: str, model: Any) -> list[Mapping[str, Any]]:
    from data.prepare_corpus import _clean_id, iter_jsonl
    records, seen = [], set()
    for query in iter_jsonl(Path(path)):
        query_id = _clean_id(query.get("query_id"), "query_id")
        if query_id in seen:
            raise LanguageIdError(f"duplicate query ID {query_id!r}")
        seen.add(query_id)
        if query.get("split") == split:
            text = query.get("text")
            normalize_processing_text(text)
            records.append({"query_id": query_id, "split": split, **_prediction_record(text, model.predict(text), model)})
    if not records:
        raise LanguageIdError(f"no queries selected for split {split!r}")
    return sorted(records, key=lambda record: record["query_id"])


def main(argv: Sequence[str] | None = None) -> int:
    from data.make_splits import ALL_SPLITS
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit auxiliary training token labels only")
    train.add_argument("--examples", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--validation-examples")
    train.add_argument("--epochs", type=int, default=180)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--force", action="store_true")
    predict = commands.add_parser("predict", help="predict tokens; omit --model for script baseline")
    predict.add_argument("--model")
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--queries")
    predict.add_argument("--split", choices=ALL_SPLITS)
    predict.add_argument("--output")
    predict.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            model = TokenLanguageIdentifier.fit(_read_examples(args.examples), LanguageIdConfig(epochs=args.epochs, seed=args.seed))
            if args.validation_examples:
                model.calibrate(_read_examples(args.validation_examples))
            model.save_json(args.output, overwrite=args.force)
            print(json.dumps({"output": args.output, "manifest": model.manifest}, ensure_ascii=False, sort_keys=True))
        else:
            model = TokenLanguageIdentifier.load_json(args.model) if args.model else ScriptEvidenceBaseline()
            if args.queries:
                if not args.split or not args.output:
                    raise LanguageIdError("batch prediction requires --split and --output")
                records = _batch_predictions(args.queries, args.split, model)
            else:
                records = [_prediction_record(args.text, model.predict(args.text), model)]
            if args.output:
                from retrieval.bm25 import _atomic_write_jsonl
                output = Path(args.output)
                if output.exists() and not args.force:
                    raise LanguageIdError(f"refusing to overwrite {output}")
                _atomic_write_jsonl(output, records)
            else:
                sys.stdout.write("".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in records))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
