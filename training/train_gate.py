"""Train a leakage-aware, dependency-free linear softmax retrieval gate.

The retriever, indexes, and query resources are frozen. Relevance is consulted
only while creating fusion-training targets and validation reports; the saved
inference model receives observable query views and branch rankings only.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.generate_variants import VariantError, generate_variants
from data.make_splits import FUSION_TRAIN, OFFICIAL_TRAIN, VALIDATION
from data.prepare_corpus import CorpusPreparationError, Qrel, _clean_id, iter_jsonl, load_qrels
from evaluation.evaluate import EvaluationError
from nlp.language_id import LanguageIdError, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator, QueryViewError, QueryViews
from nlp.text import script_bucket
from retrieval.bm25 import BM25Error, BM25Index, _atomic_write_json, _file_sha256, load_corpus as load_bm25_corpus
from retrieval.char_tfidf import CharTfidfError, CharTfidfIndex, load_corpus as load_char_corpus
from retrieval.dense import DenseConfig, DenseError, DenseIndex, INDEX_MANIFEST_NAME, _add_encoder_arguments, _encoder_from_arguments
from retrieval.fusion import BRANCHES, BranchHit, FusionError, RetrievalPipeline, _finite, _fingerprint, _ranking, fuse_rankings


SCHEMA_VERSION = 1
FROZEN_PIPELINE_FIELDS = ("schema_version", "retriever", "branches", "config", "view_config", "indexes", "resources", "fixture_only")
_PAIRS = tuple(itertools.combinations(BRANCHES, 2))
FEATURE_NAMES = (
    "latin_character_fraction", "devanagari_character_fraction", "bengali_character_fraction",
    "mean_token_entropy", "uncertain_token_fraction", "token_count", "character_count", "language_switch_count",
    "native_changed_token_fraction", "roman_changed_token_fraction", "native_edit_distance", "roman_edit_distance",
    "protected_attempted_rewrite_count", "protected_token_fraction", "native_view_count", "roman_available",
    *(f"{branch}_{field}" for branch in BRANCHES for field in ("active", "candidate_fraction", "score_gap", "score_dispersion")),
    *(f"overlap_{left}_{right}" for left, right in _PAIRS),
)


class GateError(ValueError):
    """Invalid supervision, partition isolation, feature values, or artifact."""


def _checked_excluded_features(values: Sequence[str]) -> tuple[str, ...]:
    """Validate a feature ablation and return canonical feature-schema order."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GateError("excluded_features must be a sequence of unique known feature names")
    if any(not isinstance(name, str) or name not in FEATURE_NAMES for name in values):
        raise GateError("excluded_features must contain only known string feature names")
    if len(set(values)) != len(values):
        raise GateError("excluded_features must not contain duplicate feature names")
    selected = set(values)
    return tuple(name for name in FEATURE_NAMES if name in selected)


@dataclass(frozen=True)
class GateConfig:
    temperature: float = 0.25
    epochs: int = 300
    learning_rate: float = 0.05
    l2: float = 0.001

    def validate(self) -> None:
        for name in ("temperature", "learning_rate"):
            if not _finite(getattr(self, name)) or getattr(self, name) <= 0:
                raise GateError(f"{name} must be finite and positive")
        if not _finite(self.l2) or self.l2 < 0:
            raise GateError("l2 must be finite and nonnegative")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or not 1 <= self.epochs <= 100000:
            raise GateError("epochs must be an integer in [1, 100000]")


@dataclass(frozen=True)
class GateExample:
    query_id: str
    parent_query_id: str
    group_id: str
    split: str
    views: QueryViews
    branch_results: Mapping[str, Sequence[BranchHit]]
    judgments: Mapping[str, int]
    verified: bool = True
    benchmark_language: str | None = None
    judgment_query_id: str | None = None


@dataclass(frozen=True)
class GateTrainingResult:
    gate: LinearSoftmaxGate
    report: Mapping[str, Any]


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise GateError("records must be mappings or dataclasses")
    return dict(value)


def frozen_pipeline_manifest(manifest: Any) -> dict[str, Any]:
    """Bind only frozen retrieval behavior, never the inference gate itself."""
    if isinstance(manifest, RetrievalPipeline):
        manifest = manifest.manifest
    if not isinstance(manifest, Mapping) or any(key not in manifest for key in FROZEN_PIPELINE_FIELDS):
        raise GateError("a complete frozen retrieval pipeline manifest is required")
    try:
        payload = json.loads(json.dumps({key: manifest[key] for key in FROZEN_PIPELINE_FIELDS}, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GateError("pipeline metadata must contain finite JSON values") from exc
    if payload["branches"] != list(BRANCHES) or not isinstance(payload["fixture_only"], bool):
        raise GateError("pipeline branch order or fixture declaration is invalid")
    return payload


def _branches(results: Mapping[str, Sequence[Any]]) -> dict[str, tuple[BranchHit, ...]]:
    if not isinstance(results, Mapping) or set(results) - set(BRANCHES):
        raise GateError("rankings must contain only the four known branches")
    try:
        checked = {branch: _ranking(results.get(branch, ()), label=branch) for branch in BRANCHES}
    except FusionError as exc:
        raise GateError(str(exc)) from exc
    if any(len(hits) > 100 for hits in checked.values()):
        raise GateError("branch rankings exceed the frozen 100-candidate budget")
    return checked


def _judgments(values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise GateError("judgments must be a passage-to-relevance mapping")
    result = {}
    for doc_id, value in values.items():
        try:
            doc_id = _clean_id(doc_id, "doc_id")
        except CorpusPreparationError as exc:
            raise GateError(str(exc)) from exc
        # Larger gains cannot be represented by the evaluator's finite float.
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1023:
            raise GateError("relevance must be an integer in [0, 1023]")
        result[doc_id] = value
    return result


def _ndcg_at_10(ranking: Sequence[Any], judgments: Mapping[str, int]) -> float:
    """Graded nDCG with a common gain scale to avoid IDCG overflow."""
    maximum = max(judgments.values(), default=0)
    if maximum <= 0:
        raise GateError("nDCG requires at least one positive relevance judgment")
    offset = math.ldexp(1.0, -maximum)
    def dcg(grades: Sequence[int]) -> float:
        return math.fsum((math.ldexp(1.0, grade - maximum) - offset) / math.log2(rank + 1) for rank, grade in enumerate(grades[:10], 1))
    ideal = dcg(sorted(judgments.values(), reverse=True))
    return dcg([judgments.get(hit.doc_id, 0) for hit in ranking[:10]]) / ideal


def _softmax(logits: Sequence[float], active: Sequence[bool]) -> tuple[float, ...]:
    if len(logits) != len(BRANCHES) or len(active) != len(BRANCHES) or any(not _finite(value) for value in logits):
        raise GateError("gate logits must be four finite numbers")
    selected = [value for value, enabled in zip(logits, active) if enabled]
    if not selected:
        return (0.0,) * len(BRANCHES)
    highest = max(selected)
    # Subtraction may become -inf at opposite finite float extremes, for which
    # exp correctly underflows to zero. At least one active logit stays zero.
    values = [math.exp(value - highest) if enabled else 0.0 for value, enabled in zip(logits, active)]
    total = math.fsum(values)
    return tuple(value / total for value in values)


def utility_targets(
    branch_results: Mapping[str, Sequence[Any]], judgments: Mapping[str, int], *, temperature: float = 0.25,
) -> dict[str, float] | None:
    """Temperature-softmax branch nDCG@10, masked before normalization.

    No-positive or all-zero-utility queries return None, not an artificial
    uniform preference. Validation still scores zero-retrieval queries.
    """
    if not _finite(temperature) or temperature <= 0:
        raise GateError("temperature must be finite and positive")
    branches, relevance = _branches(branch_results), _judgments(judgments)
    if not any(relevance.values()):
        return None
    utilities = [_ndcg_at_10(branches[branch], relevance) for branch in BRANCHES]
    if max(utilities, default=0.0) == 0:
        return None
    # Subtract before dividing so any finite positive temperature is valid.
    highest = max(utilities)
    logits = [max(-1e300, (utility - highest) / temperature) for utility in utilities]
    return dict(zip(BRANCHES, _softmax(logits, [bool(branches[branch]) for branch in BRANCHES])))


def _normalized_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for row, character in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (character != other)))
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def extract_features(views: QueryViews, branch_results: Mapping[str, Sequence[Any]]) -> dict[str, float]:
    """Observable features only; branch evidence uses averaged RRF scores.

    Agreement is top-10 Jaccard overlap of branch lists, including the original
    versus rewritten-native branch pair. Raw retriever scores are not fused.
    """
    if not isinstance(views, QueryViews):
        raise GateError("features require QueryViews")
    branches = _branches(branch_results)
    if not isinstance(views.processing_text, str) or len(views.processing_text) > 4096:
        raise GateError("query processing text exceeds the query-view safety limit")
    letters = [character for character in views.processing_text if character.isalpha()]
    denominator = max(1, len(letters))
    tokens = views.tokens
    count = len(tokens)
    decisions = [prediction.decision_label for prediction in tokens if not prediction.uncertain and prediction.decision_label != "other"]
    token_denominator = max(1, count)
    native_changed = max((len(view.edits) for view in views.native_views), default=0) / token_denominator
    roman_changed = len(views.roman_view.edits) / token_denominator if views.roman_view else 0.0
    protected_tokens = sum(any(token.span.start < span.end and token.span.end > span.start for span in views.protections) for token in tokens)
    values = [
        *(sum(script_bucket(character) == script for character in letters) / denominator for script in ("latin", "devanagari", "bengali")),
        math.fsum(prediction.entropy for prediction in tokens) / token_denominator,
        sum(prediction.uncertain for prediction in tokens) / token_denominator,
        float(count), float(len(views.processing_text)), float(sum(left != right for left, right in zip(decisions, decisions[1:]))),
        native_changed, roman_changed,
        max((_normalized_edit_distance(views.processing_text, view.text) for view in views.native_views), default=0.0),
        _normalized_edit_distance(views.processing_text, views.roman_view.text) if views.roman_view else 0.0,
        views.diagnostics.get("protected_attempted_rewrite_count", 0.0),
        protected_tokens / token_denominator, float(len(views.native_views)), float(views.roman_view is not None),
    ]
    for branch in BRANCHES:
        hits = branches[branch]
        scale = max((abs(hit.score) for hit in hits[:10]), default=0.0)
        scores = [hit.score / scale for hit in hits[:10]] if scale else [0.0 for _ in hits[:10]]
        gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        values.extend((float(bool(hits)), len(hits) / 100.0, gap, statistics.pstdev(scores) if scores else 0.0))
    for left, right in _PAIRS:
        left_ids, right_ids = {hit.doc_id for hit in branches[left][:10]}, {hit.doc_id for hit in branches[right][:10]}
        values.append(len(left_ids & right_ids) / max(1, len(left_ids | right_ids)))
    if len(values) != len(FEATURE_NAMES) or any(not _finite(value) for value in values):
        raise GateError("observable features must be finite numeric values")
    return dict(zip(FEATURE_NAMES, map(float, values)))


def _checked_examples(examples: Sequence[GateExample], *, split: str) -> tuple[GateExample, ...]:
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise GateError("examples must be a sequence")
    seen = set()
    result = []
    for example in examples:
        if not isinstance(example, GateExample) or example.split != split:
            raise GateError(f"expected only {split} GateExample records")
        for key in ("query_id", "parent_query_id", "group_id"):
            try:
                _clean_id(getattr(example, key), key)
            except CorpusPreparationError as exc:
                raise GateError(str(exc)) from exc
        if example.query_id in seen:
            raise GateError("duplicate query ID in gate examples")
        seen.add(example.query_id)
        if example.verified is not True:
            raise GateError("unverified variants cannot provide gate supervision")
        if example.benchmark_language is not None and example.benchmark_language not in ("en", "hi", "bn"):
            raise GateError("benchmark_language must be en, hi, bn, or None")
        if example.judgment_query_id is not None:
            try:
                _clean_id(example.judgment_query_id, "judgment_query_id")
            except CorpusPreparationError as exc:
                raise GateError(str(exc)) from exc
        _branches(example.branch_results)
        _judgments(example.judgments)
        result.append(example)
    return tuple(sorted(result, key=lambda example: example.query_id))


class LinearSoftmaxGate:
    """Portable, immutable-numeric inference provider with no relevance input."""

    feature_names = FEATURE_NAMES

    def __init__(
        self, weights: Sequence[Sequence[float]], bias: Sequence[float], feature_mean: Sequence[float],
        feature_scale: Sequence[float], *, pipeline_manifest: Mapping[str, Any], training_report: Mapping[str, Any],
    ) -> None:
        width = len(FEATURE_NAMES)
        if len(weights) != len(BRANCHES) or any(len(row) != width for row in weights) or len(bias) != len(BRANCHES) or len(feature_mean) != width or len(feature_scale) != width:
            raise GateError("gate parameter shapes do not match the feature schema")
        numbers = [*bias, *feature_mean, *feature_scale, *(value for row in weights for value in row)]
        if any(not _finite(value) for value in numbers) or any(value <= 0 for value in feature_scale):
            raise GateError("gate parameters must be finite with positive feature scales")
        self.weights = tuple(tuple(map(float, row)) for row in weights)
        self.bias = tuple(map(float, bias))
        self.feature_mean = tuple(map(float, feature_mean))
        self.feature_scale = tuple(map(float, feature_scale))
        self._pipeline_manifest = frozen_pipeline_manifest(pipeline_manifest)
        self.pipeline_fingerprint = _fingerprint(self._pipeline_manifest)
        # Round trip copies nested structures and rejects non-finite metadata.
        try:
            self._training_report = json.loads(json.dumps(dict(training_report), allow_nan=False))
        except (TypeError, ValueError, OverflowError) as exc:
            raise GateError("training report must contain finite JSON metadata") from exc
        # Old artifacts lack this report field and retain their exact payload.
        self._excluded_features = _checked_excluded_features(self._training_report.get("excluded_features", []))

    @property
    def excluded_features(self) -> tuple[str, ...]:
        return self._excluded_features

    @property
    def training_report(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._training_report)

    def __call__(self, views: QueryViews, branch_results: Mapping[str, Sequence[Any]]) -> Mapping[str, float]:
        features = extract_features(views, branch_results)
        excluded = set(self.excluded_features)
        scaled = [0.0 if name in excluded else (features[name] - mean) / scale
                  for name, mean, scale in zip(FEATURE_NAMES, self.feature_mean, self.feature_scale)]
        try:
            logits = [math.fsum(weight * value for weight, value in zip(row, scaled)) + bias for row, bias in zip(self.weights, self.bias)]
        except (OverflowError, ValueError) as exc:
            raise GateError("standardized features overflowed the trained gate") from exc
        probabilities = _softmax(logits, [bool(branch_results.get(branch)) for branch in BRANCHES])
        return dict(zip(BRANCHES, probabilities))

    def assert_compatible(self, pipeline_or_manifest: Any) -> None:
        if _fingerprint(frozen_pipeline_manifest(pipeline_or_manifest)) != self.pipeline_fingerprint:
            raise GateError("gate is incompatible with the frozen pipeline/index/resource fingerprint")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "model_type": "linear_softmax_gate", "branches": list(BRANCHES),
            "feature_names": list(FEATURE_NAMES), "weights": [list(row) for row in self.weights], "bias": list(self.bias),
            "feature_mean": list(self.feature_mean), "feature_scale": list(self.feature_scale),
            "frozen_pipeline": self._pipeline_manifest, "frozen_pipeline_fingerprint": self.pipeline_fingerprint,
            "training_report": self.training_report,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "linear_softmax_gate", "schema_version": SCHEMA_VERSION,
            "artifact_sha256": _fingerprint(self._payload()), "frozen_pipeline_fingerprint": self.pipeline_fingerprint,
            "fixture_only": self._pipeline_manifest["fixture_only"], "features": list(FEATURE_NAMES),
            "excluded_features": list(self.excluded_features),
        }

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.metadata

    def save_json(self, path: Path, *, overwrite: bool = False) -> None:
        path = Path(path)
        if path.is_symlink():
            raise GateError("gate artifact must not be a symbolic link")
        if path.exists() and not overwrite:
            raise GateError("refusing to overwrite an existing gate artifact")
        payload = self._payload()
        _atomic_write_json(path, {**payload, "artifact_sha256": _fingerprint(payload)})

    @classmethod
    def load_json(cls, path: Path, *, expected_pipeline_manifest: Mapping[str, Any] | None = None) -> LinearSoftmaxGate:
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value = {}
            for key, item in pairs:
                if key in value:
                    raise GateError(f"duplicate JSON field {key!r}")
                value[key] = item
            return value
        path = Path(path)
        if path.is_symlink():
            raise GateError("gate artifact must not be a symbolic link")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
            if not isinstance(payload, dict):
                raise GateError("gate artifact must contain an object")
            checksum = payload.pop("artifact_sha256", None)
            if checksum != _fingerprint(payload):
                raise GateError("gate artifact checksum mismatch")
            expected = {"schema_version", "model_type", "branches", "feature_names", "weights", "bias", "feature_mean", "feature_scale", "frozen_pipeline", "frozen_pipeline_fingerprint", "training_report"}
            if set(payload) != expected or type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION or payload["model_type"] != "linear_softmax_gate" or payload["branches"] != list(BRANCHES) or payload["feature_names"] != list(FEATURE_NAMES):
                raise GateError("unsupported gate artifact schema or feature/branch order")
            result = cls(payload["weights"], payload["bias"], payload["feature_mean"], payload["feature_scale"], pipeline_manifest=payload["frozen_pipeline"], training_report=payload["training_report"])
            if result.pipeline_fingerprint != payload["frozen_pipeline_fingerprint"]:
                raise GateError("frozen pipeline checksum mismatch")
            if expected_pipeline_manifest is not None:
                result.assert_compatible(expected_pipeline_manifest)
            return result
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, GateError):
                raise
            raise GateError(f"invalid gate artifact: {exc}") from exc


def _objective(
    matrix: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], masks: Sequence[Sequence[bool]],
    weights: Sequence[Sequence[float]], bias: Sequence[float], l2: float,
) -> float:
    losses = []
    for row, target, mask in zip(matrix, targets, masks):
        logits = [math.fsum(weight * value for weight, value in zip(branch, row)) + intercept for branch, intercept in zip(weights, bias)]
        probabilities = _softmax(logits, mask)
        losses.append(-math.fsum(value * math.log(max(probability, 1e-305)) for value, probability in zip(target, probabilities) if value > 0))
    penalty = 0.5 * l2 * math.fsum(value * value for branch in weights for value in branch)
    loss = statistics.fmean(losses) + penalty
    if not _finite(loss):
        raise GateError("gate training objective became non-finite")
    return loss


def _mean_ndcg(examples: Sequence[GateExample], *, config: Any, weights: Mapping[str, float] | None = None, gate: LinearSoftmaxGate | None = None) -> dict[str, Any]:
    scores = []
    languages: dict[str, list[float]] = {}
    for example in examples:
        if not any(example.judgments.values()):
            continue
        selected = gate(example.views, example.branch_results) if gate is not None else weights
        rankings = fuse_rankings(example.branch_results, selected, config=config)
        score = _ndcg_at_10(rankings.hits, example.judgments)
        scores.append(score)
        languages.setdefault(example.benchmark_language or "unspecified", []).append(score)
    per_language = {language: statistics.fmean(values) for language, values in sorted(languages.items())}
    return {"ndcg_at_10": statistics.fmean(scores) if scores else None, "language_macro_ndcg_at_10": statistics.fmean(per_language.values()) if per_language else None, "per_language_ndcg_at_10": per_language, "query_count": len(scores), "excluded_no_positive_judgments_count": len(examples) - len(scores)}


def compare_validation(gate: LinearSoftmaxGate, examples: Sequence[GateExample], *, grid_resolution: int = 4) -> dict[str, Any]:
    """Compare actual fused nDCG against equal and validation-tuned fixed RRF."""
    from retrieval.fusion import FusionConfig
    examples = _checked_examples(examples, split=VALIDATION)
    if isinstance(grid_resolution, bool) or not isinstance(grid_resolution, int) or not 1 <= grid_resolution <= 12:
        raise GateError("fixed-weight grid resolution must be an integer in [1, 12]")
    config = FusionConfig(**gate._pipeline_manifest["config"])
    # Gate labels are always nDCG@10; a shorter serving depth must not truncate
    # validation scores. All other frozen fusion parameters are preserved.
    config = FusionConfig(config.per_branch_top_k, config.per_view_top_k, 10, config.k_view, config.k)
    equal = {branch: 0.25 for branch in BRANCHES}
    equal_report = _mean_ndcg(examples, config=config, weights=equal)
    best_weights, best_report = equal, equal_report
    candidates = 0
    for integer_weights in itertools.product(range(grid_resolution + 1), repeat=len(BRANCHES)):
        if sum(integer_weights) != grid_resolution:
            continue
        weights = {branch: value / grid_resolution for branch, value in zip(BRANCHES, integer_weights)}
        # A fixed baseline must be valid on every active-branch pattern.
        if any(any(example.branch_results.values()) and not any(weights[branch] > 0 and example.branch_results.get(branch) for branch in BRANCHES) for example in examples):
            continue
        candidates += 1
        report = _mean_ndcg(examples, config=config, weights=weights)
        if report["language_macro_ndcg_at_10"] is not None and (best_report["language_macro_ndcg_at_10"] is None or report["language_macro_ndcg_at_10"] > best_report["language_macro_ndcg_at_10"] + 1e-12):
            best_weights, best_report = weights, report
    return {
        "split": VALIDATION, "equal_weights": {**equal_report, "weights": {branch: 0.25 for branch in BRANCHES}},
        "tuned_fixed_weights": {**best_report, "weights": best_weights, "grid_resolution": grid_resolution, "evaluated_candidates": candidates},
        "gate": _mean_ndcg(examples, config=config, gate=gate),
        "selection_metric": "language_macro_actual_fused_ndcg_at_10",
    }


def train_gate(
    examples: Sequence[GateExample], *, pipeline_manifest: Mapping[str, Any], config: GateConfig = GateConfig(),
    validation_examples: Sequence[GateExample] = (), excluded_features: Sequence[str] = (),
) -> GateTrainingResult:
    """Fit on fusion_train; feature ablations are retrained with zeroed columns."""
    config.validate()
    excluded = _checked_excluded_features(excluded_features)
    frozen = frozen_pipeline_manifest(pipeline_manifest)
    examples = _checked_examples(examples, split=FUSION_TRAIN)
    validation = _checked_examples(validation_examples, split=VALIDATION)
    if not examples:
        raise GateError("no fusion_train examples were supplied")
    for field in ("query_id", "parent_query_id", "group_id", "judgment_query_id"):
        if {getattr(example, field) for example in examples if getattr(example, field) is not None} & {getattr(example, field) for example in validation if getattr(example, field) is not None}:
            raise GateError(f"fusion-training and validation {field} values must be disjoint")
    included, targets, masks, raw = [], [], [], []
    no_positive, all_zero = [], []
    for example in examples:
        if not any(example.judgments.values()):
            no_positive.append(example.query_id)
            continue
        target = utility_targets(example.branch_results, example.judgments, temperature=config.temperature)
        if target is None:
            all_zero.append(example.query_id)
            continue
        included.append(example)
        targets.append([target[branch] for branch in BRANCHES])
        masks.append([bool(example.branch_results.get(branch)) for branch in BRANCHES])
        features = extract_features(example.views, example.branch_results)
        raw.append([0.0 if name in excluded else features[name] for name in FEATURE_NAMES])
    if not included:
        raise GateError("no useful fusion-training examples remain after excluding no-positive/all-zero-utility queries")
    means = [statistics.fmean(column) for column in zip(*raw)]
    scales = [statistics.pstdev(column) or 1.0 for column in zip(*raw)]
    matrix = [[(value - mean) / scale for value, mean, scale in zip(row, means, scales)] for row in raw]
    width = len(FEATURE_NAMES)
    weights, bias = [[0.0] * width for _ in BRANCHES], [0.0] * len(BRANCHES)
    loss = _objective(matrix, targets, masks, weights, bias, config.l2)
    history = [loss]
    for _ in range(config.epochs):
        gradient_w, gradient_b = [[0.0] * width for _ in BRANCHES], [0.0] * len(BRANCHES)
        for row, target, mask in zip(matrix, targets, masks):
            probabilities = _softmax([math.fsum(weight * value for weight, value in zip(branch, row)) + intercept for branch, intercept in zip(weights, bias)], mask)
            for number, (probability, desired) in enumerate(zip(probabilities, target)):
                difference = (probability - desired) / len(included)
                gradient_b[number] += difference
                for column, value in enumerate(row):
                    gradient_w[number][column] += difference * value
        for number in range(len(BRANCHES)):
            for column in range(width):
                gradient_w[number][column] += config.l2 * weights[number][column]
        step = config.learning_rate
        for _attempt in range(60):
            candidate_w = [[value - step * gradient for value, gradient in zip(row, gradients)] for row, gradients in zip(weights, gradient_w)]
            candidate_b = [value - step * gradient for value, gradient in zip(bias, gradient_b)]
            try:
                candidate_loss = _objective(matrix, targets, masks, candidate_w, candidate_b, config.l2)
            except (GateError, OverflowError, ValueError):
                candidate_loss = math.inf
            if candidate_loss <= loss + 1e-14:
                weights, bias, loss = candidate_w, candidate_b, candidate_loss
                break
            step *= 0.5
        else:
            raise GateError("line search could not find a finite non-increasing training step")
        history.append(loss)
    report = {
        "schema_version": SCHEMA_VERSION, "training_split": FUSION_TRAIN, "scaler_fit_split": FUSION_TRAIN,
        "config": asdict(config), "training_query_ids": [example.query_id for example in included],
        "training_parent_query_ids": sorted({example.parent_query_id for example in included}),
        "training_group_ids": sorted({example.group_id for example in included}),
        "supplied_training_query_count": len(examples), "training_query_count": len(included),
        "excluded_no_positive_judgments_count": len(no_positive), "excluded_no_positive_judgments_ids": no_positive,
        "excluded_all_zero_utility_count": len(all_zero), "excluded_all_zero_utility_ids": all_zero,
        "exclusion_policy": "exclude_no_positive_judgments_and_all_zero_branch_ndcg_at_10",
        "inactive_branch_policy": "zero_target_and_zero_prediction_masked_before_softmax",
        "initial_loss": history[0], "final_loss": history[-1], "loss_history": history,
        "loss_definition": "mean_masked_soft_target_cross_entropy_plus_half_l2_weight_norm_squared",
        "feature_schema": list(FEATURE_NAMES), "feature_evidence_source": "averaged_reciprocal_rank_branch_scores",
        "excluded_features": list(excluded),
        "feature_ablation_policy": "zero_raw_columns_before_scaler_and_training_then_mask_standardized_inference_columns",
        "validation_query_ids": [example.query_id for example in validation],
        "validation": None,
        "training_data_fingerprint": _fingerprint([
            {"query_id": example.query_id, "parent_query_id": example.parent_query_id, "group_id": example.group_id,
             "features": row, "target": target, "active": mask, "judgments": dict(sorted(example.judgments.items()))}
            for example, row, target, mask in zip(included, raw, targets, masks)
        ]),
    }
    gate = LinearSoftmaxGate(weights, bias, means, scales, pipeline_manifest=frozen, training_report=report)
    if validation:
        report["validation"] = compare_validation(gate, validation)
        gate = LinearSoftmaxGate(weights, bias, means, scales, pipeline_manifest=frozen, training_report=report)
    return GateTrainingResult(gate, report)


def collect_examples(
    pipeline: RetrievalPipeline, query_records: Sequence[Any], qrels: Sequence[Qrel | Mapping[str, Any]], *, split: str = FUSION_TRAIN,
) -> tuple[GateExample, ...]:
    """Validate split/review assignments and retrieve frozen branch examples.

    Complete split/qrel artifacts are accepted; only selected judgment IDs are
    read as labels. Encoders are never trained or mutated by this operation.
    """
    if split not in (FUSION_TRAIN, VALIDATION):
        raise GateError("gate collection is restricted to fusion_train or validation")
    if pipeline.weight_provider is not None:
        raise GateError("collect gate targets using a frozen pipeline without an inference weight provider")
    inputs = [_mapping(record) for record in query_records]
    selected = [record for record in inputs if record.get("split") == split]
    if not selected:
        raise GateError(f"no {split} queries were supplied")
    for record in selected:
        if record.get("source_partition") != OFFICIAL_TRAIN:
            raise GateError("gate training/validation requires official_train query groups")
        if record.get("variant_type", "original") != "original" and record.get("verified") is True and not all(key in record for key in ("candidate_id", "candidate_sha256", "review")):
            raise GateError("verified variants require a content-bound review")
    try:
        validated = generate_variants(inputs, typo_variants=0, max_variants_per_parent=max(1, len(selected)), split=split)
    except (VariantError, CorpusPreparationError, ValueError) as exc:
        raise GateError(f"invalid assigned/reviewed gate queries: {exc}") from exc
    selected = list(validated.queries)
    judgment_ids = {str(record["judgment_query_id"]) for record in selected}
    groups = {record["group_id"] for record in selected}
    for record in inputs:
        if record.get("split") != split and (record.get("judgment_query_id") in judgment_ids or record.get("group_id") in groups):
            raise GateError("selected judgment/group IDs overlap a different split")
    scoped: dict[str, dict[str, int]] = {query_id: {} for query_id in judgment_ids}
    corpus_ids = getattr(pipeline, "_corpus_ids", None)
    for value in qrels:
        record = _mapping(value)
        query_id = record.get("query_id")
        if query_id not in judgment_ids:
            continue
        relevance = _judgments({record.get("doc_id"): record.get("relevance")})
        doc_id, grade = next(iter(relevance.items()))
        if corpus_ids is not None and doc_id not in corpus_ids:
            raise GateError("selected judgment refers to a document outside the frozen corpus")
        if doc_id in scoped[query_id] and scoped[query_id][doc_id] != grade:
            raise GateError("conflicting selected relevance judgments")
        scoped[query_id][doc_id] = grade
    before = _fingerprint(frozen_pipeline_manifest(pipeline.manifest))
    observed: list[tuple[QueryViews, Mapping[str, Sequence[BranchHit]]]] = []
    def capture(views: QueryViews, rankings: Mapping[str, Sequence[BranchHit]]) -> Mapping[str, float]:
        observed.append((views, rankings))
        return {branch: 1.0 for branch in BRANCHES}
    # A separate collector shares frozen indexes/resources without replacing
    # the caller's provider or generating query hypotheses a second time.
    collector = RetrievalPipeline(
        pipeline.bm25_index, pipeline.char_index, pipeline.dense_index,
        pipeline.language_identifier, pipeline.transliterator, pipeline.config,
        pipeline.view_config, weight_provider=capture,
    )
    result = []
    for record in selected:
        collector.search(record["text"])
        if len(observed) != 1:
            raise GateError("collector must capture exactly one frozen query view and branch result")
        views, rankings = observed.pop()
        result.append(GateExample(record["query_id"], record["parent_query_id"], record["group_id"], split, views, rankings, scoped[str(record["judgment_query_id"])], benchmark_language=record["benchmark_language"], judgment_query_id=str(record["judgment_query_id"])))
    if _fingerprint(frozen_pipeline_manifest(pipeline.manifest)) != before:
        raise GateError("retrieval pipeline changed while collecting frozen examples")
    return _checked_examples(result, split=split)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "queries", "qrels", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--lid-model", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--validation", action="store_true", help="compare gate/equal/tuned-fixed fusion on validation only")
    parser.add_argument("--temperature-candidates", type=float, nargs="+", help="select temperature by validation fused nDCG@10")
    parser.add_argument("--l2-candidates", type=float, nargs="+", help="select regularization by validation fused nDCG@10")
    parser.add_argument("--exclude-features", choices=FEATURE_NAMES, nargs="+", default=(),
                        help="retrain the gate with these observable feature columns disabled")
    parser.add_argument("--force", action="store_true")
    _add_encoder_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report_path = args.report or args.output.with_suffix(".report.json")
        outputs = [args.output, report_path]
        inputs = [args.corpus, args.queries, args.qrels, args.lid_model, args.lexicon]
        if args.dense_index:
            inputs.extend((args.dense_index / INDEX_MANIFEST_NAME, args.dense_index / "vectors.jsonl"))
        if args.checkpoint:
            inputs.extend(path for path in args.checkpoint.rglob("*") if path.is_file())
        for directory in (args.checkpoint, args.dense_index):
            if directory is not None and any(path.resolve().is_relative_to(directory.resolve()) for path in outputs):
                raise GateError("gate/report outputs must not be inside frozen checkpoint or dense-index directories")
        if len({path.resolve() for path in outputs}) != len(outputs) or {path.resolve() for path in outputs} & {path.resolve() for path in inputs if path is not None}:
            raise GateError("gate/report outputs must be distinct and must not overwrite input artifacts")
        if not args.force and any(path.exists() for path in outputs):
            raise GateError("refusing to overwrite existing gate/report artifacts; pass --force")
        if (args.temperature_candidates or args.l2_candidates) and not args.validation:
            raise GateError("hyperparameter candidate selection requires --validation")
        dense = None
        if args.dense_index:
            dense = DenseIndex.load(args.dense_index, _encoder_from_arguments(args), expected_config=DenseConfig(max_length=args.max_length, batch_size=args.batch_size))
        pipeline = RetrievalPipeline(BM25Index(load_bm25_corpus(args.corpus)), CharTfidfIndex(load_char_corpus(args.corpus)), dense, TokenLanguageIdentifier.load_json(args.lid_model) if args.lid_model else None, LexiconTransliterator.from_jsonl(args.lexicon) if args.lexicon else None)
        records, qrels = list(iter_jsonl(args.queries)), load_qrels(args.qrels)
        examples = collect_examples(pipeline, records, qrels)
        validation = collect_examples(pipeline, records, qrels, split=VALIDATION) if args.validation else ()
        candidates = []
        for temperature, l2 in itertools.product(args.temperature_candidates or [args.temperature], args.l2_candidates or [args.l2]):
            result = train_gate(examples, pipeline_manifest=pipeline.manifest, config=GateConfig(temperature, args.epochs, args.learning_rate, l2), validation_examples=validation, excluded_features=args.exclude_features)
            candidates.append(result)
        if validation and not any(result.report["validation"]["gate"]["query_count"] for result in candidates):
            raise GateError("validation-based selection requires positive validation judgments")
        best = max(candidates, key=lambda result: (result.report["validation"]["gate"]["language_macro_ndcg_at_10"] if validation else -result.report["final_loss"]))
        report = dict(best.report)
        report["hyperparameter_selection"] = {"split": VALIDATION if validation else None, "metric": "language_macro_actual_fused_ndcg_at_10" if validation else None, "candidates": [{"config": result.report["config"], "excluded_features": result.report["excluded_features"], "validation_ndcg_at_10": result.report["validation"]["gate"]["ndcg_at_10"] if validation else None, "validation_language_macro_ndcg_at_10": result.report["validation"]["gate"]["language_macro_ndcg_at_10"] if validation else None} for result in candidates]}
        report["input_sha256"] = {str(path): _file_sha256(path) for path in inputs if path is not None}
        gate = LinearSoftmaxGate(best.gate.weights, best.gate.bias, best.gate.feature_mean, best.gate.feature_scale, pipeline_manifest=pipeline.manifest, training_report=report)
        gate.save_json(args.output, overwrite=args.force)
        _atomic_write_json(report_path, report)
    except (GateError, FusionError, BM25Error, CharTfidfError, DenseError, QueryViewError, LanguageIdError, EvaluationError, CorpusPreparationError, OSError, TypeError, ValueError) as exc:
        print(f"train_gate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"gate": str(args.output), "report": str(report_path), "fixture_only": pipeline.manifest["fixture_only"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
