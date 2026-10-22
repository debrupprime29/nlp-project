"""Independent numeric, leakage, artifact, and CLI checks for adaptive fusion."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from data.generate_variants import generate_variants
from data.make_splits import (
    ENCODER_TRAIN, FINAL_EVALUATION, FUSION_TRAIN, OFFICIAL_DEV,
    OFFICIAL_TRAIN, VALIDATION, canonical_query,
)
from data.prepare_corpus import Qrel
from nlp.language_id import ScriptEvidenceBaseline
from nlp.query_views import build_query_views
from retrieval.bm25 import BM25Document, BM25Index
from retrieval.char_tfidf import CharDocument, CharTfidfIndex
from retrieval.fusion import BRANCHES, BranchHit, FusionError, RetrievalPipeline
from training.train_gate import (
    FEATURE_NAMES, GateConfig, GateError, GateExample, LinearSoftmaxGate,
    collect_examples, extract_features, train_gate, utility_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def ranking(*doc_ids, scores=None):
    scores = scores if scores is not None else [1.0 / index for index in range(1, len(doc_ids) + 1)]
    return tuple(BranchHit(doc_id, rank, score) for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), 1))


def branches(original=(), native=(), character=(), dense=()):
    return dict(zip(BRANCHES, (original, native, character, dense)))


def original(query_id="q-fault", text="earthquake faults", *, split=FUSION_TRAIN, partition=OFFICIAL_TRAIN, group=None):
    query = canonical_query({"query_id": query_id, "text": text, "languages": ["en"], "group_id": group or query_id}, source_partition=partition)
    return asdict(replace(query, split=split))


def fixture_pipeline():
    documents = (
        ("d-fault", "earthquake faults", "earthquake faults"),
        ("d-rain", "rain clouds water", "rain clouds water"),
        ("d-hi", "भूकंप क्यों आते हैं", "bhukamp kyu aate hain"),
    )
    lexical = BM25Index([BM25Document(doc_id, "", text, "en", "gate-fixture") for doc_id, text, _ in documents])
    character = CharTfidfIndex([CharDocument(doc_id, roman, "en", "gate-fixture", "manual_fixture") for doc_id, _, roman in documents])
    return RetrievalPipeline(lexical, character)


def learnable_examples():
    examples = []
    for index, text in enumerate(("earthquake faults", "rain clouds", "भूकंप", "बारिश")):
        # Two script-dependent branch preferences, identical retrieval shapes.
        relevant = "d-left" if index < 2 else "d-right"
        examples.append(GateExample(
            query_id=f"q-{index}", parent_query_id=f"q-{index}", group_id=f"g-{index}", split=FUSION_TRAIN,
            views=build_query_views(text),
            branch_results=branches(ranking("d-left", "d-right"), character=ranking("d-right", "d-left")),
            judgments={relevant: 1},
        ))
    return tuple(examples)


def configuration(**changes):
    return replace(GateConfig(), **{"epochs": 180, "learning_rate": 0.1, "temperature": 0.05, "l2": 0.0, **changes})


def write_records(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


class GateUtilityTests(unittest.TestCase):
    def test_graded_ndcg_targets_match_hand_calculation_and_mask_inactive_branches(self):
        rankings = branches(ranking("high", "low"), character=ranking("low", "high"))
        temperature = 0.25
        target = utility_targets(rankings, {"high": 2, "low": 1}, temperature=temperature)
        ideal = 3.0 + 1.0 / math.log2(3)
        inferior = (1.0 + 3.0 / math.log2(3)) / ideal
        expected_original = 1 / (1 + math.exp((inferior - 1.0) / temperature))
        self.assertAlmostEqual(target["bm25_original"], expected_original)
        self.assertAlmostEqual(target["char_tfidf"], 1 - expected_original)
        self.assertEqual(target["bm25_native"], 0)
        self.assertEqual(target["dense"], 0)
        self.assertAlmostEqual(sum(target.values()), 1)

    def test_targets_use_ndcg_at_ten_and_record_no_preference_as_none(self):
        beyond_ten = ranking(*(f"irrelevant-{index}" for index in range(10)), "relevant")
        for rankings, judgments in (
            (branches(beyond_ten), {"relevant": 1}),
            (branches(ranking("nonrelevant")), {"relevant": 2}),
            (branches(), {"relevant": 1}),
            (branches(ranking("nonrelevant")), {"nonrelevant": 0}),
        ):
            with self.subTest(rankings=rankings, judgments=judgments):
                self.assertIsNone(utility_targets(rankings, judgments))

    def test_utility_softmax_is_stable_at_tiny_temperature(self):
        target = utility_targets(branches(ranking("yes"), dense=ranking("no")), {"yes": 1}, temperature=1e-300)
        self.assertEqual(target["bm25_original"], 1)
        self.assertLess(target["dense"], 1e-300)
        self.assertTrue(all(math.isfinite(value) for value in target.values()))

    def test_large_graded_relevance_does_not_overflow_ndcg_targets(self):
        rankings = branches(ranking("high", "low"), character=ranking("low", "high"))
        ordinary = utility_targets(rankings, {"high": 2, "low": 2})
        extreme = utility_targets(rankings, {"high": 1023, "low": 1023})
        self.assertEqual(extreme, ordinary)
        self.assertEqual(extreme["bm25_original"], 0.5)
        self.assertTrue(all(math.isfinite(value) for value in extreme.values()))

    def test_invalid_targets_and_training_settings_fail_explicitly(self):
        for temperature in (0, -1, True, math.nan, math.inf):
            with self.subTest(temperature=temperature), self.assertRaises(GateError):
                utility_targets(branches(ranking("yes")), {"yes": 1}, temperature=temperature)
        for judgments in ({"yes": -1}, {"yes": True}, {"yes": math.nan}):
            with self.subTest(judgments=judgments), self.assertRaises(GateError):
                utility_targets(branches(ranking("yes")), judgments)
        for changes in ({"epochs": 0}, {"epochs": True}, {"learning_rate": 0}, {"learning_rate": math.inf}, {"l2": -1}, {"l2": math.nan}, {"temperature": 0}):
            with self.subTest(changes=changes), self.assertRaises(GateError):
                configuration(**changes).validate()

    def test_malformed_rankings_are_rejected_for_both_features_and_targets(self):
        malformed = (
            branches((BranchHit("a", 2, 1.0),)),
            branches((BranchHit("a", 1, math.nan),)),
            branches((BranchHit("a", 1, True),)),
            branches((BranchHit("a", 1, 1.0), BranchHit("a", 2, 0.5))),
            {**branches(ranking("a")), "unknown_branch": ranking("a")},
        )
        for rankings in malformed:
            with self.subTest(rankings=rankings), self.assertRaises(GateError):
                utility_targets(rankings, {"a": 1})
            with self.subTest(features=rankings), self.assertRaises(GateError):
                extract_features(build_query_views("earthquake"), rankings)


class GateFeatureTests(unittest.TestCase):
    def test_schema_is_deterministic_finite_and_query_labels_are_not_inputs(self):
        views = build_query_views("hello भूकंप ভূমিকম্প 2026")
        rankings = branches(ranking("a", "b", scores=[1e308, -1e308]), character=ranking("b", "a"))
        features = extract_features(views, rankings)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features, extract_features(views, dict(reversed(tuple(rankings.items())))))
        self.assertEqual(len(features), len(set(FEATURE_NAMES)))
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertFalse(any("relevance" in name or "utility" in name or "judgment" in name for name in FEATURE_NAMES))

    def test_empty_query_and_empty_rankings_have_valid_fixed_width_features(self):
        features = extract_features(build_query_views(""), branches())
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))

    def test_protected_attempts_are_observable_and_nonconstant(self):
        views = build_query_views("NASA 2026")
        altered = replace(views, diagnostics={**views.diagnostics, "protected_attempted_rewrite_count": 3})
        base = extract_features(views, branches())
        changed = extract_features(altered, branches())
        different = {name for name in FEATURE_NAMES if base[name] != changed[name]}
        self.assertTrue(different)
        self.assertTrue(any("attempt" in name for name in different))

    def test_agreement_features_change_when_branch_overlap_changes(self):
        views = build_query_views("earthquake")
        shared = extract_features(views, branches(ranking("a", "b"), character=ranking("a", "b")))
        disjoint = extract_features(views, branches(ranking("a", "b"), character=ranking("c", "d")))
        different = {name for name in FEATURE_NAMES if shared[name] != disjoint[name]}
        self.assertTrue(any("overlap" in name or "agreement" in name for name in different))


class GateLearningTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = fixture_pipeline()
        self.examples = learnable_examples()

    def test_real_linear_learning_reduces_cross_entropy_and_tracks_query_preferences(self):
        result = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration())
        self.assertLess(result.report["final_loss"], result.report["initial_loss"] - 0.05)
        for example in self.examples:
            prediction = result.gate(example.views, example.branch_results)
            preferred = "bm25_original" if example.query_id in {"q-0", "q-1"} else "char_tfidf"
            self.assertGreater(prediction[preferred], 0.7)
            self.assertEqual(prediction["dense"], 0)
            self.assertEqual(prediction["bm25_native"], 0)
            self.assertAlmostEqual(sum(prediction.values()), 1)

    def test_training_is_reproducible_under_input_order_changes(self):
        first = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration())
        second = train_gate(tuple(reversed(self.examples)), pipeline_manifest=self.pipeline.manifest, config=configuration())
        for example in self.examples:
            self.assertEqual(first.gate(example.views, example.branch_results), second.gate(example.views, example.branch_results))
        self.assertEqual(first.report["initial_loss"], second.report["initial_loss"])
        self.assertEqual(first.report["final_loss"], second.report["final_loss"])

    def test_no_utility_examples_are_excluded_but_all_unusable_training_fails(self):
        unused = replace(self.examples[0], query_id="q-unused", parent_query_id="q-unused", group_id="g-unused", judgments={"never-retrieved": 1})
        result = train_gate((*self.examples, unused), pipeline_manifest=self.pipeline.manifest, config=configuration())
        self.assertEqual(result.report["excluded_all_zero_utility_count"], 1)
        baseline = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration())
        self.assertEqual(result.gate.weights, baseline.gate.weights)
        self.assertEqual(result.gate.feature_mean, baseline.gate.feature_mean)
        with self.assertRaises(GateError):
            train_gate([unused], pipeline_manifest=self.pipeline.manifest, config=configuration())

    def test_scaler_is_fitted_exactly_to_retained_training_observations(self):
        result = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration())
        rows = [extract_features(example.views, example.branch_results) for example in self.examples]
        for column, name in enumerate(FEATURE_NAMES):
            expected_mean = math.fsum(row[name] for row in rows) / len(rows)
            variance = math.fsum((row[name] - expected_mean) ** 2 for row in rows) / len(rows)
            expected_scale = math.sqrt(variance) if variance > 0 else 1.0
            self.assertAlmostEqual(result.gate.feature_mean[column], expected_mean)
            self.assertAlmostEqual(result.gate.feature_scale[column], expected_scale)
        self.assertEqual(result.report["scaler_fit_split"], FUSION_TRAIN)

    def test_inference_masks_newly_available_and_missing_branches_without_qrels(self):
        gate = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration()).gate
        views = build_query_views("brand new unseen question")
        for name in BRANCHES:
            values = branches()
            values[name] = ranking("unseen-document")
            prediction = gate(views, values)
            self.assertEqual(prediction[name], 1.0)
            self.assertEqual(sum(prediction.values()), 1.0)
            self.assertTrue(all(value == 0 for branch, value in prediction.items() if branch != name))
        empty = gate(views, branches())
        self.assertTrue(not empty or all(value == 0 for value in empty.values()))

    def test_training_rejects_wrong_splits_and_group_overlap_with_validation(self):
        for split in (ENCODER_TRAIN, VALIDATION, FINAL_EVALUATION):
            with self.subTest(split=split), self.assertRaises(GateError):
                train_gate([replace(self.examples[0], split=split)], pipeline_manifest=self.pipeline.manifest, config=configuration())
        heldout = replace(self.examples[0], query_id="q-validation", parent_query_id="q-validation", split=VALIDATION)
        with self.assertRaises(GateError):
            train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration(), validation_examples=[heldout])

    def test_validation_features_and_labels_cannot_change_the_trained_gate(self):
        heldout = replace(self.examples[0], query_id="q-validation", parent_query_id="q-validation", group_id="g-validation", split=VALIDATION)
        changed = replace(heldout, views=build_query_views("x " * 500), judgments={"d-right": 3})
        first = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration(), validation_examples=[heldout])
        second = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration(), validation_examples=[changed])
        self.assertEqual(first.gate.weights, second.gate.weights)
        self.assertEqual(first.gate.bias, second.gate.bias)
        self.assertEqual(first.gate.feature_mean, second.gate.feature_mean)
        self.assertEqual(first.gate.feature_scale, second.gate.feature_scale)
        for example in self.examples:
            self.assertEqual(first.gate(example.views, example.branch_results), second.gate(example.views, example.branch_results))


class GateFeatureAblationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = fixture_pipeline()
        self.examples = learnable_examples()
        self.excluded = ("mean_token_entropy", "uncertain_token_fraction")
        self.config = configuration(epochs=12)

    def fit(self, excluded=...):
        return train_gate(self.examples, pipeline_manifest=self.pipeline.manifest,
                          config=self.config,
                          excluded_features=self.excluded if excluded is ... else excluded)

    def test_retrained_ablation_has_zero_learned_columns_and_zero_one_scaler(self):
        result = self.fit(tuple(reversed(self.excluded)))
        self.assertEqual(result.report["excluded_features"], list(self.excluded))
        self.assertEqual(result.gate.excluded_features, self.excluded)
        self.assertIn("before_scaler_and_training", result.report["feature_ablation_policy"])
        for name in self.excluded:
            column = FEATURE_NAMES.index(name)
            self.assertEqual(result.gate.feature_mean[column], 0.0)
            self.assertEqual(result.gate.feature_scale[column], 1.0)
            self.assertTrue(all(row[column] == 0.0 for row in result.gate.weights))
        self.assertLess(result.report["final_loss"], result.report["initial_loss"])

    def test_masked_raw_features_cannot_change_training_weights_scaler_or_fingerprint(self):
        first = self.fit()

        def changed(views, results):
            values = extract_features(views, results)
            return {**values, "mean_token_entropy": 1e200,
                    "uncertain_token_fraction": 0.99 if views.processing_text.isascii() else 0.01}

        with patch("training.train_gate.extract_features", side_effect=changed):
            second = self.fit()
        self.assertEqual(first.gate.weights, second.gate.weights)
        self.assertEqual(first.gate.bias, second.gate.bias)
        self.assertEqual(first.gate.feature_mean, second.gate.feature_mean)
        self.assertEqual(first.gate.feature_scale, second.gate.feature_scale)
        self.assertEqual(first.report, second.report)

    def test_inference_zeroes_excluded_standardized_features_regardless_of_input(self):
        column = FEATURE_NAMES.index("mean_token_entropy")
        weights = [[0.0] * len(FEATURE_NAMES) for _ in BRANCHES]
        weights[0][column] = 3.0
        weights[2][column] = -3.0
        means = [0.0] * len(FEATURE_NAMES)
        means[column] = 2.0
        scales = [1.0] * len(FEATURE_NAMES)
        scales[column] = 0.1
        # Deliberately nonzero numeric columns prove the runtime mask itself,
        # separately from the zero columns produced by the retraining test.
        gate = LinearSoftmaxGate(weights, [0.0] * len(BRANCHES), means, scales,
                                 pipeline_manifest=self.pipeline.manifest,
                                 training_report={"excluded_features": ["mean_token_entropy"]})
        example = self.examples[0]
        baseline = gate(example.views, example.branch_results)
        features = extract_features(example.views, example.branch_results)
        with patch("training.train_gate.extract_features", return_value={**features, "mean_token_entropy": 1e300}):
            changed = gate(example.views, example.branch_results)
        self.assertEqual(baseline, changed)
        self.assertEqual(baseline["bm25_original"], 0.5)
        self.assertEqual(baseline["char_tfidf"], 0.5)

    def test_all_feature_exclusion_is_a_finite_retrained_bias_only_gate(self):
        result = self.fit(FEATURE_NAMES)
        self.assertEqual(result.gate.feature_mean, (0.0,) * len(FEATURE_NAMES))
        self.assertEqual(result.gate.feature_scale, (1.0,) * len(FEATURE_NAMES))
        self.assertTrue(all(value == 0 for row in result.gate.weights for value in row))
        self.assertTrue(all(math.isfinite(value) for value in result.report["loss_history"]))
        for example in self.examples:
            self.assertAlmostEqual(sum(result.gate(example.views, example.branch_results).values()), 1.0)

    def test_invalid_exclusion_api_and_training_report_values_are_rejected(self):
        for values in (None, True, False, 1, "mean_token_entropy", b"mean_token_entropy", {},
                       {"mean_token_entropy"}, [True], [1], [""], ["unknown"],
                       ["mean_token_entropy", "mean_token_entropy"]):
            with self.subTest(values=values), self.assertRaises(GateError):
                self.fit(values)
            with self.subTest(report=values), self.assertRaises(GateError):
                LinearSoftmaxGate([[0.0] * len(FEATURE_NAMES) for _ in BRANCHES],
                                  [0.0] * len(BRANCHES), [0.0] * len(FEATURE_NAMES),
                                  [1.0] * len(FEATURE_NAMES), pipeline_manifest=self.pipeline.manifest,
                                  training_report={"excluded_features": values})

    def test_exclusion_roundtrip_preserves_frozen_compatibility_and_content_metadata(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            result.gate.save_json(path)
            restored = LinearSoftmaxGate.load_json(path, expected_pipeline_manifest=self.pipeline.manifest)
            self.assertEqual(restored.excluded_features, self.excluded)
            self.assertEqual(restored.metadata["excluded_features"], list(self.excluded))
            self.assertEqual(result.gate.metadata, restored.metadata)
            for example in self.examples:
                self.assertEqual(result.gate(example.views, example.branch_results), restored(example.views, example.branch_results))
            payload = json.loads(path.read_text())
            self.assertEqual(set(payload), {"schema_version", "model_type", "branches", "feature_names", "weights",
                                           "bias", "feature_mean", "feature_scale", "frozen_pipeline",
                                           "frozen_pipeline_fingerprint", "training_report", "artifact_sha256"})

    def test_invalid_checksummed_and_unchecksummed_report_masks_cannot_load(self):
        result = self.fit()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            result.gate.save_json(path)
            original_payload = json.loads(path.read_text())
            for exclusions in (["unknown"], [True], "mean_token_entropy", ["mean_token_entropy"] * 2, None):
                payload = copy.deepcopy(original_payload)
                payload["training_report"]["excluded_features"] = exclusions
                for reseal in (False, True):
                    if reseal:
                        payload["artifact_sha256"] = hashlib.sha256(json.dumps(
                            {key: value for key, value in payload.items() if key != "artifact_sha256"},
                            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.subTest(exclusions=exclusions, reseal=reseal), self.assertRaises(GateError):
                        LinearSoftmaxGate.load_json(path)

    def test_legacy_report_without_exclusions_retains_payload_identity(self):
        result = self.fit(())
        report = dict(result.report)
        report.pop("excluded_features")
        gate = LinearSoftmaxGate(result.gate.weights, result.gate.bias,
                                 result.gate.feature_mean, result.gate.feature_scale,
                                 pipeline_manifest=self.pipeline.manifest, training_report=report)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-gate.json"
            gate.save_json(path)
            payload = json.loads(path.read_text())
            restored = LinearSoftmaxGate.load_json(path)
            self.assertEqual(restored.excluded_features, ())
            self.assertNotIn("excluded_features", restored.training_report)
            self.assertEqual(restored.metadata["artifact_sha256"], payload["artifact_sha256"])

    def test_public_training_report_is_a_copy_and_cannot_change_mask_or_checksum(self):
        result = self.fit()
        before = result.gate.metadata
        report = result.gate.training_report
        report["excluded_features"].clear()
        report["loss_history"][0] = -100
        self.assertEqual(result.gate.excluded_features, self.excluded)
        self.assertEqual(result.gate.metadata, before)


class GateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = fixture_pipeline()
        self.records = [original(), original("q-rain", "rain clouds")]
        self.qrels = [Qrel("q-fault", "d-fault", 1), Qrel("q-rain", "d-rain", 1)]

    def test_collects_only_requested_split_and_ignores_heldout_qrels(self):
        records = [*self.records, original("q-encoder", "training private", split=ENCODER_TRAIN), original("q-validation", "validation private", split=VALIDATION), original("q-final", "final private", split=FINAL_EVALUATION, partition=OFFICIAL_DEV)]
        first = collect_examples(self.pipeline, records, self.qrels)
        poisoned = [*self.qrels, {"query_id": "q-validation", "doc_id": "missing-private", "relevance": "invalid-but-out-of-scope"}]
        second = collect_examples(self.pipeline, records, poisoned)
        self.assertEqual({example.query_id for example in first}, {"q-fault", "q-rain"})
        self.assertEqual([(example.query_id, example.judgments) for example in first], [(example.query_id, example.judgments) for example in second])
        self.assertTrue(all(example.split == FUSION_TRAIN for example in first))

    def test_collect_rejects_cross_split_group_and_judgment_leakage(self):
        group_overlap = original("q-validation", "private", split=VALIDATION, group="q-fault")
        judgment_overlap = {**original("q-validation", "private", split=VALIDATION), "judgment_query_id": "q-fault"}
        for heldout in (group_overlap, judgment_overlap):
            with self.subTest(heldout=heldout), self.assertRaises(GateError):
                collect_examples(self.pipeline, [*self.records, heldout], self.qrels)

    def test_official_development_queries_cannot_be_relabeled_as_gate_training(self):
        forged = {**self.records[0], "source_partition": OFFICIAL_DEV}
        with self.assertRaises(GateError):
            collect_examples(self.pipeline, [forged, self.records[1]], self.qrels)

    def test_collector_features_use_the_same_view_generation_as_branch_rankings(self):
        identifier = ScriptEvidenceBaseline()
        pipeline = RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, language_identifier=identifier)
        with patch.object(identifier, "predict", wraps=identifier.predict) as predict:
            examples = collect_examples(pipeline, self.records, self.qrels)
        self.assertEqual(len(examples), 2)
        self.assertEqual(predict.call_count, 2, "Each query must produce exactly one set of views, shared by features and retrieval")

    def test_generated_variants_require_content_bound_reviews(self):
        queued = generate_variants(self.records, split=FUSION_TRAIN, seed=41)
        chosen = queued.review_queue[0]
        reviews = [{"candidate_id": chosen["candidate_id"], "candidate_sha256": chosen["candidate_sha256"], "parent_query_id": chosen["parent_query_id"], "reviewer_id": "independent-fixture-reviewer", "decision": "verified", "meaning_preserved": True}]
        reviewed = list(generate_variants(self.records, split=FUSION_TRAIN, seed=41, reviews=reviews).queries)
        examples = collect_examples(self.pipeline, reviewed, self.qrels)
        self.assertEqual(len(examples), 3)
        self.assertTrue(any(example.query_id != example.parent_query_id for example in examples))
        tampered = copy.deepcopy(reviewed)
        next(record for record in tampered if record["variant_type"] != "original")["text"] += " altered meaning"
        with self.assertRaises(GateError):
            collect_examples(self.pipeline, tampered, self.qrels)
        unreviewed = collect_examples(self.pipeline, [*self.records, *queued.review_queue], self.qrels)
        self.assertEqual(len(unreviewed), 2)

    def test_conflicting_or_foreign_training_judgments_are_rejected(self):
        for qrels in ([*self.qrels, Qrel("q-fault", "d-fault", 2)], [*self.qrels, Qrel("q-fault", "not-in-corpus", 1)]):
            with self.subTest(qrels=qrels), self.assertRaises(GateError):
                collect_examples(self.pipeline, self.records, qrels)

    def test_missing_positive_judgments_have_an_explicit_exclusion_policy(self):
        examples = collect_examples(self.pipeline, self.records, [self.qrels[0]])
        result = train_gate(examples, pipeline_manifest=self.pipeline.manifest, config=configuration())
        self.assertEqual(result.report["excluded_no_positive_judgments_count"], 1)
        self.assertEqual(result.report["training_query_ids"], ["q-fault"])


class GateArtifactTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = fixture_pipeline()
        self.examples = learnable_examples()
        self.gate = train_gate(self.examples, pipeline_manifest=self.pipeline.manifest, config=configuration()).gate

    def test_json_roundtrip_deterministic_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            self.gate.save_json(path)
            before = path.read_bytes()
            restored = LinearSoftmaxGate.load_json(path, expected_pipeline_manifest=self.pipeline.manifest)
            restored.assert_compatible(self.pipeline)
            for example in self.examples:
                self.assertEqual(self.gate(example.views, example.branch_results), restored(example.views, example.branch_results))
            with self.assertRaises(GateError):
                self.gate.save_json(path)
            self.assertEqual(path.read_bytes(), before)
            self.gate.save_json(path, overwrite=True)
            self.assertEqual(path.read_bytes(), before)

    def test_tampered_and_nonfinite_artifacts_cannot_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            self.gate.save_json(path)
            original_payload = json.loads(path.read_text(encoding="utf-8"))
            mutations = []
            # Altering any existing scalar must either break the content seal or
            # fail schema validation; a valid artifact has no label-time input.
            for key, value in original_payload.items():
                if isinstance(value, str):
                    mutations.append({**original_payload, key: value + "tampered"})
            self.assertTrue(mutations)
            mutations.append({**original_payload, "unknown_schema_field": True})
            for payload in mutations:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(GateError):
                    LinearSoftmaxGate.load_json(path)

    def test_changed_corpus_or_retrieval_settings_are_incompatible(self):
        changed_config = RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, fusion_config=replace(self.pipeline.config, k=61))
        altered = BM25Index([BM25Document("d-fault", "", "entirely changed", "en", "gate-fixture")])
        for pipeline in (changed_config, RetrievalPipeline(altered)):
            with self.subTest(pipeline=pipeline), self.assertRaises(GateError):
                self.gate.assert_compatible(pipeline)

    def test_correctly_checksummed_but_invalid_parameters_and_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            self.gate.save_json(path)
            original_payload = json.loads(path.read_text(encoding="utf-8"))
            for mutation in ("boolean_schema_version", "zero_scale", "negative_scale", "boolean_weight", "string_weight", "wrong_shape", "branch_order", "feature_order", "frozen_resource"):
                payload = copy.deepcopy(original_payload)
                payload.pop("artifact_sha256")
                if mutation == "boolean_schema_version":
                    payload["schema_version"] = True
                elif mutation == "zero_scale":
                    payload["feature_scale"][0] = 0
                elif mutation == "negative_scale":
                    payload["feature_scale"][0] = -1
                elif mutation == "boolean_weight":
                    payload["weights"][0][0] = True
                elif mutation == "string_weight":
                    payload["weights"][0][0] = "NaN"
                elif mutation == "wrong_shape":
                    payload["weights"][0].pop()
                elif mutation == "branch_order":
                    payload["branches"].reverse()
                elif mutation == "feature_order":
                    payload["feature_names"].reverse()
                else:
                    payload["frozen_pipeline"]["resources"] = {"tampered": True}
                checksum = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                path.write_text(json.dumps({**payload, "artifact_sha256": checksum}), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(GateError):
                    LinearSoftmaxGate.load_json(path)

    def test_nonfinite_numeric_parameters_are_rejected_even_before_serialization(self):
        for value in (math.nan, math.inf, -math.inf):
            weights = [list(row) for row in self.gate.weights]
            weights[0][0] = value
            with self.subTest(value=value), self.assertRaises(GateError):
                LinearSoftmaxGate(weights, self.gate.bias, self.gate.feature_mean, self.gate.feature_scale, pipeline_manifest=self.pipeline.manifest, training_report=self.gate.training_report)

    def test_integrated_pipeline_checks_gate_compatibility_at_construction_and_search(self):
        with self.assertRaises((GateError, FusionError)):
            RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, fusion_config=replace(self.pipeline.config, k=61), weight_provider=self.gate)
        pipeline = RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, weight_provider=self.gate)
        self.assertTrue(pipeline.search("earthquake faults").hits)
        pipeline.config = replace(pipeline.config, k=61)
        with self.assertRaises((GateError, FusionError)):
            pipeline.search("earthquake faults")
        pipeline = RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, weight_provider=self.gate)
        pipeline.bm25_index = BM25Index([BM25Document("d-fault", "", "changed corpus text", "en", "gate-fixture")])
        with self.assertRaises((GateError, FusionError)):
            pipeline.search("earthquake faults")

    def test_duplicate_json_fields_nonfinite_json_and_symlink_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gate.json"
            self.gate.save_json(path)
            serialized = path.read_text(encoding="utf-8")
            linked = root / "link.json"
            linked.symlink_to(path)
            with self.assertRaises(GateError):
                LinearSoftmaxGate.load_json(linked)
            with self.assertRaises(GateError):
                self.gate.save_json(linked, overwrite=True)
            self.assertEqual(path.read_text(encoding="utf-8"), serialized)
            for malformed in ('{"schema_version": 1, "schema_version": 1}', '{"weights": NaN}'):
                path.write_text(malformed, encoding="utf-8")
                with self.subTest(malformed=malformed), self.assertRaises(GateError):
                    LinearSoftmaxGate.load_json(path)


class GateCliTests(unittest.TestCase):
    def fixture(self, root):
        corpus, queries, qrels = root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.jsonl"
        write_records(corpus, [
            {"doc_id": "d-fault", "title": "", "text": "earthquake faults", "romanized_text": "earthquake faults", "romanization_status": "manual_fixture", "language": "en", "source": "gate-fixture"},
            {"doc_id": "d-rain", "title": "", "text": "rain clouds", "romanized_text": "rain clouds", "romanization_status": "manual_fixture", "language": "en", "source": "gate-fixture"},
        ])
        write_records(queries, [original(), original("q-rain", "rain clouds"), original("q-validation", "earthquake", split=VALIDATION)])
        write_records(qrels, [{"query_id": "q-fault", "doc_id": "d-fault", "relevance": 1}, {"query_id": "q-rain", "doc_id": "d-rain", "relevance": 1}, {"query_id": "q-validation", "doc_id": "d-fault", "relevance": 1}])
        return [sys.executable, "-m", "training.train_gate", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels), "--epochs", "5"], corpus

    def test_cli_trains_saves_loads_and_protects_existing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, corpus = self.fixture(root)
            output = root / "gate.json"
            completed = subprocess.run([*command, "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            gate = LinearSoftmaxGate.load_json(output)
            self.assertTrue(callable(gate))
            before = output.read_bytes()
            denied = subprocess.run([*command, "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            original_corpus = corpus.read_bytes()
            denied_input = subprocess.run([*command, "--output", str(corpus), "--force"], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(denied_input.returncode, 0)
            self.assertEqual(corpus.read_bytes(), original_corpus)

    def test_cli_force_cannot_write_inside_frozen_encoder_or_dense_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _ = self.fixture(root)
            for option, name in (("--checkpoint", "checkpoint"), ("--dense-index", "dense")):
                directory = root / name
                directory.mkdir()
                sentinel = directory / "config.json"
                sentinel.write_text('{"preserve":true}', encoding="utf-8")
                for output in (sentinel, directory / "new-gate.json"):
                    denied = subprocess.run([*command, option, str(directory), "--output", str(output), "--force"], cwd=ROOT, capture_output=True, text=True)
                    self.assertNotEqual(denied.returncode, 0)
                    self.assertIn("inside frozen", denied.stderr.lower())
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"preserve":true}')
                    self.assertFalse((directory / "new-gate.json").exists())


if __name__ == "__main__":
    unittest.main()
