"""Leakage checks plus optional, actual Torch training on explicit fixtures."""

from __future__ import annotations

import copy
import importlib.util
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
from data.make_splits import ENCODER_TRAIN, FINAL_EVALUATION, FUSION_TRAIN, OFFICIAL_DEV, OFFICIAL_TRAIN, VALIDATION, canonical_query
from data.prepare_corpus import Qrel
from evaluation.evaluate import load_run
from retrieval.dense import DenseCompatibilityError, DenseConfig, DenseDocument, DenseError, DenseIndex, DenseQuery, TrainedCheckpointEncoder, format_passage, format_query, run_queries
from training.checkpoints import CheckpointError, verify_checkpoint, write_checkpoint_manifest
from training.train_encoder import (
    EncoderTrainingConfig,
    EncoderTrainingError,
    TorchFixtureTrainingEncoder,
    build_training_batch_masks,
    consistency_loss,
    multi_positive_retrieval_loss,
    prepare_encoder_training,
    save_trained_checkpoint,
    train_encoder,
    _parameter_delta_l2,
    PARAMETER_DELTA_CHUNK_ELEMENTS,
)

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def original(query_id="q-earthquake", text="earthquake causes", *, split=ENCODER_TRAIN, partition=OFFICIAL_TRAIN, group=None):
    query = canonical_query({"query_id": query_id, "text": text, "languages": ["en"], "group_id": group or query_id}, source_partition=partition)
    return asdict(replace(query, split=split))


def fixture_records(*, reviewed=True):
    parents = [original(), original("q-rain", "rainfall clouds")]
    generated = generate_variants(parents, split=ENCODER_TRAIN, seed=31)
    if not reviewed:
        return [*parents, *generated.review_queue]
    chosen = next(candidate for candidate in generated.review_queue if candidate["parent_query_id"] == "q-earthquake")
    reviews = [{"candidate_id": chosen["candidate_id"], "candidate_sha256": chosen["candidate_sha256"], "parent_query_id": chosen["parent_query_id"], "reviewer_id": "independent-fixture-reviewer", "decision": "verified", "meaning_preserved": True}]
    return list(generate_variants(parents, split=ENCODER_TRAIN, seed=31, reviews=reviews).queries)


def fixture_documents():
    return [
        DenseDocument("d-earthquake", "", "earthquake faults causes", "en", "training-fixture"),
        DenseDocument("d-earthquake-also", "", "earthquake plate movement", "en", "training-fixture"),
        DenseDocument("d-rain", "", "rainfall clouds water", "en", "training-fixture"),
        DenseDocument("d-distractor", "", "earthquake rainfall unrelated history", "en", "training-fixture"),
    ]


def fixture_qrels():
    return [Qrel("q-earthquake", "d-earthquake", 1), Qrel("q-earthquake", "d-earthquake-also", 2), Qrel("q-earthquake", "d-rain", 0), Qrel("q-rain", "d-rain", 1)]


def fixture_config(**changes):
    settings = {"epochs": 1, "batch_size": 2, "random_negatives": 2, "bm25_negatives": 1, "temperature": 0.5, "seed": 31}
    settings.update(changes)
    return replace(EncoderTrainingConfig(), **settings)


def write_records(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


class EncoderDatasetTests(unittest.TestCase):
    def test_training_partition_only_and_heldout_qrels_do_not_change_mining(self):
        heldout = [original("q-validation", "earthquake hidden answer", split=VALIDATION), original("q-fusion", "rain hidden answer", split=FUSION_TRAIN), original("q-final", "final hidden answer", split=FINAL_EVALUATION, partition=OFFICIAL_DEV)]
        records = [*fixture_records(), *heldout]
        config = fixture_config()
        baseline = prepare_encoder_training(records, fixture_qrels(), fixture_documents(), config)
        altered = prepare_encoder_training(records, [*fixture_qrels(), Qrel("q-validation", "d-rain", 1), Qrel("q-fusion", "d-earthquake", 1), Qrel("q-final", "d-distractor", 3)], fixture_documents(), config)
        self.assertEqual(baseline.examples, altered.examples)
        self.assertEqual(baseline.manifest["training_fingerprint"], altered.manifest["training_fingerprint"])
        self.assertTrue(set(baseline.manifest["judgment_query_ids"]) <= {"q-earthquake", "q-rain"})
        self.assertTrue(all(example.original_query_id in {"q-earthquake", "q-rain"} for example in baseline.examples))
        self.assertEqual(baseline.manifest["negative_policy"]["mining_queries_split"], ENCODER_TRAIN)

    def test_known_positives_never_become_random_or_mined_negatives(self):
        config = fixture_config()
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        expected = {"q-earthquake": {"d-earthquake", "d-earthquake-also"}, "q-rain": {"d-rain"}}
        for example in data.examples:
            self.assertEqual(set(example.positive_doc_ids), expected[example.original_query_id])
            self.assertFalse(set(example.negative_doc_ids) & expected[example.original_query_id])
            self.assertEqual(len(example.negative_doc_ids), len(set(example.negative_doc_ids)))
        self.assertTrue(data.manifest["negative_policy"]["known_positives_excluded"])
        self.assertTrue(data.manifest["negative_policy"]["judged_nonrelevant_preferred"])

    def test_mining_is_reproducible_under_input_order_changes(self):
        records, qrels, documents, config = fixture_records(), fixture_qrels(), fixture_documents(), fixture_config()
        first = prepare_encoder_training(records, qrels, documents, config)
        second = prepare_encoder_training(list(reversed(records)), list(reversed(qrels)), list(reversed(documents)), config)
        self.assertEqual(first.examples, second.examples)
        self.assertEqual(first.manifest["training_fingerprint"], second.manifest["training_fingerprint"])

    def test_unreviewed_variants_cannot_inherit_labels_or_make_pairs(self):
        records = fixture_records(reviewed=False)
        data = prepare_encoder_training(records, fixture_qrels(), fixture_documents(), fixture_config())
        self.assertEqual(data.manifest["variant_pair_count"], 0)
        self.assertTrue(all(example.variant_query_id is None and example.variant_text is None for example in data.examples))
        self.assertEqual(set(data.manifest["training_query_ids"]), {"q-earthquake", "q-rain"})

    def test_verified_variants_are_content_bound_and_stale_reviews_are_rejected(self):
        original_records = fixture_records()
        variant = next(record for record in original_records if record["variant_type"] != "original")
        data = prepare_encoder_training(original_records, fixture_qrels(), fixture_documents(), fixture_config())
        self.assertEqual(data.manifest["variant_pair_count"], 1)
        paired = next(example for example in data.examples if example.variant_query_id)
        self.assertEqual(paired.variant_query_id, variant["query_id"])
        self.assertEqual(paired.original_text, "earthquake causes")
        self.assertEqual(paired.variant_text, variant["text"])
        for field in ("text", "candidate_sha256", "review"):
            records = copy.deepcopy(original_records)
            candidate = next(record for record in records if record["variant_type"] != "original")
            if field == "text":
                candidate["text"] += " unrelated"
            elif field == "candidate_sha256":
                candidate["candidate_sha256"] = "0" * 64
            else:
                candidate["review"]["candidate_sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(EncoderTrainingError):
                prepare_encoder_training(records, fixture_qrels(), fixture_documents(), fixture_config())

    def test_invalid_group_splits_and_missing_training_judgments_fail(self):
        cases = [
            [original(), original("q-related", "other question", split=VALIDATION, group="q-earthquake")],
            [original("q-heldout", "heldout only", split=VALIDATION)],
            [{**original(), "source_partition": OFFICIAL_DEV}],
        ]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(EncoderTrainingError):
                prepare_encoder_training(records, fixture_qrels(), fixture_documents(), fixture_config())
        with self.assertRaises(EncoderTrainingError):
            prepare_encoder_training(fixture_records(), [], fixture_documents(), fixture_config())

    def test_invalid_training_configuration_is_rejected_before_training(self):
        for settings in (
            {"epochs": 0}, {"batch_size": True}, {"max_length": 513},
            {"learning_rate": 0}, {"temperature": math.nan},
            {"consistency_weight": -1}, {"random_negatives": -1},
            {"in_batch_negatives": "yes"}, {"optimizer": "unrecognized"},
        ):
            with self.subTest(settings=settings), self.assertRaises(EncoderTrainingError):
                fixture_config(**settings).validate()

    def test_prepared_negatives_cannot_be_reported_under_different_mining_settings(self):
        config = fixture_config()
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        for settings in ({"random_negatives": config.random_negatives + 1}, {"bm25_negatives": config.bm25_negatives + 1}, {"seed": config.seed + 1}):
            with self.subTest(settings=settings), self.assertRaisesRegex(EncoderTrainingError, "negative-mining settings"):
                # A deliberately unusable encoder proves validation happens
                # before neural dependencies, parameters, or optimizer updates.
                train_encoder(data, object(), replace(config, **settings))


class CheckpointSafetyTests(unittest.TestCase):
    def sealed_fixture(self, directory):
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"integrity-only fixture bytes")
        (directory / "config.json").write_text('{"fixture":true}', encoding="utf-8")
        metadata = {"backend": "fixture_torch", "model_id": "fixture/trainable-torch", "init_revision": "fixture-v1", "dimension": 16, "fine_tuned": True, "fixture_only": True, "preprocessing": DenseConfig().compatibility_mapping(), "training": {"fixture_only": True, "seed": 31}}
        return write_checkpoint_manifest(directory, metadata)

    def test_seal_is_deterministic_content_addressed_and_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.sealed_fixture(root / "first")
            second = self.sealed_fixture(root / "second")
            self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
            self.assertEqual(verify_checkpoint(root / "first"), first)
            with self.assertRaises(CheckpointError):
                write_checkpoint_manifest(root / "first", first["metadata"])

    def test_changed_extra_missing_and_symlink_artifacts_are_rejected(self):
        for mutation in ("changed", "extra", "missing", "symlink", "pickle"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / "checkpoint"
                self.sealed_fixture(directory)
                if mutation == "changed":
                    (directory / "model.safetensors").write_bytes(b"modified bytes")
                elif mutation == "extra":
                    (directory / "extra.json").write_text("{}", encoding="utf-8")
                elif mutation == "missing":
                    (directory / "config.json").unlink()
                elif mutation == "symlink":
                    (directory / "linked.json").symlink_to(directory / "config.json")
                else:
                    (directory / "unsafe.pt").write_bytes(b"not a permissible weight format")
                with self.assertRaises(CheckpointError):
                    verify_checkpoint(directory)

    def test_manifest_path_traversal_and_backend_forgery_are_rejected(self):
        for mutation in ("traversal", "fixture_claims_e5", "hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "checkpoint"
                manifest = self.sealed_fixture(directory)
                if mutation == "traversal":
                    manifest["files"]["../outside.safetensors"] = "0" * 64
                elif mutation == "fixture_claims_e5":
                    manifest["metadata"]["fixture_only"] = False
                else:
                    manifest["checkpoint_sha256"] = "0" * 64
                (directory / "encoder_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(CheckpointError):
                    verify_checkpoint(directory)


@unittest.skipIf(torch is None, "optional Torch dependency is unavailable")
class EncoderNumericalTests(unittest.TestCase):
    def test_chunked_parameter_delta_matches_previous_formula_and_bounds_double_temporaries(self):
        parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, PARAMETER_DELTA_CHUNK_ELEMENTS + 19))
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(0.0125)
        expected = math.sqrt(float(((parameter.detach() - before).double() ** 2).sum()))
        original_to = torch.Tensor.to
        sizes = []
        def tracked_to(tensor, *args, **kwargs):
            if kwargs.get("dtype") == torch.float64:
                sizes.append(tensor.numel())
            return original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "to", new=tracked_to):
            actual = _parameter_delta_l2([parameter], [before])
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(sizes, [PARAMETER_DELTA_CHUNK_ELEMENTS, 19])
        self.assertIsNone(parameter.grad)

    def test_delta_chunks_handle_noncontiguous_multiple_and_empty_parameters(self):
        first = torch.arange(35, dtype=torch.float64).reshape(5, 7).transpose(0, 1)
        second = torch.arange(19, dtype=torch.float32)
        parameters = [first + 0.125, second + 0.2, torch.empty(0)]
        snapshots = [first, second, torch.empty(0)]
        expected = math.sqrt(sum(float(((parameter - before).double() ** 2).sum()) for parameter, before in zip(parameters, snapshots)))
        for chunk_elements in (1, 7, PARAMETER_DELTA_CHUNK_ELEMENTS):
            self.assertAlmostEqual(_parameter_delta_l2(parameters, snapshots, chunk_elements=chunk_elements), expected, places=12)
        for invalid in (0, True, PARAMETER_DELTA_CHUNK_ELEMENTS + 1):
            with self.subTest(invalid=invalid), self.assertRaises(EncoderTrainingError):
                _parameter_delta_l2(parameters, snapshots, chunk_elements=invalid)
        with self.assertRaises(EncoderTrainingError):
            _parameter_delta_l2(parameters, snapshots[:1])
        with self.assertRaises(EncoderTrainingError):
            _parameter_delta_l2([torch.ones(2)], [torch.ones(3)])

    def test_gradients_are_released_before_final_evaluation_without_changing_training_evidence(self):
        class RecordingEncoder(TorchFixtureTrainingEncoder):
            def __init__(self):
                super().__init__(16, 512, seed=31, device="cpu", allow_fixture=True)
                self.evaluation_grad_states = []
            def eval(self):
                self.evaluation_grad_states.append(all(parameter.grad is None for parameter in self.parameters()))
                return super().eval()
        config = fixture_config(epochs=2, learning_rate=0.05, consistency_weight=0.0)
        records = [record for record in fixture_records() if record["variant_type"] == "original"]
        data = prepare_encoder_training(records, fixture_qrels(), fixture_documents(), config)
        encoder = RecordingEncoder()
        result = train_encoder(data, encoder, config)
        self.assertEqual(encoder.evaluation_grad_states, [True, True])
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertEqual(result.manifest["variant_pair_count"], 0)
        self.assertFalse(result.manifest["lambda_zero_ablation"])
        self.assertFalse(result.manifest["consistency_applicable"])
        self.assertEqual(result.manifest["training_regime"], "original_only_retrieval")
        self.assertGreater(result.manifest["parameter_delta_l2"], 0)
        self.assertTrue(any(step["gradient_norm"] > 0 for step in result.history))

    def test_multi_positive_loss_matches_hand_calculation_and_has_finite_gradients(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        documents = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True)
        positives = torch.tensor([[True, True, False], [False, True, False]])
        candidates = torch.tensor([[True, True, True], [False, True, True]])
        loss = multi_positive_retrieval_loss(queries, documents, positives, candidate_mask=candidates, temperature=1.0)
        expected_first = math.log(math.e + 1 + math.exp(-1)) - math.log(math.e + 1)
        expected_second = math.log(math.e + 1) - 1
        self.assertAlmostEqual(float(loss.detach()), (expected_first + expected_second) / 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(documents.grad).all())

    def test_excluded_negatives_do_not_change_loss_or_receive_gradient(self):
        queries = torch.tensor([[1.0, 0.0]], requires_grad=True)
        documents = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]], requires_grad=True)
        positives = torch.tensor([[True, False, False]])
        candidates = torch.tensor([[True, True, False]])
        loss = multi_positive_retrieval_loss(queries, documents, positives, candidate_mask=candidates, temperature=1.0)
        changed = documents.detach().clone()
        changed[2] = torch.tensor([1.0, 0.0])
        comparison = multi_positive_retrieval_loss(queries, changed, positives, candidate_mask=candidates, temperature=1.0)
        self.assertAlmostEqual(float(loss.detach()), float(comparison.detach()), places=7)
        loss.backward()
        self.assertTrue(torch.equal(documents.grad[2], torch.zeros(2)))

    def test_invalid_retrieval_masks_and_temperatures_are_rejected(self):
        queries, documents = torch.eye(2), torch.eye(2)
        for positives, allowed, temperature in (
            (torch.zeros((2, 2), dtype=torch.bool), None, 1.0),
            (torch.eye(2, dtype=torch.bool), torch.tensor([[False, True], [True, True]]), 1.0),
            (torch.eye(2, dtype=torch.bool), None, 0.0),
            (torch.eye(2, dtype=torch.bool), None, math.nan),
            (torch.ones((1, 2), dtype=torch.bool), None, 1.0),
        ):
            with self.subTest(temperature=temperature), self.assertRaises(EncoderTrainingError):
                multi_positive_retrieval_loss(queries, documents, positives, candidate_mask=allowed, temperature=temperature)

    def test_batch_masks_include_all_positives_and_never_label_them_negative(self):
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), fixture_config())
        doc_ids = [document.doc_id for document in fixture_documents()]
        positive_mask, allowed_mask = build_training_batch_masks(data.examples, doc_ids, in_batch_negatives=True)
        self.assertEqual(positive_mask.dtype, torch.bool)
        for row, example in enumerate(data.examples):
            for column, doc_id in enumerate(doc_ids):
                self.assertEqual(bool(positive_mask[row, column]), doc_id in example.positive_doc_ids)
                if doc_id in example.positive_doc_ids:
                    self.assertTrue(bool(allowed_mask[row, column]))
        direct_positive, direct_allowed = build_training_batch_masks(data.examples, doc_ids, in_batch_negatives=False)
        self.assertTrue(torch.equal(direct_positive, positive_mask))
        for row, example in enumerate(data.examples):
            permitted = set(example.positive_doc_ids) | set(example.negative_doc_ids)
            self.assertEqual({doc_ids[column] for column in range(len(doc_ids)) if direct_allowed[row, column]}, permitted)

    def test_consistency_is_pair_aligned_permutation_invariant_and_masked(self):
        original_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
        variant_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, -1.0]], requires_grad=True)
        mask = torch.tensor([True, True, False])
        loss = consistency_loss(original_embeddings, variant_embeddings, pair_mask=mask)
        self.assertAlmostEqual(float(loss.detach()), 0.5, places=6)
        permutation = torch.tensor([2, 0, 1])
        permuted = consistency_loss(original_embeddings[permutation], variant_embeddings[permutation], pair_mask=mask[permutation])
        self.assertAlmostEqual(float(loss.detach()), float(permuted.detach()), places=7)
        loss.backward()
        self.assertTrue(torch.equal(variant_embeddings.grad[2], torch.zeros(2)))
        empty = consistency_loss(original_embeddings, variant_embeddings, pair_mask=torch.zeros(3, dtype=torch.bool))
        self.assertEqual(float(empty.detach()), 0.0)
        self.assertTrue(empty.requires_grad)
        with self.assertRaises(EncoderTrainingError):
            consistency_loss(original_embeddings, variant_embeddings[:2])

    def make_encoder(self, *, seed=31):
        return TorchFixtureTrainingEncoder(16, 512, seed=seed, device="cpu", allow_fixture=True)

    def test_tiny_neural_training_changes_parameters_and_reduces_loss(self):
        config = fixture_config(epochs=25, learning_rate=0.08, weight_decay=0.0)
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        encoder = self.make_encoder()
        before = [parameter.detach().clone() for parameter in encoder.parameters()]
        result = train_encoder(data, encoder, config)
        self.assertEqual(len(result.history), 25)
        self.assertEqual(result.manifest["optimizer_updates"], 25)
        self.assertGreater(result.manifest["parameter_delta_l2"], 0.01)
        self.assertLess(result.manifest["final_loss"], 0.9 * result.manifest["initial_loss"])
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, encoder.parameters(), strict=True)))
        self.assertTrue(all(math.isfinite(step[key]) for step in result.history for key in ("loss", "original_retrieval_loss", "variant_retrieval_loss", "consistency_loss", "gradient_norm")))
        self.assertTrue(any(step["gradient_norm"] > 0 for step in result.history))
        self.assertTrue(result.manifest["gradients_finite"])
        self.assertTrue(result.manifest["encoder"]["fixture_only"])
        self.assertEqual(result.manifest["training_data"]["judgment_query_ids"], ["q-earthquake", "q-rain"])
        self.assertEqual(result.manifest["evaluation_policy"], "training_objective_only_no_heldout_labels")

    def test_seeded_cpu_training_is_reproducible(self):
        config = fixture_config(epochs=4, learning_rate=0.05)
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        first_encoder, second_encoder = self.make_encoder(), self.make_encoder()
        first = train_encoder(data, first_encoder, config)
        second = train_encoder(data, second_encoder, config)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.manifest, second.manifest)
        for first_parameter, second_parameter in zip(first_encoder.parameters(), second_encoder.parameters(), strict=True):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_lambda_zero_keeps_variant_retrieval_and_removes_only_consistency(self):
        config = fixture_config(epochs=1, learning_rate=0.02, consistency_weight=0.0)
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        encoder = self.make_encoder()
        doc_by_id = {document.doc_id: document for document in data.documents}
        doc_ids = sorted({doc_id for example in data.examples for doc_id in (*example.positive_doc_ids, *example.negative_doc_ids)})
        with torch.no_grad():
            queries = encoder.encode_training([format_query(example.original_text) for example in data.examples], max_length=config.max_length)
            passages = encoder.encode_training([format_passage(doc_by_id[doc_id].title, doc_by_id[doc_id].text) for doc_id in doc_ids], max_length=config.max_length)
            positive, allowed = build_training_batch_masks(data.examples, doc_ids)
            original_loss = multi_positive_retrieval_loss(queries, passages, positive, allowed, temperature=config.temperature)
            paired_row = next(row for row, example in enumerate(data.examples) if example.variant_text is not None)
            variant = encoder.encode_training([format_query(data.examples[paired_row].variant_text)], max_length=config.max_length)
            variant_loss = multi_positive_retrieval_loss(variant, passages, positive[paired_row:paired_row + 1], allowed[paired_row:paired_row + 1], temperature=config.temperature)
            consistency = consistency_loss(queries[paired_row:paired_row + 1], variant)
            expected_zero = float(original_loss + variant_loss / len(data.examples))
            expected_weighted = expected_zero + 0.7 * float(consistency) / len(data.examples)
        zero = train_encoder(data, encoder, config)
        weighted = train_encoder(data, self.make_encoder(), replace(config, consistency_weight=0.7))
        self.assertAlmostEqual(zero.manifest["initial_loss"], expected_zero, places=6)
        self.assertAlmostEqual(weighted.manifest["initial_loss"], expected_weighted, places=6)
        self.assertGreater(zero.history[0]["variant_retrieval_loss"], 0)
        self.assertTrue(zero.manifest["lambda_zero_ablation"])
        self.assertFalse(weighted.manifest["lambda_zero_ablation"])
        self.assertTrue(zero.manifest["consistency_applicable"])
        self.assertEqual(zero.manifest["training_regime"], "reviewed_variant_retrieval_consistency")

    def test_neural_fixture_requires_explicit_opt_in(self):
        with self.assertRaises(EncoderTrainingError):
            TorchFixtureTrainingEncoder(device="cpu")


@unittest.skipUnless(torch is not None and HAS_SAFETENSORS, "optional Torch and safetensors dependencies are unavailable")
class EncoderCheckpointIntegrationTests(unittest.TestCase):
    def trained(self):
        config = fixture_config(epochs=4, learning_rate=0.08, weight_decay=0.0)
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        encoder = TorchFixtureTrainingEncoder(16, 512, seed=31, device="cpu", allow_fixture=True)
        return encoder, train_encoder(data, encoder, config), data, config

    def test_safetensors_round_trip_preserves_embeddings_and_dense_rankings(self):
        encoder, result, _, _ = self.trained()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = save_trained_checkpoint(encoder, result, root / "checkpoint")
            self.assertEqual(verify_checkpoint(root / "checkpoint"), checkpoint)
            with self.assertRaises(DenseError):
                TrainedCheckpointEncoder(root / "checkpoint", device="cpu")
            loaded = TrainedCheckpointEncoder(root / "checkpoint", device="cpu", allow_fixture=True)
            texts = [format_query("earthquake causes"), format_passage("", "rainfall clouds water")]
            before = encoder.encode(texts, batch_size=2, max_length=512)
            after = loaded.encode(texts, batch_size=2, max_length=512)
            self.assertTrue(torch.allclose(torch.tensor(before), torch.tensor(after), atol=1e-7, rtol=1e-6))
            self.assertEqual(loaded.metadata["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
            index = DenseIndex(fixture_documents(), loaded)
            self.assertTrue(index.manifest["fine_tuned"])
            self.assertEqual(index.manifest["baseline_kind"], "trained_neural_test_fixture")
            index.save(root / "dense")
            reloaded = DenseIndex.load(root / "dense", loaded)
            self.assertEqual(index.search("rainfall clouds", top_k=4), reloaded.search("rainfall clouds", top_k=4))
            run = run_queries(reloaded, [DenseQuery("q", "rainfall clouds", VALIDATION)])
            self.assertTrue(run)
            self.assertEqual({entry.branch for entry in run}, {"dense_fixture_trained"})
            for label in ("dense_e5_frozen", "dense_e5_trained"):
                with self.subTest(label=label), self.assertRaises(DenseError):
                    run_queries(reloaded, [DenseQuery("q", "rainfall clouds", VALIDATION)], branch=label)
            with self.assertRaises(DenseCompatibilityError):
                DenseIndex(fixture_documents(), loaded, DenseConfig(max_length=128))
            with self.assertRaises(EncoderTrainingError):
                save_trained_checkpoint(encoder, result, root / "checkpoint")

    def test_changed_checkpoint_weights_invalidate_an_existing_dense_index(self):
        encoder, result, data, config = self.trained()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = save_trained_checkpoint(encoder, result, root / "first")
            first_encoder = TrainedCheckpointEncoder(root / "first", device="cpu", allow_fixture=True)
            index = DenseIndex(fixture_documents(), first_encoder)
            index.save(root / "dense")
            changed_result = train_encoder(data, encoder, config)
            second = save_trained_checkpoint(encoder, changed_result, root / "second")
            self.assertNotEqual(first["files"]["fixture_model.safetensors"], second["files"]["fixture_model.safetensors"])
            self.assertNotEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
            second_encoder = TrainedCheckpointEncoder(root / "second", device="cpu", allow_fixture=True)
            with self.assertRaises(DenseCompatibilityError):
                DenseIndex.load(root / "dense", second_encoder)
            self.assertEqual(index.search("earthquake causes"), DenseIndex.load(root / "dense", first_encoder).search("earthquake causes"))
            (root / "first" / "fixture_model.safetensors").write_bytes(b"changed weights")
            with self.assertRaises(DenseCompatibilityError):
                TrainedCheckpointEncoder(root / "first", device="cpu", allow_fixture=True)

    def test_verified_manifest_cannot_hide_invalid_tensor_dimensions(self):
        from safetensors.torch import save_file
        encoder, result, _, _ = self.trained()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            manifest = save_trained_checkpoint(encoder, result, checkpoint)
            save_file({"embedding.weight": torch.ones((3, 3))}, str(checkpoint / "fixture_model.safetensors"))
            (checkpoint / "encoder_manifest.json").unlink()
            write_checkpoint_manifest(checkpoint, manifest["metadata"])
            with self.assertRaises(DenseCompatibilityError):
                TrainedCheckpointEncoder(checkpoint, device="cpu", allow_fixture=True)

    def run_cli(self, arguments):
        completed = subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120)
        self.assertEqual(completed.returncode, 0, f"{' '.join(arguments)}\n{completed.stderr}")
        return completed

    def test_cli_training_dense_fusion_and_serialized_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, queries, qrels = root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.jsonl"
            records = [*fixture_records(), original("q-eval-earthquake", "earthquake plate movement", split=FINAL_EVALUATION, partition=OFFICIAL_DEV), original("q-eval-rain", "rainfall clouds water", split=FINAL_EVALUATION, partition=OFFICIAL_DEV)]
            write_records(queries, records)
            write_records(corpus, [{**asdict(document), "romanized_text": document.text, "romanization_status": "manual_fixture"} for document in fixture_documents()])
            write_records(qrels, [asdict(qrel) for qrel in [*fixture_qrels(), Qrel("q-eval-earthquake", "d-earthquake-also", 2), Qrel("q-eval-rain", "d-rain", 1)]])
            checkpoint, dense_directory = root / "checkpoint", root / "dense"
            completed = self.run_cli(["training.train_encoder", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels), "--output-dir", str(checkpoint), "--backend", "fixture-torch", "--allow-fixture-encoder", "--device", "cpu", "--seed", "31", "--epochs", "8", "--batch-size", "2", "--learning-rate", "0.08", "--temperature", "0.5", "--random-negatives", "2", "--bm25-negatives", "1", "--max-length", "64"])
            trained = json.loads(completed.stdout)
            self.assertTrue(trained["fixture_only"])
            self.assertGreater(trained["optimizer_updates"], 0)
            self.assertLess(trained["final_loss"], trained["initial_loss"])
            manifest = verify_checkpoint(checkpoint)
            self.assertEqual(manifest["metadata"]["training"]["training_data"]["judgment_query_ids"], ["q-earthquake", "q-rain"])
            self.assertEqual(manifest["metadata"]["training"]["variant_pair_count"], 1)
            encoder_args = ["--backend", "e5-trained", "--checkpoint", str(checkpoint), "--allow-fixture-encoder", "--device", "cpu", "--max-length", "64"]
            self.run_cli(["retrieval.dense", "build-index", "--corpus", str(corpus), "--index-dir", str(dense_directory), *encoder_args])
            loaded = TrainedCheckpointEncoder(checkpoint, device="cpu", allow_fixture=True)
            index = DenseIndex.load(dense_directory, loaded, expected_config=DenseConfig(max_length=64))
            self.assertEqual(index.document_count, 4)
            self.assertEqual(index.manifest["encoder"]["checkpoint_sha256"], trained["checkpoint_sha256"])
            dense_run, fusion_run, trace = root / "dense.jsonl", root / "fusion.jsonl", root / "trace.jsonl"
            self.run_cli(["retrieval.dense", "search", "--index-dir", str(dense_directory), "--queries", str(queries), "--split", FINAL_EVALUATION, "--output", str(dense_run), *encoder_args])
            self.run_cli(["retrieval.fusion", "--corpus", str(corpus), "--queries", str(queries), "--split", FINAL_EVALUATION, "--dense-index", str(dense_directory), "--output", str(fusion_run), "--trace", str(trace), *encoder_args])
            for run_path, label in ((dense_run, "dense_fixture_trained"), (fusion_run, "fusion_fixture")):
                entries = load_run(run_path)
                self.assertEqual({entry.query_id for entry in entries}, {"q-eval-earthquake", "q-eval-rain"})
                self.assertEqual({entry.branch for entry in entries}, {label})
                report_path = root / f"{label}.evaluation.json"
                self.run_cli(["evaluation.evaluate", "--queries", str(queries), "--qrels", str(qrels), "--run", str(run_path), "--split", FINAL_EVALUATION, "--output", str(report_path), "--bootstrap-samples", "10"])
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["evaluated_query_count"], 2)
                self.assertEqual(report["overall"]["recall_at_100"], 1.0)
                self.assertEqual(report["overall"]["mrr_at_10"], 1.0)
                self.assertEqual(report["overall"]["ndcg_at_10"], 1.0)
            traces = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(traces), 2)
            self.assertTrue(all(record["diagnostics"]["branch_diagnostics"]["dense"]["search_count"] == 1 for record in traces))
            dense_manifest = json.loads(dense_run.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            fusion_manifest = json.loads(fusion_run.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(dense_manifest["fixture_only"])
            self.assertTrue(fusion_manifest["fixture_only"])
            self.assertTrue(fusion_manifest["pipeline"]["indexes"]["dense"]["fine_tuned"])


if __name__ == "__main__":
    unittest.main()
