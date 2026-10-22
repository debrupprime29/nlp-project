"""Prepared multilingual fixtures through gate training and held-out retrieval.

These tests establish integration and leakage boundaries, not benchmark quality.
"""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.generate_variants import generate_variants, load_assigned_queries
from data.make_splits import (
    FINAL_EVALUATION, FUSION_TRAIN, OFFICIAL_DEV, OFFICIAL_TRAIN, VALIDATION,
    SplitRatios, canonical_query, make_splits, write_split_artifacts,
)
from data.prepare_corpus import CorpusSource, iter_jsonl, prepare_corpus, write_prepared_corpus
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator
from retrieval.bm25 import BM25Index, RetrievalQuery, load_corpus, write_run_artifacts
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as load_char_corpus
from retrieval.dense import DenseConfig, DenseIndex, TrainedCheckpointEncoder, load_corpus as load_dense_corpus
from retrieval.fusion import BRANCHES, BranchHit, RetrievalPipeline
from training.train_gate import GateConfig, GateExample, LinearSoftmaxGate, collect_examples, compare_validation, train_gate


ROOT = Path(__file__).resolve().parents[1]
NEURAL_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))


class TopicFixtureEncoder:
    """Deterministic injected encoder; no downloaded model or quality claim."""

    dimension = 4
    metadata = {
        "backend": "gate_pipeline_topic_fixture", "model_id": "fixture/gate-topic-table",
        "resolved_revision": "v1", "dimension": 4, "fixture_only": True,
    }

    def encode(self, texts, *, batch_size, max_length):
        vectors = []
        for text in texts:
            if "bhukamp" in text or "भूकंप" in text:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "bhumikompo" in text or "ভূমিকম্প" in text:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "earthquake" in text:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return vectors


def write_records(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def prepared_fixture(root, *, dense=True):
    texts = {
        "en": ("earthquake causes", "earthquake causes"),
        "hi": ("भूकंप क्यों आते हैं", "bhukamp kyu aate hain"),
        "bn": ("ভূমিকম্প কেন হয়", "bhumikompo keno hoy"),
    }
    sources, train_queries, final_queries = [], [], []
    for language, (native, roman) in texts.items():
        passages, qrels_path = root / f"{language}-passages.jsonl", root / f"{language}-qrels.jsonl"
        write_records(passages, [
            {"docid": "relevant", "title": "", "text": native, "romanized_text": roman, "romanization_status": "manual_fixture"},
            {"docid": "other", "title": "", "text": "weather", "romanized_text": "weather", "romanization_status": "manual_fixture"},
        ])
        write_records(qrels_path, [
            {"query_id": f"train-{language}-{index}", "doc_id": "relevant", "relevance": 2}
            for index in range(3)
        ] + [{"query_id": f"final-{language}", "doc_id": "relevant", "relevance": 2}])
        sources.append(CorpusSource(language, passages, qrels_path, "manual_gate_fixture"))
        for index in range(3):
            train_queries.append(canonical_query({"query_id": f"train-{language}-{index}", "text": roman, "languages": [language]}, source_partition=OFFICIAL_TRAIN))
        final_queries.append(canonical_query({"query_id": f"final-{language}", "text": roman, "languages": [language]}, source_partition=OFFICIAL_DEV))
    prepared = prepare_corpus(sources, target_size=2, dataset_version="gate-pipeline-fixture")
    corpus_paths = write_prepared_corpus(prepared, root / "corpus")
    split_paths = write_split_artifacts(make_splits(train_queries, final_queries, ratios=SplitRatios(1 / 3, 1 / 3, 1 / 3), seed=23), root / "splits")
    assigned = load_assigned_queries(split_paths["queries"])
    proposed = generate_variants(assigned, split=FUSION_TRAIN, seed=23)
    candidate = proposed.review_queue[0]
    reviewed = generate_variants(assigned, split=FUSION_TRAIN, seed=23, reviews=[{
        "candidate_id": candidate["candidate_id"], "candidate_sha256": candidate["candidate_sha256"],
        "parent_query_id": candidate["parent_query_id"], "reviewer_id": "explicit-fixture-reviewer",
        "decision": "verified", "meaning_preserved": True,
    }])
    records = [*reviewed.queries, *(record for record in assigned if record["split"] != FUSION_TRAIN)]
    identifier = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in iter_jsonl(ROOT / "examples/tiny/lid_train.jsonl")])
    identifier.save_json(root / "lid.json")
    transliterator = LexiconTransliterator.from_jsonl(ROOT / "examples/tiny/xlit_lexicon.jsonl")
    dense_index = None
    if dense:
        dense_index = DenseIndex(load_dense_corpus(corpus_paths["corpus"]), TopicFixtureEncoder())
        dense_index.save(root / "dense")
        dense_index = DenseIndex.load(root / "dense", TopicFixtureEncoder())
    pipeline = RetrievalPipeline(
        BM25Index(load_corpus(corpus_paths["corpus"])), CharTfidfIndex(load_char_corpus(corpus_paths["corpus"])),
        dense_index, TokenLanguageIdentifier.load_json(root / "lid.json"), transliterator,
    )
    return prepared, corpus_paths, split_paths, records, pipeline


class GatePipelineTests(unittest.TestCase):
    def test_fixed_weight_selection_uses_language_macro_not_majority_language(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, _, _, records, pipeline = prepared_fixture(root, dense=False)
            examples = collect_examples(pipeline, records, prepared.qrels)
            fitted = train_gate(examples, pipeline_manifest=pipeline.manifest, config=GateConfig(epochs=5))
            rankings = {
                "bm25_original": (BranchHit("en:relevant", 1, 1.0), BranchHit("hi:relevant", 2, 0.5)),
                "bm25_native": (BranchHit("hi:relevant", 1, 1.0), BranchHit("en:relevant", 2, 0.5)),
                "char_tfidf": (), "dense": (),
            }
            validation = []
            for language, count in (("en", 9), ("hi", 1), ("bn", 1)):
                for index in range(count):
                    query_id = f"macro-{language}-{index}"
                    judgments = {"en:relevant" if language == "en" else "hi:relevant": 1}
                    validation.append(GateExample(query_id, query_id, query_id, VALIDATION, examples[0].views, rankings, judgments, benchmark_language=language))
            report = compare_validation(fitted.gate, validation)
            equal, tuned = report["equal_weights"], report["tuned_fixed_weights"]
            self.assertEqual(report["selection_metric"], "language_macro_actual_fused_ndcg_at_10")
            self.assertGreater(tuned["language_macro_ndcg_at_10"], equal["language_macro_ndcg_at_10"])
            self.assertLess(tuned["ndcg_at_10"], equal["ndcg_at_10"])
            self.assertGreater(tuned["weights"]["bm25_native"], tuned["weights"]["bm25_original"])
            self.assertEqual(set(tuned["per_language_ndcg_at_10"]), {"en", "hi", "bn"})

    @unittest.skipUnless(NEURAL_AVAILABLE, "optional Torch and safetensors are required for actual neural training")
    def test_actual_encoder_training_checkpoint_reindex_gate_and_heldout_pipeline(self):
        from training.train_encoder import (
            EncoderTrainingConfig, TorchFixtureTrainingEncoder, prepare_encoder_training,
            save_trained_checkpoint, train_encoder,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, corpus_paths, split_paths, records, baseline = prepared_fixture(root, dense=False)
            documents = load_dense_corpus(corpus_paths["corpus"])
            encoder_config = EncoderTrainingConfig(epochs=6, batch_size=3, max_length=64, learning_rate=0.05, temperature=0.5, random_negatives=1, seed=23, weight_decay=0.0)
            encoder_data = prepare_encoder_training(records, prepared.qrels, documents, encoder_config)
            encoder = TorchFixtureTrainingEncoder(seed=23, allow_fixture=True)
            neural_result = train_encoder(encoder_data, encoder, encoder_config)
            self.assertGreater(neural_result.manifest["parameter_delta_l2"], 0.0)
            self.assertLess(neural_result.manifest["final_loss"], neural_result.manifest["initial_loss"])
            save_trained_checkpoint(encoder, neural_result, root / "checkpoint")
            frozen_encoder = TrainedCheckpointEncoder(root / "checkpoint", device="cpu", allow_fixture=True)
            index = DenseIndex(documents, frozen_encoder, DenseConfig(max_length=64))
            index.save(root / "trained-dense")
            reloaded = DenseIndex.load(root / "trained-dense", frozen_encoder)
            pipeline = RetrievalPipeline(baseline.bm25_index, baseline.char_index, reloaded, baseline.language_identifier, baseline.transliterator)
            training = collect_examples(pipeline, records, prepared.qrels)
            validation = collect_examples(pipeline, records, prepared.qrels, split=VALIDATION)
            self.assertFalse(set(encoder_data.manifest["group_ids"]) & {example.group_id for example in training})
            dense_before = dict(reloaded.manifest)
            fitted = train_gate(training, pipeline_manifest=pipeline.manifest, config=GateConfig(epochs=30), validation_examples=validation)
            fitted.gate.save_json(root / "neural-gate.json")
            pipeline.weight_provider = LinearSoftmaxGate.load_json(root / "neural-gate.json", expected_pipeline_manifest=pipeline.manifest)
            self.assertEqual(reloaded.manifest, dense_before)
            final_records = [record for record in records if record["split"] == FINAL_EVALUATION]
            run = pipeline.run_queries([RetrievalQuery(record["query_id"], record["text"], record["split"]) for record in final_records], branch="neural_gate_pipeline_fixture")
            write_run_artifacts(run, root / "neural-run.jsonl", pipeline.manifest, root / "neural-run.manifest.json")
            report = evaluate_run(load_evaluation_queries(split_paths["queries"], split=FINAL_EVALUATION), prepared.qrels, load_run(root / "neural-run.jsonl"), bootstrap_samples=10)
            self.assertEqual(report.summary["evaluated_query_count"], 3)
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["recall_at_100"], 1.0)

    def test_prepare_split_review_all_indexes_gate_and_final_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, _, split_paths, records, pipeline = prepared_fixture(root)
            frozen_manifest = pipeline.manifest
            training = collect_examples(pipeline, records, prepared.qrels, split=FUSION_TRAIN)
            validation = collect_examples(pipeline, records, prepared.qrels, split=VALIDATION)
            self.assertEqual(len(training), 4)
            self.assertEqual(len(validation), 3)
            self.assertEqual({example.split for example in training}, {FUSION_TRAIN})
            self.assertFalse({example.group_id for example in training} & {example.group_id for example in validation})
            self.assertTrue(any(example.query_id != example.parent_query_id for example in training))
            trained = train_gate(training, pipeline_manifest=frozen_manifest, config=GateConfig(epochs=40), validation_examples=validation)
            self.assertEqual(pipeline.manifest, frozen_manifest, "gate fitting must not change any frozen index or resource")
            validation_report = trained.report["validation"]
            for name in ("equal_weights", "tuned_fixed_weights", "gate"):
                self.assertEqual(validation_report[name]["query_count"], 3)
                self.assertGreaterEqual(validation_report[name]["ndcg_at_10"], 0.0)
                self.assertLessEqual(validation_report[name]["ndcg_at_10"], 1.0)
            for name in ("equal_weights", "tuned_fixed_weights"):
                self.assertEqual(set(validation_report[name]["weights"]), set(BRANCHES))
                self.assertAlmostEqual(sum(validation_report[name]["weights"].values()), 1.0)
            self.assertGreaterEqual(validation_report["tuned_fixed_weights"]["ndcg_at_10"], validation_report["equal_weights"]["ndcg_at_10"])
            gate_path = root / "gate.json"
            trained.gate.save_json(gate_path)
            loaded = LinearSoftmaxGate.load_json(gate_path, expected_pipeline_manifest=frozen_manifest)
            pipeline.weight_provider = loaded
            heldout = [record for record in records if record["split"] == FINAL_EVALUATION]
            for record in heldout:
                result = pipeline.search(record["text"])
                self.assertEqual(result.diagnostics["weight_source"], "provider")
                self.assertAlmostEqual(sum(result.active_weights.values()), 1.0)
                self.assertEqual(set(result.active_weights), {branch for branch in BRANCHES if result.branch_results[branch]})
                self.assertEqual(result.hits[0].doc_id, f"{record['benchmark_language']}:relevant")
                self.assertLessEqual(result.diagnostics["candidate_union_count"], 400)
                for hit in result.hits:
                    self.assertAlmostEqual(hit.score, sum(hit.contributions.values()))
            run = pipeline.run_queries([RetrievalQuery(record["query_id"], record["text"], record["split"]) for record in heldout], branch="fusion_adaptive_fixture")
            write_run_artifacts(run, root / "run.jsonl", pipeline.manifest, root / "run.manifest.json")
            evaluation = evaluate_run(load_evaluation_queries(split_paths["queries"], split=FINAL_EVALUATION), prepared.qrels, load_run(root / "run.jsonl"), bootstrap_samples=10)
            self.assertEqual(evaluation.summary["evaluated_query_count"], 3)
            self.assertEqual(evaluation.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(evaluation.summary["overall"]["recall_at_100"], 1.0)

    def test_saved_gate_fusion_cli_trace_and_safe_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, corpus_paths, split_paths, records, pipeline = prepared_fixture(root, dense=False)
            training = collect_examples(pipeline, records, prepared.qrels)
            trained = train_gate(training, pipeline_manifest=pipeline.manifest, config=GateConfig(epochs=10))
            gate_path = root / "gate.json"
            trained.gate.save_json(gate_path)
            command = [
                sys.executable, "-m", "retrieval.fusion", "--corpus", str(corpus_paths["corpus"]),
                "--queries", str(split_paths["queries"]), "--split", FINAL_EVALUATION,
                "--lid-model", str(root / "lid.json"), "--lexicon", str(ROOT / "examples/tiny/xlit_lexicon.jsonl"),
                "--gate-model", str(gate_path), "--output", str(root / "run.jsonl"), "--trace", str(root / "trace.jsonl"),
            ]
            cli = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            run = load_run(root / "run.jsonl")
            self.assertEqual({entry.branch for entry in run}, {"fusion_adaptive"})
            self.assertEqual({entry.query_id for entry in run}, {record["query_id"] for record in records if record["split"] == FINAL_EVALUATION})
            traces = list(iter_jsonl(root / "trace.jsonl"))
            self.assertEqual(len(traces), 3)
            self.assertTrue(all(trace["diagnostics"]["weight_source"] == "provider" for trace in traces))
            self.assertTrue(all(hit["contributions"] for trace in traces for hit in trace["hits"]))
            manifest = json.loads((root / "run.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["weights"], "learned_gate")
            self.assertIsNotNone(manifest["pipeline"]["weight_provider"])
            existing = (root / "run.jsonl").read_bytes()
            rerun = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(rerun.returncode, 2)
            self.assertEqual((root / "run.jsonl").read_bytes(), existing)
            original_gate = gate_path.read_bytes()
            collision_command = list(command)
            collision_command[collision_command.index("--output") + 1] = str(gate_path)
            collision = subprocess.run([*collision_command, "--force"], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(collision.returncode, 2)
            self.assertEqual(gate_path.read_bytes(), original_gate)


if __name__ == "__main__":
    unittest.main()
