"""Persisted prepared-data, dense-index, and learned-gate app integration.

All data and neural models are explicit mechanics-only fixtures; no benchmark
quality or real pretrained E5 behavior is inferred from these checks.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode

from app import application_from_arguments, build_argument_parser
from data.make_splits import FINAL_EVALUATION, VALIDATION
from data.prepare_corpus import iter_jsonl
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from retrieval.bm25 import RetrievalResult, write_run_artifacts
from retrieval.dense import DenseConfig, DenseIndex, FixtureHashEncoder, TrainedCheckpointEncoder, load_corpus
from retrieval.fusion import RetrievalPipeline
from tests.test_gate_pipeline import NEURAL_AVAILABLE, ROOT, prepared_fixture
from training.train_gate import GateConfig, collect_examples, train_gate


def persisted_application_fixture(root, *, neural=False):
    prepared, corpus_paths, split_paths, records, baseline = prepared_fixture(root, dense=False)
    documents = load_corpus(corpus_paths["corpus"])
    if neural:
        from training.train_encoder import (
            EncoderTrainingConfig, TorchFixtureTrainingEncoder, prepare_encoder_training,
            save_trained_checkpoint, train_encoder,
        )

        config = EncoderTrainingConfig(epochs=6, batch_size=3, max_length=64, learning_rate=0.05, temperature=0.5, random_negatives=1, seed=23, weight_decay=0.0)
        data = prepare_encoder_training(records, prepared.qrels, documents, config)
        trainable = TorchFixtureTrainingEncoder(seed=23, allow_fixture=True)
        trained = train_encoder(data, trainable, config)
        if trained.manifest["parameter_delta_l2"] <= 0:
            raise AssertionError("fixture neural training must update real parameters")
        save_trained_checkpoint(trainable, trained, root / "checkpoint")
        encoder = TrainedCheckpointEncoder(root / "checkpoint", device="cpu", allow_fixture=True)
        dense_config = DenseConfig(max_length=64)
        encoder_args = ["--backend", "e5-trained", "--checkpoint", str(root / "checkpoint"), "--device", "cpu", "--allow-fixture-encoder", "--max-length", "64"]
    else:
        encoder = FixtureHashEncoder(64, allow_fixture=True)
        dense_config = DenseConfig()
        encoder_args = ["--backend", "fixture-hash", "--fixture-dimension", "64", "--allow-fixture-encoder"]
    index = DenseIndex(documents, encoder, dense_config)
    index.save(root / "dense")
    pipeline = RetrievalPipeline(baseline.bm25_index, baseline.char_index, DenseIndex.load(root / "dense", encoder), baseline.language_identifier, baseline.transliterator)
    gate = train_gate(
        collect_examples(pipeline, records, prepared.qrels), pipeline_manifest=pipeline.manifest,
        config=GateConfig(epochs=30), validation_examples=collect_examples(pipeline, records, prepared.qrels, split=VALIDATION),
    ).gate
    gate.save_json(root / "gate.json")
    pipeline.weight_provider = gate
    arguments = [
        "--corpus", str(corpus_paths["corpus"]), "--lid-model", str(root / "lid.json"),
        "--lexicon", str(ROOT / "examples/tiny/xlit_lexicon.jsonl"),
        "--dense-index", str(root / "dense"), "--gate-model", str(root / "gate.json"), *encoder_args,
    ]
    return prepared, split_paths, records, pipeline, arguments


def header(headers, name):
    return next(value for key, value in headers.items() if key.casefold() == name.casefold())


class AppPipelineTests(unittest.TestCase):
    def test_persisted_resources_gate_cli_html_api_and_evaluated_rankings_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, split_paths, records, pipeline, arguments = persisted_application_fixture(root)
            application = application_from_arguments(build_argument_parser().parse_args(arguments))
            self.assertEqual(application.pipeline.manifest, pipeline.manifest)
            passages = {record["doc_id"]: record for record in iter_jsonl(root / "corpus" / "corpus.jsonl")}
            final_queries = [record for record in records if record["split"] == FINAL_EVALUATION]
            app_run, direct_run = [], []
            for query in final_queries:
                result = application.search(query["text"])
                expected = pipeline.search(query["text"])
                self.assertTrue(result["fixture_only"])
                self.assertEqual(result["query"], query["text"])
                self.assertEqual(result["views"]["original"], query["text"])
                self.assertEqual(result["active_weights"], expected.active_weights)
                self.assertEqual(result["diagnostics"]["weight_source"], "provider")
                self.assertEqual([hit["doc_id"] for hit in result["hits"]], [hit.doc_id for hit in expected.hits])
                self.assertLessEqual(len(result["hits"]), 10)
                for hit, reference in zip(result["hits"], expected.hits):
                    passage = passages[hit["doc_id"]]
                    self.assertEqual(hit["text"], passage["text"])
                    self.assertEqual(hit["language"], passage["language"])
                    self.assertEqual(hit["source"], passage["source"])
                    self.assertEqual(hit["title"], passage["title"])
                    self.assertEqual(hit["rank"], reference.rank)
                    self.assertAlmostEqual(hit["score"], reference.score)
                    self.assertEqual(hit["contributions"], reference.contributions)
                    app_run.append(RetrievalResult(query["query_id"], hit["doc_id"], "app_fixture", hit["rank"], hit["score"]))
                    direct_run.append(RetrievalResult(query["query_id"], reference.doc_id, "direct_fixture", reference.rank, reference.score))
                encoded = urlencode({"q": query["text"]})
                status, headers, body = application.handle("/api/search?" + encoded)
                self.assertEqual(status, 200)
                self.assertIn("application/json", header(headers, "Content-Type"))
                api_result = json.loads(body)
                self.assertEqual(api_result["hits"], result["hits"])
                self.assertEqual(api_result["active_weights"], result["active_weights"])
                status, headers, body = application.handle("/search?" + encoded)
                self.assertEqual(status, 200)
                self.assertIn("text/html", header(headers, "Content-Type"))
                readable = html.unescape(body.decode("utf-8"))
                self.assertIn(query["text"], readable)
                self.assertIn(result["hits"][0]["text"], readable)
                for branch in result["active_weights"]:
                    self.assertIn(branch, readable)
                cli = subprocess.run([sys.executable, "-m", "app", *arguments, "--query", query["text"]], cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(cli.returncode, 0, cli.stderr)
                printed = json.loads(cli.stdout)
                self.assertEqual(printed["hits"], result["hits"])
                self.assertEqual(printed["active_weights"], result["active_weights"])
            queries = load_evaluation_queries(split_paths["queries"], split=FINAL_EVALUATION)
            write_run_artifacts(app_run, root / "app-run.jsonl", application.pipeline.manifest, root / "app-run.manifest.json")
            write_run_artifacts(direct_run, root / "direct-run.jsonl", pipeline.manifest, root / "direct-run.manifest.json")
            app_report = evaluate_run(queries, prepared.qrels, load_run(root / "app-run.jsonl"), bootstrap_samples=10)
            direct_report = evaluate_run(queries, prepared.qrels, load_run(root / "direct-run.jsonl"), bootstrap_samples=10)
            self.assertEqual(app_report.summary["overall"], direct_report.summary["overall"])
            self.assertEqual(app_report.per_query, direct_report.per_query)
            self.assertEqual(app_report.summary["evaluated_query_count"], 3)
            self.assertEqual(app_report.summary["overall"]["ndcg_at_10"], 1.0)

    @unittest.skipUnless(NEURAL_AVAILABLE, "optional Torch and safetensors are required for actual neural training")
    def test_actual_neural_checkpoint_index_and_gate_load_in_application_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, records, pipeline, arguments = persisted_application_fixture(root, neural=True)
            application = application_from_arguments(build_argument_parser().parse_args(arguments))
            self.assertTrue(application.pipeline.dense_index.manifest["encoder"]["fine_tuned"])
            self.assertEqual(application.pipeline.manifest, pipeline.manifest)
            query = next(record for record in records if record["query_id"] == "final-hi")
            result = application.search(query["text"])
            expected = pipeline.search(query["text"])
            self.assertEqual([hit["doc_id"] for hit in result["hits"]], [hit.doc_id for hit in expected.hits])
            self.assertEqual(result["hits"][0]["doc_id"], "hi:relevant")
            self.assertEqual(result["active_weights"], expected.active_weights)
            self.assertTrue(result["fixture_only"])
            cli = subprocess.run([sys.executable, "-m", "app", *arguments, "--query", query["text"]], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            printed = json.loads(cli.stdout)
            self.assertEqual(printed["hits"], result["hits"])
            self.assertEqual(printed["active_weights"], result["active_weights"])


if __name__ == "__main__":
    unittest.main()
