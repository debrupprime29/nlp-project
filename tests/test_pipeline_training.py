"""Prepared toy corpus through training and final fused fixture evaluation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from data.generate_variants import generate_variants, load_assigned_queries
from data.make_splits import ENCODER_TRAIN, FINAL_EVALUATION, OFFICIAL_DEV, OFFICIAL_TRAIN, SplitRatios, canonical_query, make_splits, write_split_artifacts
from data.prepare_corpus import CorpusSource, iter_jsonl, prepare_corpus, write_prepared_corpus
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator
from retrieval.bm25 import BM25Index, RetrievalQuery, load_corpus, write_run_artifacts
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as load_char_corpus
from retrieval.dense import DenseConfig, DenseIndex, TrainedCheckpointEncoder, load_corpus as load_dense_corpus
from retrieval.fusion import RetrievalPipeline
from training.train_encoder import EncoderTrainingConfig, TorchFixtureTrainingEncoder, prepare_encoder_training, save_trained_checkpoint, train_encoder


ROOT = Path(__file__).resolve().parents[1]
NEURAL_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))


@unittest.skipUnless(NEURAL_AVAILABLE, "optional Torch and safetensors are required for actual training")
class PreparedTrainingPipelineTests(unittest.TestCase):
    def test_preparation_splits_reviews_training_views_fusion_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptions = [
                ("train-quake", "fault energy earthquake", "earthquake", OFFICIAL_TRAIN),
                ("train-rain", "rain clouds moisture", "weather", OFFICIAL_TRAIN),
                ("eval-quake", "earthquake causes", "earthquake", OFFICIAL_DEV),
                ("eval-rain", "clouds produce rain", "weather", OFFICIAL_DEV),
            ]
            qrels_path = root / "qrels.jsonl"
            qrels_path.write_text("".join(json.dumps({"query_id": query_id, "doc_id": doc_id, "relevance": 1}) + "\n" for query_id, _, doc_id, _ in descriptions), encoding="utf-8")
            prepared = prepare_corpus([CorpusSource("en", ROOT / "examples/tiny/en_passages.jsonl", qrels_path, "synthetic-training-pipeline")], target_size=2, seed=17, dataset_version="synthetic-training-pipeline")
            corpus_paths = write_prepared_corpus(prepared, root / "corpus")
            source_queries = [(canonical_query({"query_id": query_id, "text": text, "languages": ["en"]}, source_partition=partition), partition) for query_id, text, _, partition in descriptions]
            splits = make_splits([query for query, partition in source_queries if partition == OFFICIAL_TRAIN], [query for query, partition in source_queries if partition == OFFICIAL_DEV], ratios=SplitRatios(1.0, 0.0, 0.0))
            split_paths = write_split_artifacts(splits, root / "splits")
            assigned = load_assigned_queries(split_paths["queries"])
            generated = generate_variants(assigned, split=ENCODER_TRAIN, seed=17)
            candidate = generated.review_queue[0]
            reviewed = generate_variants(assigned, split=ENCODER_TRAIN, seed=17, reviews=[{
                "candidate_id": candidate["candidate_id"], "candidate_sha256": candidate["candidate_sha256"],
                "parent_query_id": candidate["parent_query_id"], "reviewer_id": "synthetic-fixture-reviewer",
                "decision": "verified", "meaning_preserved": True,
            }])
            heldout = [query for query in assigned if query["split"] == FINAL_EVALUATION]
            config = EncoderTrainingConfig(epochs=8, batch_size=2, max_length=64, learning_rate=0.05, temperature=0.5, random_negatives=1, seed=17, weight_decay=0.0)
            documents = load_dense_corpus(corpus_paths["corpus"])
            data = prepare_encoder_training([*reviewed.queries, *heldout], prepared.qrels, documents, config)
            self.assertEqual(set(data.manifest["judgment_query_ids"]), {"train-quake", "train-rain"})
            self.assertEqual(data.manifest["variant_pair_count"], 1)
            encoder = TorchFixtureTrainingEncoder(seed=17, allow_fixture=True)
            trained = train_encoder(data, encoder, config)
            self.assertGreater(trained.manifest["parameter_delta_l2"], 0.0)
            self.assertLess(trained.manifest["final_loss"], trained.manifest["initial_loss"])
            save_trained_checkpoint(encoder, trained, root / "checkpoint")
            restored = TrainedCheckpointEncoder(root / "checkpoint", device="cpu", allow_fixture=True)
            dense = DenseIndex(documents, restored, DenseConfig(max_length=64))
            dense.save(root / "index")
            dense = DenseIndex.load(root / "index", restored)
            identifier = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in iter_jsonl(ROOT / "examples/tiny/lid_train.jsonl")])
            identifier.save_json(root / "lid.json")
            pipeline = RetrievalPipeline(
                BM25Index(load_corpus(corpus_paths["corpus"])),
                CharTfidfIndex(load_char_corpus(corpus_paths["corpus"])), dense,
                TokenLanguageIdentifier.load_json(root / "lid.json"),
                LexiconTransliterator.from_jsonl(ROOT / "examples/tiny/xlit_lexicon.jsonl"),
            )
            results = pipeline.run_queries([RetrievalQuery(record["query_id"], record["text"], record["split"]) for record in heldout], branch="fusion_trained_fixture")
            write_run_artifacts(results, root / "run.jsonl", pipeline.manifest, root / "run.manifest.json")
            loaded = load_run(root / "run.jsonl")
            self.assertEqual({entry.query_id for entry in loaded}, {"eval-quake", "eval-rain"})
            evaluation_queries = load_evaluation_queries(split_paths["queries"], split=FINAL_EVALUATION)
            report = evaluate_run(evaluation_queries, prepared.qrels, loaded, bootstrap_samples=10)
            self.assertEqual(report.summary["evaluated_query_count"], 2)
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["recall_at_100"], 1.0)


if __name__ == "__main__":
    unittest.main()
