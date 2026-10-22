"""Fixture integration across completed modules; these are not benchmark scores."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from data.generate_variants import generate_variants, load_assigned_queries, write_variant_artifacts
from data.make_splits import OFFICIAL_DEV, canonical_query, make_splits, write_split_artifacts
from data.prepare_corpus import CorpusSource, iter_jsonl, prepare_corpus, write_prepared_corpus
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator, build_query_views
from retrieval.bm25 import BM25Index, RetrievalResult, load_corpus, write_run_artifacts
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as load_char_corpus
from retrieval.dense import DenseIndex, load_corpus as load_dense_corpus
from retrieval.fusion import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[1]


class ControlledTopicEncoder:
    """Explicitly injected topic table, solely to verify dense integration."""

    dimension = 4
    metadata = {
        "backend": "controlled_topic_fixture",
        "model_id": "fixture/topic-table",
        "resolved_revision": "v1",
        "dimension": 4,
        "fixture_only": True,
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


def write_fixture(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


class FullPipelineTests(unittest.TestCase):
    def test_pooled_prepare_split_review_lid_views_all_retrievers_and_evaluation(self):
        texts = {
            "en": ("earthquake causes", "earthquake causes"),
            "hi": ("भूकंप क्यों आते हैं", "bhukamp kyu aate hain"),
            "bn": ("ভূমিকম্প কেন হয়", "bhumikompo keno hoy"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, queries = [], []
            for language, (native, roman) in texts.items():
                passages, qrels = root / f"{language}-passages.jsonl", root / f"{language}-qrels.jsonl"
                write_fixture(passages, [
                    {"docid": "relevant", "title": "", "text": native, "romanized_text": roman, "romanization_status": "manual_fixture"},
                    {"docid": "other", "title": "", "text": "weather", "romanized_text": "weather", "romanization_status": "manual_fixture"},
                ])
                write_fixture(qrels, [{"query_id": f"q-{language}", "doc_id": "relevant", "relevance": 1}])
                sources.append(CorpusSource(language, passages, qrels, "manual_fixture"))
                queries.append(canonical_query({"query_id": f"q-{language}", "text": roman, "languages": [language]}, source_partition=OFFICIAL_DEV))
            prepared = prepare_corpus(sources, target_size=2, dataset_version="full-pipeline-fixture")
            corpus_paths = write_prepared_corpus(prepared, root / "corpus")
            split_paths = write_split_artifacts(make_splits([], queries), root / "split")
            assigned = load_assigned_queries(split_paths["queries"])

            candidates = generate_variants(assigned, split="final_evaluation", seed=5)
            self.assertTrue(candidates.review_queue)
            chosen = candidates.review_queue[0]
            reviewed = generate_variants(assigned, split="final_evaluation", seed=5, reviews=[{
                "candidate_id": chosen["candidate_id"], "candidate_sha256": chosen["candidate_sha256"],
                "parent_query_id": chosen["parent_query_id"], "reviewer_id": "fixture-reviewer",
                "decision": "verified", "meaning_preserved": True,
            }])
            variant_paths = write_variant_artifacts(reviewed, root / "variants")
            self.assertTrue(all(record["verified"] for record in iter_jsonl(variant_paths["queries"])))
            self.assertEqual(len(reviewed.queries), 4)

            # This auxiliary model is trained on explicit toy labels only.
            lid = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in iter_jsonl(ROOT / "examples/tiny/lid_train.jsonl")])
            lid.save_json(root / "lid.json")
            lid = TokenLanguageIdentifier.load_json(root / "lid.json")
            transliterator = LexiconTransliterator.from_jsonl(ROOT / "examples/tiny/xlit_lexicon.jsonl")
            lexical = BM25Index(load_corpus(corpus_paths["corpus"]))
            character = CharTfidfIndex(load_char_corpus(corpus_paths["corpus"]))
            dense = DenseIndex(load_dense_corpus(corpus_paths["corpus"]), ControlledTopicEncoder())
            dense.save(root / "dense")
            dense = DenseIndex.load(root / "dense", ControlledTopicEncoder())
            pipeline = RetrievalPipeline(lexical, character, dense, lid, transliterator)

            batch_path = root / "batch-views.jsonl"
            cli = subprocess.run([
                sys.executable, "-m", "nlp.query_views", "--queries", str(variant_paths["queries"]),
                "--split", "final_evaluation", "--lid-model", str(root / "lid.json"),
                "--backend", "lexicon", "--lexicon", str(ROOT / "examples/tiny/xlit_lexicon.jsonl"),
                "--output", str(batch_path),
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            batch = list(iter_jsonl(batch_path))
            self.assertEqual(len(batch), len(reviewed.queries))
            self.assertTrue(all(record["original_text"] == record["text"] for record in batch))
            self.assertTrue(all(record["split"] == "final_evaluation" for record in batch))
            evaluation_queries = load_evaluation_queries(variant_paths["queries"], split="final_evaluation")
            runs = {"bm25_views": [], "char_tfidf": [], "dense_controlled_fixture": [], "fusion_fixture": []}
            for query in reviewed.queries:
                views = build_query_views(query["text"], language_identifier=lid, transliterator=transliterator)
                self.assertEqual(views.original_text, query["text"])
                self.assertLessEqual(len(views.native_views), 2)
                self.assertIsNotNone(views.roman_view)
                if query["variant_type"] == "original":
                    if query["benchmark_language"] == "en":
                        self.assertEqual(views.native_views, ())
                    else:
                        self.assertEqual(views.native_views[0].text, texts[query["benchmark_language"]][0])
                dense_views = [views.original.text, *(view.text for view in views.native_views)]
                lexical_views = [view.text for view in views.native_views] or [views.original.text]
                fused = pipeline.search(query["text"])
                self.assertLessEqual(fused.diagnostics["candidate_union_count"], 400)
                self.assertLessEqual(len(fused.hits), 10)
                self.assertAlmostEqual(sum(fused.active_weights.values()), 1.0)
                per_branch = {
                    "bm25_views": lexical.search_views(lexical_views).hits,
                    "char_tfidf": character.search(views.roman_view.text),
                    "dense_controlled_fixture": dense.search_views(dense_views).hits,
                    "fusion_fixture": fused.hits,
                }
                for branch, hits in per_branch.items():
                    if query["variant_type"] == "original" or branch == "char_tfidf":
                        self.assertEqual(hits[0].doc_id, f"{query['benchmark_language']}:relevant")
                    self.assertLessEqual(len(hits), 100)
                    runs[branch].extend(RetrievalResult(query["query_id"], hit.doc_id, branch, hit.rank, hit.score) for hit in hits)
            for branch, results in runs.items():
                run_path = root / f"{branch}.jsonl"
                write_run_artifacts(results, run_path, {"fixture_only": True}, run_path.with_suffix(".manifest.json"))
                loaded = load_run(run_path)
                self.assertTrue(all(entry.doc_id in {passage.doc_id for passage in prepared.passages} for entry in loaded))
                report = evaluate_run(evaluation_queries, prepared.qrels, loaded, bootstrap_samples=20)
                self.assertEqual(len(report.per_query), 4)
                self.assertTrue(all(score.status == "evaluated" for score in report.per_query))
                for score in report.per_query:
                    if score.variant_type == "original" or branch == "char_tfidf":
                        self.assertEqual(score.ndcg_at_10, 1.0)
                        self.assertEqual(score.mrr_at_10, 1.0)
                        self.assertEqual(score.recall_at_100, 1.0)


if __name__ == "__main__":
    unittest.main()
