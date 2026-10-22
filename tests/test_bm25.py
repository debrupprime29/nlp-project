from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.make_splits import OFFICIAL_DEV, Query, make_splits
from data.prepare_corpus import CorpusSource, prepare_corpus
from evaluation.evaluate import EvaluationQuery, evaluate_run
from retrieval.bm25 import (
    BM25Config,
    BM25Document,
    BM25Error,
    BM25Index,
    RetrievalQuery,
    run_queries,
    tokenize,
)


class BM25Tests(unittest.TestCase):
    def test_script_aware_tokenizer_preserves_indic_marks_and_boundaries(self) -> None:
        self.assertEqual(
            tokenize("Earthquakeक्यों आते हैं 2026"),
            ("earthquake", "क्यों", "आते", "हैं", "2026"),
        )
        self.assertEqual(tokenize("ভূমিকম্প কেন হয়"), ("ভূমিকম্প", "কেন", "হয়"))

    def test_nfc_equivalent_query_matches(self) -> None:
        composed = "café"
        decomposed = "cafe\u0301"
        index = BM25Index([BM25Document("d", "", composed, "en", "fixture")])
        self.assertEqual(index.search(decomposed)[0].doc_id, "d")

    def test_hand_calculated_okapi_score(self) -> None:
        index = BM25Index(
            [BM25Document("d", "", "term term", "en", "fixture")],
            BM25Config(k1=1.2, b=0.75, title_weight=1.0),
        )
        hit = index.search("term")[0]
        expected_idf = math.log(1 + 0.5 / 1.5)
        expected = expected_idf * (2 * 2.2 / (2 + 1.2))
        self.assertAlmostEqual(hit.score, expected)

    def test_title_and_body_are_indexed_with_configurable_weight(self) -> None:
        index = BM25Index(
            [
                BM25Document("title", "earthquake", "unrelated", "en", "fixture"),
                BM25Document("body", "unrelated", "earthquake", "en", "fixture"),
            ],
            BM25Config(title_weight=3.0),
        )
        hits = index.search("earthquake")
        self.assertEqual([hit.doc_id for hit in hits], ["title", "body"])

    def test_length_normalization_and_term_saturation(self) -> None:
        index = BM25Index(
            [
                BM25Document("short", "", "term", "en", "fixture"),
                BM25Document("long", "", "term filler filler filler", "en", "fixture"),
            ]
        )
        hits = index.search("term")
        self.assertEqual([hit.doc_id for hit in hits], ["short", "long"])

        once = BM25Index([BM25Document("d", "", "term", "en", "fixture")]).search("term")[0].score
        thrice = BM25Index([BM25Document("d", "", "term term term", "en", "fixture")]).search("term")[0].score
        self.assertLess(thrice, 3 * once)

    def test_hindi_bengali_negation_and_numbers_are_searchable(self) -> None:
        index = BM25Index(
            [
                BM25Document("hi", "", "भूकंप नहीं आया 2026", "hi", "fixture"),
                BM25Document("bn", "", "ভূমিকম্প কেন হয়", "bn", "fixture"),
            ]
        )
        self.assertEqual(index.search("नहीं 2026")[0].doc_id, "hi")
        self.assertEqual(index.search("ভূমিকম্প")[0].doc_id, "bn")

    def test_empty_oov_topk_and_empty_index_are_safe(self) -> None:
        index = BM25Index(
            [BM25Document(f"d{number}", "", "shared", "en", "fixture") for number in range(5)]
        )
        self.assertEqual(index.search(""), ())
        self.assertEqual(index.search("absent"), ())
        self.assertEqual(len(index.search("shared", top_k=2)), 2)
        self.assertEqual(BM25Index([]).search("anything"), ())

    def test_duplicate_ids_fail_and_ties_use_doc_id(self) -> None:
        with self.assertRaisesRegex(BM25Error, "Duplicate document"):
            BM25Index(
                [
                    BM25Document("same", "", "one", "en", "fixture"),
                    BM25Document("same", "", "two", "en", "fixture"),
                ]
            )
        index = BM25Index(
            [
                BM25Document("z", "", "same", "en", "fixture"),
                BM25Document("a", "", "same", "en", "fixture"),
            ]
        )
        self.assertEqual([hit.doc_id for hit in index.search("same")], ["a", "z"])

    def test_multiview_search_deduplicates_and_averages(self) -> None:
        index = BM25Index(
            [
                BM25Document("en", "", "earthquake causes", "en", "fixture"),
                BM25Document("hi", "", "भूकंप कारण", "hi", "fixture"),
            ]
        )
        result = index.search_views(
            ["earthquake", "earthquake", "भूकंप"], top_k=10, per_view_top_k=10
        )
        self.assertEqual(result.unique_view_count, 2)
        self.assertEqual(result.search_count, 2)
        self.assertEqual(result.intermediate_candidate_count, 2)
        self.assertEqual({hit.doc_id for hit in result.hits}, {"en", "hi"})

    def test_runner_emits_stable_shared_result_contract(self) -> None:
        index = BM25Index([BM25Document("d", "", "earthquake", "en", "fixture")])
        results = run_queries(
            index,
            [RetrievalQuery("q", "earthquake")],
            branch="bm25_original",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            (results[0].query_id, results[0].doc_id, results[0].branch, results[0].rank),
            ("q", "d", "bm25_original", 1),
        )

    def test_cli_run_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            corpus.write_text(
                json.dumps(
                    {
                        "doc_id": "d",
                        "title": "Earthquake",
                        "text": "Causes and faults",
                        "language": "en",
                        "source": "fixture",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps(
                    {"query_id": "q", "text": "earthquake causes", "split": "validation"}
                )
                + "\n",
                encoding="utf-8",
            )
            outputs = []
            for name in ("first.jsonl", "second.jsonl"):
                output = root / name
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "retrieval.bm25",
                        "--corpus",
                        str(corpus),
                        "--queries",
                        str(queries),
                        "--split",
                        "validation",
                        "--output",
                        str(output),
                        "--top-k",
                        "100",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                outputs.append(output.read_bytes())
                manifest = output.with_suffix(".manifest.json")
                self.assertTrue(manifest.exists())
                self.assertEqual(json.loads(manifest.read_text())["search_count"], 1)
            self.assertEqual(outputs[0], outputs[1])


class EndToEndPipelineTests(unittest.TestCase):
    def test_prepare_split_bm25_evaluate_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            passages_path = root / "passages.jsonl"
            qrels_path = root / "qrels.jsonl"
            passages_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"docid": "relevant", "title": "Earthquakes", "text": "Fault movement causes earthquakes."}
                        ),
                        json.dumps(
                            {"docid": "other", "title": "Weather", "text": "Clouds create rain."}
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text(
                json.dumps({"query_id": "q", "doc_id": "relevant", "relevance": 1}) + "\n",
                encoding="utf-8",
            )
            prepared = prepare_corpus(
                [CorpusSource("en", passages_path, qrels_path, "fixture")],
                target_size=2,
                seed=3,
            )

            final_query = Query(
                query_id="q",
                parent_query_id="q",
                group_id="q",
                text="what causes earthquakes",
                languages=("en",),
                benchmark_language="en",
                source_partition=OFFICIAL_DEV,
                split=None,
                variant_type="original",
                verified=True,
                review_status="verified",
                judgment_query_id="q",
            )
            split_result = make_splits([], [final_query])
            assigned = split_result.queries[0]
            index = BM25Index(
                [
                    BM25Document(item.doc_id, item.title, item.text, item.language, item.source)
                    for item in prepared.passages
                ]
            )
            retrieval_results = run_queries(
                index, [RetrievalQuery(assigned.query_id, assigned.text, assigned.split)]
            )
            evaluation_query_record = EvaluationQuery(
                query_id=assigned.query_id,
                parent_query_id=assigned.parent_query_id,
                judgment_query_id=assigned.judgment_query_id,
                benchmark_language=assigned.benchmark_language,
                languages=assigned.languages,
                split=assigned.split or "",
                variant_type=assigned.variant_type,
                verified=assigned.verified,
                review_status=assigned.review_status,
                metadata={},
            )
            report = evaluate_run(
                [evaluation_query_record],
                prepared.qrels,
                retrieval_results,
                bootstrap_samples=20,
                seed=3,
            )
            self.assertEqual(retrieval_results[0].doc_id, "relevant")
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["mrr_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["recall_at_100"], 1.0)


if __name__ == "__main__":
    unittest.main()

