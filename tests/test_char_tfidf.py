from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from data.make_splits import OFFICIAL_DEV, Query, make_splits
from data.prepare_corpus import CorpusSource, prepare_corpus
from evaluation.evaluate import EvaluationQuery, evaluate_run, load_run
from retrieval.bm25 import RetrievalQuery
from retrieval.char_tfidf import (
    CharDocument,
    CharTfidfConfig,
    CharTfidfError,
    CharTfidfIndex,
    character_ngrams,
    normalize_romanized_text,
    run_queries,
)


class CharTfidfTests(unittest.TestCase):
    def test_normalization_and_exact_ngram_range(self) -> None:
        self.assertEqual(normalize_romanized_text("  CAFÉ\t Test!  "), "café test!")
        self.assertEqual(
            character_ngrams("Abcd", min_n=3, max_n=4),
            ("abc", "bcd", "abcd"),
        )

    def test_nfc_equivalent_query_matches(self) -> None:
        index = CharTfidfIndex(
            [CharDocument("d", "café", "en", "fixture", "supplied")]
        )
        self.assertEqual(index.search("cafe\u0301")[0].doc_id, "d")

    def test_hand_checked_tfidf_cosine(self) -> None:
        index = CharTfidfIndex(
            [
                CharDocument("d1", "abcd", "en", "fixture", "supplied"),
                CharDocument("d2", "bcdx", "en", "fixture", "supplied"),
            ],
            CharTfidfConfig(3, 3),
        )
        hit = next(item for item in index.search("abc") if item.doc_id == "d1")
        idf_unique = math.log(3 / 2) + 1
        idf_shared = 1.0
        expected = idf_unique / math.sqrt(idf_unique**2 + idf_shared**2)
        self.assertAlmostEqual(hit.score, expected)

    def test_spelling_variation_retrieves_expected_passage(self) -> None:
        index = CharTfidfIndex(
            [
                CharDocument("quake", "earthquake causes", "en", "fixture", "identity"),
                CharDocument("rain", "clouds produce rainfall", "en", "fixture", "identity"),
            ]
        )
        self.assertEqual(index.search("earthquke")[0].doc_id, "quake")

    def test_pending_and_non_roman_documents_are_never_native_fallbacks(self) -> None:
        index = CharTfidfIndex(
            [
                CharDocument("pending", "", "hi", "fixture", "pending"),
                CharDocument("bad", "भूकंप", "hi", "fixture", "supplied"),
                CharDocument("active", "bhukamp", "hi", "fixture", "supplied"),
            ]
        )
        self.assertEqual(index.document_count, 1)
        self.assertEqual(index.manifest["empty_romanized_document_count"], 1)
        self.assertEqual(index.manifest["non_roman_document_count"], 1)
        self.assertEqual(index.search("भूकंप"), ())
        self.assertEqual(index.search("bhukamp")[0].doc_id, "active")

    def test_empty_oov_short_topk_and_empty_index(self) -> None:
        index = CharTfidfIndex(
            [
                CharDocument(f"d{number}", "shared text", "en", "fixture", "identity")
                for number in range(4)
            ]
        )
        self.assertEqual(index.search(""), ())
        self.assertEqual(index.search("zzzzzz"), ())
        self.assertEqual(index.search("a"), ())
        self.assertEqual(len(index.search("shared", top_k=2)), 2)
        self.assertEqual(CharTfidfIndex([]).search("shared"), ())

    def test_duplicate_ids_invalid_config_and_deterministic_ties(self) -> None:
        duplicate = CharDocument("same", "earthquake", "en", "fixture", "identity")
        with self.assertRaisesRegex(CharTfidfError, "Duplicate document"):
            CharTfidfIndex([duplicate, duplicate])
        with self.assertRaisesRegex(CharTfidfError, "greater than"):
            CharTfidfIndex([], CharTfidfConfig(5, 3))
        index = CharTfidfIndex(
            [
                CharDocument("z", "identical", "en", "fixture", "identity"),
                CharDocument("a", "identical", "en", "fixture", "identity"),
            ]
        )
        self.assertEqual([hit.doc_id for hit in index.search("identical")], ["a", "z"])

    def test_runner_uses_shared_contract_and_stable_query_order(self) -> None:
        index = CharTfidfIndex(
            [CharDocument("d", "earthquake", "en", "fixture", "identity")]
        )
        results = run_queries(
            index,
            [RetrievalQuery("z", "earthquke"), RetrievalQuery("a", "earthquake")],
        )
        self.assertEqual([item.query_id for item in results], ["a", "z"])
        self.assertTrue(all(item.branch == "char_tfidf" for item in results))
        self.assertTrue(all(item.rank == 1 for item in results))

    def test_cli_is_reproducible_and_reports_unavailable_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "doc_id": "en:d",
                                "romanized_text": "earthquake causes",
                                "romanization_status": "identity",
                                "language": "en",
                                "source": "fixture",
                            }
                        ),
                        json.dumps(
                            {
                                "doc_id": "hi:d",
                                "romanized_text": "",
                                "romanization_status": "pending",
                                "language": "hi",
                                "source": "fixture",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps(
                    {"query_id": "q", "text": "earthquke", "split": "validation"}
                )
                + "\n",
                encoding="utf-8",
            )
            run_bytes = []
            for name in ("first.jsonl", "second.jsonl"):
                output = root / name
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "retrieval.char_tfidf",
                        "--corpus",
                        str(corpus),
                        "--queries",
                        str(queries),
                        "--split",
                        "validation",
                        "--output",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                run_bytes.append(output.read_bytes())
                manifest = json.loads(
                    output.with_suffix(".manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    manifest["index"]["empty_romanized_document_count"], 1
                )
                self.assertEqual(manifest["active_roman_query_count"], 1)
            self.assertEqual(run_bytes[0], run_bytes[1])


class CharTfidfIntegrationTests(unittest.TestCase):
    def test_prepare_split_char_run_roundtrip_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            passages = root / "passages.jsonl"
            qrels = root / "qrels.jsonl"
            passages.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "docid": "quake",
                                "title": "भूकंप",
                                "text": "भूकंप क्यों आते हैं",
                                "romanized_text": "bhukamp kyu aate hain",
                                "romanization_status": "human_verified",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "docid": "rain",
                                "title": "बारिश",
                                "text": "बारिश कैसे होती है",
                                "romanized_text": "barish kaise hoti hai",
                                "romanization_status": "human_verified",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels.write_text(
                json.dumps({"query_id": "q", "doc_id": "quake", "relevance": 1})
                + "\n",
                encoding="utf-8",
            )
            prepared = prepare_corpus(
                [CorpusSource("hi", passages, qrels, "fixture")],
                target_size=2,
                seed=7,
                prefix_doc_ids=True,
            )
            source_query = Query(
                query_id="q",
                parent_query_id="q",
                group_id="q",
                text="bhookamp kyu ate hain",
                languages=("hi",),
                benchmark_language="hi",
                source_partition=OFFICIAL_DEV,
                split=None,
                variant_type="original",
                verified=True,
                review_status="verified",
                judgment_query_id="q",
            )
            assigned = make_splits([], [source_query]).queries[0]
            index = CharTfidfIndex(
                [
                    CharDocument(
                        item.doc_id,
                        item.romanized_text,
                        item.language,
                        item.source,
                        item.romanization_status,
                    )
                    for item in prepared.passages
                ]
            )
            results = run_queries(
                index, [RetrievalQuery(assigned.query_id, assigned.text, assigned.split)]
            )
            run_path = root / "char.jsonl"
            with run_path.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
            loaded = load_run(run_path)
            evaluation_query = EvaluationQuery(
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
                [evaluation_query], prepared.qrels, loaded, bootstrap_samples=20
            )
            self.assertEqual(loaded[0].doc_id, "hi:quake")
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["mrr_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["recall_at_100"], 1.0)


if __name__ == "__main__":
    unittest.main()
