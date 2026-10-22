from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.prepare_corpus import Qrel
from evaluation.evaluate import (
    EvaluationError,
    EvaluationQuery,
    RunEntry,
    bootstrap_parent_groups,
    compare_reports,
    evaluate_run,
)


def evaluation_query(
    query_id: str,
    *,
    parent: str | None = None,
    judgment: str | None = None,
    language: str = "en",
    variant_type: str = "original",
    verified: bool = True,
    metadata: dict[str, object] | None = None,
) -> EvaluationQuery:
    parent_id = parent or query_id
    judgment_id = (judgment or parent_id) if verified else None
    return EvaluationQuery(
        query_id=query_id,
        parent_query_id=parent_id,
        judgment_query_id=judgment_id,
        benchmark_language=language,
        languages=(language,),
        split="validation",
        variant_type=variant_type,
        verified=verified,
        review_status="verified" if verified else "unreviewed",
        metadata=metadata or {},
    )


def entries(query_id: str, doc_ids: list[str], branch: str = "fixture") -> list[RunEntry]:
    return [
        RunEntry(query_id, doc_id, branch, rank, float(len(doc_ids) - rank + 1))
        for rank, doc_id in enumerate(doc_ids, start=1)
    ]


class EvaluateTests(unittest.TestCase):
    def test_single_relevant_at_rank_two_has_known_metrics(self) -> None:
        report = evaluate_run(
            [evaluation_query("q")],
            [Qrel("q", "relevant", 1)],
            entries("q", ["unjudged", "relevant"]),
            bootstrap_samples=0,
        )
        metrics = report.summary["overall"]
        self.assertAlmostEqual(metrics["ndcg_at_10"], 1 / math.log2(3))
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertEqual(metrics["recall_at_100"], 1.0)

    def test_graded_ndcg_and_multiple_positive_recall(self) -> None:
        report = evaluate_run(
            [evaluation_query("q")],
            [Qrel("q", "high", 2), Qrel("q", "low", 1)],
            entries("q", ["low"]),
            bootstrap_samples=0,
        )
        ideal = 3.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(report.summary["overall"]["ndcg_at_10"], 1.0 / ideal)
        self.assertEqual(report.summary["overall"]["mrr_at_10"], 1.0)
        self.assertEqual(report.summary["overall"]["recall_at_100"], 0.5)

    def test_cutoffs_are_enforced(self) -> None:
        first_hundred = [f"d{number}" for number in range(1, 101)]
        first_hundred[10] = "relevant-11"
        report_11 = evaluate_run(
            [evaluation_query("q")],
            [Qrel("q", "relevant-11", 1)],
            entries("q", first_hundred),
            bootstrap_samples=0,
        )
        self.assertEqual(report_11.summary["overall"]["ndcg_at_10"], 0.0)
        self.assertEqual(report_11.summary["overall"]["mrr_at_10"], 0.0)
        self.assertEqual(report_11.summary["overall"]["recall_at_100"], 1.0)

        at_101 = [f"x{number}" for number in range(1, 101)] + ["relevant-101"]
        report_101 = evaluate_run(
            [evaluation_query("q")],
            [Qrel("q", "relevant-101", 1)],
            entries("q", at_101),
            bootstrap_samples=0,
        )
        self.assertEqual(report_101.summary["overall"]["recall_at_100"], 0.0)

    def test_missing_run_is_an_empty_ranking_not_a_missing_query(self) -> None:
        report = evaluate_run(
            [evaluation_query("q")],
            [Qrel("q", "relevant", 1)],
            [],
            bootstrap_samples=0,
        )
        self.assertEqual(
            report.summary["overall"],
            {"ndcg_at_10": 0.0, "mrr_at_10": 0.0, "recall_at_100": 0.0},
        )
        self.assertEqual(report.summary["missing_run_query_count"], 1)

    def test_no_positive_and_unverified_queries_are_reported(self) -> None:
        queries = [
            evaluation_query("no-positive"),
            evaluation_query("variant", parent="p", verified=False, variant_type="typo"),
        ]
        report = evaluate_run(
            queries,
            [Qrel("no-positive", "d", 0)],
            [],
            bootstrap_samples=0,
        )
        self.assertIsNone(report.summary["overall"])
        self.assertEqual(
            report.summary["status_counts"],
            {"evaluated": 0, "skipped_unverified": 1, "skipped_no_positive": 1},
        )

    def test_verified_variant_inherits_explicit_judgments(self) -> None:
        queries = [
            evaluation_query("parent"),
            evaluation_query("variant", parent="parent", variant_type="romanized"),
        ]
        run = entries("parent", ["d"]) + entries("variant", ["d"])
        report = evaluate_run(
            queries,
            [Qrel("parent", "d", 1)],
            run,
            bootstrap_samples=0,
        )
        self.assertEqual(report.summary["evaluated_query_count"], 2)
        self.assertEqual(report.summary["robustness"]["paired_parent_count"], 1)
        self.assertEqual(report.summary["robustness"]["mean_drop"], 0.0)

    def test_language_macro_is_not_query_weighted(self) -> None:
        queries = [evaluation_query(f"en-{i}") for i in range(3)] + [
            evaluation_query("hi", language="hi")
        ]
        qrels = [Qrel(item.query_id, f"rel-{item.query_id}", 1) for item in queries]
        run = []
        for item in queries[:3]:
            run.extend(entries(item.query_id, [f"rel-{item.query_id}"]))
        report = evaluate_run(queries, qrels, run, bootstrap_samples=0)

        self.assertEqual(report.summary["overall"]["ndcg_at_10"], 0.75)
        self.assertEqual(report.summary["per_language"]["en"]["metrics"]["ndcg_at_10"], 1.0)
        self.assertEqual(report.summary["per_language"]["hi"]["metrics"]["ndcg_at_10"], 0.0)
        self.assertEqual(report.summary["macro_languages"]["ndcg_at_10"], 0.5)

    def test_clean_to_noisy_drop_is_parent_grouped(self) -> None:
        queries = [
            evaluation_query("p1"),
            evaluation_query("p1-noisy", parent="p1", variant_type="typo"),
            evaluation_query("p2"),
            evaluation_query("p2-noisy", parent="p2", variant_type="romanized"),
        ]
        qrels = [Qrel("p1", "d1", 1), Qrel("p2", "d2", 1)]
        run = entries("p1", ["d1"]) + entries("p2", ["d2"]) + entries("p2-noisy", ["d2"])
        report = evaluate_run(queries, qrels, run, bootstrap_samples=0)

        self.assertEqual(report.summary["robustness"]["paired_parent_count"], 2)
        self.assertEqual(report.summary["robustness"]["per_parent"], {"p1": 1.0, "p2": 0.0})
        self.assertEqual(report.summary["robustness"]["mean_drop"], 0.5)

    def test_bootstrap_is_seeded_and_uses_parent_groups(self) -> None:
        queries = [
            evaluation_query("p"),
            evaluation_query("v", parent="p", variant_type="typo"),
        ]
        qrels = [Qrel("p", "d", 1)]
        run = entries("p", ["d"])
        first = evaluate_run(queries, qrels, run, bootstrap_samples=50, seed=27)
        second = evaluate_run(queries, qrels, run, bootstrap_samples=50, seed=27)

        self.assertEqual(first.summary["bootstrap"], second.summary["bootstrap"])
        self.assertEqual(first.summary["bootstrap"]["parent_group_count"], 1)
        ndcg_interval = first.summary["bootstrap"]["intervals"]["ndcg_at_10"]
        self.assertEqual(ndcg_interval["lower"], 0.5)
        self.assertEqual(ndcg_interval["upper"], 0.5)

    def test_normalization_harm_rate_compares_same_queries(self) -> None:
        queries = [evaluation_query("q1"), evaluation_query("q2")]
        qrels = [Qrel("q1", "d1", 1), Qrel("q2", "d2", 1)]
        baseline = evaluate_run(
            queries,
            qrels,
            entries("q1", ["d1"]) + entries("q2", ["d2"]),
            bootstrap_samples=0,
        )
        comparator = evaluate_run(
            queries,
            qrels,
            entries("q1", ["wrong"]) + entries("q2", ["d2"]),
            bootstrap_samples=0,
        )
        comparison = compare_reports(baseline, comparator)
        self.assertEqual(comparison["harmed_count"], 1)
        self.assertEqual(comparison["tied_count"], 1)
        self.assertEqual(comparison["harm_rate"], 0.5)

    def test_invalid_runs_fail_loudly(self) -> None:
        queries = [evaluation_query("q")]
        qrels = [Qrel("q", "d", 1)]
        with self.assertRaisesRegex(EvaluationError, "duplicate documents"):
            evaluate_run(
                queries,
                qrels,
                [RunEntry("q", "d", "x", 1, 2.0), RunEntry("q", "d", "x", 2, 1.0)],
                bootstrap_samples=0,
            )
        with self.assertRaisesRegex(EvaluationError, "contiguous"):
            evaluate_run(
                queries,
                qrels,
                [RunEntry("q", "d", "x", 2, 1.0)],
                bootstrap_samples=0,
            )
        with self.assertRaisesRegex(EvaluationError, "outside the declared universe"):
            evaluate_run(
                queries,
                qrels,
                [RunEntry("unknown", "d", "x", 1, 1.0)],
                bootstrap_samples=0,
            )

    def test_cli_scores_declared_split_and_writes_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries_path = root / "queries.jsonl"
            qrels_path = root / "qrels.jsonl"
            run_path = root / "run.jsonl"
            output_path = root / "evaluation.json"
            queries_path.write_text(
                json.dumps(
                    {
                        "query_id": "q",
                        "parent_query_id": "q",
                        "judgment_query_id": "q",
                        "benchmark_language": "en",
                        "languages": ["en"],
                        "split": "validation",
                        "variant_type": "original",
                        "verified": True,
                        "review_status": "verified",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text(
                json.dumps({"query_id": "q", "doc_id": "d", "relevance": 1}) + "\n",
                encoding="utf-8",
            )
            run_path.write_text(
                json.dumps(
                    {"query_id": "q", "doc_id": "d", "branch": "bm25", "rank": 1, "score": 1.0}
                )
                + "\n",
                encoding="utf-8",
            )
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.evaluate",
                    "--queries",
                    str(queries_path),
                    "--qrels",
                    str(qrels_path),
                    "--run",
                    str(run_path),
                    "--output",
                    str(output_path),
                    "--split",
                    "validation",
                    "--bootstrap-samples",
                    "20",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report["bootstrap"]["unit"], "parent_query_id")
            self.assertNotIn("NaN", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

