"""Hand-computed fusion checks and fixture-only pipeline integration."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from data.prepare_corpus import Qrel, iter_jsonl
from evaluation.evaluate import EvaluationQuery, evaluate_run, load_run
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator, QueryView, build_query_views
from retrieval.bm25 import BM25Document, BM25Index, RetrievalQuery, write_run_artifacts
from retrieval.char_tfidf import CharDocument, CharTfidfIndex
from retrieval.dense import DenseDocument, DenseIndex, FixtureHashEncoder
from retrieval.fusion import (
    BRANCHES,
    BranchHit,
    FusionConfig,
    FusionError,
    RetrievalPipeline,
    average_rankings,
    fuse_rankings,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DOCUMENTS = (
    ("en:relevant", "earthquake causes", "earthquake causes", "en"),
    ("hi:relevant", "भूकंप क्यों आते हैं", "bhukamp kyu aate hain", "hi"),
    ("bn:relevant", "ভূমিকম্প কেন হয়", "bhumikompo keno hoy", "bn"),
    ("en:weather", "clouds create rain", "clouds create rain", "en"),
)


def weights(**overrides):
    return {branch: overrides.get(branch, 1.0) for branch in BRANCHES}


def ranking(*doc_ids, score=1.0):
    return tuple(BranchHit(doc_id, rank, score) for rank, doc_id in enumerate(doc_ids, 1))


def write_records(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def fixture_indexes():
    records = FIXTURE_DOCUMENTS
    lexical = BM25Index([BM25Document(doc_id, "", text, language, "fusion-fixture") for doc_id, text, _, language in records])
    character = CharTfidfIndex([CharDocument(doc_id, roman, language, "fusion-fixture", "manual_fixture") for doc_id, _, roman, language in records])
    dense = DenseIndex([DenseDocument(doc_id, "", text, language, "fusion-fixture") for doc_id, text, _, language in records], FixtureHashEncoder(64, allow_fixture=True))
    return lexical, character, dense


def fixture_lid_and_transliterator():
    lid = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in iter_jsonl(ROOT / "examples/tiny/lid_train.jsonl")])
    transliterator = LexiconTransliterator.from_jsonl(ROOT / "examples/tiny/xlit_lexicon.jsonl")
    return lid, transliterator


class FusionRankingTests(unittest.TestCase):
    def test_weighted_rrf_uses_ranks_and_masks_empty_branches(self):
        rankings = {
            "bm25_original": (BranchHit("a", 1, 1e9), BranchHit("b", 2, -20.0)),
            "bm25_native": (BranchHit("b", 1, -999.0), BranchHit("c", 2, 1e12)),
            "char_tfidf": (),
        }
        result = fuse_rankings(rankings, weights=weights(bm25_original=2.0, bm25_native=1.0, char_tfidf=1000.0, dense=2000.0))
        self.assertEqual(result.active_weights, {"bm25_original": 2 / 3, "bm25_native": 1 / 3})
        expected = {"a": (2 / 3) / 61, "b": (2 / 3) / 62 + (1 / 3) / 61, "c": (1 / 3) / 62}
        self.assertEqual([hit.doc_id for hit in result.hits], ["b", "a", "c"])
        for hit in result.hits:
            self.assertAlmostEqual(hit.score, expected[hit.doc_id])
            self.assertAlmostEqual(sum(hit.contributions.values()), hit.score)
        changed_scores = {branch: tuple(replace(hit, score=-hit.score * 17) for hit in hits) for branch, hits in rankings.items()}
        self.assertEqual(result.hits, fuse_rankings(changed_scores, weights=weights(bm25_original=2.0, bm25_native=1.0, char_tfidf=1000.0, dense=2000.0)).hits)

    def test_no_results_is_an_empty_valid_run(self):
        result = fuse_rankings({branch: () for branch in BRANCHES})
        self.assertEqual(result.hits, ())
        self.assertEqual(result.active_weights, {})
        self.assertEqual(result.diagnostics["candidate_union_count"], 0)

    def test_equal_rrf_ties_use_document_ids(self):
        result = fuse_rankings({"bm25_original": ranking("z", "a"), "dense": ranking("a", "z")})
        self.assertEqual([hit.doc_id for hit in result.hits], ["a", "z"])
        self.assertEqual([hit.rank for hit in result.hits], [1, 2])
        self.assertEqual(result.hits[0].score, result.hits[1].score)

    def test_within_branch_is_an_average_and_missing_results_contribute_zero(self):
        result = average_rankings({"view one": ranking("a", "b"), "view two": ranking("b", "c"), "empty view": ()})
        expected = {"a": (1 / 61) / 3, "b": (1 / 62 + 1 / 61) / 3, "c": (1 / 62) / 3}
        self.assertEqual([hit.doc_id for hit in result], ["b", "a", "c"])
        for hit in result:
            self.assertAlmostEqual(hit.score, expected[hit.doc_id])

    def test_within_branch_deduplicates_nfc_and_whitespace_views(self):
        baseline = average_rankings({"café query": ranking("a", "b")})
        repeated = average_rankings({"café query": ranking("a", "b"), " cafe\u0301   query ": ranking("a", "b")})
        self.assertEqual(repeated, baseline)
        # Transformer case variants can encode differently and remain distinct.
        case_views = average_rankings({"Case": ranking("a"), "case": ranking("b")})
        self.assertAlmostEqual(case_views[0].score, 0.5 / 61)
        self.assertAlmostEqual(case_views[1].score, 0.5 / 61)

    def test_default_branch_caps_union_and_final_depth(self):
        rankings = {branch: ranking(*(f"{branch}-{number:03}" for number in range(150))) for branch in BRANCHES}
        result = fuse_rankings(rankings)
        self.assertEqual({branch: len(hits) for branch, hits in result.branch_results.items()}, {branch: 100 for branch in BRANCHES})
        self.assertEqual(result.diagnostics["candidate_union_count"], 400)
        self.assertEqual(len(result.hits), 10)
        self.assertEqual(len({hit.doc_id for hit in result.hits}), 10)
        within = average_rankings({"first": ranking(*(f"a-{number:03}" for number in range(150))), "second": ranking(*(f"b-{number:03}" for number in range(150)))})
        self.assertEqual(len(within), 100)
        self.assertEqual([hit.rank for hit in within], list(range(1, 101)))

    def test_invalid_configuration_is_rejected(self):
        for changes in ({"k": -1}, {"k_view": math.inf}, {"final_top_k": 401}, {"per_branch_top_k": 101}, {"per_view_top_k": 101}, {"final_top_k": True}):
            with self.subTest(changes=changes), self.assertRaises(FusionError):
                replace(FusionConfig(), **changes).validate()

    def test_invalid_weights_are_rejected_before_fusion(self):
        rankings = {"bm25_original": ranking("a"), "dense": ranking("b")}
        bad = [weights(dense=-1), weights(dense=math.nan), weights(dense=math.inf), weights(dense=True), {"bm25_original": 1}, {**weights(), "unknown": 1}, {branch: 0 for branch in BRANCHES}]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(FusionError):
                fuse_rankings(rankings, weights=values)

    def test_invalid_rankings_are_rejected_even_beyond_candidate_cap(self):
        invalid = [
            (BranchHit("a", 0, 1.0),),
            (BranchHit("a", True, 1.0),),
            (BranchHit("a", 1, math.nan),),
            (BranchHit("a", 1, math.inf),),
            (BranchHit("a", 1, True),),
            (BranchHit("a", 1, 1.0), BranchHit("a", 2, 2.0)),
            (BranchHit("a", 1, 1.0), BranchHit("b", 1, 2.0)),
            (BranchHit("a", 2, 1.0),),
            (BranchHit("", 1, 1.0),),
            (*ranking(*(f"d-{number}" for number in range(100))), BranchHit("d-0", 101, 2.0)),
        ]
        for hits in invalid:
            with self.subTest(hits=hits), self.assertRaises(FusionError):
                fuse_rankings({"bm25_original": hits})
            with self.subTest(within=hits), self.assertRaises(FusionError):
                average_rankings({"view": hits})
        with self.assertRaises(FusionError):
            fuse_rankings({"unrecognized": ranking("a")})


class FusionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lid, cls.transliterator = fixture_lid_and_transliterator()

    def test_real_views_all_retrievers_actual_cost_and_serialized_evaluation(self):
        lexical, character, dense = fixture_indexes()
        pipeline = RetrievalPipeline(lexical, character, dense, self.lid, self.transliterator)
        searches, hit_counts = [], []
        def record_search(branch, search):
            def wrapped(text, *, top_k=100):
                hits = search(text, top_k=top_k)
                searches.append((branch, text, top_k))
                hit_counts.append(len(hits))
                return hits
            return wrapped
        with patch.object(lexical, "search", side_effect=record_search("bm25", lexical.search)), patch.object(character, "search", side_effect=record_search("char", character.search)), patch.object(dense, "search", side_effect=record_search("dense", dense.search)):
            result = pipeline.search("bhukamp kyu aate hain")
        self.assertEqual(result.hits[0].doc_id, "hi:relevant")
        self.assertEqual(result.diagnostics["views"]["native"], ["भूकंप क्यों आते हैं"])
        self.assertEqual(result.diagnostics["search_count"], len(searches))
        self.assertEqual(result.diagnostics["intermediate_candidate_count"], sum(hit_counts))
        self.assertEqual(len(searches), 5)  # original/native BM25, Roman TF-IDF, original/native dense
        self.assertTrue(result.diagnostics["fixture_only"])
        self.assertLessEqual(result.diagnostics["candidate_union_count"], 400)
        self.assertAlmostEqual(sum(result.active_weights.values()), 1.0)
        for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds"):
            self.assertGreaterEqual(result.diagnostics[key], 0.0)
        self.assertGreaterEqual(result.diagnostics["total_seconds"], result.diagnostics["view_generation_seconds"] + result.diagnostics["retrieval_seconds"] + result.diagnostics["fusion_seconds"])
        queries = [RetrievalQuery("q-hi", "bhukamp kyu aate hain", "final_evaluation")]
        run = pipeline.run_queries(queries, branch="fusion_fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run_artifacts(run, root / "run.jsonl", {"fixture_only": True}, root / "manifest.json")
            loaded = load_run(root / "run.jsonl")
            self.assertEqual([entry.rank for entry in loaded], list(range(1, len(loaded) + 1)))
            report = evaluate_run([EvaluationQuery("q-hi", "q-hi", "q-hi", "hi", ("hi",), "final_evaluation", "original", True, "original", {})], [Qrel("q-hi", "hi:relevant", 1)], loaded, bootstrap_samples=10)
            self.assertEqual(report.per_query[0].ndcg_at_10, 1.0)
            self.assertEqual(report.per_query[0].mrr_at_10, 1.0)
            self.assertEqual(report.per_query[0].recall_at_100, 1.0)

    def test_no_native_rewrite_does_not_repeat_original_bm25(self):
        lexical, _, _ = fixture_indexes()
        pipeline = RetrievalPipeline(lexical, language_identifier=self.lid, transliterator=self.transliterator)
        with patch.object(lexical, "search", wraps=lexical.search) as search:
            result = pipeline.search("earthquake causes")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result.active_weights, {"bm25_original": 1.0})
        self.assertEqual(result.branch_results["bm25_native"], ())
        self.assertEqual(result.diagnostics["search_count"], 1)
        self.assertEqual(result.diagnostics["branch_diagnostics"]["bm25_native"]["search_count"], 0)
        self.assertIsNotNone(result.diagnostics["branch_diagnostics"]["char_tfidf"]["unavailable_reason"])
        self.assertIsNotNone(result.diagnostics["branch_diagnostics"]["dense"]["unavailable_reason"])

    def test_token_equivalent_native_hypotheses_do_not_double_count_bm25(self):
        lexical, _, _ = fixture_indexes()
        original = build_query_views("earthquake causes")
        repeated = replace(original, native_views=(QueryView("native", "EARTHQUAKE causes"), QueryView("native", " earthquake   causes ")))
        pipeline = RetrievalPipeline(lexical)
        with patch("retrieval.fusion.build_query_views", return_value=repeated), patch.object(lexical, "search", wraps=lexical.search) as search:
            result = pipeline.search("earthquake causes")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result.branch_results["bm25_native"], ())
        self.assertEqual(result.active_weights, {"bm25_original": 1.0})

    def test_empty_roman_index_is_unavailable_and_weights_renormalize(self):
        lexical, _, _ = fixture_indexes()
        character = CharTfidfIndex([CharDocument("hi:relevant", "", "hi", "fixture", "pending")])
        pipeline = RetrievalPipeline(lexical, character)
        with patch.object(character, "search", wraps=character.search) as search:
            result = pipeline.search("earthquake")
        self.assertEqual(search.call_count, 0)
        self.assertEqual(result.active_weights, {"bm25_original": 1.0})
        self.assertIsNotNone(result.diagnostics["branch_diagnostics"]["char_tfidf"]["unavailable_reason"])

    def test_incompatible_document_universes_are_rejected(self):
        lexical, _, _ = fixture_indexes()
        foreign_character = CharTfidfIndex([CharDocument("foreign", "earthquake", "en", "fixture", "manual_fixture")])
        incomplete_dense = DenseIndex([DenseDocument("hi:relevant", "", "भूकंप", "hi", "fixture")], FixtureHashEncoder(allow_fixture=True))
        for character, dense in ((foreign_character, None), (None, incomplete_dense)):
            with self.subTest(character=character, dense=dense), self.assertRaises(FusionError):
                RetrievalPipeline(lexical, character, dense)

    def test_weight_provider_receives_observable_retrieval_and_explicit_override_wins(self):
        lexical, character, dense = fixture_indexes()
        observations = []
        def provider(views, branch_results):
            observations.append((views.original_text, set(branch_results)))
            return weights(bm25_original=0, bm25_native=0, char_tfidf=0, dense=1)
        pipeline = RetrievalPipeline(lexical, character, dense, self.lid, self.transliterator, weight_provider=provider)
        result = pipeline.search("bhukamp kyu aate hain")
        self.assertEqual(observations, [("bhukamp kyu aate hain", set(BRANCHES))])
        self.assertEqual(result.active_weights.get("dense"), 1.0)
        self.assertEqual(result.diagnostics["weight_source"], "provider")
        explicit = pipeline.search("bhukamp kyu aate hain", weights=weights())
        self.assertEqual(len(observations), 1)
        self.assertEqual(explicit.diagnostics["weight_source"], "explicit")

    def test_batch_contract_is_stable_and_rejects_duplicate_ids_or_unmarked_fixtures(self):
        lexical, character, dense = fixture_indexes()
        pipeline = RetrievalPipeline(lexical, character, dense)
        queries = [RetrievalQuery("q-z", "earthquake"), RetrievalQuery("q-a", "rain")]
        first = pipeline.run_queries(queries, branch="fusion_fixture")
        second = pipeline.run_queries(list(reversed(queries)), branch="fusion_fixture")
        self.assertEqual(first, second)
        self.assertEqual(first[0].query_id, "q-a")
        self.assertTrue(all(entry.branch == "fusion_fixture" for entry in first))
        with self.assertRaises(FusionError):
            pipeline.run_queries(queries)
        with self.assertRaises(FusionError):
            pipeline.run_queries([RetrievalQuery("q", "rain"), RetrievalQuery("q", "earthquake")], branch="fusion_fixture")

    def cli_fixture(self, root):
        corpus, queries, lid_path = root / "corpus.jsonl", root / "queries.jsonl", root / "lid.json"
        write_records(corpus, [{"doc_id": doc_id, "title": "", "text": text, "romanized_text": roman, "language": language, "source": "fusion-fixture", "romanization_status": "manual_fixture"} for doc_id, text, roman, language in FIXTURE_DOCUMENTS])
        write_records(queries, [
            {"query_id": "q-hi", "text": "bhukamp kyu aate hain", "split": "validation"},
            {"query_id": "ignored", "text": "private training query", "split": "encoder_training"},
            {"query_id": "q-en", "text": "earthquake causes", "split": "validation"},
        ])
        self.lid.save_json(lid_path)
        _, _, dense = fixture_indexes()
        dense.save(root / "dense")
        command = [sys.executable, "-m", "retrieval.fusion", "--corpus", str(corpus), "--queries", str(queries), "--split", "validation", "--lid-model", str(lid_path), "--lexicon", str(ROOT / "examples/tiny/xlit_lexicon.jsonl"), "--dense-index", str(root / "dense"), "--backend", "fixture-hash", "--allow-fixture-encoder"]
        return command, corpus, queries, lid_path

    def run_cli(self, command):
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    def test_cli_split_provenance_trace_overwrite_and_reproducible_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, corpus, queries, lid_path = self.cli_fixture(root)
            outputs = []
            for name in ("first", "second"):
                output, trace = root / f"{name}.jsonl", root / f"{name}.trace.jsonl"
                completed = self.run_cli([*command, "--output", str(output), "--trace", str(trace)])
                self.assertEqual(completed.returncode, 0, completed.stderr)
                outputs.append(output.read_bytes())
                entries = load_run(output)
                self.assertEqual({entry.query_id for entry in entries}, {"q-en", "q-hi"})
                self.assertEqual(entries[0].query_id, "q-en")
                self.assertTrue(all(entry.branch == "fusion_fixture" for entry in entries))
                manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["split"], "validation")
                self.assertEqual(manifest["query_count"], 2)
                self.assertTrue(manifest["fixture_only"])
                self.assertEqual(manifest["search_count"], 8)
                self.assertEqual(set(manifest["input_sha256"]), {str(corpus), str(queries), str(lid_path), str(ROOT / "examples/tiny/xlit_lexicon.jsonl"), str(root / "dense/dense_manifest.json"), str(root / "dense/vectors.jsonl")})
                traces = list(iter_jsonl(trace))
                self.assertEqual([record["query_id"] for record in traces], ["q-en", "q-hi"])
                self.assertEqual(manifest["search_count"], sum(record["diagnostics"]["search_count"] for record in traces))
                self.assertEqual(manifest["intermediate_candidate_count"], sum(record["diagnostics"]["intermediate_candidate_count"] for record in traces))
                self.assertGreaterEqual(manifest["latency_seconds"]["total_seconds"], manifest["latency_seconds"]["view_generation_seconds"])
            self.assertEqual(outputs[0], outputs[1])
            output = root / "first.jsonl"
            denied = self.run_cli([*command, "--output", str(output)])
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("overwrite", denied.stderr.lower())
            self.assertEqual(output.read_bytes(), outputs[0])
            forced = self.run_cli([*command, "--output", str(output), "--force"])
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertEqual(output.read_bytes(), outputs[0])

    def test_cli_rejects_missing_split_input_overwrite_and_fixture_mislabel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, corpus, _, _ = self.cli_fixture(root)
            no_split = list(command)
            split_position = no_split.index("--split")
            del no_split[split_position:split_position + 2]
            missing = self.run_cli([*no_split, "--output", str(root / "missing.jsonl")])
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse((root / "missing.jsonl").exists())
            before = corpus.read_bytes()
            overwrite_input = self.run_cli([*command, "--output", str(corpus), "--force"])
            self.assertNotEqual(overwrite_input.returncode, 0)
            self.assertEqual(corpus.read_bytes(), before)
            mislabeled = self.run_cli([*command, "--output", str(root / "mislabeled.jsonl"), "--branch", "fusion_fixed"])
            self.assertNotEqual(mislabeled.returncode, 0)
            self.assertIn("fixture", mislabeled.stderr.lower())
            self.assertFalse((root / "mislabeled.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
