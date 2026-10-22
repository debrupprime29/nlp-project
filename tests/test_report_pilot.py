"""Tiny sealed fixtures only: no public final data or neural model is opened."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from evaluation.report_pilot import LANGUAGES, SEEDS, VARIANTS, main, report_pilot
from evaluation.study_runs import StudyError, _digest, _sha256, process_study
from retrieval.fusion import BRANCHES, FusionConfig, fuse_rankings
from training.train_gate import FEATURE_NAMES, LinearSoftmaxGate


def writejson(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def writerows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def readjson(path):
    return json.loads(path.read_text(encoding="utf-8"))


class PilotReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the 27-track integration fixture once. Between cases restore
        # changed/deleted files and remove newly created outputs, keeping the
        # absolute provenance paths stable without thousands of repeated fits.
        fixture = cls("test_complete_report_macro_seed_variation_costs_and_qualifications")
        fixture.create_fixture()
        cls.addClassCleanup(fixture.temp.cleanup)
        cls.fixture_attributes = {key: getattr(fixture, key) for key in
                                  ("root", "study", "output", "scope", "lock", "profiles")}
        cls.fixture_files = {path: path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()}
        cls.fixture_directories = {path for path in fixture.root.rglob("*") if path.is_dir()}

    def setUp(self):
        self.__dict__.update(self.fixture_attributes)
        for path in sorted(self.root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            if path.is_symlink() or (path.is_file() and path not in self.fixture_files):
                path.unlink()
            elif path.is_dir() and path not in self.fixture_directories:
                path.rmdir()
        for path, content in self.fixture_files.items():
            if not path.exists() or path.read_bytes() != content:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

    def create_fixture(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.output = self.root / "report"
        self.scope = self.root / "scope.json"
        self.lock = self.study / "final-lock.json"
        writejson(self.scope, {"scope": "public_data_only_clean_query_study", "scope_approved_by_user": "fixture",
                              "human_challenge_set_available": False,
                              "unreviewed_variants_used_for_supervised_training": False,
                              "unreviewed_variants_used_for_scored_evaluation": False,
                              "prior_e5_miracl_training_exposure_disclosed": True,
                              "consistency_weight": 0, "pilot": {"target_passages_per_language": 1500,
                              "max_length_tokens": 128, "document_encoding_batch_size": 8,
                              "encoder_training_seeds": list(SEEDS)}})
        self.profiles = {}
        for language in LANGUAGES:
            profile = self.root / "profile" / language
            corpus, queries, qrels = [profile / name for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")]
            writerows(corpus, [{"doc_id": "a", "text": "Alpha", "title": "", "language": language},
                                {"doc_id": "b", "text": "Beta", "title": "", "language": language}])
            records = []
            for split in ("encoder_train", "fusion_train", "validation", "final_evaluation"):
                for number in range(2 if split == "final_evaluation" else 1):
                    qid = f"{language}:{split}:{number}"
                    records.append({"query_id": qid, "parent_query_id": qid, "judgment_query_id": qid,
                                    "group_id": qid, "text": qid + " fixture query", "languages": [language],
                                    "benchmark_language": language, "split": split,
                                    "source_partition": "official_dev" if split == "final_evaluation" else "official_train",
                                    "variant_type": "original", "verified": True, "review_status": "verified"})
            writerows(queries, records)
            writerows(qrels, [{"query_id": row["query_id"], "doc_id": "a", "relevance": 1} for row in records])
            self.profiles[language] = corpus, queries, qrels, records
        for model in ("frozen", *(f"adapted-seed-{seed}" for seed in SEEDS)):
            adapted = model != "frozen"
            if adapted:
                writejson(self.study / model / "checkpoint" / "weights.json", {"fixture_only": True, "model": model})
            for language in LANGUAGES:
                track = self.study / model / language
                writejson(track / "dense" / "dense_manifest.json", {"fixture_only": True, "model": model})
                for variant in VARIANTS if adapted else ("base",):
                    self.make_track(model, language, variant)
        self.freeze()

    def make_track(self, model, language, variant):
        track = self.study / model / language
        result = track if variant == "base" else track / f"{variant}-gate"
        corpus, queries, qrels, records = self.profiles[language]
        adapted = model != "frozen"
        config = FusionConfig()
        char = {"source_document_count": 2, "indexed_document_count": 0 if language == "bn" else 2}
        pipeline = {"schema_version": 1, "retriever": "multi_view_rrf", "branches": list(BRANCHES),
                    "config": asdict(config), "view_config": {}, "resources": {"fixture_only": True},
                    "fixture_only": True, "indexes": {"bm25": {"fixture_only": True}, "char_tfidf": char,
                    "dense": {"document_count": 2, "index_fingerprint": model + ":" + language,
                    "fine_tuned": adapted, "fixture_only": True, "config": {"max_length": 128, "batch_size": 8,
                    "truncation_side": "right", "truncation_policy": "single_original_passage_head"}}}}
        writejson(track / "dense" / "dense_manifest.json", pipeline["indexes"]["dense"])
        if variant != "base":
            excluded = ([] if variant == "full" else ["mean_token_entropy", "uncertain_token_fraction"]
                        if variant == "no_uncertainty" else [name for name in FEATURE_NAMES if name.startswith("overlap_")])
            width = len(FEATURE_NAMES)
            report = {"excluded_features": excluded, "training_split": "fusion_train", "scaler_fit_split": "fusion_train",
                      "training_query_ids": [row["query_id"] for row in records if row["split"] == "fusion_train"],
                      "validation_query_ids": [row["query_id"] for row in records if row["split"] == "validation"],
                      "hyperparameter_selection": {"split": "validation"}, "fixture_only": True}
            gate = LinearSoftmaxGate([[0] * width for _ in BRANCHES], [0] * len(BRANCHES), [0] * width, [1] * width,
                                    pipeline_manifest=pipeline, training_report=report)
            gate_path = track / "gates" / f"{variant}.json"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate.save_json(gate_path)
            pipeline["weight_provider"] = gate.metadata
        pipeline["pipeline_fingerprint"] = _digest(pipeline)
        for split in ("validation", "final_evaluation"):
            traces = []
            for row in records:
                if row["split"] != split:
                    continue
                branches = {}
                for branch in BRANCHES:
                    ids = [] if branch == "bm25_native" or (language == "bn" and branch == "char_tfidf") else ["a", "b"]
                    if branch == "dense" and model == "adapted-seed-29":
                        ids = list(reversed(ids))
                    branches[branch] = [{"doc_id": doc, "rank": rank, "score": 1 / rank} for rank, doc in enumerate(ids, 1)]
                fused = fuse_rankings(branches, config=config)
                info = {branch: {"views": [row["text"]] if hits else [], "unique_view_count": int(bool(hits)),
                                 "search_count": int(bool(hits)), "intermediate_candidate_count": len(hits),
                                 "candidate_count": len(hits), "seconds": .01,
                                 "unavailable_reason": None if hits else "fixture_unavailable"}
                        for branch, hits in branches.items()}
                diagnostics = {"fixture_only": True, "branch_diagnostics": info,
                               "views": {"original": row["text"], "native": [], "roman": row["text"]},
                               "search_count": sum(item["search_count"] for item in info.values()),
                               "intermediate_candidate_count": sum(item["intermediate_candidate_count"] for item in info.values()),
                               "view_generation_seconds": .01, "retrieval_seconds": .04,
                               "fusion_seconds": .01, "total_seconds": .06,
                               "weight_source": "equal" if variant == "base" else "provider"}
                traces.append({"query_id": row["query_id"], "branch_results": branches,
                               "active_weights": dict(fused.active_weights), "hits": [asdict(hit) for hit in fused.hits],
                               "diagnostics": diagnostics})
            trace, manifest = result / f"{split}.trace.jsonl", result / f"{split}.run.manifest.json"
            writerows(trace, traces)
            writejson(manifest, {"schema_version": 1, "pipeline": pipeline, "fixture_only": True,
                                "split": split, "trace": str(trace), "weights": "equal" if variant == "base" else "learned_gate",
                                "query_count": len(traces), "result_count": sum(len(t["hits"]) for t in traces),
                                "input_sha256": {str(corpus): _sha256(corpus), str(queries): _sha256(queries)},
                                "search_count": sum(t["diagnostics"]["search_count"] for t in traces),
                                "intermediate_candidate_count": sum(t["diagnostics"]["intermediate_candidate_count"] for t in traces),
                                "latency_seconds": {key: sum(t["diagnostics"][key] for t in traces)
                                for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds")}})
            kwargs = {"selection_path": result / "validation-analysis" / "validation-selection.json"} if split == "final_evaluation" else {}
            process_study(trace, manifest, queries, qrels, corpus,
                          result / ("final-analysis" if split == "final_evaluation" else "validation-analysis"),
                          split=split, allow_fixture=True, bootstrap_samples=4, grid_resolution=1, **kwargs)

    def freeze(self):
        # Final outputs are deliberately not in the pre-final input lock.
        files = {str(path): _sha256(path) for path in self.root.rglob("*") if path.is_file()
                 and path != self.lock and "final-analysis" not in path.parts
                 and not path.name.startswith("final_evaluation") and self.output not in path.parents}
        lock = {"files_sha256": files, "deployment_rule": "seed 13, full gate; report both seeds"}
        lock["lock_sha256"] = _digest(lock)
        writejson(self.lock, lock)

    def aggregate(self, **kwargs):
        return report_pilot(self.study, self.output, scope_config=self.scope, allow_fixture=True, **kwargs)

    def change_summary(self, callback, *, model="frozen", language="en", variant="base"):
        result = self.study / model / language
        if variant != "base":
            result /= f"{variant}-gate"
        path = result / "final-analysis" / "study_manifest.json"
        value = readjson(path)
        callback(value, path.parent)
        value["study_sha256"] = _digest({key: item for key, item in value.items() if key != "study_sha256"})
        writejson(path, value)

    def test_complete_report_macro_seed_variation_costs_and_qualifications(self):
        report = self.aggregate()
        self.assertEqual(report["actual_track_count"], 27)
        self.assertEqual(len(report["tracks"]), 9)
        self.assertTrue(report["fixture_only"])
        self.assertEqual(report["report_sha256"], _digest({key: item for key, item in report.items() if key != "report_sha256"}))
        frozen = report["tracks"]["frozen"]
        self.assertEqual(frozen["languages"]["bn"]["evaluations"]["B2_protected_roman_view_character"]["missing_run_query_count"], 2)
        self.assertEqual(frozen["macro_languages"]["B2_protected_roman_view_character"]["missing_run_query_count"], 2)
        self.assertEqual(frozen["languages"]["bn"]["actual_trace_cost"]["per_branch"]["char_tfidf"]["masked_query_count"], 2)
        self.assertEqual(frozen["languages"]["en"]["actual_trace_cost"]["total_seconds_per_query"], .06)
        variation = report["across_seed_descriptive"]["base"]["B3_dense_original"]["macro_languages"]
        self.assertEqual(variation["seed_count"], 2)
        self.assertGreater(variation["metrics"]["ndcg_at_10"]["range"], .3)
        self.assertIn("report both seeds", report["pre_final_reporting_rule"]["deployment_rule"])
        text = (self.output / "report.md").read_text()
        for phrase in ("zero-shot", "Bengali gold", "not independent", "not standalone", "50,000", "Fixture-only"):
            self.assertIn(phrase, text)
        self.assertEqual(readjson(self.output / "report.json")["report_sha256"], report["report_sha256"])

    def test_fixture_opt_in_and_cli_refusal(self):
        with self.assertRaisesRegex(StudyError, "Fixture-only"):
            report_pilot(self.study, self.output, scope_config=self.scope)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--study-root", str(self.study), "--output-dir", str(self.output), "--scope-config", str(self.scope)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_lock_seal_rejected_before_final_reads(self):
        value = readjson(self.lock); value["deployment_rule"] = "pick final winner"; writejson(self.lock, value)
        with patch("evaluation.report_pilot._track", side_effect=AssertionError("final read too early")):
            with self.assertRaisesRegex(StudyError, "lock content seal"):
                self.aggregate()

    def test_changed_locked_input_rejected(self):
        self.scope.write_text("{}")
        with self.assertRaisesRegex(StudyError, "Changed final lock file"):
            self.aggregate()

    def test_missing_frozen_closure_rejected(self):
        lock = readjson(self.lock)
        path = self.study / "adapted-seed-29" / "bn" / "gates" / "no_agreement.json"
        del lock["files_sha256"][str(path)]
        lock["lock_sha256"] = _digest({key: item for key, item in lock.items() if key != "lock_sha256"})
        writejson(self.lock, lock)
        with self.assertRaisesRegex(StudyError, "missing frozen"):
            self.aggregate()

    def test_missing_track_is_not_silently_omitted(self):
        (self.study / "adapted-seed-29" / "bn" / "full-gate" / "final-analysis" / "study_manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.aggregate()

    def test_changed_final_artifact_rejected(self):
        path = self.study / "frozen" / "en" / "final-analysis" / "B1_bm25_original.evaluation.json"
        value = readjson(path); value["overall"]["ndcg_at_10"] = .3; writejson(path, value)
        with self.assertRaisesRegex(StudyError, "Changed final artifact"):
            self.aggregate()

    def test_resealed_summary_cannot_hide_missing_method(self):
        self.change_summary(lambda value, _: value["evaluations"].pop("B3_dense_original"))
        with self.assertRaisesRegex(StudyError, "Missing required"):
            self.aggregate()

    def test_resealed_summary_cannot_change_fixed_selection(self):
        self.change_summary(lambda value, _: value["weights"]["fixed_selected"].update(dense=.123))
        with self.assertRaisesRegex(StudyError, "locked validation selection"):
            self.aggregate()

    def test_resealed_evaluation_cannot_change_per_query_mean(self):
        def change(value, directory):
            path = directory / "B1_bm25_original.evaluation.json"
            evaluation = readjson(path); evaluation["per_query"][0]["ndcg_at_10"] = .25; writejson(path, evaluation)
            value["artifacts"][path.name] = _sha256(path)
        self.change_summary(change)
        with self.assertRaisesRegex(StudyError, "per-query means"):
            self.aggregate()

    def test_resealed_evaluation_cannot_change_missing_run_count(self):
        def change(value, directory):
            path = directory / "B1_bm25_original.evaluation.json"
            evaluation = readjson(path); evaluation["missing_run_query_count"] = 1; writejson(path, evaluation)
            value["artifacts"][path.name] = _sha256(path)
        self.change_summary(change)
        with self.assertRaisesRegex(StudyError, "Missing-run count"):
            self.aggregate()

    def test_boolean_query_count_rejected(self):
        self.change_summary(lambda value, _: value.update(query_count=True))
        with self.assertRaisesRegex(StudyError, "Invalid final query count"):
            self.aggregate()

    def test_nonfinite_metric_rejected(self):
        def change(value, directory):
            path = directory / "B1_bm25_original.evaluation.json"
            evaluation = readjson(path); evaluation["per_query"][0]["ndcg_at_10"] = float("nan"); writejson(path, evaluation)
            value["artifacts"][path.name] = _sha256(path)
        self.change_summary(change)
        with self.assertRaisesRegex(StudyError, "Invalid ndcg_at_10"):
            self.aggregate()

    def test_unsafe_artifact_name_rejected(self):
        self.change_summary(lambda value, _: value["artifacts"].update({"../escape.json": "a" * 64}))
        with self.assertRaisesRegex(StudyError, "artifact hash/path"):
            self.aggregate()

    def test_scope_protocol_change_rejected_even_when_relocked(self):
        value = readjson(self.scope); value["pilot"]["max_length_tokens"] = 512; writejson(self.scope, value)
        self.freeze()
        with self.assertRaisesRegex(StudyError, "approved public CPU-pilot scope"):
            self.aggregate()

    def test_boolean_only_fixture_opt_in_and_nested_model_output_guard(self):
        with self.assertRaisesRegex(StudyError, "must be boolean"):
            report_pilot(self.study, self.output, scope_config=self.scope, allow_fixture="yes")
        with self.assertRaisesRegex(StudyError, "nested inside"):
            report_pilot(self.study, self.study / "frozen" / "en" / "dense" / "report",
                         scope_config=self.scope, allow_fixture=True)

    def test_path_symlinks_and_overwrites_rejected(self):
        self.aggregate()
        with self.assertRaisesRegex(StudyError, "fresh"):
            self.aggregate()
        with self.assertRaisesRegex(StudyError, "fresh"):
            report_pilot(self.study, self.root, scope_config=self.scope, allow_fixture=True)
        link = self.root / "symlink-lock.json"
        link.symlink_to(self.lock)
        with self.assertRaisesRegex(StudyError, "Symlink"):
            report_pilot(self.study, self.root / "other", final_lock=link, scope_config=self.scope, allow_fixture=True)

    def test_input_mutation_during_aggregation_does_not_publish(self):
        import evaluation.report_pilot as module
        original = module._markdown
        def mutate(report):
            text = original(report)
            self.scope.write_text("changed while reporting")
            return text
        with patch("evaluation.report_pilot._markdown", side_effect=mutate):
            with self.assertRaisesRegex(StudyError, "changed during aggregation"):
                self.aggregate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".report.report-*")))

    def test_no_training_selection_or_retrieval_during_reporting(self):
        with patch("evaluation.study_runs.select_fixed_weights", side_effect=AssertionError("must not select")), \
             patch("training.train_gate.train_gate", side_effect=AssertionError("must not fit")):
            self.aggregate()


if __name__ == "__main__":
    unittest.main()
