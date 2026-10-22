"""Offline trace-replay fixtures: no downloaded model or benchmark claims."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from evaluation.study_runs import StudyError, _digest, main, process_study, select_fixed_weights
from retrieval.fusion import BRANCHES, FusionConfig, fuse_rankings


def writejson(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def writerows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def readjson(path):
    return json.loads(path.read_text(encoding="utf-8"))


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudyRunsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        self.queries = self.root / "queries.jsonl"
        self.qrels = self.root / "qrels.jsonl"
        self.output = self.root / "study"
        self.config = FusionConfig()
        writerows(self.corpus, [{"doc_id": "a", "text": "Alpha", "title": "", "language": "en", "source": "explicit_fixture"},
                                {"doc_id": "b", "text": "Beta", "title": "", "language": "hi", "source": "explicit_fixture"}])
        self.records = []
        for split, languages in (("encoder_train", ["en"]), ("fusion_train", ["hi"]),
                                 ("validation", ["en", "en", "en", "hi", "bn"]),
                                 ("final_evaluation", ["en", "hi", "bn"])):
            for number, language in enumerate(languages):
                qid = f"{split}-{number}"
                duplicate = split == "validation" and number == 1
                group = f"{split}-g-{0 if duplicate else number}"
                self.records.append({"query_id": qid, "parent_query_id": qid, "judgment_query_id": qid,
                                     "group_id": group, "text": f"{split} query {0 if duplicate else number}",
                                     "languages": [language], "benchmark_language": language, "split": split,
                                     "source_partition": "official_dev" if split == "final_evaluation" else "official_train",
                                     "variant_type": "original", "verified": True, "review_status": "verified"})
        writerows(self.queries, self.records)
        self.judgments = [{"query_id": row["query_id"], "doc_id": "a" if row["benchmark_language"] == "en" else "b", "relevance": 1}
                          for row in self.records]
        writerows(self.qrels, self.judgments)
        self.pipeline = {"schema_version": 1, "retriever": "multi_view_rrf", "branches": list(BRANCHES),
                         "config": asdict(self.config), "view_config": {"fixture": "explicit_test_only"},
                         "indexes": {"bm25": {"fixture_only": True}, "char_tfidf": {"source_document_count": 2, "indexed_document_count": 2},
                                     "dense": {"encoder": {"fixture_only": True}}},
                         "resources": {"fixture_only": True}, "fixture_only": True}
        self.pipeline["pipeline_fingerprint"] = _digest(self.pipeline)
        self.paths = {}
        for split in ("validation", "final_evaluation"):
            self.make_trace(split)

    def make_trace(self, split, *, rankings=None, adaptive=False, multiview=False, mask_char=False):
        trace_path, manifest_path = self.root / f"{split}.trace.jsonl", self.root / f"{split}.manifest.json"
        pipeline = json.loads(json.dumps(self.pipeline))
        if adaptive:
            pipeline.pop("pipeline_fingerprint")
            pipeline["weight_provider"] = {"kind": "explicit_fixture_gate", "fixture_only": True}
            pipeline["pipeline_fingerprint"] = _digest(pipeline)
        default = {"bm25_original": ["a", "b"], "bm25_native": [], "char_tfidf": ["b", "a"], "dense": ["a", "b"]}
        rankings = rankings if rankings is not None else default
        traces = []
        for query in self.records:
            if query["split"] != split:
                continue
            branches = {branch: [{"doc_id": docid, "rank": rank, "score": 1.0 / rank} for rank, docid in enumerate(rankings[branch], 1)]
                        for branch in BRANCHES}
            if mask_char:
                branches["char_tfidf"] = []
            weights = {branch: (0.7 if branch == "char_tfidf" else 0.1) for branch in BRANCHES} if adaptive else None
            fused = fuse_rankings(branches, weights, config=self.config)
            branch_info = {}
            for branch in BRANCHES:
                views = ([query["text"], "native fixture"] if multiview and branch == "dense"
                         else [] if branch == "bm25_native" or (mask_char and branch == "char_tfidf")
                         else [query["text"]])
                branch_info[branch] = {"views": views, "unique_view_count": len(views), "search_count": len(views),
                                       "intermediate_candidate_count": len(branches[branch]), "candidate_count": len(branches[branch]),
                                       "seconds": 0.01, "unavailable_reason": None if branches[branch] else "explicit_fixture_empty"}
            diagnostics = {"fixture_only": True, "search_count": sum(d["search_count"] for d in branch_info.values()),
                           "intermediate_candidate_count": sum(d["intermediate_candidate_count"] for d in branch_info.values()),
                           "view_generation_seconds": 0.02, "retrieval_seconds": 0.05, "fusion_seconds": 0.01, "total_seconds": 0.08,
                           "branch_diagnostics": branch_info, "views": {"original": query["text"], "native": ["native fixture"] if multiview else [], "roman": query["text"]},
                           "weight_source": "provider" if adaptive else "equal"}
            traces.append({"query_id": query["query_id"], "branch_results": branches,
                           "active_weights": dict(fused.active_weights), "hits": [asdict(h) for h in fused.hits], "diagnostics": diagnostics})
        writerows(trace_path, traces)
        manifest = {"schema_version": 1, "pipeline": pipeline, "fixture_only": True, "split": split,
                    "weights": "learned_gate" if adaptive else "equal", "query_count": len(traces),
                    "result_count": sum(len(t["hits"]) for t in traces), "trace": str(trace_path),
                    "input_sha256": {str(self.corpus): filehash(self.corpus), str(self.queries): filehash(self.queries)},
                    "search_count": sum(t["diagnostics"]["search_count"] for t in traces),
                    "intermediate_candidate_count": sum(t["diagnostics"]["intermediate_candidate_count"] for t in traces),
                    "latency_seconds": {key: sum(t["diagnostics"][key] for t in traces)
                                        for key in ("view_generation_seconds", "retrieval_seconds", "fusion_seconds", "total_seconds")}}
        writejson(manifest_path, manifest)
        self.paths[split] = trace_path, manifest_path
        return traces, manifest

    def run_study(self, split="validation", *, output=None, **kwargs):
        trace, manifest = self.paths[split]
        return process_study(trace, manifest, self.queries, self.qrels, self.corpus, output or self.output,
                             split=split, bootstrap_samples=12, allow_fixture=True, **kwargs)

    def mutate_trace(self, callback, split="validation"):
        path, _ = self.paths[split]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        callback(rows)
        writerows(path, rows)

    def test_validation_metrics_macro_selector_group_bootstrap_and_outputs(self):
        summary = self.run_study()
        self.assertEqual(summary["query_count"], 5)
        self.assertEqual(summary["group_count"], 4)
        self.assertTrue(summary["fixture_only"])
        self.assertEqual(summary["serving_final_top_k"], 10)
        self.assertEqual(summary["evaluation_final_top_k"], 100)
        self.assertEqual(summary["actual_trace_cost"]["search_count"], 15)
        self.assertEqual(summary["actual_trace_cost"]["per_branch"]["bm25_native"]["masked_query_count"], 5)
        self.assertGreater(summary["evaluations"]["fixed_selected"]["macro_languages"]["ndcg_at_10"],
                           summary["evaluations"]["B5_equal"]["macro_languages"]["ndcg_at_10"])
        selection = readjson(self.output / "validation-selection.json")
        self.assertEqual(selection["attempted_candidates"], 35)
        self.assertEqual(selection["source_split"], "validation")
        self.assertEqual(selection["selection_metric"], "language_macro_actual_fused_ndcg_at_10")
        report = readjson(self.output / "B1_bm25_original.evaluation.json")
        self.assertEqual(report["bootstrap"]["parent_group_count"], 4)
        self.assertEqual(report["bootstrap"]["unit"], "information_need_group_id")
        self.assertEqual(report["bootstrap"]["samples"], 12)
        self.assertIn("B2_protected_roman_view_character", summary["evaluations"])
        self.assertIn("B3_dense_original", summary["evaluations"])
        self.assertIn("no_character_fixed", summary["evaluations"])
        for name, sha in summary["artifacts"].items():self.assertEqual(filehash(self.output / name), sha)
        self.assertEqual(summary["study_sha256"], _digest({k:v for k,v in summary.items() if k != "study_sha256"}))

    def test_selector_ignores_all_final_qrels(self):
        self.run_study()
        first = (self.output / "validation-selection.json").read_bytes()
        modified = [{**row, "doc_id": "not-in-corpus", "relevance": "INVALID_FINAL_LABEL"} if row["query_id"].startswith("final_evaluation") else row
                    for row in self.judgments]
        writerows(self.qrels, modified)
        second = self.root / "second"
        self.run_study(output=second)
        self.assertEqual(first, (second / "validation-selection.json").read_bytes())

    def test_final_requires_frozen_selection_and_never_calls_selector(self):
        with self.assertRaisesRegex(StudyError, "requires a frozen"):
            self.run_study("final_evaluation")
        self.run_study()
        frozen = self.output / "validation-selection.json"
        with patch("evaluation.study_runs.select_fixed_weights", side_effect=AssertionError("final must not tune")):
            summary = self.run_study("final_evaluation", output=self.root / "final", selection_path=frozen)
        self.assertEqual(summary["split"], "final_evaluation")
        self.assertEqual(summary["weights"]["fixed_selected"], readjson(frozen)["weights"])
        self.assertFalse((self.root / "final" / "validation-selection.json").exists())

    def test_modified_selection_seal_rejected(self):
        self.run_study()
        path = self.output / "validation-selection.json";selection = readjson(path)
        selection["weights"]["dense"] += 0.25;writejson(path, selection)
        with self.assertRaisesRegex(StudyError, "content seal"):
            self.run_study("final_evaluation", output=self.root / "final", selection_path=path)

    def test_invalid_final_selection_rejected_before_label_parsing(self):
        self.run_study();path=self.output/'validation-selection.json';selection=readjson(path)
        selection['source_split']='final_evaluation';writejson(path,selection)
        with patch('evaluation.study_runs._scoped_qrels',side_effect=AssertionError('final labels parsed too early')):
            with self.assertRaisesRegex(StudyError,'content seal'):
                self.run_study('final_evaluation',output=self.root/'final',selection_path=path)

    def test_frozen_selection_rejects_changed_pipeline(self):
        self.run_study()
        self.make_trace("final_evaluation", adaptive=True)
        with self.assertRaisesRegex(StudyError, "incompatible"):
            self.run_study("final_evaluation", output=self.root / "final", selection_path=self.output / "validation-selection.json")

    def test_fixture_is_api_opt_in_only_and_cli_rejects(self):
        trace, manifest = self.paths["validation"]
        with self.assertRaisesRegex(StudyError, "Fixture traces"):
            process_study(trace, manifest, self.queries, self.qrels, self.corpus, self.output, split="validation")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--trace", str(trace), "--run-manifest", str(manifest), "--queries", str(self.queries),
                  "--qrels", str(self.qrels), "--corpus", str(self.corpus), "--output-dir", str(self.output), "--split", "validation"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_trace_query_membership_missing_or_duplicate_rejected(self):
        for duplicate in (False, True):
            self.make_trace("validation")
            self.mutate_trace(lambda rows: rows.append(rows[0]) if duplicate else rows.pop())
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(StudyError, "query|Trace"):
                self.run_study()

    def test_branch_keys_ranks_scores_and_doc_ids_are_strict(self):
        changes = [lambda rows: rows[0]["branch_results"].pop("dense"),
                   lambda rows: rows[0]["branch_results"]["dense"][0].update(rank=2),
                   lambda rows: rows[0]["branch_results"]["dense"][0].update(doc_id="outside"),
                   lambda rows: rows[0]["branch_results"]["dense"][0].update(score=float("nan")),
                   lambda rows: rows[0]["branch_results"]["dense"].append(dict(rows[0]["branch_results"]["dense"][0]))]
        for change in changes:
            self.make_trace("validation");self.mutate_trace(change)
            with self.subTest(change=change), self.assertRaises(ValueError):self.run_study()
        self.assertFalse(self.output.exists())

    def test_serving_hit_or_active_weight_tampering_rejected(self):
        for change in (lambda rows: rows[0]["hits"][0].update(score=42),
                       lambda rows: rows[0]["active_weights"].update(char_tfidf=0.8)):
            self.make_trace("validation");self.mutate_trace(change)
            with self.assertRaises(StudyError):self.run_study()

    def test_hash_and_advertised_trace_path_mismatch_rejected(self):
        trace, manifest_path = self.paths["validation"]
        manifest = readjson(manifest_path);manifest["trace"] = str(self.root / "other.jsonl");writejson(manifest_path, manifest)
        with self.assertRaisesRegex(StudyError, "advertised"):
            self.run_study()
        self.make_trace("validation")
        self.corpus.write_text(self.corpus.read_text() + "\n")
        with self.assertRaisesRegex(StudyError, "hash/path mismatch"):
            self.run_study()

    def test_no_retrieval_provenance_qrels(self):
        _, manifest_path = self.paths["validation"];m = readjson(manifest_path)
        m["input_sha256"][str(self.qrels)] = filehash(self.qrels);writejson(manifest_path,m)
        with self.assertRaisesRegex(StudyError, "relevance judgments"):
            self.run_study()

    def test_queries_are_verified_original_and_group_isolated(self):
        for change in (lambda rows: rows[0].update(verified=False),
                       lambda rows: rows[0].update(variant_type="romanized"),
                       lambda rows: rows[-1].update(group_id=rows[0]["group_id"]),
                       lambda rows: rows[-1].update(source_partition="official_train")):
            records = json.loads(json.dumps(self.records));change(records);writerows(self.queries,records)
            self.make_trace("validation")
            with self.assertRaises(StudyError):self.run_study()

    def test_masked_character_branch_keeps_empty_run_and_reports_availability(self):
        self.make_trace("validation", mask_char=True)
        report = self.run_study()
        self.assertEqual(report["evaluations"]["B2_protected_roman_view_character"]["missing_run_query_count"],5)
        self.assertEqual((self.output / "B2_protected_roman_view_character.run.jsonl").read_bytes(),b"")
        self.assertEqual(report["actual_trace_cost"]["per_branch"]["char_tfidf"]["masked_query_count"],5)

    def test_adaptive_trace_is_preserved_and_multiview_is_not_called_b3(self):
        self.make_trace("validation", adaptive=True, multiview=True)
        report = self.run_study()
        self.assertEqual(report["trace_weight_method"], "adaptive_learned_gate")
        self.assertIn("dense_views",report["evaluations"])
        self.assertNotIn("B3_dense_original",report["evaluations"])
        self.assertIn("B3_dense_original",report["not_available"])

    def test_top100_replay_does_not_change_top10_serving_pipeline(self):
        writerows(self.corpus,[{"doc_id":f"d{i:03}","text":"fixture","language":"en"} for i in range(110)])
        self.judgments=[{"query_id":r["query_id"],"doc_id":"d050","relevance":1} for r in self.records];writerows(self.qrels,self.judgments)
        ranks={b:([f"d{i:03}" for i in range(100)] if b=="bm25_original" else []) for b in BRANCHES}
        traces,_=self.make_trace("validation",rankings=ranks)
        self.assertTrue(all(len(t["hits"])==10 for t in traces))
        summary=self.run_study()
        self.assertEqual(summary["evaluations"]["fixed_selected"]["overall"]["recall_at_100"],1.0)
        self.assertEqual(summary["evaluations"]["fixed_selected"]["overall"]["ndcg_at_10"],0.0)
        self.assertEqual(summary["source_pipeline"]["config"]["final_top_k"],10)

    def test_existing_output_and_input_overlap_never_overwritten(self):
        self.output.mkdir();sentinel=self.output/"user.txt";sentinel.write_text("keep")
        with self.assertRaisesRegex(StudyError,"fresh"):
            self.run_study()
        self.assertEqual(sentinel.read_text(),"keep")
        with self.assertRaisesRegex(StudyError,"fresh"):
            self.run_study(output=self.root)

    def test_publication_failure_leaves_no_partial_output(self):
        with patch("evaluation.study_runs._write_run",side_effect=OSError("disk error")),self.assertRaises(OSError):
            self.run_study()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".study.study-*")))

    def test_optional_original_dense_run_requires_matching_manifest(self):
        self.make_trace("validation",multiview=True)
        dense_run=self.root/'dense.run.jsonl';dense_manifest=dense_run.with_suffix('.manifest.json')
        entries=[{"query_id":r['query_id'],"doc_id":"a","branch":"explicit_fixture_dense","rank":1,"score":0.9}
                 for r in self.records if r['split']=='validation']
        writerows(dense_run,entries)
        writejson(dense_manifest,{"index":self.pipeline['indexes']['dense'],"selected_split":"validation",
             "queries_path":str(self.queries),"queries_file_sha256":filehash(self.queries),"query_count":5,
             "fixture_only":True,"top_k":100,"branch":"explicit_fixture_dense","result_count":5,
             "search_count":5,"search_seconds":0.5,"latency_includes_view_generation":False})
        summary=self.run_study(dense_original_run=dense_run)
        self.assertIn('B3_dense_original',summary['evaluations'])
        self.assertIn('dense_views',summary['evaluations'])
        self.assertEqual(summary['actual_trace_cost']['separate_original_dense_run']['search_seconds'],0.5)

    def test_selector_refuses_final_queries_even_when_called_directly(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(StudyError,'validation-only'):
            select_fixed_weights([], [SimpleNamespace(split='final_evaluation')], [], self.config)

    def test_candidate_budget_and_actual_call_totals_are_enforced(self):
        for change in (lambda rows: rows[0]['branch_results']['dense'].extend(
                {'doc_id':f'outside-{i}','rank':i+3,'score':0.1} for i in range(99)),
                       lambda rows: rows[0]['diagnostics'].update(search_count=999),
                       lambda rows: rows[0]['diagnostics']['branch_diagnostics']['dense'].update(search_count=4)):
            self.make_trace('validation');self.mutate_trace(change)
            with self.assertRaises(StudyError):self.run_study()

    def test_no_character_ablation_is_only_renormalized_selected_fixed_weights(self):
        summary=self.run_study()
        tuned=summary['weights']['fixed_selected'];ablated=summary['weights']['no_character_fixed']
        self.assertEqual(ablated['char_tfidf'],0)
        remaining=sum(tuned[b] for b in BRANCHES if b!='char_tfidf')
        for b in BRANCHES:
            if b!='char_tfidf':self.assertAlmostEqual(ablated[b],tuned[b]/remaining)

    def test_input_change_before_publication_aborts_atomically(self):
        from evaluation import study_runs
        original=study_runs._write_run
        def mutate_after_write(path,entries):
            original(path,entries)
            self.qrels.write_text(self.qrels.read_text()+'\n')
        with patch('evaluation.study_runs._write_run',side_effect=mutate_after_write),self.assertRaisesRegex(StudyError,'changed'):
            self.run_study()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.study.study-*')))

    def test_live_fixture_pipeline_trace_contract_integrates_with_replay(self):
        from tests.test_gate_pipeline import prepared_fixture
        folder=self.root/'live';folder.mkdir()
        prepared,corpus_paths,split_paths,_,pipeline=prepared_fixture(folder)
        query_rows=[json.loads(line) for line in split_paths['queries'].read_text().splitlines()]
        for row in query_rows:row['text']+=' '+row['query_id']
        writerows(split_paths['queries'],query_rows)
        selected=[row for row in query_rows if row['split']=='validation'];traces=[]
        for row in selected:
            result=pipeline.search(row['text'])
            traces.append({'query_id':row['query_id'],'active_weights':dict(result.active_weights),
                'hits':[asdict(hit) for hit in result.hits],
                'branch_results':{b:[asdict(hit) for hit in hits] for b,hits in result.branch_results.items()},
                'diagnostics':dict(result.diagnostics)})
        trace=folder/'live.trace.jsonl';metadata=folder/'live.manifest.json';writerows(trace,traces)
        writejson(metadata,{'pipeline':pipeline.manifest,'fixture_only':True,'split':'validation','weights':'equal',
              'query_count':len(selected),'result_count':sum(len(t['hits']) for t in traces),'trace':str(trace),
              'input_sha256':{str(corpus_paths['corpus']):filehash(corpus_paths['corpus']),str(split_paths['queries']):filehash(split_paths['queries'])},
              'search_count':sum(t['diagnostics']['search_count'] for t in traces),
              'intermediate_candidate_count':sum(t['diagnostics']['intermediate_candidate_count'] for t in traces),
              'latency_seconds':{k:sum(t['diagnostics'][k] for t in traces) for k in ('view_generation_seconds','retrieval_seconds','fusion_seconds','total_seconds')}})
        report=process_study(trace,metadata,split_paths['queries'],corpus_paths['qrels'],corpus_paths['corpus'],folder/'output',
                             split='validation',allow_fixture=True,bootstrap_samples=10)
        self.assertEqual(report['query_count'],3)
        self.assertTrue(report['fixture_only'])

    def test_equal_validation_ties_preserve_equal_and_match_gate_fixed_selector(self):
        from nlp.language_id import ScriptEvidenceBaseline
        from nlp.query_views import build_query_views
        from training.train_gate import GateExample, compare_validation
        ranks={branch:(['a','b'] if branch!='bm25_native' else []) for branch in BRANCHES}
        traces,_=self.make_trace('validation',rankings=ranks)
        summary=self.run_study()
        self.assertEqual(summary['weights']['fixed_selected'],{branch:0.25 for branch in BRANCHES})
        selection=readjson(self.output/'validation-selection.json')
        self.assertEqual(selection['tie_policy'],'equal_first_then_lexicographic_simplex_tuple')
        class ExplicitEqualFixtureGate:
            _pipeline_manifest={'config':asdict(self.config)}
            def __call__(self,views,branch_results):return {branch:0.25 for branch in BRANCHES}
        examples=[]
        selected_rows={r['query_id']:r for r in self.records if r['split']=='validation'}
        judgments={r['query_id']:{r['doc_id']:r['relevance']} for r in self.judgments}
        for trace in traces:
            row=selected_rows[trace['query_id']]
            examples.append(GateExample(row['query_id'],row['parent_query_id'],row['group_id'],'validation',
                build_query_views(row['text'],ScriptEvidenceBaseline()),trace['branch_results'],judgments[row['query_id']],
                benchmark_language=row['benchmark_language']))
        compared=compare_validation(ExplicitEqualFixtureGate(),examples)
        self.assertEqual(compared['tuned_fixed_weights']['weights'],selection['weights'])
        self.assertAlmostEqual(compared['tuned_fixed_weights']['language_macro_ndcg_at_10'],selection['language_macro_ndcg_at_10'])

    def test_nontrivial_fixed_selector_matches_gate_validation_protocol(self):
        from nlp.language_id import ScriptEvidenceBaseline
        from nlp.query_views import build_query_views
        from training.train_gate import GateExample, compare_validation
        traces=[json.loads(line) for line in self.paths['validation'][0].read_text().splitlines()]
        summary=self.run_study();selection=readjson(self.output/'validation-selection.json')
        class ExplicitEqualFixtureGate:
            _pipeline_manifest={'config':asdict(self.config)}
            def __call__(self,views,branch_results):return {branch:0.25 for branch in BRANCHES}
        by_query={r['query_id']:r for r in self.records};judgments={r['query_id']:{r['doc_id']:r['relevance']} for r in self.judgments}
        examples=[GateExample(t['query_id'],t['query_id'],by_query[t['query_id']]['group_id'],'validation',
                    build_query_views(by_query[t['query_id']]['text'],ScriptEvidenceBaseline()),t['branch_results'],judgments[t['query_id']],
                    benchmark_language=by_query[t['query_id']]['benchmark_language']) for t in traces]
        compared=compare_validation(ExplicitEqualFixtureGate(),examples)
        self.assertEqual(compared['tuned_fixed_weights']['weights'],selection['weights'])
        self.assertAlmostEqual(compared['tuned_fixed_weights']['language_macro_ndcg_at_10'],selection['language_macro_ndcg_at_10'])


if __name__ == '__main__':
    unittest.main()
