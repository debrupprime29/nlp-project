"""Pure Python batch integration fixtures: no Torch, network or real models."""

from __future__ import annotations

import gc
import io
import json
import os
import tempfile
import unittest
import weakref
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data.prepare_corpus import iter_jsonl
from evaluation import pilot_workflow as workflow
from evaluation.study_runs import process_study
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator
from retrieval.bm25 import BM25Index, load_corpus as bm25_corpus
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as char_corpus
from retrieval.dense import DenseConfig, DenseIndex, FixtureHashEncoder, load_corpus as dense_corpus
from retrieval.fusion import RetrievalPipeline
from training.checkpoints import verify_checkpoint, write_checkpoint_manifest


def writejson(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def writerows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sealed(document, key="manifest_sha256"):
    return {**document, key: workflow._digest(document)}


class InjectedEncoder(FixtureHashEncoder):
    def __init__(self, checkpoint=None):
        super().__init__(16, allow_fixture=True)
        self.checkpoint = verify_checkpoint(checkpoint) if checkpoint else None

    @property
    def metadata(self):
        if self.checkpoint is None:
            return super().metadata
        manifest = self.checkpoint
        fingerprint = manifest["checkpoint_sha256"]
        return {"backend": "fixture_torch_trained", "model_id": "fixture/hash-encoder", "dimension": 16,
                "resolved_revision": fingerprint, "tokenizer_revision": fingerprint, "fixture_only": True,
                "fine_tuned": True, "checkpoint_sha256": fingerprint,
                "preprocessing": manifest["metadata"]["preprocessing"]}


class PilotWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile, self.study = self.root / "profile", self.root / "study"
        self.lid, self.lexicon = self.root / "lid.json", self.root / "lexicon.jsonl"
        self.languages, self.seeds = ("en", "hi"), (13, 29)
        self.config = DenseConfig(max_length=128, batch_size=8)
        self.calls, self.references = [], []
        identifier = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in
                                                  iter_jsonl(workflow.ROOT / "examples/tiny/lid_train.jsonl")])
        identifier.save_json(self.lid)
        self.lexicon.write_bytes((workflow.ROOT / "examples/tiny/xlit_lexicon.jsonl").read_bytes())
        self.make_profile()
        for seed in self.seeds:
            checkpoint = self.study / f"adapted-seed-{seed}" / "checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "weights.safetensors").write_bytes(f"explicit fixture seed {seed}".encode())
            write_checkpoint_manifest(checkpoint, {
                "backend": "fixture_torch", "model_id": "fixture/hash-encoder", "init_revision": "fixture-v1",
                "dimension": 16, "fine_tuned": True, "fixture_only": True,
                "preprocessing": self.config.compatibility_mapping(), "training": {"seed": seed, "fixture_only": True},
            })
        for kind, seed, model_root, checkpoint in workflow._model_tracks(self.study, self.seeds, "final"):
            for language in self.languages:
                corpus = self.profile / language / "corpus.jsonl"
                index = DenseIndex(dense_corpus(corpus), InjectedEncoder(checkpoint), self.config)
                index.save(model_root / language / "dense", corpus_path=corpus)

    def make_profile(self):
        config = {"schema_version": 1, "fixture_only": True, "profile": "public_data_only_cpu_pilot"}
        writejson(self.profile / "profile_config.json", config)
        tracks = {}
        for language in self.languages:
            root = self.profile / language
            corpus = [{"doc_id": f"{language}:a", "text": "earthquake भूकंप", "title": "", "language": language,
                       "source": "explicit_workflow_fixture", "romanized_text": "earthquake bhukamp", "romanization_status": "complete"},
                      {"doc_id": f"{language}:b", "text": "weather rain", "title": "", "language": language,
                       "source": "explicit_workflow_fixture", "romanized_text": "weather rain", "romanization_status": "complete"}]
            rows = []
            for number, split in enumerate(("encoder_train", "fusion_train", "validation", "final_evaluation")):
                qid = f"{language}:{split}"
                rows.append({"query_id": qid, "parent_query_id": qid, "judgment_query_id": qid, "group_id": f"{qid}:g",
                             "text": f"earthquake {split} {number}", "languages": [language], "benchmark_language": language,
                             "split": split, "source_partition": "official_dev" if split == "final_evaluation" else "official_train",
                             "variant_type": "original", "verified": True, "review_status": "verified"})
            writerows(root / "corpus.jsonl", corpus)
            writerows(root / "queries.jsonl", rows)
            writerows(root / "qrels.jsonl", [{"query_id": row["query_id"], "doc_id": f"{language}:a", "relevance": 1} for row in rows])
            writejson(root / "split_manifest.json", sealed({"official_dev_policy": "locked_final_evaluation",
                       "group_assignments": [{"group_id": row["group_id"], "split": row["split"], "query_ids": [row["query_id"]]} for row in rows]}))
            writejson(root / "corpus_manifest.json", sealed({"corpus_sha256": workflow._sha(root / "corpus.jsonl"),
                                                               "qrels_sha256": workflow._sha(root / "qrels.jsonl")}))
            document = sealed({"language": language, "config_sha256": workflow._digest(config),
                               "selected_query_ids": sorted(row["query_id"] for row in rows),
                               "selected_group_ids": sorted(row["group_id"] for row in rows),
                               "artifacts": {name: {"sha256": workflow._sha(root / name), "size": (root / name).stat().st_size}
                                             for name in workflow.PROFILE_FILES[1:]}})
            writejson(root / "profile_manifest.json", document)
            tracks[language] = {"manifest_sha256": document["manifest_sha256"]}
        writejson(self.profile / "study_manifest.json", sealed({"languages": list(self.languages), "tracks": tracks,
                    "config_sha256": workflow._digest(config), "official_dev_policy": "locked_final_evaluation",
                    "profile_config": {"sha256": workflow._sha(self.profile / "profile_config.json")}}))

    def factory(self, **kwargs):
        gc.collect()
        self.assertTrue(all(reference() is None for reference in self.references), "previous model retained")
        self.calls.append((kwargs["kind"], kwargs["seed"]))
        encoder = InjectedEncoder(kwargs["checkpoint"])
        self.references.append(weakref.ref(encoder))
        return encoder

    def run_workflow(self, mode, **kwargs):
        with redirect_stdout(io.StringIO()):
            return workflow.run_pilot_workflow(self.profile, self.study, self.lid, self.lexicon,
                       mode=mode, seeds=self.seeds, encoder_factory=self.factory, allow_fixture=True,
                       gate_epochs=2, temperatures=(0.25,), l2s=(0.001,), bootstrap_samples=2, **kwargs)

    def frozen_validation(self):
        identifier = TokenLanguageIdentifier.load_json(self.lid)
        transliterator = LexiconTransliterator.from_jsonl(self.lexicon)
        encoder = InjectedEncoder()
        for language in self.languages:
            track, source = self.study / "frozen" / language, self.profile / language
            dense = DenseIndex.load(track / "dense", encoder, expected_config=self.config)
            pipeline = RetrievalPipeline(BM25Index(bm25_corpus(source / "corpus.jsonl")),
                                         CharTfidfIndex(char_corpus(source / "corpus.jsonl")), dense, identifier, transliterator)
            inputs = workflow._retrieval_inputs(source, track, self.lid, self.lexicon, None)
            original, metadata = workflow._original_run(dense, encoder, track, source / "queries.jsonl", "validation", "validation")
            trace, manifest = workflow._search(pipeline, source / "queries.jsonl", "validation", track, "validation", inputs, adaptive=False)
            process_study(trace, manifest, source / "queries.jsonl", source / "qrels.jsonl", source / "corpus.jsonl",
                          track / "validation-analysis", split="validation", bootstrap_samples=2, allow_fixture=True,
                          dense_original_run=original, dense_original_manifest=metadata)

    def lock(self, *, omit=None):
        members = workflow.required_input_paths(self.profile, self.study, self.lid, self.lexicon,
                                                seeds=self.seeds, allow_fixture=True)
        document = {"files_sha256": {str(path): workflow._sha(path) for path in members if path != omit}, "protocol": "explicit_fixture"}
        path = self.root / "final-lock.json"
        writejson(path, sealed(document, "lock_sha256"))
        return path

    def prepare_final(self):
        self.run_workflow("gates")
        self.frozen_validation()
        self.calls.clear()
        return self.lock()

    def refresh_profile_seals(self, language):
        track = self.profile / language
        corpus = workflow._json(track / "corpus_manifest.json")
        corpus.pop("manifest_sha256")
        corpus["qrels_sha256"] = workflow._sha(track / "qrels.jsonl")
        writejson(track / "corpus_manifest.json", sealed(corpus))
        manifest = workflow._json(track / "profile_manifest.json")
        manifest.pop("manifest_sha256")
        manifest["artifacts"] = {name: {"sha256": workflow._sha(track / name), "size": (track / name).stat().st_size}
                                 for name in workflow.PROFILE_FILES[1:]}
        manifest = sealed(manifest)
        writejson(track / "profile_manifest.json", manifest)
        parent = workflow._json(self.profile / "study_manifest.json")
        parent.pop("manifest_sha256")
        parent["tracks"][language]["manifest_sha256"] = manifest["manifest_sha256"]
        writejson(self.profile / "study_manifest.json", sealed(parent))

    def test_gates_and_matched_final_end_to_end_model_reuse_and_source_seals(self):
        initial = {str(path): workflow._sha(path) for path in workflow.required_input_paths(
            self.profile, self.study, self.lid, self.lexicon, mode="gates", seeds=self.seeds, allow_fixture=True)}
        original_collect, collections = workflow.collect_examples, []
        def counted_collect(*args, **kwargs):
            collections.append(kwargs["split"])
            return original_collect(*args, **kwargs)
        with patch.object(workflow, "collect_examples", new=counted_collect), \
             patch.object(TokenLanguageIdentifier, "load_json", wraps=TokenLanguageIdentifier.load_json) as lid:
            gate_report = self.run_workflow("gates")
        self.assertEqual(len(collections), len(self.seeds) * len(self.languages) * 2)
        self.assertEqual(lid.call_count, 1)
        self.assertEqual(self.calls, [("adapted", 13), ("adapted", 29)])
        self.assertEqual(gate_report["report_sha256"], workflow._digest({key: value for key, value in gate_report.items() if key != "report_sha256"}))
        self.frozen_validation()
        lock = self.lock()
        self.calls.clear()
        with patch.object(workflow, "train_gate", side_effect=AssertionError("No final fitting")), \
             patch.object(workflow, "collect_examples", side_effect=AssertionError("No final collection")), \
             patch("evaluation.study_runs.select_fixed_weights", side_effect=AssertionError("No final selection")):
            report = self.run_workflow("final", final_lock=lock)
        self.assertEqual(self.calls, [("frozen", None), ("adapted", 13), ("adapted", 29)])
        self.assertEqual(len(report["results"]), 6)
        for result in report["results"]:
            self.assertIsNone(result["gate_training"])
            for analysis in result["analyses"].values():
                summary = workflow._json(Path(analysis["output"]) / "study_manifest.json", seal="study_sha256")
                self.assertEqual(summary["split"], "final_evaluation")
                self.assertIn("B3_dense_original", summary["evaluations"])
                self.assertEqual(summary["serving_final_top_k"], 10)
                self.assertGreater(summary["actual_trace_cost"]["search_count"], 0)
        self.assertTrue(all(workflow._sha(Path(path)) == sha for path, sha in initial.items()))
        self.assertTrue(all(workflow._sha(Path(path)) == sha for path, sha in report["files_sha256"].items()))
        workflow.verify_final_lock(lock)

    def test_missing_and_mutated_final_lock_reject_before_models_or_labels(self):
        with patch("evaluation.study_runs._scoped_qrels", side_effect=AssertionError("labels must stay unopened")):
            with self.assertRaisesRegex(workflow.WorkflowError, "requires --final-lock"):
                self.run_workflow("final")
        marker = self.root / "marker.json"
        writejson(marker, {"fixture": 1})
        lock = self.root / "bad-lock.json"
        writejson(lock, sealed({"files_sha256": {str(marker): "0" * 64}}, "lock_sha256"))
        with patch.object(workflow, "_queries", side_effect=AssertionError("queries must stay unopened")):
            with self.assertRaisesRegex(workflow.WorkflowError, "checksum mismatch"):
                self.run_workflow("final", final_lock=lock)
        self.assertFalse(self.calls)

    def test_incomplete_closure_rejects_before_models_and_final_qrels(self):
        lock = self.prepare_final()
        omitted = self.study / "adapted-seed-29" / "hi" / "gates" / "no_agreement.json"
        self.lock(omit=omitted)
        with patch("evaluation.study_runs._scoped_qrels", side_effect=AssertionError("No final labels")):
            with self.assertRaisesRegex(workflow.WorkflowError, "missing required inputs"):
                self.run_workflow("final", final_lock=lock)
        self.assertFalse(self.calls)

    def test_changed_final_selection_and_checkpoint_fail_before_models(self):
        lock = self.prepare_final()
        selection = self.study / "adapted-seed-29" / "hi" / "validation-analysis" / "validation-selection.json"
        selection.write_bytes(selection.read_bytes() + b" ")
        with self.assertRaisesRegex(workflow.WorkflowError, "checksum mismatch"):
            self.run_workflow("final", final_lock=lock)
        self.assertFalse(self.calls)

    def test_late_missing_index_is_preflighted_before_earlier_training(self):
        path = self.study / "adapted-seed-29" / "hi" / "dense" / "vectors.jsonl"
        path.rename(path.with_suffix(".backup"))
        with self.assertRaisesRegex(workflow.WorkflowError, "missing"):
            self.run_workflow("gates")
        self.assertFalse(self.calls)
        self.assertFalse((self.study / "adapted-seed-13" / "en" / "gates").exists())

    def test_no_overwrites_and_cli_never_allows_fixtures(self):
        target = self.study / "adapted-seed-29" / "hi" / "validation.trace.jsonl"
        target.write_text("user artifact\n", encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "overwrite"):
            self.run_workflow("gates")
        self.assertEqual(target.read_text(), "user artifact\n")
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                workflow.main(["--mode", "gates", "--profile", str(self.profile), "--study-root", str(self.study),
                               "--lid-model", str(self.lid), "--lexicon", str(self.lexicon), "--allow-fixture"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.calls)

    def test_duplicate_keys_bad_seal_noncanonical_and_symlink_lock_rejected(self):
        path = self.root / "lock.json"
        path.write_text('{"files_sha256":{},"files_sha256":{}}', encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "Duplicate"):
            workflow.verify_final_lock(path)
        writejson(path, {"files_sha256": {}, "lock_sha256": "0" * 64})
        with self.assertRaisesRegex(workflow.WorkflowError, "seal"):
            workflow.verify_final_lock(path)
        link = self.root / "linked-lid.json"
        link.symlink_to(self.lid)
        writejson(path, sealed({"files_sha256": {str(link): workflow._sha(self.lid)}}, "lock_sha256"))
        with self.assertRaisesRegex(workflow.WorkflowError, "Symlink"):
            workflow.verify_final_lock(path)

    def test_gates_do_not_parse_any_final_relevance_values(self):
        for language in self.languages:
            path = self.profile / language / "qrels.jsonl"
            rows = list(iter_jsonl(path))
            for row in rows:
                if "final_evaluation" in row["query_id"]:
                    row["relevance"], row["doc_id"] = {"must_not_be_parsed": True}, "not-a-corpus-document"
            writerows(path, rows)
            self.refresh_profile_seals(language)
        report = self.run_workflow("gates")
        self.assertEqual(report["mode"], "gates")

    def test_resealed_incompatible_late_selection_fails_before_any_model(self):
        self.prepare_final()
        path = self.study / "adapted-seed-29" / "hi" / "no_agreement-gate" / "validation-analysis" / "validation-selection.json"
        document = workflow._json(path)
        document.pop("selection_sha256")
        document["query_source_sha256"] = "0" * 64
        writejson(path, sealed(document, "selection_sha256"))
        # Re-seal its parent too: semantic mismatch must fail, not just a checksum.
        parent = path.parent / "study_manifest.json"
        document = workflow._json(parent)
        document.pop("study_sha256")
        document["artifacts"][path.name] = workflow._sha(path)
        document["selection_sha256"] = workflow._json(path)["selection_sha256"]
        writejson(parent, sealed(document, "study_sha256"))
        lock = self.lock()
        with self.assertRaisesRegex(workflow.WorkflowError, "selection/profile identity"):
            self.run_workflow("final", final_lock=lock)
        self.assertFalse(self.calls)

    def test_resealed_wrong_ablation_fails_before_any_model(self):
        self.prepare_final()
        track = self.study / "adapted-seed-29" / "hi" / "gates"
        (track / "no_agreement.json").write_bytes((track / "full.json").read_bytes())
        (track / "no_agreement.report.json").write_bytes((track / "full.report.json").read_bytes())
        lock = self.lock()
        with self.assertRaisesRegex(workflow.WorkflowError, "ablation identity"):
            self.run_workflow("final", final_lock=lock)
        self.assertFalse(self.calls)

    def test_resealed_wrong_checkpoint_seed_fails_before_any_model(self):
        path = self.study / "adapted-seed-29" / "checkpoint" / "encoder_manifest.json"
        document = workflow._json(path)
        document["metadata"]["training"]["seed"] = 13
        document["checkpoint_sha256"] = workflow._digest({key: document[key] for key in ("metadata", "files")})
        writejson(path, document)
        with self.assertRaisesRegex(workflow.WorkflowError, "training seed"):
            self.run_workflow("gates")
        self.assertFalse(self.calls)

    def test_conflicting_model_cache_overrides_rejected(self):
        with patch.dict(workflow.os.environ, {"HF_HUB_CACHE": str(self.root / "sealed-cache"),
                                              "TRANSFORMERS_CACHE": str(self.root / "different-cache")}):
            with self.assertRaisesRegex(workflow.WorkflowError, "Conflicting TRANSFORMERS_CACHE"):
                workflow._frozen_cache()

    def test_nested_symlink_directory_cannot_escape_closure(self):
        directory = self.study / "adapted-seed-29" / "checkpoint"
        (directory / "hidden").symlink_to(self.profile, target_is_directory=True)
        with self.assertRaisesRegex(workflow.WorkflowError, "Symlink artifacts"):
            workflow.required_input_paths(self.profile, self.study, self.lid, self.lexicon,
                                          mode="gates", seeds=self.seeds, allow_fixture=True)
        self.assertFalse(self.calls)

    def test_existing_cli_relative_validation_input_paths_are_supported(self):
        self.frozen_validation()
        identifier = TokenLanguageIdentifier.load_json(self.lid)
        transliterator = LexiconTransliterator.from_jsonl(self.lexicon)
        for language in self.languages:
            track, source = self.study / "frozen" / language, self.profile / language
            path = track / "validation.run.manifest.json"
            manifest = workflow._json(path)
            manifest["input_sha256"] = {os.path.relpath(name): sha for name, sha in manifest["input_sha256"].items()}
            manifest["trace"] = os.path.relpath(manifest["trace"])
            writejson(path, manifest)
            dense = workflow._index_preflight(track, source / "corpus.jsonl", self.config, None, True)
            workflow._final_track_preflight(track, source, None, dense, identifier, transliterator, {})
        self.assertFalse(self.calls)

    def test_midrun_input_mutation_is_detected_and_no_completion_published(self):
        original = workflow._run_track
        count = [0]
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            count[0] += 1
            if count[0] == len(self.seeds) * len(self.languages):
                self.lexicon.write_bytes(self.lexicon.read_bytes() + b"\n")
            return result
        with patch.object(workflow, "_run_track", new=mutate):
            with self.assertRaisesRegex(workflow.WorkflowError, "changed during"):
                self.run_workflow("gates")
        self.assertFalse((self.study / "gates-workflow-report.json").exists())

    def test_provider_restored_after_search_failure(self):
        # Capture real pipelines and inject only the actual adaptive search fault.
        self.run_workflow("gates")
        self.frozen_validation()
        lock = self.lock()
        pipelines = []
        original = workflow._search
        def fail(pipeline, *args, **kwargs):
            if kwargs["adaptive"]:
                pipelines.append(pipeline)
                raise RuntimeError("injected adaptive failure")
            return original(pipeline, *args, **kwargs)
        with patch.object(workflow, "_search", new=fail):
            with self.assertRaisesRegex(RuntimeError, "injected adaptive"):
                self.run_workflow("final", final_lock=lock)
        self.assertTrue(pipelines)
        self.assertTrue(all(pipeline.weight_provider is None for pipeline in pipelines))
        self.assertFalse((self.study / "final-workflow-report.json").exists())


if __name__ == "__main__":
    unittest.main()
