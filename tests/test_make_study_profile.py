from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from data.make_splits import ALL_SPLITS, FINAL_EVALUATION, OFFICIAL_DEV, OFFICIAL_TRAIN
from data.make_study_profile import StudyProfileError, _digest, _seal, make_study_profile
from data.prepare_corpus import iter_jsonl, load_qrels
from evaluation.evaluate import RunEntry, evaluate_run, load_evaluation_queries
from retrieval.bm25 import BM25Index, load_corpus
from retrieval.char_tfidf import CharTfidfIndex, load_corpus as load_char_corpus


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def metadata(path: Path) -> dict:
    return {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class MakeStudyProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.romanized = self.root / "romanized"
        self.output = self.root / "pilot"
        self.queries: list[dict] = []
        self.qrels: list[dict] = []
        self.passages: list[dict] = []
        for split in ALL_SPLITS:
            for number in range(3):
                identifier = f"en:{split}:{number}"
                group = f"en:group:{split}:{number}"
                token = f"topic{split}{number}"
                self.queries.append({"query_id": identifier, "parent_query_id": identifier,
                                     "group_id": group, "text": token, "languages": ["en"],
                                     "benchmark_language": "en", "source_partition": OFFICIAL_DEV if split == FINAL_EVALUATION else OFFICIAL_TRAIN,
                                     "split": split, "variant_type": "original", "verified": True,
                                     "review_status": "verified", "judgment_query_id": identifier})
                for relevance, suffix in ((1, "positive"), (0, "zero")):
                    doc_id = identifier + ":" + suffix
                    text = token if relevance else f"unrelated negative passage {number}"
                    self.passages.append({"doc_id": doc_id, "language": "en", "title": "",
                                          "text": text, "romanized_text": text, "source": "controlled-fixture",
                                          "source_doc_id": doc_id, "romanization_status": "identity"})
                    self.qrels.append({"query_id": identifier, "doc_id": doc_id, "relevance": relevance})
        for number in range(12):
            self.passages.append({"doc_id": f"en:distractor:{number}", "language": "en", "title": "",
                                  "text": f"filler distractor {number}", "romanized_text": f"filler distractor {number}",
                                  "source": "controlled-fixture", "source_doc_id": f"distractor:{number}",
                                  "romanization_status": "identity"})
        # A second member must be kept whenever its preassigned group is chosen.
        variant = {**self.queries[0], "query_id": self.queries[0]["query_id"] + ":variant",
                   "variant_type": "public_paraphrase", "text": self.queries[0]["text"] + " details"}
        self.queries.append(variant)
        self.save_sources()

    def save_sources(self):
        track = self.prepared / "en"
        write_jsonl(track / "queries.jsonl", self.queries)
        write_jsonl(track / "qrels.jsonl", self.qrels)
        write_jsonl(track / "corpus.jsonl", self.passages)
        canonical = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                             for row in self.passages)
        write_json(track / "corpus_manifest.json", {"corpus_sha256": hashlib.sha256(canonical).hexdigest()})
        groups = {}
        for row in self.queries:
            groups.setdefault(row["group_id"], []).append(row)
        assignments = [{"group_id": group, "benchmark_language": "en", "split": rows[0]["split"],
                        "source_partition": rows[0]["source_partition"],
                        "query_ids": sorted(row["query_id"] for row in rows),
                        "parent_query_ids": sorted({row["parent_query_id"] for row in rows})}
                       for group, rows in sorted(groups.items())]
        split = {"official_dev_policy": "locked_final_evaluation", "group_assignments": assignments,
                 "input_sha256": "b" * 64, "config_hash": "c" * 64}
        write_json(track / "split_manifest.json", split)
        source = {"schema_version": 1, "language": "en", "dataset_version": "controlled-fixture-v1",
                  "artifacts": {name: metadata(track / name) for name in
                                ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "corpus_manifest.json", "split_manifest.json")}}
        write_json(track / "import_manifest.json", _seal(source))
        parent = {"schema_version": 1, "official_dev_policy": "locked_final_evaluation",
                  "tracks": {"en": {"path": "en", "manifest_sha256": source["manifest_sha256"]}}}
        write_json(self.prepared / "import_manifest.json", _seal(parent))
        rows = [{**row, "romanized_title": "", "romanization_provenance": {"fixture": True}}
                for row in self.passages]
        write_jsonl(self.romanized / "en.jsonl", rows)
        write_jsonl(self.romanized / "transliteration.jsonl", [])
        roman = {"schema_version": 1, "corpora": {
            "en": {"output": "en.jsonl", "document_count": len(rows),
                   "input_sha256": metadata(track / "corpus.jsonl")["sha256"],
                   "output_sha256": metadata(self.romanized / "en.jsonl")["sha256"]}},
                 "lexicon": {"path": "transliteration.jsonl",
                             "sha256": metadata(self.romanized / "transliteration.jsonl")["sha256"]}}
        write_json(self.romanized / "romanization_manifest.json", _seal(roman))

    def refresh_track_seals(self):
        track = self.prepared / "en"
        source = json.loads((track / "import_manifest.json").read_text())
        source["artifacts"] = {name: metadata(track / name) for name in source["artifacts"]}
        write_json(track / "import_manifest.json", _seal(source))
        parent = json.loads((self.prepared / "import_manifest.json").read_text())
        parent["tracks"]["en"]["manifest_sha256"] = source["manifest_sha256"]
        write_json(self.prepared / "import_manifest.json", _seal(parent))

    def refresh_roman_seal(self):
        path = self.romanized / "romanization_manifest.json"
        roman = json.loads(path.read_text())
        roman["corpora"]["en"]["output_sha256"] = metadata(self.romanized / "en.jsonl")["sha256"]
        write_json(path, _seal(roman))

    def run_profile(self, output=None, **kwargs):
        options = {"languages": ["en"], "encoder_groups": 1, "fusion_groups": 1,
                   "validation_groups": 1, "final_groups": 1, "target_size": 10, **kwargs}
        return make_study_profile(self.prepared, self.romanized, output or self.output, **options)

    def profile(self):
        return json.loads((self.output / "en" / "profile_manifest.json").read_text())

    def assert_unpublished(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".study-profile-*")))

    def test_preserves_whole_groups_parent_ids_source_rows_and_final_assignments(self):
        report = self.run_profile(encoder_groups=3)
        queries = list(iter_jsonl(self.output / "en" / "queries.jsonl"))
        by_id = {row["query_id"]: row for row in self.queries}
        selected = {row["query_id"] for row in queries}
        for row in queries:
            self.assertEqual(row, by_id[row["query_id"]])
            self.assertIn(row["parent_query_id"], selected)
            self.assertEqual(row["split"] == FINAL_EVALUATION, row["source_partition"] == OFFICIAL_DEV)
        self.assertIn(self.queries[-1]["query_id"], selected)
        for group in {row["group_id"] for row in queries}:
            self.assertEqual({row["query_id"] for row in queries if row["group_id"] == group},
                             {row["query_id"] for row in self.queries if row["group_id"] == group})
        self.assertEqual(report["official_dev_policy"], "locked_final_evaluation")
        split = json.loads((self.output / "en" / "split_manifest.json").read_text())
        self.assertFalse(split["assignments_changed"])
        self.assertEqual(split["source_fullsplit_manifest"]["sha256"],
                         metadata(self.prepared / "en" / "split_manifest.json")["sha256"])

    def test_all_judgments_including_zero_preserved_and_corpus_expands(self):
        self.run_profile(target_size=1)
        queries = list(iter_jsonl(self.output / "en" / "queries.jsonl"))
        ids = {row["query_id"] for row in queries}
        qrels = list(iter_jsonl(self.output / "en" / "qrels.jsonl"))
        self.assertEqual(qrels, sorted([row for row in self.qrels if row["query_id"] in ids],
                                       key=lambda row: (row["query_id"], row["doc_id"])))
        corpus_ids = {row["doc_id"] for row in iter_jsonl(self.output / "en" / "corpus.jsonl")}
        self.assertEqual(corpus_ids, {row["doc_id"] for row in qrels})
        self.assertTrue(any(row["relevance"] == 0 for row in qrels))
        self.assertTrue(self.profile()["target_expanded_for_judgments"])
        self.assertEqual(len(corpus_ids), self.profile()["effective_target_size"])

    def test_reproducibility_and_fixed_source_distractors(self):
        first = self.run_profile()
        second_output = self.root / "pilot-second"
        second = self.run_profile(second_output)
        self.assertEqual(first, second)
        for relative in ("en/corpus.jsonl", "en/queries.jsonl", "en/qrels.jsonl", "profile_config.json", "study_manifest.json"):
            self.assertEqual((self.output / relative).read_bytes(), (second_output / relative).read_bytes())
        ids = {row["doc_id"] for row in iter_jsonl(self.output / "en" / "corpus.jsonl")}
        self.assertEqual(len(ids), 10)
        self.assertTrue(ids.issubset({row["doc_id"] for row in self.passages}))
        self.assertTrue(self.profile()["same_fixed_corpus_for_all_methods"])

    def test_grade_changes_do_not_change_group_selection(self):
        self.run_profile()
        selected = self.profile()["selected_group_ids"]
        self.qrels = [{**row, "relevance": 0 if row["relevance"] else 2} for row in self.qrels]
        self.save_sources()
        other = self.root / "other"
        self.run_profile(other)
        other_profile = json.loads((other / "en" / "profile_manifest.json").read_text())
        self.assertEqual(selected, other_profile["selected_group_ids"])

    def test_romanized_passage_fields_preserved_without_hidden_fallback(self):
        roman = list(iter_jsonl(self.romanized / "en.jsonl"))
        roman[0]["romanized_text"] = ""
        roman[0]["romanization_status"] = "unavailable"
        write_jsonl(self.romanized / "en.jsonl", roman)
        self.refresh_roman_seal()
        self.run_profile(encoder_groups=3, target_size=100)
        output = {row["doc_id"]: row for row in iter_jsonl(self.output / "en" / "corpus.jsonl")}
        self.assertEqual(output, {row["doc_id"]: row for row in roman})
        self.assertEqual(output[roman[0]["doc_id"]]["romanized_text"], "")

    def test_seals_artifact_inventory_and_reducedscope_config_are_rich_and_valid(self):
        self.run_profile()
        for path in (self.output / "study_manifest.json", self.output / "en" / "profile_manifest.json",
                     self.output / "en" / "corpus_manifest.json", self.output / "en" / "split_manifest.json"):
            document = json.loads(path.read_text())
            self.assertEqual(document["manifest_sha256"], _digest({key: value for key, value in document.items()
                                                                 if key != "manifest_sha256"}))
        for name, expected in self.profile()["artifacts"].items():
            self.assertEqual(metadata(self.output / "en" / name), expected)
        config = json.loads((self.output / "profile_config.json").read_text())
        self.assertFalse(config["resplit_existing_groups"])
        self.assertFalse(config["query_sampling_uses_qrels_or_relevance"])
        self.assertIn("not_full_MIRACL_or_full_50k", config["scope"])
        self.assertEqual(self.profile()["excluded_query_count"], len(self.queries) - self.profile()["selected_query_count"])

    def test_target_and_group_shortfalls_recorded_without_synthesizing_data(self):
        self.run_profile(encoder_groups=20, target_size=100)
        profile = self.profile()
        self.assertEqual(profile["selection"]["encoder_train"]["group_shortfall"], 17)
        self.assertEqual(profile["selected_passage_count"], len(self.passages))
        self.assertEqual(profile["target_shortfall"], 100 - len(self.passages))

    def test_multiple_language_tracks_remain_independent(self):
        def remap(value):
            if isinstance(value, str):
                return "hi" if value == "en" else value.replace("en:", "hi:")
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value

        track = self.prepared / "hi"
        for name, rows in (("queries.jsonl", self.queries), ("qrels.jsonl", self.qrels),
                           ("corpus.jsonl", self.passages)):
            write_jsonl(track / name, remap(rows))
        split = remap(json.loads((self.prepared / "en" / "split_manifest.json").read_text()))
        write_json(track / "split_manifest.json", split)
        canonical = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                             for row in remap(self.passages))
        write_json(track / "corpus_manifest.json", {"corpus_sha256": hashlib.sha256(canonical).hexdigest()})
        source = remap(json.loads((self.prepared / "en" / "import_manifest.json").read_text()))
        source["artifacts"] = {name: metadata(track / name) for name in source["artifacts"]}
        write_json(track / "import_manifest.json", _seal(source))
        parent = json.loads((self.prepared / "import_manifest.json").read_text())
        parent["tracks"]["hi"] = {"path": "hi", "manifest_sha256": source["manifest_sha256"]}
        write_json(self.prepared / "import_manifest.json", _seal(parent))
        write_jsonl(self.romanized / "hi.jsonl", remap(list(iter_jsonl(self.romanized / "en.jsonl"))))
        roman = json.loads((self.romanized / "romanization_manifest.json").read_text())
        roman["corpora"]["hi"] = {"output": "hi.jsonl", "document_count": len(self.passages),
                                     "input_sha256": metadata(track / "corpus.jsonl")["sha256"],
                                     "output_sha256": metadata(self.romanized / "hi.jsonl")["sha256"]}
        write_json(self.romanized / "romanization_manifest.json", _seal(roman))
        report = self.run_profile(languages=["en", "hi"])
        self.assertEqual(set(report["tracks"]), {"en", "hi"})
        for language in ("en", "hi"):
            passages = list(iter_jsonl(self.output / language / "corpus.jsonl"))
            self.assertEqual(len(passages), 10)
            self.assertTrue(all(row["language"] == language and row["doc_id"].startswith(language + ":")
                                for row in passages))

    def test_source_top_or_track_manifest_tampering_rejected(self):
        for path in (self.prepared / "import_manifest.json", self.prepared / "en" / "import_manifest.json",
                     self.romanized / "romanization_manifest.json"):
            original = path.read_bytes()
            value = json.loads(original)
            value["tampered"] = True
            write_json(path, value)
            with self.subTest(path=path), self.assertRaisesRegex(StudyProfileError, "seal"):
                self.run_profile()
            self.assert_unpublished()
            path.write_bytes(original)

    def test_file_corruption_is_rejected_before_publication(self):
        for path in (self.prepared / "en" / "queries.jsonl", self.prepared / "en" / "qrels.jsonl",
                     self.prepared / "en" / "corpus.jsonl", self.romanized / "en.jsonl",
                     self.romanized / "transliteration.jsonl"):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaisesRegex(StudyProfileError, "checksum"):
                self.run_profile()
            self.assert_unpublished()
            path.write_bytes(original)

    def test_romanized_source_lineage_and_original_fields_must_match(self):
        rows = list(iter_jsonl(self.romanized / "en.jsonl"))
        rows[0]["text"] = "changed original passage"
        write_jsonl(self.romanized / "en.jsonl", rows)
        self.refresh_roman_seal()
        with self.assertRaisesRegex(StudyProfileError, "changed original"):
            self.run_profile()
        self.assert_unpublished()
        self.save_sources()
        path = self.romanized / "romanization_manifest.json"
        document = json.loads(path.read_text())
        document["corpora"]["en"]["input_sha256"] = "0" * 64
        write_json(path, _seal(document))
        with self.assertRaisesRegex(StudyProfileError, "input lineage"):
            self.run_profile()

    def test_duplicate_missing_or_unknown_romanized_documents_rejected(self):
        original = list(iter_jsonl(self.romanized / "en.jsonl"))
        cases = [original + [original[0]], original[:-1], [{**original[0], "doc_id": "unknown"}, *original[1:]]]
        for rows in cases:
            write_jsonl(self.romanized / "en.jsonl", rows)
            self.refresh_roman_seal()
            with self.subTest(count=len(rows)), self.assertRaises(StudyProfileError):
                self.run_profile()
            self.assert_unpublished()

    def test_existing_group_assignment_changes_parent_loss_and_cross_split_are_rejected(self):
        original = copy.deepcopy(self.queries)
        cases = [lambda rows: rows[0].update(split="fusion_train"),
                 lambda rows: rows[0].update(parent_query_id="missing-parent"),
                 lambda rows: rows[0].update(source_partition=OFFICIAL_DEV)]
        for mutate in cases:
            rows = copy.deepcopy(original)
            mutate(rows)
            write_jsonl(self.prepared / "en" / "queries.jsonl", rows)
            self.refresh_track_seals()
            with self.subTest(rows=rows[0]), self.assertRaises(StudyProfileError):
                self.run_profile()
            self.assert_unpublished()
            self.save_sources()

    def test_unmanifested_query_and_missing_group_members_rejected(self):
        path = self.prepared / "en" / "split_manifest.json"
        document = json.loads(path.read_text())
        document["group_assignments"] = document["group_assignments"][1:]
        write_json(path, document)
        self.refresh_track_seals()
        with self.assertRaisesRegex(StudyProfileError, "cover all"):
            self.run_profile()

    def test_source_query_or_passage_missing_for_qrels_is_rejected(self):
        for change in ({"query_id": "missing-query"}, {"doc_id": "missing-document"}):
            rows = [dict(self.qrels[0], **change), *self.qrels[1:]]
            write_jsonl(self.prepared / "en" / "qrels.jsonl", rows)
            self.refresh_track_seals()
            with self.subTest(change=change), self.assertRaisesRegex(StudyProfileError, "missing|absent"):
                self.run_profile()
            self.assert_unpublished()
            self.save_sources()

    def test_original_canonical_corpus_manifest_digest_is_verified(self):
        path = self.prepared / "en" / "corpus_manifest.json"
        write_json(path, {"corpus_sha256": "0" * 64})
        self.refresh_track_seals()
        with self.assertRaisesRegex(StudyProfileError, "canonical corpus digest"):
            self.run_profile()

    def test_original_canonical_qrels_digest_and_cross_manifest_lineage_verified(self):
        track = self.prepared / "en"
        corpus_path = track / "corpus_manifest.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["qrels_sha256"] = "0" * 64
        write_json(corpus_path, corpus)
        self.refresh_track_seals()
        with self.assertRaisesRegex(StudyProfileError, "canonical qrels digest"):
            self.run_profile()
        self.save_sources()
        source_path = track / "import_manifest.json"
        source = json.loads(source_path.read_text())
        source["split_config_hash"] = "different-from-source-split"
        write_json(source_path, _seal(source))
        self.refresh_track_seals()
        with self.assertRaisesRegex(StudyProfileError, "full split lineage"):
            self.run_profile()

    def test_symlinks_unsafe_manifest_paths_and_sources_as_outputs_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "link"
        link.symlink_to(self.romanized, target_is_directory=True)
        with self.assertRaisesRegex(StudyProfileError, "Symlinks"):
            make_study_profile(self.prepared, link, self.output, languages=["en"])
        with self.assertRaisesRegex(StudyProfileError, "outside the source"):
            self.run_profile(self.prepared / "child")
        path = self.prepared / "import_manifest.json"
        document = json.loads(path.read_text())
        document["tracks"]["en"]["path"] = "../outside"
        write_json(path, _seal(document))
        with self.assertRaisesRegex(StudyProfileError, "Unsafe relative"):
            self.run_profile()
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_output_preserved_and_atomic_failures_clean_staging_only(self):
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(StudyProfileError, "new output"):
            self.run_profile()
        self.assertEqual(sentinel.read_text(), "keep")
        other = self.root / "failed"
        with patch("data.make_study_profile.os.rename", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                self.run_profile(other)
        self.assertFalse(other.exists())
        self.assertFalse(list(self.root.glob(".study-profile-*")))

    def test_invalid_configuration_rejected(self):
        for options in ({"encoder_groups": 0}, {"fusion_groups": True}, {"target_size": -1},
                        {"final_groups": 100001}, {"seed": True}, {"languages": ["en", "en"]},
                        {"languages": ["xx"]}, {"languages": []}):
            with self.subTest(options=options), self.assertRaises(StudyProfileError):
                self.run_profile(**options)
        self.assert_unpublished()

    def test_cpu_profile_feeds_bm25_character_retrieval_and_final_evaluation(self):
        self.run_profile()
        track = self.output / "en"
        corpus = load_corpus(track / "corpus.jsonl")
        lexical = BM25Index(corpus)
        character = CharTfidfIndex(load_char_corpus(track / "corpus.jsonl"))
        queries = load_evaluation_queries(track / "queries.jsonl", split=FINAL_EVALUATION)
        texts = {row["query_id"]: row["text"] for row in iter_jsonl(track / "queries.jsonl")}
        qrels = load_qrels(track / "qrels.jsonl")
        for name, retriever in (("bm25", lexical), ("character", character)):
            entries = []
            for query in queries:
                hits = retriever.search(texts[query.query_id], top_k=10)
                entries.extend(RunEntry(query.query_id, hit.doc_id, name, rank, hit.score)
                               for rank, hit in enumerate(hits, start=1))
            report = evaluate_run(queries, qrels, entries)
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)

    def test_cli_publishes_named_pilot_not_full_benchmark(self):
        command = [sys.executable, "-m", "data.make_study_profile", "--prepared-root", str(self.prepared),
                   "--romanized-root", str(self.romanized), "--output-dir", str(self.output), "--languages", "en",
                   "--encoder-groups", "1", "--fusion-groups", "1", "--validation-groups", "1",
                   "--final-groups", "1", "--target-size", "10", "--seed", "13"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("en", json.loads(result.stdout)["tracks"])
        config = json.loads((self.output / "profile_config.json").read_text())
        self.assertEqual(config["profile"], "public_data_only_cpu_pilot")
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 2)


if __name__ == "__main__":
    unittest.main()
