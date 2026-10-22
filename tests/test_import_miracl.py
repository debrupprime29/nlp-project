"""Offline tests using the actual official MIRACL topic/qrel/gzip layout."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from data.import_miracl import MiraclImportError, import_miracl, load_topics, main
from data.make_splits import FINAL_EVALUATION
from data.prepare_corpus import CorpusPreparationError, _sample_priority
from data.prepare_corpus import load_qrels
from evaluation.evaluate import evaluate_run, load_evaluation_queries
from retrieval.bm25 import BM25Index, load_corpus, load_retrieval_queries, run_queries
from retrieval.dense import load_corpus as load_dense_corpus
from training.train_encoder import EncoderTrainingConfig, prepare_encoder_training


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MiraclImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / "raw"
        self.sources.mkdir()
        self.output = self.root / "prepared"

    def fixture(self, language="en", *, duplicate_train_dev=False, duplicate_train=False):
        dataset = self.sources / f"miracl-v1.0-{language}"
        (dataset / "topics").mkdir(parents=True)
        (dataset / "qrels").mkdir()
        corpus = self.sources / f"miracl-corpus-v1.0-{language}"
        corpus.mkdir()
        train = [(str(index), f"{language} training information need {index}") for index in range(1, 11)]
        dev = [("1", f"{language} final independent information need")]
        if duplicate_train_dev:
            train.append(("11", "  " + dev[0][1].upper().replace(" ", "  ") + "  "))
        if duplicate_train:
            train.append(("12", train[0][1].upper()))
        for partition, topics in (("train", train), ("dev", dev)):
            (dataset / "topics" / f"topics.miracl-v1.0-{language}-{partition}.tsv").write_text(
                "".join(f"{identifier}\t{text}\n" for identifier, text in topics), encoding="utf-8")
            (dataset / "qrels" / f"qrels.miracl-v1.0-{language}-{partition}.tsv").write_text(
                "".join(f"{identifier}\tQ0\t{'p' if partition == 'train' else 'd'}\t1\n"
                        for identifier, _ in topics)
                + ("1\tQ0\tz\t0\n" if partition == "train" else ""), encoding="utf-8")
        passages = [{"docid": identifier, "title": f"title {identifier}", "text": f"{language} passage {identifier}"}
                    for identifier in ["p", "d", "z", *[f"n{i}" for i in range(20)]]]
        self.write_shard(corpus / "docs-0.jsonl.gz", passages[:7])
        self.write_shard(corpus / "docs-1.jsonl.gz", passages[7:])
        return dataset, corpus

    @staticmethod
    def write_shard(path, records):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_import(self, **kwargs):
        return import_miracl(self.sources, self.output, languages=["en"], target_size=8, **kwargs)

    def test_real_formats_ids_all_judgments_and_seals(self):
        self.fixture()
        result = self.run_import()
        track = self.output / "en"
        corpus = _jsonl(track / "corpus.jsonl")
        qrels = _jsonl(track / "qrels.jsonl")
        queries = _jsonl(track / "queries.jsonl")
        self.assertEqual(len(corpus), 8)
        self.assertTrue(all(row["doc_id"].startswith("en:") for row in corpus))
        self.assertEqual({row["doc_id"] for row in corpus} & {"en:p", "en:d", "en:z"}, {"en:p", "en:d", "en:z"})
        self.assertEqual(len(qrels), 12)
        self.assertTrue(any(row["relevance"] == 0 for row in qrels))
        self.assertEqual({row["query_id"] for row in queries}, {f"en:train:{index}" for index in range(1, 11)} | {"en:dev:1"})
        self.assertEqual(next(row["split"] for row in queries if row["query_id"] == "en:dev:1"), FINAL_EVALUATION)
        self.assertEqual(result["manifest"]["evaluation_tracks"], "separate_monolingual_not_pooled")
        manifest = json.loads((track / "import_manifest.json").read_text())
        self.assertEqual(manifest["source_passage_count"], 23)
        self.assertEqual(len(manifest["source_files"]), 6)
        self.assertEqual(manifest["query_counts_by_split"], {"encoder_train": 7, "fusion_train": 2, "validation": 1, "final_evaluation": 1})
        for name, metadata in manifest["artifacts"].items():
            self.assertEqual(hashlib.sha256((track / name).read_bytes()).hexdigest(), metadata["sha256"])

    def test_sampling_covers_all_shards_matches_existing_priority(self):
        self.fixture()
        self.run_import(seed=79)
        chosen = {row["source_doc_id"] for row in _jsonl(self.output / "en" / "corpus.jsonl")}
        expected = sorted([f"n{i}" for i in range(20)], key=lambda identifier: _sample_priority(79, "en", identifier))[:5]
        self.assertEqual(chosen, {"p", "d", "z", *expected})

    def test_imported_artifacts_feed_retrieval_evaluation_and_scoped_training(self):
        self.fixture()
        self.run_import()
        track = self.output / "en"
        qrels = load_qrels(track / "qrels.jsonl")
        index = BM25Index(load_corpus(track / "corpus.jsonl"))
        validation_queries = load_retrieval_queries(track / "queries.jsonl", split="validation")
        run = run_queries(index, validation_queries, branch="bm25_original")
        report = evaluate_run(load_evaluation_queries(track / "queries.jsonl", split="validation"), qrels, run,
                              bootstrap_samples=10)
        self.assertEqual(len(report.per_query), 1)
        self.assertEqual(report.per_query[0].status, "evaluated")
        prepared = prepare_encoder_training(
            _jsonl(track / "queries.jsonl"), qrels, load_dense_corpus(track / "corpus.jsonl"),
            EncoderTrainingConfig(consistency_weight=0.0),
        )
        self.assertEqual(prepared.manifest["selected_split"], "encoder_train")
        self.assertEqual(prepared.manifest["ignored_qrel_count"], sum(
            row.query_id not in {example.original_query_id for example in prepared.examples} for row in qrels))
        self.assertTrue(all(example.original_query_id.startswith("en:train:") for example in prepared.examples))
        self.assertTrue(all("en:d" not in example.positive_doc_ids for example in prepared.examples))

    def test_all_judged_passages_expand_target(self):
        self.fixture()
        import_miracl(self.sources, self.output, languages=["en"], target_size=1)
        self.assertEqual(len(_jsonl(self.output / "en" / "corpus.jsonl")), 3)
        manifest = json.loads((self.output / "en" / "corpus_manifest.json").read_text())
        self.assertTrue(manifest["sources"][0]["target_expanded_for_judgments"])

    def test_duplicate_groups_and_development_precedence(self):
        self.fixture(duplicate_train_dev=True, duplicate_train=True)
        self.run_import()
        queries = {row["query_id"]: row for row in _jsonl(self.output / "en" / "queries.jsonl")}
        self.assertNotIn("en:train:11", queries)
        self.assertEqual(queries["en:train:1"]["split"], queries["en:train:12"]["split"])
        self.assertEqual(queries["en:train:1"]["group_id"], queries["en:train:12"]["group_id"])
        manifest = json.loads((self.output / "en" / "import_manifest.json").read_text())
        self.assertEqual(manifest["query_policy"]["excluded_training_query_ids"], ["en:train:11"])
        self.assertFalse(any(row["query_id"] == "en:train:11" for row in _jsonl(self.output / "en" / "qrels.jsonl")))

    def test_monolingual_tracks_namespace_same_raw_ids(self):
        self.fixture("en")
        self.fixture("hi")
        self.fixture("bn")
        result = import_miracl(self.sources, self.output, target_size=4)
        self.assertEqual(set(result["manifest"]["tracks"]), {"en", "hi", "bn"})
        all_ids = []
        for language in ("en", "hi", "bn"):
            track = self.output / language
            passages = _jsonl(track / "corpus.jsonl")
            self.assertEqual({row["language"] for row in passages}, {language})
            self.assertEqual({row["romanization_status"] for row in passages}, {"identity" if language == "en" else "pending"})
            all_ids.extend(row["query_id"] for row in _jsonl(track / "queries.jsonl"))
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_reproducible_bytes_and_query_limits(self):
        self.fixture()
        self.run_import(train_query_limit=4, dev_query_limit=1)
        second = self.root / "second"
        import_miracl(self.sources, second, languages=["en"], target_size=8, train_query_limit=4, dev_query_limit=1)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(self.output)).read_bytes())
        report = json.loads((self.output / "import_manifest.json").read_text())
        self.assertEqual(report["profile"], "pilot")
        counts = report["tracks"]["en"]["query_counts_by_split"]
        self.assertEqual(sum(counts.values()), 5)
        self.assertEqual(counts["final_evaluation"], 1)

    def test_development_precedence_applies_before_pilot_limits(self):
        dataset, _ = self.fixture(duplicate_train_dev=True)
        topics = dataset / "topics" / "topics.miracl-v1.0-en-dev.tsv"
        topics.write_text(topics.read_text() + "2\tanother final need\n")
        qrels = dataset / "qrels" / "qrels.miracl-v1.0-en-dev.tsv"
        qrels.write_text(qrels.read_text() + "2\tQ0\td\t1\n")
        self.run_import(dev_query_limit=1)
        queries = _jsonl(self.output / "en" / "queries.jsonl")
        self.assertFalse(any(row["query_id"] == "en:train:11" for row in queries))

    def test_invalid_configuration_never_creates_output(self):
        self.fixture()
        for kwargs in ({"target_size": 0}, {"train_query_limit": 0}, {"dev_query_limit": 0},
                       {"languages": []}, {"languages": ["en", "english"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises((MiraclImportError, CorpusPreparationError)):
                import_miracl(self.sources, self.output, **kwargs)
        self.assertFalse(self.output.exists())

    def test_missing_judged_document_fails_without_partial_publication(self):
        _, corpus = self.fixture()
        self.write_shard(corpus / "docs-0.jsonl.gz", [{"docid": "p", "text": "p"}, {"docid": "d", "text": "d"}])
        with self.assertRaisesRegex(MiraclImportError, "judged documents missing"):
            self.run_import()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".prepared.import-*")))

    def test_duplicate_document_across_shards_rejected_even_if_not_selected(self):
        _, corpus = self.fixture()
        self.write_shard(corpus / "docs-2.jsonl.gz", [{"docid": "n19", "text": "duplicate"}])
        with self.assertRaisesRegex(MiraclImportError, "Duplicate passage ID"):
            self.run_import()
        self.assertFalse(self.output.exists())

    def test_malformed_or_empty_passage_is_not_silently_skipped(self):
        _, corpus = self.fixture()
        self.write_shard(corpus / "docs-2.jsonl.gz", [{"docid": "bad", "text": ""}])
        with self.assertRaisesRegex(CorpusPreparationError, "empty text"):
            self.run_import()

    def test_malformed_gzip_json(self):
        _, corpus = self.fixture()
        with gzip.open(corpus / "docs-2.jsonl.gz", "wt") as handle:
            handle.write("not JSON\n")
        with self.assertRaisesRegex(MiraclImportError, "Invalid JSON"):
            self.run_import()

    def test_unknown_qrel_query_rejected(self):
        dataset, _ = self.fixture()
        path = dataset / "qrels" / "qrels.miracl-v1.0-en-train.tsv"
        path.write_text(path.read_text() + "unknown\tQ0\tp\t1\n")
        with self.assertRaisesRegex(MiraclImportError, "unknown topic"):
            self.run_import()

    def test_query_without_positive_judgment_rejected(self):
        dataset, _ = self.fixture()
        path = dataset / "qrels" / "qrels.miracl-v1.0-en-dev.tsv"
        path.write_text("1\tQ0\td\t0\n")
        with self.assertRaisesRegex(MiraclImportError, "lack positive"):
            self.run_import()

    def test_malformed_and_duplicate_topic_ids(self):
        path = self.root / "bad.tsv"
        for text in ("id no tab\n", "1\tfirst\n1\tsecond\n", "1\t   \n"):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(MiraclImportError):
                load_topics(path, "en", "train")

    def test_missing_official_file_or_shard_rejected(self):
        dataset, _ = self.fixture()
        (dataset / "topics" / "topics.miracl-v1.0-en-dev.tsv").unlink()
        with self.assertRaisesRegex(MiraclImportError, "Missing official source"):
            self.run_import()

    def test_safe_overwrite_and_recoverable_forced_replacement(self):
        self.fixture()
        self.run_import()
        original = (self.output / "import_manifest.json").read_bytes()
        with self.assertRaisesRegex(MiraclImportError, "Refusing to overwrite"):
            self.run_import()
        self.assertEqual((self.output / "import_manifest.json").read_bytes(), original)
        result = self.run_import(force=True)
        backup = Path(result["previous_output_backup"])
        self.assertEqual((backup / "import_manifest.json").read_bytes(), original)

    def test_force_never_replaces_unowned_directory_or_input(self):
        self.fixture()
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("preserve me")
        with self.assertRaisesRegex(MiraclImportError, "importer-owned"):
            self.run_import(force=True)
        self.assertEqual(sentinel.read_text(), "preserve me")
        for destination in (self.sources, self.sources / "nested", self.root):
            with self.assertRaisesRegex(MiraclImportError, "disjoint"):
                import_miracl(self.sources, destination, languages=["en"], force=True)

    def make_lock(self, git=False):
        assets = []
        for path in sorted(self.sources.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            assets.append({
                "name": path.name, "url": "https://example.invalid/unused", "path": path.relative_to(self.sources).as_posix(),
                "size": len(data), "license": "CC-BY-SA-4.0", "revision": "pinned-test",
                **({"git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
                   if git else {"sha256": hashlib.sha256(data).hexdigest()}),
            })
        lock = self.root / "resources.lock.json"
        lock.write_text(json.dumps({"assets": assets}))
        return lock

    def test_locked_source_checksums_sha256_and_git_blobs(self):
        self.fixture()
        for git in (False, True):
            lock = self.make_lock(git=git)
            result = self.run_import(lock_path=lock, force=self.output.exists())
            self.assertEqual(result["manifest"]["resource_lock_sha256"], hashlib.sha256(lock.read_bytes()).hexdigest())

    def test_lock_detects_modified_source_and_missing_shard(self):
        dataset, corpus = self.fixture()
        lock = self.make_lock()
        (corpus / "docs-1.jsonl.gz").unlink()
        with self.assertRaisesRegex(MiraclImportError, "Locked corpus shard is missing"):
            self.run_import(lock_path=lock)
        lock = self.make_lock()
        path = dataset / "topics" / "topics.miracl-v1.0-en-dev.tsv"
        data = path.read_bytes()
        path.write_bytes(data.replace(b"final", b"FINAL"))
        with self.assertRaisesRegex(MiraclImportError, "SHA-256 mismatch"):
            self.run_import(lock_path=lock)

    def test_cli_offline_smoke(self):
        self.fixture()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--source-dir", str(self.sources), "--output-dir", str(self.output),
                                   "--languages", "en", "--target-size", "5"]), 0)
        self.assertTrue((self.output / "en" / "queries.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
