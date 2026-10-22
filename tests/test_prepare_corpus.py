from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from data.prepare_corpus import (
    CorpusPreparationError,
    CorpusSource,
    prepare_corpus,
    write_prepared_corpus,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class PrepareCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source(
        self,
        language: str,
        passages: list[dict[str, object]],
        qrels: list[dict[str, object]] | None = None,
    ) -> CorpusSource:
        passage_path = self.root / f"{language}-passages.jsonl"
        write_jsonl(passage_path, passages)
        qrels_path = None
        if qrels is not None:
            qrels_path = self.root / f"{language}-qrels.jsonl"
            write_jsonl(qrels_path, qrels)
        return CorpusSource(language, passage_path, qrels_path, "fixture")

    def test_keeps_all_judged_documents_and_expands_target(self) -> None:
        source = self.make_source(
            "hi",
            [
                {"docid": "d1", "title": "एक", "text": "पहला पाठ"},
                {"docid": "d2", "title": "दो", "text": "दूसरा पाठ"},
                {"docid": "d3", "title": "तीन", "text": "तीसरा पाठ"},
            ],
            [
                {"query_id": "q1", "doc_id": "d1", "relevance": 2},
                {"query_id": "q1", "doc_id": "d2", "relevance": 0},
            ],
        )
        prepared = prepare_corpus([source], target_size=1, seed=7)

        self.assertEqual([item.doc_id for item in prepared.passages], ["d1", "d2"])
        self.assertEqual([item.relevance for item in prepared.qrels], [2, 0])
        self.assertTrue(prepared.manifest["sources"][0]["target_expanded_for_judgments"])
        self.assertTrue(all(item.romanized_text == "" for item in prepared.passages))
        self.assertTrue(
            all(item.romanization_status == "pending" for item in prepared.passages)
        )

    def test_pooled_ids_and_qrels_are_rewritten_together(self) -> None:
        en = self.make_source(
            "en",
            [{"docid": "same", "title": "Earth", "text": "earthquake"}],
            [{"query_id": "en-q", "doc_id": "same", "relevance": 1}],
        )
        bn = self.make_source(
            "bn",
            [{"docid": "same", "title": "পৃথিবী", "text": "ভূমিকম্প"}],
            [{"query_id": "bn-q", "doc_id": "same", "relevance": 2}],
        )
        prepared = prepare_corpus([en, bn], target_size=1)

        self.assertEqual(
            [item.doc_id for item in prepared.passages], ["bn:same", "en:same"]
        )
        self.assertEqual(
            {(item.query_id, item.doc_id) for item in prepared.qrels},
            {("en-q", "en:same"), ("bn-q", "bn:same")},
        )
        english = next(item for item in prepared.passages if item.language == "en")
        self.assertEqual(english.romanized_text, "earthquake")
        self.assertEqual(english.romanization_status, "identity")

    def test_hash_sample_is_reproducible_across_source_order(self) -> None:
        records = [
            {"docid": f"d{number}", "title": "", "text": f"text {number}"}
            for number in range(20)
        ]
        first = self.make_source("en", records)
        selected_first = prepare_corpus([first], target_size=5, seed=99)
        second_path = self.root / "reordered.jsonl"
        write_jsonl(second_path, list(reversed(records)))
        second = CorpusSource("en", second_path, None, "fixture")
        selected_second = prepare_corpus([second], target_size=5, seed=99)

        self.assertEqual(
            [item.doc_id for item in selected_first.passages],
            [item.doc_id for item in selected_second.passages],
        )
        self.assertEqual(
            selected_first.manifest["corpus_sha256"],
            selected_second.manifest["corpus_sha256"],
        )

    def test_missing_judged_document_fails(self) -> None:
        source = self.make_source(
            "en",
            [{"docid": "present", "text": "text"}],
            [{"query_id": "q", "doc_id": "missing", "relevance": 1}],
        )
        with self.assertRaisesRegex(CorpusPreparationError, "missing"):
            prepare_corpus([source], target_size=5)

    def test_duplicate_passage_id_fails(self) -> None:
        source = self.make_source(
            "en",
            [
                {"docid": "duplicate", "text": "first"},
                {"docid": "duplicate", "text": "second"},
            ],
        )
        with self.assertRaisesRegex(CorpusPreparationError, "Duplicate passage ID"):
            prepare_corpus([source], target_size=5)

    def test_artifacts_are_complete_and_overwrite_is_explicit(self) -> None:
        source = self.make_source("en", [{"docid": "d", "text": "café"}])
        prepared = prepare_corpus([source], target_size=1, dataset_version="fixture-v1")
        output = self.root / "output"
        paths = write_prepared_corpus(prepared, output)

        self.assertEqual(set(paths), {"corpus", "qrels", "id_map", "manifest"})
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_version"], "fixture-v1")
        self.assertEqual(manifest["passage_count"], 1)
        with self.assertRaisesRegex(CorpusPreparationError, "Refusing to overwrite"):
            write_prepared_corpus(prepared, output)
        write_prepared_corpus(prepared, output, overwrite=True)

    def test_prepared_corpus_can_be_reprocessed_without_double_prefixing(self) -> None:
        source = self.make_source("en", [{"docid": "d", "text": "text"}])
        prepared = prepare_corpus([source], target_size=1, prefix_doc_ids=True)
        output = self.root / "first"
        paths = write_prepared_corpus(prepared, output)
        reprocessed = prepare_corpus(
            [CorpusSource("en", paths["corpus"], None, "fixture")],
            target_size=1,
            prefix_doc_ids=True,
        )
        self.assertEqual(reprocessed.passages[0].doc_id, "en:d")

    def test_cli_smoke(self) -> None:
        source = self.make_source("en", [{"docid": "d", "text": "earthquake"}])
        output = self.root / "cli-output"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "data.prepare_corpus",
                "--passages",
                f"en={source.passages_path}",
                "--output-dir",
                str(output),
                "--target-size",
                "1",
                "--dataset-version",
                "fixture",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        reported = json.loads(process.stdout)
        self.assertTrue(Path(reported["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
