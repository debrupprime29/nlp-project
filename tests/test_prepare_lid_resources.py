from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path
from unittest.mock import patch

from data.fetch_dakshina import (
    ARCHIVE_GENERATION, ARCHIVE_MD5_BASE64, ARCHIVE_SIZE, ARCHIVE_URL,
    LICENSE, MANIFEST_NAME, PINNED_URL, WANTED_MEMBERS,
    DakshinaAcquisitionError, _digest,
)
from data.prepare_lid_resources import (
    ANNOTATORS, AUXILIARY_TEST, AnnotationRejection, LidResourceError,
    _group, _hash, parse_comi_row, prepare_lid_resources, write_lid_resources,
)
from nlp.language_id import (
    AUXILIARY_TRAIN, AUXILIARY_VALIDATION, LanguageIdError,
    _read_examples, _validated_examples,
)
from nlp.text import tokenize_with_spans


def annotated_row(text: str, keys: list[str] | None = None,
                  labels: list[str] | None = None) -> dict[str, str]:
    keys = keys if keys is not None else text.split()
    labels = labels if labels is not None else ["hi"] * len(keys)
    annotations = [{"key": key, "value": label, "index": index}
                   for index, (key, label) in enumerate(zip(keys, labels))]
    return {"Sentences": text, "Predicted tags": "THIS IS NOT AN ANNOTATION",
            **{name: repr(annotations) for name in ANNOTATORS}}


class PrepareLidResourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.comi = self.root / "comi"
        self.comi.mkdir()
        self.lock = self.root / "lock.json"
        self.train_rows = [annotated_row(f"नमस्ते world {number}", labels=["hi", "en", "ot"])
                           for number in range(100)]
        self.test_rows = [annotated_row(f"परीक्षण test {number}", labels=["hi", "en", "ot"])
                          for number in range(10)]
        self.write_sources()

    def write_sources(self) -> None:
        assets = []
        for name, rows in (("LID_train.csv", self.train_rows), ("LID_test.csv", self.test_rows)):
            path = self.comi / name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Sentences", "Predicted tags", *ANNOTATORS])
                writer.writeheader()
                writer.writerows(rows)
            data = path.read_bytes()
            assets.append({"name": name, "path": name, "url": "https://example.org/v1/" + name,
                           "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                           "license": "CC-BY-4.0", "revision": "v1"})
        self.lock.write_text(json.dumps({"assets": assets}), encoding="utf-8")

    def prepare(self, **kwargs: object):
        options = {"train_limit": 30, "validation_limit": 15, "test_limit": 12, **kwargs}
        return prepare_lid_resources(self.comi, self.lock, **options)

    def dakshina_cache(self) -> Path:
        root = self.root / "dakshina"
        members = []
        for number, name in enumerate(WANTED_MEMBERS):
            if name.endswith("bn.translit.sampled.train.tsv"):
                data = "বাংলা\tbangla\t3\nশব্দ\tshobdo\t2\nবাংলা\tBangla\t1\n".encode()
            elif "/hi/" in name:
                data = "शब्द\tshabd\t3\n".encode()
            else:
                data = "পরীক্ষা\theldoutonly\t3\n".encode()
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            members.append({"path": name, "size": len(data), "header_offset": number * 1024,
                            "body_offset": number * 1024 + 512, "source_url": PINNED_URL,
                            "sha256": hashlib.sha256(data).hexdigest(), "header_sha256": "a" * 64})
        manifest = {"schema_version": 1, "license": LICENSE, "members": members,
                    "archive": {"url": ARCHIVE_URL, "pinned_url": PINNED_URL,
                                "generation": ARCHIVE_GENERATION, "size": ARCHIVE_SIZE,
                                "md5_base64": ARCHIVE_MD5_BASE64, "full_archive_md5_verified": False}}
        manifest["manifest_sha256"] = _digest(manifest)
        (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_human_majority_overrides_predicted_and_counts_disagreement(self) -> None:
        row = annotated_row("कंप्यूटर works", labels=["en", "en"])
        row["Predicted tags"] = repr([{"key": "कंप्यूटर", "value": "hi", "index": 0}])
        minority = [{"key": "कंप्यूटर", "value": "hi", "index": 0},
                    {"key": "works", "value": "en", "index": 1}]
        row[ANNOTATORS[1]] = repr(minority)
        parsed = parse_comi_row(row)
        self.assertEqual([label for _, _, label in parsed.labels], ["en", "en"])
        self.assertEqual(parsed.counts["human_disagreement_tokens"], 1)

    def test_no_majority_is_rejected_per_token(self) -> None:
        row = annotated_row("hello नमस्ते", labels=["hi", "hi"])
        for column, label in zip(ANNOTATORS, ["hi", "en", "ot"]):
            data = [{"key": "hello", "value": label, "index": 0},
                    {"key": "नमस्ते", "value": "hi", "index": 1}]
            row[column] = repr(data)
        parsed = parse_comi_row(row)
        self.assertEqual(len(parsed.labels), 1)
        self.assertEqual(parsed.counts["rejected_no_human_majority"], 1)

    def test_missing_one_human_cannot_fall_back_to_predicted(self) -> None:
        row = annotated_row("hello")
        row[ANNOTATORS[0]] = ""
        with self.assertRaisesRegex(AnnotationRejection, "missing_human"):
            parse_comi_row(row)

    def test_literal_eval_never_executes_annotation_code(self) -> None:
        row = annotated_row("hello")
        sentinel = self.root / "executed"
        row[ANNOTATORS[0]] = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('bad')"
        with self.assertRaisesRegex(AnnotationRejection, "invalid_annotation_literal"):
            parse_comi_row(row)
        self.assertFalse(sentinel.exists())

    def test_bad_indices_labels_shapes_and_different_keys_rejected(self) -> None:
        invalid = ([{"key": "hello", "value": "en", "index": 1}],
                   [{"key": "hello", "value": "en", "index": True}],
                   [{"key": "hello", "value": "unknown", "index": 0}],
                   [{"key": "hello", "value": [], "index": 0}],
                   [{"key": "hello\x00", "value": "en", "index": 0}],
                   [{"key": "different", "value": "en", "index": 0}],
                   [], {"wrong": "shape"})
        for annotation in invalid:
            row = annotated_row("hello", labels=["en"])
            row[ANNOTATORS[1]] = repr(annotation)
            with self.subTest(annotation=annotation), self.assertRaises(AnnotationRejection):
                parse_comi_row(row)

    def test_nfc_attached_punctuation_and_contractions_keep_complete_spans(self) -> None:
        text = "cafe\u0301 don't सीखेंगे|"
        row = annotated_row(text, labels=["en", "en", "hi"])
        parsed = parse_comi_row(row)
        self.assertEqual(parsed.text, unicodedata.normalize("NFC", text))
        actual = [(parsed.text[start:end], label) for start, end, label in parsed.labels]
        self.assertEqual(actual, [("café", "en"), ("don", "en"), ("t", "en"), ("सीखेंगे", "hi")])
        spans = {(item.start, item.end) for item in tokenize_with_spans(parsed.text)}
        self.assertTrue(all((start, end) in spans for start, end, _ in parsed.labels))

    def test_ambiguous_mixed_script_or_numeric_annotation_rejected(self) -> None:
        parsed = parse_comi_row(annotated_row("helloहाय COVID19 fine", labels=["en", "en", "en"]))
        self.assertEqual([parsed.text[start:end] for start, end, _ in parsed.labels], ["fine"])
        self.assertEqual(parsed.counts["rejected_mixed_script_annotation"], 2)

    def test_unmatched_tokens_and_unannotated_lexical_tail_rejected(self) -> None:
        for row in (annotated_row("hello", keys=["goodbye"], labels=["en"]),
                    annotated_row("hello extra", keys=["hello"], labels=["en"]),
                    annotated_row("prefix hello", keys=["hello"], labels=["en"])):
            with self.subTest(row=row), self.assertRaises(AnnotationRejection):
                parse_comi_row(row)

    def test_source_token_cannot_take_a_partial_shared_token(self) -> None:
        parsed = parse_comi_row(annotated_row("hello", keys=["hel", "lo"], labels=["en", "hi"]))
        self.assertFalse(parsed.labels)
        self.assertEqual(parsed.counts["rejected_incomplete_shared_token"], 2)

    def test_punctuation_only_source_token_is_recorded_not_used_as_training_span(self) -> None:
        parsed = parse_comi_row(annotated_row("hello !", labels=["en", "ot"]))
        self.assertEqual(len(parsed.labels), 1)
        self.assertEqual(parsed.counts["rejected_punctuation_without_shared_token"], 1)

    def test_preparation_is_deterministic_bounded_label_balanced_and_compatible(self) -> None:
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(first.manifest["partitions"][AUXILIARY_TRAIN]["examples"], 30)
        self.assertEqual(first.manifest["partitions"][AUXILIARY_TRAIN]["labels"],
                         {"en": 10, "hi": 10, "other": 10})
        self.assertEqual(len(first.partitions[AUXILIARY_VALIDATION]), 15)
        self.assertEqual(len(first.partitions[AUXILIARY_TEST]), 12)
        for partition in (AUXILIARY_TRAIN, AUXILIARY_VALIDATION):
            self.assertEqual(len(_validated_examples(first.partitions[partition], partition)),
                             len(first.partitions[partition]))
        train = {example.group_id for example in first.partitions[AUXILIARY_TRAIN]}
        validation = {example.group_id for example in first.partitions[AUXILIARY_VALIDATION]}
        test = {example.group_id for example in first.partitions[AUXILIARY_TEST]}
        self.assertFalse(train & validation or train & test or validation & test)
        with self.assertRaisesRegex(LanguageIdError, "expected auxiliary_train"):
            _validated_examples(first.partitions[AUXILIARY_TEST], AUXILIARY_TRAIN)
        self.assertFalse(first.manifest["annotation_policy"]["predicted_tags_used"])

    def test_official_test_overlap_is_quarantined_even_when_test_annotation_invalid(self) -> None:
        overlap = annotated_row("Overlap नमस्ते", labels=["en", "hi"])
        self.train_rows.append(overlap)
        invalid_test = copy.deepcopy(overlap)
        invalid_test[ANNOTATORS[0]] = ""
        self.test_rows.append(invalid_test)
        self.write_sources()
        result = self.prepare(train_limit=1000, validation_limit=1000, test_limit=1000)
        self.assertEqual(result.manifest["counts"]["train_rows_quarantined_for_official_test_overlap"], 1)
        self.assertFalse(any(item.text == overlap["Sentences"]
                             for partition in result.partitions.values() for item in partition))

    def test_content_groups_normalize_case_and_whitespace_duplicates(self) -> None:
        duplicate = copy.deepcopy(self.train_rows[0])
        self.train_rows.append(duplicate)
        self.write_sources()
        result = self.prepare()
        self.assertEqual(result.manifest["counts"]["LID_train.csv:duplicate_sentence_groups"], 1)
        self.assertEqual(_group(" HELLO  world "), _group("hello world"))

    def test_rejected_rows_counted_without_predicted_fallback(self) -> None:
        row = annotated_row("malformed annotation")
        row[ANNOTATORS[0]] = "unparseable"
        self.train_rows.append(row)
        self.write_sources()
        result = self.prepare()
        self.assertEqual(result.manifest["counts"]["LID_train.csv:rejected_invalid_annotation_literal"], 1)

    def test_input_lock_corruption_rejected_offline(self) -> None:
        path = self.comi / "LID_train.csv"
        path.write_text(path.read_text() + "changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cached asset"):
            self.prepare()

    def test_source_license_revision_and_required_files_are_validated(self) -> None:
        original = json.loads(self.lock.read_text())
        for change in ("license", "revision", "required_file"):
            document = copy.deepcopy(original)
            if change == "required_file":
                document["assets"] = document["assets"][:1]
            else:
                document["assets"][0][change] = "incorrect"
            self.lock.write_text(json.dumps(document), encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(LidResourceError):
                self.prepare()

    def test_weak_bengali_uses_verified_train_only_and_has_reduced_explicit_weight(self) -> None:
        cache = self.dakshina_cache()
        result = self.prepare(dakshina_root=cache, weak_bn_limit=3)
        weak = [item for item in result.partitions[AUXILIARY_TRAIN] if item.label == "bn"]
        self.assertEqual(len(weak), 3)
        self.assertTrue(all(item.weight == 0.25 and "weak_lexicon_language" in item.source for item in weak))
        self.assertTrue(all("train.tsv" in item.source for item in weak))
        self.assertFalse(any(item.text == "heldoutonly" for item in weak))
        self.assertTrue(all(item.label != "bn" for partition in (AUXILIARY_VALIDATION, AUXILIARY_TEST)
                            for item in result.partitions[partition]))
        self.assertEqual(result.manifest["sources"]["dakshina_bn_weak_train"]["license"], "CC-BY-SA-4.0")
        self.assertFalse(result.manifest["sources"]["dakshina_bn_weak_train"]["dev_test_used_for_fit"])
        _validated_examples(result.partitions[AUXILIARY_TRAIN], AUXILIARY_TRAIN)

    def test_weak_source_seal_and_all_members_verified_before_preparation(self) -> None:
        cache = self.dakshina_cache()
        test_member = next(name for name in WANTED_MEMBERS if name.endswith("bn.translit.sampled.test.tsv"))
        (cache / test_member).write_text("corrupt", encoding="utf-8")
        with self.assertRaisesRegex(DakshinaAcquisitionError, "checksum mismatch"):
            self.prepare(dakshina_root=cache)

    def test_missing_dakshina_cache_is_not_created_or_downloaded(self) -> None:
        missing = self.root / "missing-dakshina"
        with self.assertRaisesRegex(LidResourceError, "verified sealed"):
            self.prepare(dakshina_root=missing)
        self.assertFalse(missing.exists())

    def test_weak_isolated_word_cannot_overlap_public_heldout_sentence(self) -> None:
        self.test_rows.append(annotated_row("bangla", labels=["en"]))
        self.write_sources()
        result = self.prepare(dakshina_root=self.dakshina_cache(), weak_bn_limit=100)
        self.assertFalse(any(item.label == "bn" and item.text.casefold() == "bangla"
                             for item in result.partitions[AUXILIARY_TRAIN]))
        self.assertGreater(result.manifest["counts"]["dakshina_bn_train:rejected_heldout_text_overlap"], 0)

    def test_weak_invalid_rows_and_mixed_words_rejected(self) -> None:
        cache = self.dakshina_cache()
        path = cache / "dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("wrong\nবাংলা\tbangla\t0\nnotbengali\troman\t1\nবাংলা\ttwo words\t2\n")
        manifest_path = cache / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        member = next(item for item in manifest["members"] if item["path"] == path.relative_to(cache).as_posix())
        member.update(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        manifest_path.write_text(json.dumps(manifest))
        result = self.prepare(dakshina_root=cache)
        self.assertEqual(result.manifest["counts"]["dakshina_bn_train:rejected_invalid_row"], 2)
        self.assertEqual(result.manifest["counts"]["dakshina_bn_train:rejected_nonword_or_script"], 2)

    def test_invalid_limits_and_split_configuration_rejected(self) -> None:
        for kwargs in ({"train_limit": 0}, {"train_limit": True}, {"weak_bn_limit": -1},
                       {"validation_fraction": 0}, {"validation_fraction": 1},
                       {"validation_fraction": float("nan")}, {"weak_weight": 0},
                       {"weak_weight": True}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LidResourceError):
                self.prepare(**kwargs)

    def test_empty_human_partition_fails_explicitly(self) -> None:
        self.train_rows = self.train_rows[:1]
        self.write_sources()
        with self.assertRaisesRegex(LidResourceError, "partition is empty"):
            self.prepare()

    def test_writer_is_new_output_only_and_jsonl_fits_existing_model_loader(self) -> None:
        prepared = self.prepare()
        output = self.root / "prepared"
        paths = write_lid_resources(prepared, output)
        loaded = _read_examples(paths[AUXILIARY_TRAIN])
        self.assertEqual(loaded, prepared.partitions[AUXILIARY_TRAIN])
        manifest = json.loads(paths["manifest"].read_text())
        self.assertEqual(manifest["manifest_sha256"], _hash({key: value for key, value in manifest.items()
                                                          if key != "manifest_sha256"}))
        with self.assertRaisesRegex(LidResourceError, "already exists"):
            write_lid_resources(prepared, output)
        self.assertEqual(loaded, _read_examples(paths[AUXILIARY_TRAIN]))
        self.assertFalse(list(self.root.glob(".lid-resources-*")))

    def test_writer_rejects_symlink_root_and_cleans_own_staging_on_failure(self) -> None:
        prepared = self.prepare()
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(LidResourceError, "Symlinks"):
            write_lid_resources(prepared, link / "output")
        with patch("data.prepare_lid_resources.os.rename", side_effect=OSError("failure")):
            with self.assertRaises(OSError):
                write_lid_resources(prepared, self.root / "failed")
        self.assertFalse((self.root / "failed").exists())
        self.assertFalse(list(self.root.glob(".lid-resources-*")))

    def test_cli_prepares_offline_without_training(self) -> None:
        output = self.root / "cli"
        command = [sys.executable, "-m", "data.prepare_lid_resources", "--comi-root", str(self.comi),
                   "--source-lock", str(self.lock), "--output-dir", str(output),
                   "--train-limit", "30", "--validation-limit", "15", "--test-limit", "12"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["partitions"][AUXILIARY_TRAIN]["examples"], 30)
        self.assertFalse(any("model" in path.name for path in output.iterdir()))


if __name__ == "__main__":
    unittest.main()
