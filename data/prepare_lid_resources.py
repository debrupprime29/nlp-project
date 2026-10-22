"""Prepare bounded auxiliary token labels from public, human COMI annotations.

Only the three human annotation columns are read. The official test sentences
are quarantined before splitting training-source sentences by content group.
Dakshina Bengali *training* lexicon labels are optional, weak, and never added
to calibration or the public human test. This module does not fit a model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import heapq
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from data.acquire_resources import acquire_resources, load_lock
from data.fetch_dakshina import verify_cache
from nlp.language_id import AUXILIARY_TRAIN, AUXILIARY_VALIDATION, LabeledTokenExample
from nlp.text import TextPreparationError, normalize_processing_text, tokenize_with_spans


AUXILIARY_TEST = "auxiliary_test"
ANNOTATORS = tuple(f"Annotated by: Annotator {number}" for number in (1, 2, 3))
LABEL_MAP = {"hi": "hi", "en": "en", "ot": "other", "other": "other"}
MAX_CELL_CHARACTERS = 1024 * 1024
MAX_ANNOTATED_TOKENS = 4096
SCHEMA_VERSION = 1


class LidResourceError(ValueError):
    """An unsafe source, annotation, or resource preparation configuration."""


class AnnotationRejection(LidResourceError):
    """A row rejection with a stable reason suitable for provenance counts."""


@dataclass(frozen=True)
class AnnotatedSentence:
    text: str
    labels: tuple[tuple[int, int, str], ...]
    counts: Mapping[str, int]


@dataclass(frozen=True)
class PreparedLidResources:
    partitions: Mapping[str, tuple[LabeledTokenExample, ...]]
    manifest: Mapping[str, Any]


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group(text: str) -> str:
    return _hash(" ".join(normalize_processing_text(text).casefold().split()))


def _rows(path: Path) -> Iterator[tuple[int, Mapping[str, str]]]:
    previous = csv.field_size_limit()
    csv.field_size_limit(MAX_CELL_CHARACTERS)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"Sentences", *ANNOTATORS}.issubset(reader.fieldnames):
                raise LidResourceError("COMI CSV is missing sentence/human annotation columns")
            for number, row in enumerate(reader, start=1):
                if None in row or any(row.get(key) is None for key in ("Sentences", *ANNOTATORS)):
                    raise LidResourceError(f"Malformed COMI CSV record {number}")
                yield number, row
    except (csv.Error, UnicodeError) as exc:
        raise LidResourceError("Invalid or oversized COMI CSV record") from exc
    finally:
        csv.field_size_limit(previous)


def _human_annotations(value: str) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationRejection("missing_human_annotation")
    if len(value) > MAX_CELL_CHARACTERS:
        raise AnnotationRejection("oversized_annotation")
    try:
        annotation = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
        raise AnnotationRejection("invalid_annotation_literal") from exc
    if not isinstance(annotation, list) or not annotation or len(annotation) > MAX_ANNOTATED_TOKENS:
        raise AnnotationRejection("invalid_annotation_list")
    for position, item in enumerate(annotation):
        if (not isinstance(item, dict) or type(item.get("index")) is not int
                or item["index"] != position or not isinstance(item.get("key"), str)
                or not item["key"] or not isinstance(item.get("value"), str)
                or item["value"] not in LABEL_MAP):
            raise AnnotationRejection("invalid_annotation_token_index_or_label")
    return annotation


def parse_comi_row(row: Mapping[str, str]) -> AnnotatedSentence:
    """Majority-vote humans and project exact source tokens to shared spans.

    Attached punctuation and same-script contractions may project to multiple
    complete shared tokens. A mixed-script/numeric annotated token is ambiguous
    and is rejected instead of assigning its one label to unlike subparts.
    """
    try:
        text = normalize_processing_text(row.get("Sentences"))
    except TextPreparationError as exc:
        raise AnnotationRejection("invalid_sentence") from exc
    if not text.strip():
        raise AnnotationRejection("empty_sentence")
    annotations = [_human_annotations(row.get(column, "")) for column in ANNOTATORS]
    reference = annotations[0]
    if any(len(items) != len(reference) for items in annotations):
        raise AnnotationRejection("annotator_token_alignment_mismatch")
    try:
        for position, item in enumerate(reference):
            key = normalize_processing_text(item["key"])
            if any(normalize_processing_text(items[position]["key"]) != key for items in annotations):
                raise AnnotationRejection("annotator_token_alignment_mismatch")
    except TextPreparationError as exc:
        raise AnnotationRejection("invalid_annotated_token_text") from exc
    spans = tokenize_with_spans(text)
    labels: list[tuple[int, int, str]] = []
    counts: Counter[str] = Counter()
    cursor = 0
    for position, item in enumerate(reference):
        counts["source_annotation_tokens"] += 1
        key = normalize_processing_text(item["key"])
        start = text.find(key, cursor)
        if start < 0 or tokenize_with_spans(text[cursor:start]):
            raise AnnotationRejection("unmatched_source_token")
        end = start + len(key)
        cursor = end
        overlapping = [span for span in spans if span.start < end and span.end > start]
        if any(span.start < start or span.end > end for span in overlapping):
            counts["rejected_incomplete_shared_token"] += 1
            continue
        votes = [LABEL_MAP[items[position]["value"]] for items in annotations]
        if len(set(votes)) > 1:
            counts["human_disagreement_tokens"] += 1
        label, number = Counter(votes).most_common(1)[0]
        if number < 2:
            counts["rejected_no_human_majority"] += 1
            continue
        if not overlapping:
            counts["rejected_punctuation_without_shared_token"] += 1
            continue
        if len({span.script for span in overlapping}) > 1:
            counts["rejected_mixed_script_annotation"] += 1
            continue
        if len(overlapping) > 1:
            counts["projected_multiple_same_script_tokens"] += 1
        labels.extend((span.start, span.end, label) for span in overlapping)
    if tokenize_with_spans(text[cursor:]):
        raise AnnotationRejection("unannotated_sentence_tail")
    if len({(start, end) for start, end, _ in labels}) != len(labels):
        raise AnnotationRejection("duplicate_projected_span")
    counts["accepted_shared_tokens"] = len(labels)
    return AnnotatedSentence(text, tuple(labels), dict(counts))


class _Sampler:
    """Bounded per-label hash reservoirs; balanced fill when labels are scarce."""

    def __init__(self, limit: int, seed: int):
        self.limit, self.seed = limit, seed
        self.heaps: dict[str, list[tuple[int, str, LabeledTokenExample]]] = {}

    def add(self, example: LabeledTokenExample) -> None:
        if not self.limit:
            return
        identity = _hash([example.text, example.start, example.end, example.label])
        priority = int(_hash([self.seed, identity]), 16)
        heap = self.heaps.setdefault(example.label, [])
        item = (-priority, identity, example)
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def selected(self) -> tuple[LabeledTokenExample, ...]:
        ordered = {label: sorted(heap, key=lambda item: (-item[0], item[1]))
                   for label, heap in self.heaps.items()}
        offsets = Counter()
        result: list[LabeledTokenExample] = []
        while len(result) < self.limit:
            changed = False
            for label in sorted(ordered):
                if offsets[label] < len(ordered[label]) and len(result) < self.limit:
                    result.append(ordered[label][offsets[label]][2])
                    offsets[label] += 1
                    changed = True
            if not changed:
                break
        return tuple(sorted(result, key=lambda item: (item.group_id, item.start, item.end, item.source)))


def _comi_sources(comi_root: Path, source_lock: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    assets = load_lock(source_lock)
    acquire_resources(assets, comi_root, max_bytes=0, verify_only=True)
    required = [asset for asset in assets if asset["path"] in ("LID_train.csv", "LID_test.csv")]
    if len(required) != 2 or any(asset["license"] != "CC-BY-4.0" for asset in required):
        raise LidResourceError("Lock must verify both CC-BY-4.0 COMI LID source files")
    if len({asset["revision"] for asset in required}) != 1:
        raise LidResourceError("COMI source revisions must match")
    checksums = {asset["path"]: asset.get("sha256", "git:" + asset.get("git_blob_sha1", ""))
                 for asset in required}
    return checksums, required


def prepare_lid_resources(comi_root: str | Path, source_lock: str | Path, *,
                          dakshina_root: str | Path | None = None,
                          train_limit: int = 6000, weak_bn_limit: int = 2000,
                          validation_limit: int = 2000, test_limit: int = 2000,
                          seed: int = 13, validation_fraction: float = 0.2,
                          weak_weight: float = 0.25) -> PreparedLidResources:
    """Prepare auxiliary train/calibration/test labels entirely offline."""
    limits = (train_limit, weak_bn_limit, validation_limit, test_limit)
    if any(type(value) is not int or not 0 <= value <= 100000 for value in limits):
        raise LidResourceError("Token limits must be integers between 0 and 100000")
    if not all((train_limit, validation_limit, test_limit)) or type(seed) is not int:
        raise LidResourceError("Human partition limits must be positive and seed an integer")
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float))
            or not 0 < validation_fraction < 1 or isinstance(weak_weight, bool)
            or not isinstance(weak_weight, (int, float)) or not 0 < weak_weight <= 1):
        raise LidResourceError("Invalid validation fraction or weak-label weight")
    root = Path(comi_root)
    checksums, source_assets = _comi_sources(root, Path(source_lock))
    revision = source_assets[0]["revision"]
    counts: Counter[str] = Counter()
    # Read no test labels during quarantine: every official-test information
    # source is excluded from fitting, including unselected/invalid test rows.
    test_groups: set[str] = set()
    for _, row in _rows(root / "LID_test.csv"):
        try:
            test_groups.add(_group(row["Sentences"]))
        except TextPreparationError:
            counts["test_quarantine_invalid_sentence"] += 1
    samplers = {AUXILIARY_TRAIN: _Sampler(train_limit, seed),
                AUXILIARY_VALIDATION: _Sampler(validation_limit, seed),
                AUXILIARY_TEST: _Sampler(test_limit, seed)}
    validation_groups: set[str] = set()
    for filename in ("LID_train.csv", "LID_test.csv"):
        seen: set[str] = set()
        for number, row in _rows(root / filename):
            counts[f"{filename}:rows_seen"] += 1
            try:
                group = _group(row["Sentences"])
            except TextPreparationError:
                counts[f"{filename}:rejected_invalid_sentence"] += 1
                continue
            if filename == "LID_train.csv":
                if group in test_groups:
                    counts["train_rows_quarantined_for_official_test_overlap"] += 1
                    continue
                fraction = int(_hash([seed, group, "auxiliary_split"]), 16) / (1 << 256)
                partition = AUXILIARY_VALIDATION if fraction < validation_fraction else AUXILIARY_TRAIN
                if partition == AUXILIARY_VALIDATION:
                    validation_groups.add(group)
            else:
                partition = AUXILIARY_TEST
            if group in seen:
                counts[f"{filename}:duplicate_sentence_groups"] += 1
                continue
            seen.add(group)
            try:
                sentence = parse_comi_row(row)
            except AnnotationRejection as exc:
                counts[f"{filename}:rejected_{exc}"] += 1
                continue
            counts[f"{filename}:rows_parsed"] += 1
            counts.update({f"{filename}:{key}": value for key, value in sentence.counts.items()})
            for start, end, label in sentence.labels:
                source = f"COMI-LINGUA@{revision}:{filename}:row={number}:human_majority_2of3"
                example = LabeledTokenExample(sentence.text, start, end, label,
                                              partition, "comi:" + group, source)
                samplers[partition].add(example)
    partitions = {key: sampler.selected() for key, sampler in samplers.items()}
    weak_manifest: Mapping[str, Any] | None = None
    if dakshina_root is not None:
        cache = verify_cache(Path(dakshina_root))
        if cache is None:
            raise LidResourceError("Dakshina root must contain a verified sealed cache")
        relative = "dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv"
        member = next(item for item in cache["members"] if item["path"] == relative)
        weak_manifest = {"license": cache["license"], "archive": cache["archive"],
                         "member": member, "manifest_sha256": cache["manifest_sha256"],
                         "supervision": "weak_isolated_word_language_from_lexicon_membership",
                         "weight": weak_weight, "dev_test_used_for_fit": False}
        sampler = _Sampler(weak_bn_limit, seed)
        seen_weak: set[str] = set()
        with (Path(dakshina_root) / relative).open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                counts["dakshina_bn_train:rows_seen"] += 1
                if len(row) != 3 or not row[2].isdigit() or int(row[2]) < 1:
                    counts["dakshina_bn_train:rejected_invalid_row"] += 1
                    continue
                try:
                    native, roman = (normalize_processing_text(value) for value in row[:2])
                except TextPreparationError:
                    counts["dakshina_bn_train:rejected_invalid_text"] += 1
                    continue
                native_spans, roman_spans = tokenize_with_spans(native), tokenize_with_spans(roman)
                if (len(native_spans) != 1 or native_spans[0].text != native
                        or native_spans[0].script != "bengali" or len(roman_spans) != 1
                        or roman_spans[0].text != roman or roman_spans[0].script != "latin"):
                    counts["dakshina_bn_train:rejected_nonword_or_script"] += 1
                    continue
                for text in (native, roman):
                    identity = _group(text)
                    if identity in test_groups or identity in validation_groups:
                        counts["dakshina_bn_train:rejected_heldout_text_overlap"] += 1
                        continue
                    if identity in seen_weak:
                        continue
                    seen_weak.add(identity)
                    source = f"Dakshina-v1.0:{relative}:row={number}:weak_lexicon_language"
                    sampler.add(LabeledTokenExample(text, 0, len(text), "bn", AUXILIARY_TRAIN,
                                                    "dakshina-bn:" + _group(native), source, weak_weight))
        partitions[AUXILIARY_TRAIN] = (*partitions[AUXILIARY_TRAIN], *sampler.selected())
    if any(not partitions[partition] for partition in samplers):
        raise LidResourceError("A human partition is empty; inspect annotations or increase source/split limits")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "sources": {"comi_lingua": {"revision": revision, "assets": source_assets,
                                    "checksums": checksums, "license": "CC-BY-4.0"},
                    "dakshina_bn_weak_train": weak_manifest},
        "annotation_policy": {"human_columns": list(ANNOTATORS), "predicted_tags_used": False,
                              "majority_required": 2, "nfc_before_span_alignment": True,
                              "ambiguous_mixed_script_annotations": "reject",
                              "casefold_whitespace_sentence_groups": True},
        "sampling": {"method": "bounded_per_label_sha256_reservoir_balanced_fill",
                     "human_train_limit": train_limit, "weak_bn_train_limit": weak_bn_limit,
                     "validation_limit": validation_limit, "test_limit": test_limit,
                     "validation_fraction_of_training_source_groups": validation_fraction},
        "isolation": {"official_test_partition": AUXILIARY_TEST,
                      "official_test_groups_quarantined": len(test_groups),
                      "training_source_validation_groups": len(validation_groups),
                      "heldout_labels_used_for_training": False},
        "counts": dict(sorted(counts.items())),
        "partitions": {partition: {"examples": len(examples),
                                   "labels": dict(Counter(item.label for item in examples)),
                                   "scripts": dict(Counter(tokenize_with_spans(item.text[item.start:item.end])[0].script for item in examples)),
                                   "groups": len({item.group_id for item in examples}),
                                   "examples_sha256": _hash([asdict(item) for item in examples])}
                       for partition, examples in partitions.items()},
        "limitations": ["Human Hindi-English sentence labels, not manually labeled retrieval queries",
                        "No human Bengali code-mixed calibration/test labels",
                        "Dakshina lexicon membership is weak supervision, not contextual gold",
                        "No generated retrieval-query semantic review or human challenge set is supplied"],
    }
    manifest["manifest_sha256"] = _hash(manifest)
    return PreparedLidResources(partitions, manifest)


def write_lid_resources(prepared: PreparedLidResources, output_dir: str | Path) -> Mapping[str, Path]:
    """Publish a new artifact directory; existing output is never replaced."""
    output = Path(output_dir).absolute()
    for path in [*reversed(output.parents), output]:
        if path.is_symlink():
            raise LidResourceError("Symlinks are forbidden in the output path")
    if output.exists():
        raise LidResourceError("Output already exists; choose a new artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".lid-resources-", dir=output.parent))
    try:
        for partition, examples in prepared.partitions.items():
            if partition not in (AUXILIARY_TRAIN, AUXILIARY_VALIDATION, AUXILIARY_TEST):
                raise LidResourceError("Unexpected resource partition")
            with (stage / f"{partition}.jsonl").open("x", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n")
        with (stage / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(prepared.manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        if output.exists():
            raise LidResourceError("Output appeared during preparation; not replacing it")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**{partition: output / f"{partition}.jsonl" for partition in prepared.partitions},
            "manifest": output / "manifest.json"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comi-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dakshina-root", type=Path)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--weak-bn-limit", type=int, default=2000)
    parser.add_argument("--validation-limit", type=int, default=2000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        prepared = prepare_lid_resources(args.comi_root, args.source_lock,
                                         dakshina_root=args.dakshina_root,
                                         train_limit=args.train_limit, weak_bn_limit=args.weak_bn_limit,
                                         validation_limit=args.validation_limit, test_limit=args.test_limit,
                                         seed=args.seed, validation_fraction=args.validation_fraction)
        paths = write_lid_resources(prepared, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"LID resource preparation error: {exc}\n")
    print(json.dumps({"paths": {key: str(path) for key, path in paths.items()},
                      "partitions": prepared.manifest["partitions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
