"""Prepare explicit passage Romanizations and one shared offline word resource.

Only the pinned Dakshina training lexicons provide attested pairs. Optional
uroman mappings come from unlabeled corpus words, are algorithmic rather than
human-attested, and are persisted for the same passage/query lookup policy.
No query labels, relevance judgments, model downloads, or network are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.fetch_dakshina import MANIFEST_NAME as DAKSHINA_MANIFEST_NAME
from data.fetch_dakshina import DakshinaAcquisitionError, verify_cache
from data.prepare_corpus import CorpusPreparationError, _clean_id, iter_jsonl
from nlp.query_views import LexiconTransliterator, QueryViewError, TARGET_SCRIPTS, _letter_token
from nlp.text import TextPreparationError, tokenize_with_spans
from retrieval.char_tfidf import is_roman_script_view


SCHEMA_VERSION = 1
MANIFEST_NAME = "romanization_manifest.json"
LEXICON_NAME = "transliteration.jsonl"
POLICY_VERSION = "dakshina_train_then_uroman_single_native_token_v1"
UROMAN_VERSION = "1.3.1.1"
UROMAN_WHEEL_SHA256 = "394f965f7011fd56a84aca098a6c3b50082f365324f5d94c992852137918c8f5"
UROMAN_LICENSE = "USC permissive license (see bundled LICENSE.txt); publication acknowledgment required"
LANGUAGE_CODES = {"hi": "hin", "bn": "ben"}
SCRIPT_LANGUAGES = {script: language for language, script in TARGET_SCRIPTS.items()}
COUNT_FIELDS = (
    "native_token_count", "native_character_count", "attested_token_count", "attested_character_count",
    "uroman_token_count", "uroman_character_count", "unavailable_token_count", "unavailable_character_count",
    "identity_token_count", "identity_character_count",
)


class RomanizationError(ValueError):
    """Invalid sealed input, unsafe output, or incompatible local Romanizer."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _no_symlinks(path: Path) -> Path:
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise RomanizationError("Symbolic links are not allowed for sealed inputs or outputs")
    return path


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value) + b"\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for record in records:
            handle.write(_canonical(record) + b"\n")


def _read_lexicon(path: Path, language: str, *, source: str, revision: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if len(line) > 65536:
                raise RomanizationError(f"Dakshina row exceeds the line limit at {path}:{number}")
            if not line.strip():
                continue
            counts["supplied_rows"] += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise RomanizationError(f"Dakshina rows require three TSV fields at {path}:{number}")
            native, romanized, count = fields
            if not count.isascii() or not count.isdigit() or int(count) <= 0:
                raise RomanizationError(f"Dakshina attestation counts must be positive integers at {path}:{number}")
            native, romanized = unicodedata.normalize("NFC", native), unicodedata.normalize("NFC", romanized)
            if not _letter_token(native, TARGET_SCRIPTS[language]) or not _letter_token(romanized, "latin"):
                counts["excluded_non_single_script_token_rows"] += 1
                continue
            records.append({"language": language, "native": native, "romanized": romanized,
                            "source": source, "revision": revision, "count": int(count)})
            counts["accepted_rows"] += 1
    if not counts["supplied_rows"]:
        raise RomanizationError(f"Dakshina lexicon is empty: {path}")
    return records, dict(counts)


def _load_uroman() -> tuple[Any, dict[str, Any]]:
    try:
        distribution = importlib.metadata.distribution("uroman")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RomanizationError("--allow-uroman requires the explicitly installed pinned uroman==1.3.1.1; no download is attempted") from exc
    if distribution.version != UROMAN_VERSION:
        raise RomanizationError(f"uroman must be pinned to {UROMAN_VERSION}, not {distribution.version}")
    files, license_files = {}, {}
    for relative in sorted(distribution.files or (), key=str):
        # Hash the actual installed implementation and rules, not an assumed
        # wheel identity. The expected PyPI wheel hash is provenance only.
        parts = Path(str(relative)).parts
        if parts and parts[0] == "uroman" and "__pycache__" not in parts and not str(relative).endswith(".pyc"):
            path = _no_symlinks(Path(distribution.locate_file(relative)))
            if path.is_file():
                files[str(relative)] = _file_sha256(path)
        if Path(str(relative)).name.casefold().startswith("license"):
            path = _no_symlinks(Path(distribution.locate_file(relative)))
            if path.is_file():
                license_files[str(relative)] = _file_sha256(path)
    if not files:
        raise RomanizationError("Installed uroman source/data files could not be fingerprinted")
    try:
        from uroman import Uroman
        engine = Uroman()
    except Exception as exc:
        raise RomanizationError(f"The installed uroman engine could not initialize: {exc}") from exc
    return engine, {
        "package": "uroman", "version": distribution.version, "source": "https://github.com/isi-nlp/uroman",
        "license": UROMAN_LICENSE, "license_files_sha256": license_files,
        "expected_pypi_wheel_sha256": UROMAN_WHEEL_SHA256,
        "wheel_digest_verified_by_this_command": False, "installed_files_sha256": files,
        "installed_resource_sha256": _digest({"implementation_and_data": files, "license_files": license_files}), "fixture_only": False,
    }


class _Resolver:
    def __init__(self, records: Sequence[Mapping[str, Any]], engine: Any | None):
        self.records = list(records)
        self.engine = engine
        self.attested: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[(record["language"], record["native"].casefold())].append(dict(record))
        for key, values in grouped.items():
            # Same score/text/backend/revision tie policy as query candidates.
            self.attested[key] = tuple(sorted(values, key=lambda item: (-math.log(item["count"]), item["romanized"], item["source"], item["revision"])))
        self.cache: dict[tuple[str, str], tuple[str | None, str]] = {}
        self.generated: dict[tuple[str, str], dict[str, Any]] = {}
        self.fallback_unique_calls = 0

    def fallback(self, token: str, language: str) -> tuple[str | None, str]:
        if self.engine is None:
            return None, "oov_no_fallback"
        try:
            candidate = self.engine.romanize_string(token, lcode=LANGUAGE_CODES[language])
        except Exception:
            return None, "uroman_error"
        if not isinstance(candidate, str):
            return None, "uroman_non_string"
        candidate = unicodedata.normalize("NFC", candidate)
        if not _letter_token(candidate, "latin"):
            return None, "uroman_not_single_latin_token"
        return candidate, "uroman_algorithmic"

    def resolve(self, token: str, language: str) -> tuple[str | None, str]:
        key = (language, token.casefold())
        if key not in self.cache:
            attested = self.attested.get(key)
            if attested:
                self.cache[key] = attested[0]["romanized"], "attested"
            elif not _letter_token(token, TARGET_SCRIPTS[language]):
                self.cache[key] = None, "invalid_native_token"
            else:
                if self.engine is not None:
                    self.fallback_unique_calls += 1
                self.cache[key] = self.fallback(token, language)
                romanized, source = self.cache[key]
                if romanized is not None:
                    self.generated[key] = {"language": language, "native": token, "romanized": romanized,
                                           "source": source, "revision": f"{UROMAN_VERSION}:{POLICY_VERSION}", "count": 1}
        return self.cache[key]


def _romanize_text(text: str, resolver: _Resolver) -> tuple[str, str, dict[str, Any]]:
    counts: Counter[str] = Counter({name: 0 for name in COUNT_FIELDS})
    reasons: Counter[str] = Counter()
    pieces, cursor = [], 0
    for span in tokenize_with_spans(text):
        pieces.append(text[cursor:span.start])
        cursor = span.end
        if not any(unicodedata.category(character).startswith("L") for character in span.text) or span.script == "latin":
            counts["identity_token_count"] += 1
            counts["identity_character_count"] += len(span.text)
            pieces.append(span.text)
            continue
        counts["native_token_count"] += 1
        counts["native_character_count"] += len(span.text)
        language = SCRIPT_LANGUAGES.get(span.script)
        romanized, source = resolver.resolve(span.text, language) if language else (None, "unsupported_native_script")
        if romanized is None:
            counts["unavailable_token_count"] += 1
            counts["unavailable_character_count"] += len(span.text)
            reasons[source] += 1
            pieces.append("")
        else:
            prefix = "attested" if source == "attested" else "uroman"
            counts[prefix + "_token_count"] += 1
            counts[prefix + "_character_count"] += len(span.text)
            pieces.append(romanized)
    pieces.append(text[cursor:])
    result = "".join(pieces)
    if counts["unavailable_token_count"]:
        result, status = "", "unavailable"
    elif not text.strip():
        result, status = "", "empty"
    elif not is_roman_script_view(result):
        result, status = "", "unavailable"
        reasons["no_complete_roman_content"] += 1
    elif counts["uroman_token_count"]:
        status = "attested_uroman" if counts["attested_token_count"] else "uroman_algorithmic"
    else:
        status = "attested" if counts["attested_token_count"] else "identity"
    return result, status, {**dict(counts), "unavailable_reasons": dict(sorted(reasons.items()))}


def _corpus_records(path: Path):
    seen = set()
    for record in iter_jsonl(path):
        doc_id = _clean_id(record.get("doc_id"), "doc_id")
        if record.get("doc_id") != doc_id or doc_id in seen:
            raise RomanizationError("Prepared corpus document IDs must be canonical and unique")
        seen.add(doc_id)
        if record.get("language") not in ("en", "hi", "bn"):
            raise RomanizationError("Prepared corpus language must be en, hi, or bn")
        for field in ("title", "text"):
            value = record.get(field)
            if not isinstance(value, str) or "\x00" in value or unicodedata.normalize("NFC", value) != value:
                raise RomanizationError(f"Prepared corpus {field} must be an NFC string without NULs")
        if not record["text"].strip() or not isinstance(record.get("source"), str):
            raise RomanizationError("Prepared corpus requires nonempty text and a source string")
        yield record
    if not seen:
        raise RomanizationError("Prepared corpus must not be empty")


def _heldout_report(root: Path, manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]], resolver: _Resolver, resource_sha256: str) -> dict[str, Any]:
    native: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    roman: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        native[(record["language"], record["native"].casefold())].append(record)
        roman[(record["language"], record["romanized"].casefold())].append(record)
    for grouped, target in ((native, "romanized"), (roman, "native")):
        for key, values in grouped.items():
            values.sort(key=lambda item: (-math.log(item["count"]), item[target], item["source"], item["revision"]))
    reports = {}
    for member in manifest["members"]:
        name = member["path"]
        if name.endswith(".train.tsv"):
            continue
        language = Path(name).name.split(".")[0]
        split = "dev" if name.endswith(".dev.tsv") else "test"
        heldout, counts = _read_lexicon(root / name, language, source=f"Dakshina-v1-{split}", revision=member["sha256"])
        native_gold: dict[str, set[str]] = defaultdict(set)
        roman_gold: dict[str, set[str]] = defaultdict(set)
        for record in heldout:
            native_gold[record["native"]].add(record["romanized"].casefold())
            roman_gold[record["romanized"].casefold()].add(record["native"].casefold())
        direction_reports = {}
        for direction, gold, resource, target in (("native_to_roman", native_gold, native, "romanized"), ("roman_to_native", roman_gold, roman, "native")):
            matched = Counter()
            covered = 0
            for token, valid in gold.items():
                alternatives = list(dict.fromkeys(record[target].casefold() for record in resource.get((language, token.casefold()), ())))[:2]
                if not alternatives and direction == "native_to_roman":
                    candidate, _ = resolver.fallback(token, language)
                    alternatives = [candidate.casefold()] if candidate is not None else []
                covered += bool(alternatives)
                for top_k in (1, 2):
                    matched[top_k] += bool(set(alternatives[:top_k]) & valid)
            direction_reports[direction] = {
                "query_word_count": len(gold), "covered_word_count": covered,
                "top1_attested_variant_match_rate": matched[1] / len(gold) if gold else None,
                "top2_attested_variant_match_rate": matched[2] / len(gold) if gold else None,
            }
        reports[f"{language}_{split}"] = {"source_sha256": member["sha256"], "row_counts": counts, **direction_reports,
                                              "unlabeled_corpus_native_overlap_count": sum((language, token.casefold()) in resolver.generated for token in native_gold)}
    return {"schema_version": SCHEMA_VERSION, "resource_frozen_before_evaluation": True,
            "frozen_resource_sha256": resource_sha256, "heldout_labels_used_for_fit": False,
            "metric_notice": "Word-level attested-variant matching/coverage only; not semantic equivalence or retrieval quality. Algorithmic fallbacks supply at most one Roman prediction.",
            "splits": reports}


def prepare_romanized_corpora(
    corpora: Mapping[str, Path], dakshina_dir: Path, output_dir: Path, *,
    allow_uroman: bool = False, romanizer: Any | None = None, evaluate_heldout: bool = False,
) -> Mapping[str, Any]:
    """Write new corpora and a single shared resource into a fresh directory.

    Injected engines are explicit test fixtures and are labeled as such. The
    original fields and IDs are preserved; no qrel file is opened or modified.
    """
    if not isinstance(corpora, Mapping) or not corpora:
        raise RomanizationError("At least one named corpus is required")
    if type(allow_uroman) is not bool or type(evaluate_heldout) is not bool:
        raise RomanizationError("Romanizer/evaluation opt-ins must be booleans")
    output_dir, dakshina_dir = _no_symlinks(output_dir), _no_symlinks(dakshina_dir)
    if output_dir.exists():
        raise RomanizationError("Refusing to overwrite an existing output directory; choose a fresh directory")
    if output_dir.is_relative_to(dakshina_dir):
        raise RomanizationError("Outputs must not be placed inside the sealed Dakshina cache")
    sources = {}
    for name, path in sorted(corpora.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name == Path(LEXICON_NAME).stem:
            raise RomanizationError("Corpus names must be simple distinct alphanumeric/underscore/hyphen labels")
        path = _no_symlinks(path)
        if not path.is_file():
            raise RomanizationError(f"Corpus input is not a file: {path}")
        sources[name] = path
    manifest = verify_cache(dakshina_dir)
    if manifest is None:
        raise RomanizationError("A previously fetched, sealed six-file Dakshina cache is required")
    inputs = {str(path): _file_sha256(path) for path in (*sources.values(), dakshina_dir / DAKSHINA_MANIFEST_NAME)}
    records, train_counts = [], {}
    for member in manifest["members"]:
        path = dakshina_dir / member["path"]
        inputs[str(path)] = member["sha256"]
        if member["path"].endswith(".train.tsv"):
            language = path.name.split(".")[0]
            rows, counts = _read_lexicon(path, language, source="Dakshina-v1-train", revision=member["sha256"])
            records.extend(rows)
            train_counts[language] = counts
    # Validate and canonicalize every pair using the existing query resource
    # schema; query-time protection/negation rules are deliberately unchanged.
    attested = LexiconTransliterator(records)
    records = list(attested._records)
    package = {"enabled": False, "fixture_only": False}
    engine = None
    if romanizer is not None:
        if not allow_uroman or not callable(getattr(romanizer, "romanize_string", None)):
            raise RomanizationError("Injected test Romanizers require allow_uroman=True and romanize_string")
        engine, package = romanizer, {"enabled": True, "package": "injected_test_fixture", "version": UROMAN_VERSION, "fixture_only": True}
    elif allow_uroman:
        engine, package = _load_uroman()
        package["enabled"] = True
    resolver = _Resolver(records, engine)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.romanize-", dir=output_dir.parent))
    try:
        corpus_reports = {}
        for name, path in sources.items():
            totals: Counter[str] = Counter({key: 0 for key in COUNT_FIELDS})
            statuses: Counter[str] = Counter()
            reasons: Counter[str] = Counter()
            languages: Counter[str] = Counter()
            count = 0
            target = staging / f"{name}.jsonl"
            with target.open("xb") as handle:
                for record in _corpus_records(path):
                    romanized, status, counts = _romanize_text(record["text"], resolver)
                    roman_title, title_status, title_counts = _romanize_text(record["title"], resolver)
                    output = {**record, "romanized_text": romanized, "romanization_status": status,
                              "romanized_title": roman_title, "romanization_title_status": title_status,
                              "romanization_provenance": {"policy": POLICY_VERSION, "body": counts, "title": title_counts}}
                    handle.write(_canonical(output) + b"\n")
                    count += 1
                    statuses[status] += 1
                    languages[record["language"]] += 1
                    totals.update({key: counts[key] for key in COUNT_FIELDS})
                    reasons.update(counts["unavailable_reasons"])
            corpus_reports[name] = {"input": str(path), "input_sha256": inputs[str(path)], "output": target.name,
                                    "output_sha256": _file_sha256(target), "document_count": count,
                                    "available_document_count": count - statuses["unavailable"],
                                    "unavailable_document_count": statuses["unavailable"], "status_counts": dict(sorted(statuses.items())),
                                    "language_counts": dict(sorted(languages.items())), "body_counts": dict(totals),
                                    "unavailable_reasons": dict(sorted(reasons.items()))}
        combined = LexiconTransliterator([*records, *resolver.generated.values()])
        _write_jsonl(staging / LEXICON_NAME, combined._records)
        lexicon_sha256 = _file_sha256(staging / LEXICON_NAME)
        report = {
            "schema_version": SCHEMA_VERSION, "policy": POLICY_VERSION, "fixture_only": package["fixture_only"],
            "dataset": "Dakshina-v1.0", "dakshina_manifest_sha256": manifest["manifest_sha256"],
            "licenses": [{"resource": "Dakshina", "license": "CC-BY-SA-4.0", "source": "https://github.com/google-research-datasets/dakshina"},
                         {"resource": "uroman", "license": UROMAN_LICENSE, "source": "https://github.com/isi-nlp/uroman"}] if allow_uroman else [{"resource": "Dakshina", "license": "CC-BY-SA-4.0", "source": "https://github.com/google-research-datasets/dakshina"}],
            "input_files_sha256": inputs, "training_lexicon_row_counts": train_counts,
            "attested_resource_record_count": len(records), "algorithmic_resource_record_count": len(resolver.generated),
            "uroman_unique_corpus_word_calls": resolver.fallback_unique_calls,
            "unavailable_unique_supported_native_word_count": sum(value[0] is None for value in resolver.cache.values()),
            "fallback": package, "corpora": corpus_reports,
            "lexicon": {"path": LEXICON_NAME, "sha256": lexicon_sha256, "manifest": dict(combined.manifest), "fingerprint": combined.fingerprint},
            "contract": {"romanized_text_source": "passage_body_only", "romanized_title_is_auxiliary": True,
                         "original_fields_preserved": ["doc_id", "title", "text", "language", "source"],
                         "latin_and_numeric_tokens_preserved_exactly": True, "qrels_read_or_modified": False,
                         "heldout_lexicons_used_for_fit": False, "fallback_derived_only_from_unlabeled_corpus_tokens": True,
                         "algorithmic_pairs_are_human_attested": False, "native_text_used_as_hidden_character_fallback": False,
                         "query_protection_and_roman_completeness_rules_changed": False},
            "quality_notice": "Attested lexicon plus deterministic algorithmic OOV baseline, not IndicXlit execution or fluent-speaker verification. Unsupported/incomplete passages remain unavailable.",
            "heldout_evaluation": None,
        }
        if evaluate_heldout:
            heldout = _heldout_report(dakshina_dir, manifest, combined._records, resolver, lexicon_sha256)
            _write_json(staging / "heldout_lexicon_report.json", heldout)
            report["heldout_evaluation"] = {"path": "heldout_lexicon_report.json", "sha256": _file_sha256(staging / "heldout_lexicon_report.json")}
        if any(_file_sha256(Path(path)) != sha256 for path, sha256 in inputs.items()):
            raise RomanizationError("A sealed input changed during Romanization")
        if output_dir.exists() or output_dir.is_symlink():
            raise RomanizationError("Output directory appeared during processing; refusing to overwrite")
        report["manifest_sha256"] = _digest(report)
        _write_json(staging / MANIFEST_NAME, report)
        os.rename(staging, output_dir)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", required=True, help="NAME=prepared-corpus.jsonl; repeat for one shared resource")
    parser.add_argument("--dakshina-dir", type=Path, required=True, help="sealed cache produced by data.fetch_dakshina")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory; existing artifacts are never overwritten")
    parser.add_argument("--allow-uroman", action="store_true", help="opt in to an already installed pinned uroman; no downloads")
    parser.add_argument("--evaluate-heldout", action="store_true", help="report separate Dakshina word variant matching after freezing the resource")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        corpora = {}
        for value in args.corpus:
            name, path = value.split("=", 1) if "=" in value else (Path(value).stem, value)
            if name in corpora:
                raise RomanizationError("Duplicate corpus name; use distinct NAME=PATH labels")
            corpora[name] = Path(path)
        report = prepare_romanized_corpora(corpora, args.dakshina_dir, args.output_dir, allow_uroman=args.allow_uroman, evaluate_heldout=args.evaluate_heldout)
    except (RomanizationError, DakshinaAcquisitionError, CorpusPreparationError, QueryViewError, TextPreparationError, OSError, TypeError, ValueError) as exc:
        print(f"romanize_corpus: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": MANIFEST_NAME, "fixture_only": report["fixture_only"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
