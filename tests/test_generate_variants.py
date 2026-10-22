from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from data.generate_variants import (
    VariantError,
    generate_variants,
    load_assigned_queries,
    write_variant_artifacts,
)
from data.make_splits import (
    ENCODER_TRAIN,
    FINAL_EVALUATION,
    OFFICIAL_DEV,
    OFFICIAL_TRAIN,
    VALIDATION,
    canonical_query,
    make_splits,
    write_split_artifacts,
)
from data.prepare_corpus import Qrel
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from retrieval.bm25 import (
    BM25Document,
    BM25Index,
    RetrievalQuery,
    run_queries,
    write_run_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def original(
    query_id: str = "q",
    text: str = "why does Delhi not get rainfall in 2026",
    *,
    split: str | None = ENCODER_TRAIN,
    partition: str = OFFICIAL_TRAIN,
    group: str | None = None,
) -> dict:
    query = canonical_query(
        {
            "query_id": query_id,
            "text": text,
            "languages": ["en"],
            "group_id": group or query_id,
        },
        source_partition=partition,
    )
    return asdict(replace(query, split=split))


def proposal(
    text: str,
    *,
    parent: str = "q",
    kind: str = "code_mixed",
    human_author: str | None = None,
) -> dict:
    value = {
        "parent_query_id": parent,
        "text": text,
        "variant_type": kind,
        "languages": ["en", "hi"],
        "provenance": {"source": "test-fixture", "revision": "v1"},
    }
    if human_author is not None:
        value["author_id"] = human_author
    return value


def review(candidate: dict, **changes: object) -> dict:
    value = {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "parent_query_id": candidate["parent_query_id"],
        "reviewer_id": "reviewer-two",
        "decision": "verified",
        "meaning_preserved": True,
        "fluent_languages": list(candidate["languages"]),
    }
    value.update(changes)
    return value


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


class GenerateVariantsTests(unittest.TestCase):
    def test_generation_requires_complete_parent_level_split_assignment(self) -> None:
        for invalid in (
            [original(split=None)],
            [original(split="test")],
            [original(), original("related", group="q", split=VALIDATION)],
            [original(partition=OFFICIAL_DEV, split=ENCODER_TRAIN)],
        ):
            with self.subTest(records=invalid), self.assertRaises(VariantError):
                generate_variants(invalid)

    def test_duplicate_or_orphan_input_does_not_produce_variants(self) -> None:
        duplicate = original()
        orphan = {
            **original("orphan"),
            "parent_query_id": "missing",
            "variant_type": "typo_only",
            "verified": False,
            "review_status": "unreviewed",
            "judgment_query_id": None,
        }
        for invalid in ([duplicate, duplicate], [original(), orphan]):
            with self.subTest(records=invalid), self.assertRaises(VariantError):
                generate_variants(invalid)

    def test_generated_typo_is_unreviewed_and_preserves_protected_values(self) -> None:
        parent = original(text="why does Delhi not get rainfall on 2026-10-16")
        result = generate_variants([parent], seed=11)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertNotEqual(candidate["text"], parent["text"])
        for protected in ("Delhi", "not", "2026-10-16"):
            self.assertIn(protected, candidate["text"])
        self.assertEqual(candidate["variant_type"], "typo_only")
        self.assertEqual(candidate["parent_query_id"], "q")
        self.assertEqual(candidate["group_id"], "q")
        self.assertEqual(candidate["split"], ENCODER_TRAIN)
        self.assertFalse(candidate["verified"])
        self.assertEqual(candidate["review_status"], "unreviewed")
        self.assertIsNone(candidate["judgment_query_id"])
        self.assertEqual(len(candidate["candidate_sha256"]), 64)
        self.assertTrue(candidate["provenance"])
        self.assertTrue(candidate["noise_severity"])
        self.assertEqual(len(result.review_queue), 1)
        self.assertEqual({item["query_id"] for item in result.queries}, {"q"})

    def test_reproducibility_is_independent_of_parent_order(self) -> None:
        parents = [original(f"q-{number}") for number in range(5)]
        forward = generate_variants(parents, seed=42, typo_variants=2)
        reverse = generate_variants(list(reversed(parents)), seed=42, typo_variants=2)
        self.assertEqual(forward, reverse)
        different = generate_variants(parents, seed=99, typo_variants=2)
        self.assertNotEqual(
            [(item["candidate_id"], item["text"]) for item in forward.candidates],
            [(item["candidate_id"], item["text"]) for item in different.candidates],
        )

    def test_proposal_order_and_duplicates_do_not_multiply_information_needs(self) -> None:
        proposals = [
            proposal("why does Delhi not receive rainfall in 2026"),
            proposal("why does Delhi not get precipitation in 2026"),
        ]
        forward = generate_variants([original()], proposals=proposals, typo_variants=0)
        reverse = generate_variants([original()], proposals=list(reversed(proposals)), typo_variants=0)
        duplicate = generate_variants([original()], proposals=[*proposals, proposals[0]], typo_variants=0)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, duplicate)
        self.assertEqual(len({item["candidate_id"] for item in forward.candidates}), 2)

    def test_candidate_budget_counts_safe_variants_and_retains_excess_for_audit(self) -> None:
        result = generate_variants([original()], typo_variants=4, max_variants_per_parent=2)
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual(len(result.review_queue), 2)
        self.assertEqual(sum(item["safety_status"] == "safe" for item in result.candidates), 2)
        for item in result.candidates:
            if item["safety_status"] == "rejected":
                self.assertEqual(item["review_status"], "rejected")
                self.assertFalse(item["verified"])
                self.assertIsNone(item["judgment_query_id"])

    def test_romanization_preserves_language_metadata_despite_latin_script(self) -> None:
        parent = {
            **original(text="भूकंप क्यों आते हैं"),
            "languages": ["hi"],
            "benchmark_language": "hi",
        }
        proposed = {
            **proposal("bhukamp kyu aate hain", kind="romanized"),
            "languages": ["hi"],
        }
        result = generate_variants([parent], proposals=[proposed], typo_variants=0)
        self.assertEqual(result.candidates[0]["languages"], ["hi"])
        self.assertEqual(result.candidates[0]["benchmark_language"], "hi")
        with self.assertRaises(VariantError):
            generate_variants(
                [parent],
                proposals=[{**proposed, "languages": ["en"]}],
                typo_variants=0,
            )

    def test_proposal_metadata_cannot_assign_splits_or_approve_itself(self) -> None:
        proposed = {
            **proposal("why does Delhi not receive rainfall in 2026"),
            "split": FINAL_EVALUATION,
            "group_id": "foreign-group",
            "source_partition": OFFICIAL_DEV,
            "verified": True,
            "judgment_query_id": "foreign-labels",
        }
        result = generate_variants([original()], proposals=[proposed], typo_variants=0)
        candidate = result.candidates[0]
        self.assertEqual(candidate["split"], ENCODER_TRAIN)
        self.assertEqual(candidate["group_id"], "q")
        self.assertEqual(candidate["source_partition"], OFFICIAL_TRAIN)
        self.assertFalse(candidate["verified"])
        self.assertIsNone(candidate["judgment_query_id"])
        self.assertEqual({item["query_id"] for item in result.queries}, {"q"})

    def test_filter_retains_parent_assignment_and_excludes_other_partitions(self) -> None:
        parents = [
            original("train"),
            original("dev", partition=OFFICIAL_DEV, split=FINAL_EVALUATION),
        ]
        selected = generate_variants(parents, split=FINAL_EVALUATION)
        self.assertTrue(selected.candidates)
        self.assertTrue(all(item["split"] == FINAL_EVALUATION for item in selected.candidates))
        self.assertEqual({item["parent_query_id"] for item in selected.candidates}, {"dev"})
        self.assertTrue(all(item["split"] == FINAL_EVALUATION for item in selected.queries))

    def test_proposals_cannot_change_numbers_negation_or_names(self) -> None:
        parent = original()
        for text in (
            parent["text"].replace("2026", "2027"),
            parent["text"].replace(" not ", " "),
            parent["text"].replace("Delhi", "Mumbai"),
        ):
            with self.subTest(text=text):
                proposed = [proposal(text)]
                result = generate_variants([parent], proposals=proposed, typo_variants=0)
                self.assertEqual(len(result.candidates), 1)
                candidate = result.candidates[0]
                self.assertEqual(candidate["safety_status"], "rejected")
                self.assertFalse(candidate["verified"])
                self.assertIsNone(candidate["judgment_query_id"])
                self.assertTrue(candidate["rejection_reason"])
                with self.assertRaises(VariantError):
                    generate_variants(
                        [parent],
                        proposals=proposed,
                        reviews=[review(candidate)],
                        typo_variants=0,
                    )

    def test_explicit_protection_survives_and_output_offsets_match_candidate_text(self) -> None:
        parent = original()
        start = parent["text"].index("rainfall")
        parent["protected_spans"] = [{"start": start, "end": start + 8, "text": "rainfall"}]
        result = generate_variants(
            [parent],
            proposals=[proposal("why does Delhi not receive rainfall in 2026")],
            typo_variants=0,
        )
        candidate = result.candidates[0]
        self.assertEqual(candidate["safety_status"], "safe")
        self.assertTrue(candidate["protected_spans"])
        for span in candidate["protected_spans"]:
            self.assertEqual(candidate["text"][span["start"]:span["end"]], span["text"])
        for span in candidate["parent_protected_spans"]:
            self.assertEqual(parent["text"][span["start"]:span["end"]], span["text"])
        changed = generate_variants(
            [parent],
            proposals=[proposal("why does Delhi not get precipitation in 2026")],
            typo_variants=0,
        ).candidates[0]
        self.assertEqual(changed["safety_status"], "rejected")
        self.assertIsNone(changed["judgment_query_id"])

    def test_verified_review_is_bound_to_exact_candidate_and_parent(self) -> None:
        parents = [original()]
        generated = generate_variants(parents)
        candidate = generated.candidates[0]
        for changes in (
            {"candidate_sha256": "0" * 64},
            {"parent_query_id": "different-parent"},
            {"candidate_id": "different-candidate"},
            {"reviewer_id": ""},
            {"meaning_preserved": False},
            {"fluent_languages": ["bn"]},
        ):
            with self.subTest(changes=changes), self.assertRaises(VariantError):
                generate_variants(parents, reviews=[review(candidate, **changes)])
        approved = generate_variants(parents, reviews=[review(candidate)])
        reviewed = next(item for item in approved.candidates if item["candidate_id"] == candidate["candidate_id"])
        self.assertTrue(reviewed["verified"])
        self.assertEqual(reviewed["review_status"], "verified")
        self.assertEqual(reviewed["judgment_query_id"], "q")
        self.assertIn(reviewed["query_id"], {item["query_id"] for item in approved.queries})
        self.assertEqual(approved.review_queue, ())

    def test_human_challenge_requires_independent_reviewer(self) -> None:
        parents = [original()]
        human = proposal(
            "why does Delhi not receive rainfall in 2026",
            kind="human_written",
            human_author="writer-one",
        )
        generated = generate_variants(parents, proposals=[human], typo_variants=0)
        candidate = generated.candidates[0]
        self.assertEqual(candidate["generation_kind"], "human_written")
        self.assertEqual(candidate["author_id"], "writer-one")
        with self.assertRaises(VariantError):
            generate_variants(
                parents,
                proposals=[human],
                reviews=[review(candidate, reviewer_id="writer-one")],
                typo_variants=0,
            )
        approved = generate_variants(
            parents,
            proposals=[human],
            reviews=[review(candidate)],
            typo_variants=0,
        )
        self.assertTrue(approved.candidates[0]["verified"])
        without_author = {key: value for key, value in human.items() if key != "author_id"}
        with self.assertRaises(VariantError):
            generate_variants(parents, proposals=[without_author], typo_variants=0)

    def test_review_rejection_prevents_judgment_inheritance(self) -> None:
        parents = [original()]
        candidate = generate_variants(parents).candidates[0]
        result = generate_variants(
            parents,
            reviews=[review(candidate, decision="rejected", meaning_preserved=False)],
        )
        self.assertFalse(result.candidates[0]["verified"])
        self.assertEqual(result.candidates[0]["review_status"], "rejected")
        self.assertIsNone(result.candidates[0]["judgment_query_id"])
        self.assertNotIn(candidate["query_id"], {item["query_id"] for item in result.queries})
        self.assertEqual(result.review_queue, ())

    def test_generated_ids_and_review_binding_change_when_parent_text_changes(self) -> None:
        candidate = generate_variants([original()]).candidates[0]
        changed_parent = original(text="why does Delhi not get sunshine in 2026")
        with self.assertRaises(VariantError):
            generate_variants([changed_parent], reviews=[review(candidate)])

    def test_existing_bound_approval_cannot_survive_text_or_review_tampering(self) -> None:
        parent = original(text="why does river water move slowly")
        proposed = [proposal("why does river water move quickly")]
        candidate = generate_variants([parent], proposals=proposed, typo_variants=0).candidates[0]
        approved = generate_variants([parent], proposals=proposed, reviews=[review(candidate)], typo_variants=0)
        verified = next(item for item in approved.queries if item["variant_type"] != "original")
        self.assertEqual(len(generate_variants([parent, verified], typo_variants=0).queries), 2)
        for tampered in (
            {**verified, "text": "why does river water stop flowing"},
            {**verified, "review": {**verified["review"], "candidate_sha256": "0" * 64}},
            {**verified, "reviewer_id": "different-reviewer"},
        ):
            with self.subTest(tampered=tampered), self.assertRaises(VariantError):
                generate_variants([parent, tampered], typo_variants=0)

    def test_existing_variant_cannot_cross_parent_split(self) -> None:
        variant = {
            **original("variant", split=VALIDATION),
            "parent_query_id": "q",
            "group_id": "q",
            "variant_type": "typo_only",
            "verified": False,
            "review_status": "unreviewed",
            "judgment_query_id": None,
        }
        with self.assertRaises(VariantError):
            generate_variants([original(), variant], typo_variants=0)

    def test_existing_unverified_variant_never_gains_judgments_implicitly(self) -> None:
        variant = {
            **original("variant", text="why does Delhi not get rainfal in 2026"),
            "parent_query_id": "q",
            "group_id": "q",
            "variant_type": "typo_only",
            "verified": False,
            "review_status": "unreviewed",
            "judgment_query_id": None,
        }
        result = generate_variants([original(), variant], typo_variants=0)
        self.assertEqual({item["query_id"] for item in result.queries}, {"q"})
        self.assertEqual(len(result.review_queue), 1)
        self.assertFalse(result.review_queue[0]["verified"])
        self.assertIsNone(result.review_queue[0]["judgment_query_id"])

    def test_verified_existing_variant_retains_canonical_judgment_source(self) -> None:
        variant = {
            **original("variant", text="why does Delhi not get rainfal in 2026"),
            "parent_query_id": "q",
            "group_id": "q",
            "variant_type": "typo_only",
            "verified": True,
            "review_status": "verified",
            "judgment_query_id": "q",
        }
        result = generate_variants([original(), variant], typo_variants=0)
        self.assertEqual({item["query_id"] for item in result.queries}, {"q", "variant"})
        kept = next(item for item in result.queries if item["query_id"] == "variant")
        self.assertTrue(kept["verified"])
        self.assertEqual(kept["judgment_query_id"], "q")
        self.assertEqual(kept["split"], ENCODER_TRAIN)
        tampered = {**variant, "judgment_query_id": "variant"}
        with self.assertRaises(VariantError):
            generate_variants([original(), tampered], typo_variants=0)


class GenerateVariantsIntegrationTests(unittest.TestCase):
    def test_split_review_retrieval_serialization_and_scoring(self) -> None:
        source = canonical_query(
            {
                "query_id": "q-rain",
                "text": "why does rainfall occur",
                "languages": ["en"],
            },
            source_partition=OFFICIAL_TRAIN,
        )
        split_result = make_splits([source])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_paths = write_split_artifacts(split_result, root / "split")
            assigned = load_assigned_queries(split_paths["queries"])
            unreviewed = generate_variants(assigned, seed=5)
            self.assertEqual(len(unreviewed.candidates), 1)
            candidate = unreviewed.candidates[0]
            approved = generate_variants(assigned, seed=5, reviews=[review(candidate)])
            paths = write_variant_artifacts(approved, root / "variants")
            queries = load_evaluation_queries(paths["queries"], split=ENCODER_TRAIN)
            self.assertEqual(len(queries), 2)
            self.assertTrue(all(item.verified for item in queries))
            self.assertEqual({item.judgment_query_id for item in queries}, {"q-rain"})
            index = BM25Index(
                [
                    BM25Document("d-rain", "", "why does rainfall occur", "en", "fixture"),
                    BM25Document("d-space", "", "satellite astronomy", "en", "fixture"),
                ]
            )
            results = run_queries(
                index,
                [RetrievalQuery(item["query_id"], item["text"], item["split"]) for item in approved.queries],
            )
            run_path = root / "run.jsonl"
            write_run_artifacts(results, run_path, {}, root / "run-manifest.json")
            report = evaluate_run(
                queries,
                [Qrel("q-rain", "d-rain", 1)],
                load_run(run_path),
                bootstrap_samples=10,
            )
            self.assertEqual(report.summary["evaluated_query_count"], 2)
            self.assertEqual(report.summary["status_counts"]["skipped_unverified"], 0)
            self.assertTrue(all(item.ndcg_at_10 == 1.0 for item in report.per_query))
            with self.assertRaises(VariantError):
                write_variant_artifacts(approved, root / "variants")

    def test_cli_requires_split_and_emits_reproducible_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.jsonl"
            write_jsonl(query_path, [original()])
            base = [sys.executable, "-m", "data.generate_variants", "--queries", str(query_path)]
            denied = subprocess.run(
                [*base, "--output-dir", str(root / "denied")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("--split", denied.stderr)
            outputs = [root / "first", root / "second"]
            for output in outputs:
                completed = subprocess.run(
                    [*base, "--output-dir", str(output), "--split", ENCODER_TRAIN, "--seed", "19"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue((output / "queries.jsonl").is_file())
                self.assertTrue((output / "candidates.jsonl").is_file())
                self.assertTrue((output / "review_queue.jsonl").is_file())
                self.assertTrue((output / "manifest.json").is_file())
            for filename in ("queries.jsonl", "candidates.jsonl", "review_queue.jsonl"):
                self.assertEqual((outputs[0] / filename).read_bytes(), (outputs[1] / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
