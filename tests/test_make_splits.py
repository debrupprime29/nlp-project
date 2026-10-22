from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from data.make_splits import (
    ENCODER_TRAIN,
    FINAL_EVALUATION,
    FUSION_TRAIN,
    OFFICIAL_DEV,
    OFFICIAL_TRAIN,
    VALIDATION,
    Query,
    SplitError,
    SplitRatios,
    canonical_query,
    make_splits,
)


def query(
    query_id: str,
    language: str = "en",
    *,
    partition: str = OFFICIAL_TRAIN,
    parent: str | None = None,
    group: str | None = None,
    variant_type: str = "original",
    verified: bool = True,
) -> Query:
    parent_id = parent or query_id
    group_id = group or parent_id
    return Query(
        query_id=query_id,
        parent_query_id=parent_id,
        group_id=group_id,
        text=f"query text {query_id}",
        languages=(language,),
        benchmark_language=language,
        source_partition=partition,
        split=None,
        variant_type=variant_type,
        verified=verified,
        review_status="verified" if verified else "unreviewed",
        judgment_query_id=parent_id if verified else None,
    )


class MakeSplitsTests(unittest.TestCase):
    def test_language_stratification_uses_group_counts(self) -> None:
        train = [
            query(f"{language}-{number}", language)
            for language in ("en", "hi", "bn")
            for number in range(20)
        ]
        result = make_splits(train, seed=9)

        for language in ("en", "hi", "bn"):
            self.assertEqual(
                result.manifest["group_counts_by_language_and_split"][language],
                {
                    ENCODER_TRAIN: 14,
                    FUSION_TRAIN: 3,
                    VALIDATION: 3,
                    FINAL_EVALUATION: 0,
                },
            )
        self.assertEqual(len(result.queries), len(train))
        self.assertTrue(all(item.split is not None for item in result.queries))

    def test_parent_variants_and_related_parents_stay_together(self) -> None:
        train = [
            query("p1", group="related"),
            query("v1", parent="p1", group="related", variant_type="typo"),
            query("p2", group="related"),
            query("p3"),
            query("p4"),
        ]
        result = make_splits(train, seed=3)
        assigned = {item.query_id: item.split for item in result.queries}

        self.assertEqual(assigned["p1"], assigned["v1"])
        self.assertEqual(assigned["p1"], assigned["p2"])
        group = next(
            item
            for item in result.manifest["group_assignments"]
            if item["group_id"] == "related"
        )
        self.assertEqual(group["parent_query_ids"], ["p1", "p2"])

    def test_official_development_is_locked_final(self) -> None:
        train = [query(f"train-{number}") for number in range(5)]
        final = [
            query("dev", partition=OFFICIAL_DEV),
            query(
                "dev-variant",
                partition=OFFICIAL_DEV,
                parent="dev",
                variant_type="romanized",
            ),
        ]
        result = make_splits(train, final)
        assigned = {item.query_id: item.split for item in result.queries}

        self.assertEqual(assigned["dev"], FINAL_EVALUATION)
        self.assertEqual(assigned["dev-variant"], FINAL_EVALUATION)
        self.assertNotIn(FINAL_EVALUATION, {assigned[item.query_id] for item in train})

    def test_assignment_and_manifest_are_input_order_independent(self) -> None:
        train = [query(f"q-{number}", "hi") for number in range(25)]
        forward = make_splits(train, seed=81)
        reverse = make_splits(list(reversed(train)), seed=81)

        self.assertEqual(forward.queries, reverse.queries)
        self.assertEqual(forward.manifest, reverse.manifest)

    def test_tiny_stratum_reports_actual_allocation_without_dropping(self) -> None:
        result = make_splits([query("only", "bn")])
        self.assertEqual(result.queries[0].split, ENCODER_TRAIN)
        self.assertEqual(
            result.manifest["training_group_allocation_by_language"]["bn"],
            {ENCODER_TRAIN: 1, FUSION_TRAIN: 0, VALIDATION: 0},
        )

    def test_duplicate_and_unknown_parent_fail(self) -> None:
        with self.assertRaisesRegex(SplitError, "Duplicate query_id"):
            make_splits([query("same"), query("same")])
        orphan = query("variant", parent="absent", variant_type="typo")
        with self.assertRaisesRegex(SplitError, "unknown parent"):
            make_splits([orphan])

    def test_cross_partition_group_fails(self) -> None:
        train = [query("train", group="shared")]
        final = [query("dev", partition=OFFICIAL_DEV, group="shared")]
        with self.assertRaisesRegex(SplitError, "crosses official source partitions"):
            make_splits(train, final)

    def test_judgment_inheritance_cannot_cross_groups_or_partitions(self) -> None:
        parent = query("parent")
        other = query("other")
        cross_group = replace(
            query("variant", parent="parent", variant_type="typo"),
            judgment_query_id="other",
        )
        with self.assertRaisesRegex(SplitError, "crosses information-need groups"):
            make_splits([parent, other, cross_group])

        dev = query("dev", partition=OFFICIAL_DEV)
        cross_partition = replace(cross_group, judgment_query_id="dev")
        with self.assertRaisesRegex(SplitError, "crosses official partitions"):
            make_splits([parent, cross_partition], [dev])

    def test_variant_parent_must_be_an_original(self) -> None:
        self_parented_variant = query(
            "variant", parent="variant", variant_type="typo"
        )
        with self.assertRaisesRegex(SplitError, "original parent"):
            make_splits([self_parented_variant])

    def test_ratio_validation(self) -> None:
        with self.assertRaisesRegex(SplitError, "sum to 1"):
            make_splits([query("q")], ratios=SplitRatios(0.8, 0.2, 0.2))

    def test_review_status_preserves_unreviewed_variants(self) -> None:
        original = canonical_query(
            {"query_id": "p", "text": "why", "language": "en"},
            source_partition=OFFICIAL_TRAIN,
        )
        variant = canonical_query(
            {
                "query_id": "v",
                "parent_query_id": "p",
                "text": "wai",
                "language": "en",
                "variant_type": "typo",
                "verified": False,
            },
            source_partition=OFFICIAL_TRAIN,
        )
        self.assertTrue(original.verified)
        self.assertFalse(variant.verified)
        self.assertEqual(variant.review_status, "unreviewed")
        self.assertIsNone(variant.judgment_query_id)

    def test_cli_is_reproducible_and_writes_locked_dev(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"
            with train_path.open("w", encoding="utf-8") as handle:
                for number in range(6):
                    handle.write(
                        json.dumps(
                            {"query_id": f"q{number}", "text": f"text {number}"}
                        )
                        + "\n"
                    )
            dev_path.write_text(
                json.dumps({"query_id": "dev", "text": "held out"}) + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "data.make_splits",
                    "--train-queries",
                    f"en={train_path}",
                    "--dev-queries",
                    f"en={dev_path}",
                    "--output-dir",
                    str(output),
                    "--seed",
                    "5",
                    "--dataset-version",
                    "fixture-v1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            records = [
                json.loads(line)
                for line in (output / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dev = next(item for item in records if item["query_id"] == "dev")
            self.assertEqual(dev["split"], FINAL_EVALUATION)
            manifest = json.loads(
                (output / "split_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset_version"], "fixture-v1")
            self.assertEqual(manifest["query_count"], 7)


if __name__ == "__main__":
    unittest.main()
