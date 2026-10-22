"""Small offline checks of the official-source lock construction."""

from __future__ import annotations

import copy
import unittest

from data.plan_miracl import CORPUS_REVISION, TOPICS_REVISION, SourcePlanError, build_plan


def metadata():
    topics, corpus = [], []
    for language in ("en", "hi", "bn"):
        for split in ("train", "dev"):
            for kind in ("topics", "qrels"):
                topics.append({"rfilename": f"miracl-v1.0-{language}/{kind}/{kind}.miracl-v1.0-{language}-{split}.tsv", "size": 20, "blobId": "a" * 40})
        corpus.append({"rfilename": f"miracl-corpus-v1.0-{language}/docs-0.jsonl.gz", "size": 100, "lfs": {"sha256": "b" * 64, "size": 100}})
    return {"miracl/miracl": {"sha": TOPICS_REVISION, "siblings": topics}, "miracl/miracl-corpus": {"sha": CORPUS_REVISION, "siblings": corpus}}


class MiraclSourcePlanTests(unittest.TestCase):
    def test_complete_plan_is_immutable_order_independent_and_excludes_tests(self):
        source = metadata()
        expected = build_plan(source)
        self.assertEqual(len(expected["assets"]), 15)
        self.assertEqual(expected["total_download_bytes"], 540)
        self.assertTrue(all("/resolve/" in asset["url"] and "main/" not in asset["url"] for asset in expected["assets"]))
        source["miracl/miracl"]["siblings"].append({"rfilename": "miracl-v1.0-en/topics/topics.miracl-v1.0-en-test-b.tsv", "size": 20, "blobId": "c" * 40})
        for repository in source.values():
            repository["siblings"].reverse()
        self.assertEqual(build_plan(source), expected)

    def test_missing_sources_wrong_revisions_and_inconsistent_hashes_fail(self):
        for change in ("revision", "missing_topics", "missing_shard", "bad_hash", "bad_size", "duplicate"):
            source = copy.deepcopy(metadata())
            if change == "revision":
                source["miracl/miracl"]["sha"] = "c" * 40
            elif change == "missing_topics":
                source["miracl/miracl"]["siblings"].pop()
            elif change == "missing_shard":
                source["miracl/miracl-corpus"]["siblings"][0]["rfilename"] = "miracl-corpus-v1.0-en/docs-1.jsonl.gz"
            elif change == "bad_hash":
                source["miracl/miracl-corpus"]["siblings"][0]["lfs"]["sha256"] = "invalid"
            elif change == "bad_size":
                source["miracl/miracl-corpus"]["siblings"][0]["lfs"]["size"] = 1
            else:
                source["miracl/miracl"]["siblings"].append(source["miracl/miracl"]["siblings"][0])
            with self.subTest(change=change), self.assertRaises(SourcePlanError):
                build_plan(source)


if __name__ == "__main__":
    unittest.main()
