from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.make_splits import FINAL_EVALUATION
from data.prepare_corpus import Qrel
from evaluation.evaluate import evaluate_run, load_evaluation_queries, load_run
from retrieval.dense import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DenseCompatibilityError,
    DenseConfig,
    DenseDependencyError,
    DenseQuery,
    DenseError,
    DenseDocument,
    DenseIndex,
    TransformersE5Encoder,
    run_queries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TableEncoder:
    """Small injected encoder with explicit vectors and observable calls."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        dimension: int,
        revision: str = "table-v1",
    ) -> None:
        self._vectors = {text: tuple(vector) for text, vector in vectors.items()}
        self._dimension = dimension
        self._revision = revision
        self.calls: list[tuple[tuple[str, ...], int, int]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "test_table_encoder",
            "model_id": "fixture/table-encoder",
            "requested_revision": self._revision,
            "resolved_revision": self._revision,
            "tokenizer_revision": self._revision,
            "dimension": self.dimension,
            "fixture_only": True,
        }

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Sequence[Sequence[float]]:
        self.calls.append((tuple(texts), batch_size, max_length))
        return [self._vectors[text] for text in texts]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class DenseCoreTests(unittest.TestCase):
    def test_e5_adapter_rejects_nonbaseline_identity_before_optional_imports(self) -> None:
        with self.assertRaisesRegex(DenseError, "requires model"):
            TransformersE5Encoder("some/other-model", DEFAULT_MODEL_REVISION)
        with self.assertRaisesRegex(DenseCompatibilityError, "commit SHA"):
            TransformersE5Encoder(DEFAULT_MODEL_ID, "main")
        with self.assertRaisesRegex(DenseError, "device must be"):
            TransformersE5Encoder(
                DEFAULT_MODEL_ID,
                DEFAULT_MODEL_REVISION,
                device="cuda:1",
            )

    def test_fixture_labels_reserved_branch_and_manifest_copy_are_hardened(self) -> None:
        vectors = {
            "passage: body": (1.0, 0.0),
            "query: query": (1.0, 0.0),
        }
        index = DenseIndex(
            [DenseDocument("d", "", "body", "en", "fixture")],
            TableEncoder(vectors, dimension=2),
        )

        manifest = index.manifest
        self.assertEqual(manifest["retriever"], "dense_test_fixture")
        self.assertEqual(manifest["baseline_kind"], "deterministic_test_fixture")
        self.assertEqual(manifest["search_backend"], "brute_force_python")
        self.assertTrue(manifest["encoder"]["fixture_only"])
        with self.assertRaisesRegex(DenseError, "cannot use the E5 baseline label"):
            run_queries(
                index,
                [DenseQuery("q", "query")],
                branch="dense_e5_frozen",
            )

        # The property must not expose nested mutable state that could corrupt
        # a later saved manifest or alter fixture/production labeling.
        manifest["encoder"]["model_id"] = "tampered"
        manifest["config"]["max_length"] = 1
        self.assertEqual(
            index.manifest["encoder"]["model_id"], "fixture/table-encoder"
        )
        self.assertEqual(index.manifest["config"]["max_length"], 512)

    def test_empty_index_persists_without_calling_the_encoder(self) -> None:
        encoder = TableEncoder({}, dimension=2)
        index = DenseIndex([], encoder)
        self.assertEqual(encoder.calls, [])
        self.assertEqual(index.document_count, 0)
        self.assertEqual(index.manifest["document_count"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "empty-index"
            index.save(index_dir)
            self.assertEqual((index_dir / "vectors.jsonl").read_bytes(), b"")

            loaded_encoder = TableEncoder({}, dimension=2)
            loaded = DenseIndex.load(index_dir, loaded_encoder)
            self.assertEqual(loaded.search("anything"), ())
            self.assertEqual(loaded_encoder.calls, [])

    def test_invalid_vectors_and_duplicate_ids_are_rejected(self) -> None:
        document = DenseDocument("d", "", "body", "en", "fixture")
        for vector in ((0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0), (True, 1.0), (1.0,)):
            with self.subTest(vector=vector), self.assertRaises(DenseError):
                DenseIndex([document], TableEncoder({"passage: body": vector}, dimension=2))
        with self.assertRaisesRegex(DenseError, "Duplicate document"):
            DenseIndex([document, document], TableEncoder({}, dimension=2))

    def test_ties_negative_scores_empty_index_and_topk(self) -> None:
        encoder = TableEncoder({"passage: body": (1.0, 0.0), "query: opposite": (-1.0, 0.0)}, dimension=2)
        index = DenseIndex([DenseDocument(name, "", "body", "en", "fixture") for name in ("z", "a")], encoder)
        self.assertEqual([hit.doc_id for hit in index.search("opposite")], ["a", "z"])
        self.assertEqual(index.search("opposite")[0].score, -1.0)
        self.assertEqual(len(index.search("opposite", top_k=1)), 1)
        self.assertEqual(index.search(""), ())
        self.assertEqual(index.search("opposite", top_k=0), ())
        self.assertEqual(DenseIndex([], encoder).search("opposite"), ())
        for top_k in (-1, True, 1.5):
            with self.assertRaises(DenseError):
                index.search("opposite", top_k=top_k)

    def _documents(self) -> list[DenseDocument]:
        # Deliberately reverse ID order: the index must encode and rank using a
        # deterministic doc_id ordering.
        return [
            DenseDocument("z-rain", "Rain", "cloud moisture", "en", "fixture"),
            DenseDocument("a-quake", "Cafe\u0301", "fault movement", "en", "fixture"),
        ]

    def _vectors(self) -> dict[str, Sequence[float]]:
        return {
            "passage: Caf\u00e9\nfault movement": (3.0, 0.0, 0.0),
            "passage: Rain\ncloud moisture": (0.0, 4.0, 0.0),
            "query: caf\u00e9 quake": (10.0, 0.0, 0.0),
            "query: rain": (0.0, 7.0, 0.0),
        }

    def test_injected_encoder_gets_prefixed_nfc_text_and_vectors_are_normalized(self) -> None:
        encoder = TableEncoder(self._vectors(), dimension=3)
        config = DenseConfig(max_length=64, batch_size=2)
        index = DenseIndex(self._documents(), encoder, config)

        self.assertEqual(
            encoder.calls[0],
            (
                (
                    "passage: Caf\u00e9\nfault movement",
                    "passage: Rain\ncloud moisture",
                ),
                2,
                64,
            ),
        )
        hits = index.search("cafe\u0301 quake", top_k=2)
        self.assertEqual([hit.doc_id for hit in hits], ["a-quake", "z-rain"])
        self.assertEqual([hit.rank for hit in hits], [1, 2])
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertAlmostEqual(hits[1].score, 0.0, places=6)
        self.assertEqual(
            encoder.calls[-1], (("query: caf\u00e9 quake",), 2, 64)
        )

        manifest = index.manifest
        self.assertEqual(manifest["dimension"], 3)
        self.assertEqual(manifest["document_count"], 2)
        self.assertEqual(manifest["vector_storage"], "normalized_float32_json_v1")
        self.assertEqual(manifest["config"]["pooling"], "attention_mask_mean")
        self.assertFalse(manifest["fine_tuned"])

    def test_persistent_index_rejects_encoder_config_and_vector_corruption(self) -> None:
        config = DenseConfig(max_length=64, batch_size=2)
        encoder = TableEncoder(self._vectors(), dimension=3)
        index = DenseIndex(self._documents(), encoder, config)

        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "dense-index"
            index.save(index_dir)

            # Batch size is operational, not semantic, so a loaded index may
            # use a different query batch size with otherwise identical config.
            loaded_encoder = TableEncoder(self._vectors(), dimension=3)
            loaded = DenseIndex.load(
                index_dir,
                loaded_encoder,
                expected_config=DenseConfig(max_length=64, batch_size=1),
            )
            self.assertEqual(loaded.config.batch_size, 1)
            self.assertEqual(loaded.search("rain", top_k=1)[0].doc_id, "z-rain")

            with self.assertRaisesRegex(
                DenseCompatibilityError, "does not match index"
            ):
                DenseIndex.load(
                    index_dir,
                    TableEncoder(self._vectors(), dimension=3, revision="table-v2"),
                )

            with self.assertRaisesRegex(
                DenseCompatibilityError, "preprocessing configuration"
            ):
                DenseIndex.load(
                    index_dir,
                    TableEncoder(self._vectors(), dimension=3),
                    expected_config=DenseConfig(max_length=63),
                )

            manifest_path = index_dir / "dense_manifest.json"
            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field, corrupt_value, message in (
                ("config_hash", "0" * 64, "configuration hash mismatch"),
                ("baseline_kind", "off_the_shelf_pretrained_frozen", "search policy"),
                ("search_backend", "unvalidated_backend", "search policy"),
            ):
                with self.subTest(manifest_field=field):
                    corrupted_manifest = dict(original_manifest)
                    corrupted_manifest[field] = corrupt_value
                    manifest_path.write_text(
                        json.dumps(corrupted_manifest, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(DenseCompatibilityError, message):
                        DenseIndex.load(
                            index_dir,
                            TableEncoder(self._vectors(), dimension=3),
                        )
            manifest_path.write_text(
                json.dumps(original_manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            vector_path = index_dir / "vectors.jsonl"
            rows = [json.loads(line) for line in vector_path.read_text().splitlines()]
            # Preserve dimension and unit norm so this reaches the logical
            # artifact checksum rather than only the shape/norm validator.
            rows[0]["vector"] = [-value for value in rows[0]["vector"]]
            _write_jsonl(vector_path, rows)
            with self.assertRaisesRegex(DenseCompatibilityError, "checksum mismatch"):
                DenseIndex.load(
                    index_dir,
                    TableEncoder(self._vectors(), dimension=3),
                )

    def test_multiview_rrf_deduplicates_views_and_averages_contributions(self) -> None:
        vectors = {
            "passage: left": (1.0, 0.0),
            "passage: bridge": (1.0, 1.0),
            "passage: right": (0.0, 1.0),
            "query: caf\u00e9": (1.0, 0.0),
            "query: other": (0.0, 1.0),
        }
        encoder = TableEncoder(vectors, dimension=2)
        index = DenseIndex(
            [
                DenseDocument("z-right", "", "right", "en", "fixture"),
                DenseDocument("m-bridge", "", "bridge", "en", "fixture"),
                DenseDocument("a-left", "", "left", "en", "fixture"),
            ],
            encoder,
        )

        result = index.search_views(
            ["cafe\u0301", "caf\u00e9", "other"],
            top_k=3,
            per_view_top_k=2,
            k_view=60.0,
        )

        self.assertEqual(result.unique_view_count, 2)
        self.assertEqual(result.search_count, 2)
        self.assertEqual(result.intermediate_candidate_count, 4)
        self.assertEqual(
            [hit.doc_id for hit in result.hits],
            ["m-bridge", "a-left", "z-right"],
        )
        self.assertAlmostEqual(result.hits[0].score, 1.0 / 62.0, places=6)
        self.assertAlmostEqual(result.hits[1].score, 1.0 / 122.0, places=6)
        self.assertAlmostEqual(result.hits[2].score, 1.0 / 122.0, places=6)

    def test_cached_pinned_e5_smoke_never_downloads(self) -> None:
        try:
            encoder = TransformersE5Encoder(
                DEFAULT_MODEL_ID,
                DEFAULT_MODEL_REVISION,
                device="cpu",
                allow_download=False,
            )
        except DenseDependencyError as exc:
            self.skipTest(f"Pinned E5 dependencies/checkpoint not cached: {exc}")

        self.assertEqual(encoder.dimension, 384)
        self.assertEqual(
            encoder.metadata["requested_revision"], DEFAULT_MODEL_REVISION
        )
        index = DenseIndex(
            [
                DenseDocument(
                    "earthquake",
                    "Earthquakes",
                    "Fault movement releases energy and causes earthquakes.",
                    "en",
                    "fixture",
                ),
                DenseDocument(
                    "rain",
                    "Rain",
                    "Cloud moisture can produce precipitation.",
                    "en",
                    "fixture",
                ),
            ],
            encoder,
            DenseConfig(max_length=64, batch_size=2),
        )
        hits = index.search("What causes earthquakes?", top_k=2)
        self.assertEqual(hits[0].doc_id, "earthquake")
        self.assertTrue(all(math.isfinite(hit.score) for hit in hits))
        self.assertFalse(index.manifest["encoder"]["fixture_only"])


class DenseCliIntegrationTests(unittest.TestCase):
    def test_explicit_fixture_cli_round_trips_through_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            index_dir = root / "dense-index"
            run_path = root / "dense-run.jsonl"
            _write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d-earth",
                        "title": "Earthquake causes",
                        "text": "Fault movement releases earthquake energy.",
                        "language": "en",
                        "source": "fixture",
                    },
                    {
                        "doc_id": "d-rain",
                        "title": "Rain formation",
                        "text": "Cloud moisture produces precipitation.",
                        "language": "en",
                        "source": "fixture",
                    },
                ],
            )
            _write_jsonl(
                queries,
                [
                    {
                        "query_id": "q-earth",
                        "parent_query_id": "q-earth",
                        "judgment_query_id": "q-earth",
                        "text": "earthquake fault movement",
                        "benchmark_language": "en",
                        "languages": ["en"],
                        "split": FINAL_EVALUATION,
                        "variant_type": "original",
                        "verified": True,
                        "review_status": "verified",
                    }
                ],
            )

            build_command = [
                sys.executable,
                "-m",
                "retrieval.dense",
                "build-index",
                "--corpus",
                str(corpus),
                "--index-dir",
                str(index_dir),
                "--backend",
                "fixture-hash",
                "--fixture-dimension",
                "128",
            ]
            denied = subprocess.run(
                build_command,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("explicitly set allow_fixture=True", denied.stderr)
            self.assertFalse(index_dir.exists())

            built = subprocess.run(
                [*build_command, "--allow-fixture-encoder"],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            index_manifest = json.loads(
                (index_dir / "dense_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(index_manifest["encoder"]["fixture_only"])
            self.assertEqual(index_manifest["retriever"], "dense_test_fixture")
            self.assertEqual(
                index_manifest["baseline_kind"], "deterministic_test_fixture"
            )
            self.assertEqual(index_manifest["search_backend"], "brute_force_python")
            self.assertEqual(
                index_manifest["vector_storage"], "normalized_float32_json_v1"
            )
            self.assertEqual(
                index_manifest["config"]["text_normalization"], "unicode_nfc"
            )
            self.assertEqual(index_manifest["corpus_path"], str(corpus))
            self.assertEqual(
                index_manifest["corpus_file_sha256"],
                hashlib.sha256(corpus.read_bytes()).hexdigest(),
            )

            searched = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval.dense",
                    "search",
                    "--index-dir",
                    str(index_dir),
                    "--queries",
                    str(queries),
                    "--split",
                    FINAL_EVALUATION,
                    "--output",
                    str(run_path),
                    "--top-k",
                    "2",
                    "--backend",
                    "fixture-hash",
                    "--fixture-dimension",
                    "128",
                    "--allow-fixture-encoder",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(searched.returncode, 0, searched.stderr)

            run = load_run(run_path)
            self.assertEqual(len(run), 2)
            self.assertEqual(run[0].query_id, "q-earth")
            self.assertEqual(run[0].doc_id, "d-earth")
            self.assertEqual(run[0].branch, "dense_fixture_hash")
            self.assertEqual([entry.rank for entry in run], [1, 2])

            run_manifest = json.loads(
                run_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(run_manifest["fixture_only"])
            self.assertEqual(run_manifest["selected_split"], FINAL_EVALUATION)
            self.assertEqual(run_manifest["result_count"], 2)
            self.assertEqual(run_manifest["index_dir"], str(index_dir))
            self.assertEqual(
                run_manifest["index_manifest_file_sha256"],
                hashlib.sha256(
                    (index_dir / "dense_manifest.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                run_manifest["index_vector_file_sha256"],
                hashlib.sha256((index_dir / "vectors.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                run_manifest["query_encoder_operational_metadata"]["backend"],
                "fixture_hash_v1",
            )

            evaluation_queries = load_evaluation_queries(
                queries, split=FINAL_EVALUATION
            )
            report = evaluate_run(
                evaluation_queries,
                [Qrel("q-earth", "d-earth", 1)],
                run,
                bootstrap_samples=0,
            )
            self.assertEqual(report.summary["overall"]["ndcg_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["mrr_at_10"], 1.0)
            self.assertEqual(report.summary["overall"]["recall_at_100"], 1.0)

    def test_empty_fixture_index_reports_zero_actual_searches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "empty-corpus.jsonl"
            queries = root / "queries.jsonl"
            index_dir = root / "empty-index"
            run_path = root / "empty-run.jsonl"
            corpus.write_text("", encoding="utf-8")
            _write_jsonl(
                queries,
                [
                    {
                        "query_id": "q",
                        "text": "nonempty query",
                        "split": FINAL_EVALUATION,
                    }
                ],
            )
            common_encoder_arguments = [
                "--backend",
                "fixture-hash",
                "--fixture-dimension",
                "8",
                "--allow-fixture-encoder",
            ]
            built = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval.dense",
                    "build-index",
                    "--corpus",
                    str(corpus),
                    "--index-dir",
                    str(index_dir),
                    *common_encoder_arguments,
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            searched = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval.dense",
                    "search",
                    "--index-dir",
                    str(index_dir),
                    "--queries",
                    str(queries),
                    "--split",
                    FINAL_EVALUATION,
                    "--output",
                    str(run_path),
                    *common_encoder_arguments,
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(searched.returncode, 0, searched.stderr)
            self.assertEqual(run_path.read_bytes(), b"")
            run_manifest = json.loads(
                run_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["query_count"], 1)
            self.assertEqual(run_manifest["search_count"], 0)
            self.assertEqual(run_manifest["intermediate_candidate_count"], 0)
            self.assertEqual(run_manifest["result_count"], 0)


if __name__ == "__main__":
    unittest.main()
