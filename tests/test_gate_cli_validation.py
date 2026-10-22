"""Exercise persisted validation selection, not just the fitting API."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import test_train_gate as gate_fixtures
from data.make_splits import FINAL_EVALUATION, OFFICIAL_DEV
from training.train_gate import FEATURE_NAMES, LinearSoftmaxGate


ROOT = Path(__file__).resolve().parents[1]


class GateValidationCliTests(unittest.TestCase):
    def test_excluded_features_retrain_every_validation_candidate_and_persist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _ = gate_fixtures.GateCliTests().fixture(root)
            exclusion_sets = [
                ["mean_token_entropy", "uncertain_token_fraction"],
                [name for name in FEATURE_NAMES if name.startswith("overlap_")],
            ]
            for index, excluded in enumerate(exclusion_sets):
                output = root / f"ablated-{index}.json"
                completed = subprocess.run([
                    *command, "--output", str(output), "--validation",
                    "--temperature-candidates", "0.05", "0.25",
                    "--l2-candidates", "0", "0.001",
                    "--exclude-features", *reversed(excluded),
                ], cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                gate = LinearSoftmaxGate.load_json(output)
                report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["excluded_features"], excluded)
                self.assertEqual(gate.excluded_features, tuple(excluded))
                self.assertEqual(len(report["hyperparameter_selection"]["candidates"]), 4)
                for candidate in report["hyperparameter_selection"]["candidates"]:
                    self.assertEqual(candidate["excluded_features"], excluded)
                for name in excluded:
                    column = FEATURE_NAMES.index(name)
                    self.assertEqual(gate.feature_mean[column], 0.0)
                    self.assertEqual(gate.feature_scale[column], 1.0)
                    self.assertTrue(all(row[column] == 0.0 for row in gate.weights))

    def test_cli_rejects_unknown_or_duplicate_exclusions_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _ = gate_fixtures.GateCliTests().fixture(root)
            output = root / "denied.json"
            for excluded in (["unknown_feature"], ["mean_token_entropy", "mean_token_entropy"]):
                completed = subprocess.run([
                    *command, "--output", str(output), "--exclude-features", *excluded,
                ], cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".report.json").exists())

    def test_validation_grid_saves_selected_gate_and_auditable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _ = gate_fixtures.GateCliTests().fixture(root)
            query_path = root / "queries.jsonl"
            records = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
            gate_fixtures.write_records(query_path, [*records, gate_fixtures.original("q-final", "private heldout need", split=FINAL_EVALUATION, partition=OFFICIAL_DEV)])
            output = root / "selected.json"
            completed = subprocess.run([
                *command, "--output", str(output), "--validation",
                "--temperature-candidates", "0.05", "0.25",
                "--l2-candidates", "0", "0.001",
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
            selection = report["hyperparameter_selection"]
            self.assertEqual(selection["split"], "validation")
            self.assertEqual(selection["metric"], "language_macro_actual_fused_ndcg_at_10")
            self.assertEqual(len(selection["candidates"]), 4)
            self.assertEqual(set(report["training_query_ids"]), {"q-fault", "q-rain"})
            self.assertEqual(report["validation_query_ids"], ["q-validation"])
            gate = LinearSoftmaxGate.load_json(output)
            self.assertEqual(gate.training_report, report)
            self.assertEqual(report["config"], selection["candidates"][0]["config"], "ties retain the first declared candidate")
            for baseline in ("equal_weights", "tuned_fixed_weights", "gate"):
                values = report["validation"][baseline]
                self.assertEqual(values["query_count"], 1)
                self.assertEqual(values["language_macro_ndcg_at_10"], 1.0)

    def test_candidate_grid_requires_validation_and_writes_nothing_on_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _ = gate_fixtures.GateCliTests().fixture(root)
            output = root / "denied.json"
            completed = subprocess.run([
                *command, "--output", str(output), "--temperature-candidates", "0.1", "0.25",
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("--validation", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".report.json").exists())


if __name__ == "__main__":
    unittest.main()
