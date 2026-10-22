"""Protect sealed retrieval artifacts from run-output writes."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from retrieval.fusion import main


class FusionArtifactSafetyTests(unittest.TestCase):
    def test_force_cannot_add_or_replace_files_in_frozen_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for option in ("--checkpoint", "--dense-index"):
                frozen = root / option.removeprefix("--")
                frozen.mkdir()
                marker = frozen / "sealed.json"
                marker.write_text("sealed fixture\n", encoding="utf-8")
                for output in (marker, frozen / "new-run.jsonl"):
                    with self.subTest(option=option, output=output):
                        error = io.StringIO()
                        with redirect_stderr(error):
                            result = main([
                                "--corpus", str(root / "unused-corpus.jsonl"),
                                "--queries", str(root / "unused-queries.jsonl"),
                                "--split", "validation", "--output", str(output),
                                option, str(frozen), "--force",
                            ])
                        self.assertEqual(result, 2)
                        self.assertIn("frozen", error.getvalue())
                        self.assertEqual(marker.read_text(encoding="utf-8"), "sealed fixture\n")
                        self.assertEqual(set(frozen.iterdir()), {marker})


if __name__ == "__main__":
    unittest.main()
