"""Launch the genuine trained app from a verified, relocated runtime bundle."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sharing.runtime_bundle import BundleError, LANGUAGES, PROJECT_ROOT, sha256_file, verify_runtime


def get_runtime_layout(root: Path) -> dict:
    root = Path(root).absolute()
    return {"checkpoint": root / "checkpoint", "lid": root / "models/lid.json",
            "lexicon": root / "resources/transliteration.jsonl",
            "languages": {lang: {"corpus": root / f"languages/{lang}/corpus.jsonl",
                                 "dense": root / f"languages/{lang}/dense",
                                 "gate": root / f"languages/{lang}/gate.json"} for lang in LANGUAGES}}


def build_arguments(runtime: Path, language: str, query: str | None = None, port: int = 8000) -> list[str]:
    if language not in LANGUAGES:
        raise BundleError("Language must be en (English), hi (Hindi) or bn (Bengali)")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise BundleError("Port must be an integer between 1 and 65535")
    layout = get_runtime_layout(runtime)
    track = layout["languages"][language]
    result = ["--corpus", str(track["corpus"]), "--lid-model", str(layout["lid"]),
              "--lexicon", str(layout["lexicon"]), "--dense-index", str(track["dense"]),
              "--backend", "e5-trained", "--checkpoint", str(layout["checkpoint"]),
              "--gate-model", str(track["gate"]), "--device", "cpu",
              "--max-length", "128", "--batch-size", "8", "--host", "127.0.0.1", "--port", str(port)]
    if query is not None:
        result.extend(["--query", query])
    return result


class _KnownE5TokenizerWarning(logging.Filter):
    """Hide only the proven local XLM-R false-positive; never patch tokenization."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        self.path = str(checkpoint)

    def filter(self, record):
        message = record.getMessage()
        return not (f"The tokenizer you are loading from '{self.path}'" in message
                    and "with an incorrect regex pattern:" in message
                    and "fix_mistral_regex=True" in message)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "artifacts/shared-runtime")
    parser.add_argument("--language", choices=LANGUAGES, default="en")
    parser.add_argument("--query")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        print("Verifying shared runtime checksums…", file=sys.stderr, flush=True)
        manifest = verify_runtime(args.runtime_dir)
        metadata = manifest["metadata"]
        if metadata.get("name") != "cpu-pilot-runtime-v1" or metadata.get("languages") != list(LANGUAGES):
            raise BundleError("This launcher requires the full three-language CPU-pilot bundle")
        expected_code = metadata.get("source_code_sha256")
        if not isinstance(expected_code, dict) or not expected_code:
            raise BundleError("Runtime has no source-code compatibility record")
        from sharing.runtime_bundle import safe_relative_path
        for name, expected in expected_code.items():
            safe_relative_path(name)
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                raise BundleError(f"Code differs from the released runtime: {name}; use the matching release tag")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Do not set fix_mistral_regex=True on this Unigram/Metaspace tokenizer.
        # Suppress only the previously diagnosed message for the hash-verified model.
        checkpoint = get_runtime_layout(args.runtime_dir)["checkpoint"]
        warning_filter = _KnownE5TokenizerWarning(checkpoint)
        logger = logging.getLogger("transformers.tokenization_utils_base")
        logger.addFilter(warning_filter)
        try:
            print(f"Loading trained {args.language} collection (1,500 passages); no downloads or training…",
                  file=sys.stderr, flush=True)
            from app import main as app_main
            return app_main(build_arguments(args.runtime_dir, args.language, args.query, args.port))
        finally:
            logger.removeFilter(warning_filter)
    except (BundleError, OSError, ValueError) as exc:
        parser.exit(2, f"Shared app error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
