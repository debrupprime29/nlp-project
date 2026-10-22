"""Create a checksum-pinned public MIRACL download lock from official metadata.

Planning is the only network operation here. Dataset bytes are acquired by
``data.acquire_resources`` after a separate explicit download opt-in/budget.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.bm25 import _atomic_write_json


TOPICS_REVISION = "5be20db9509754dadad47689368639fcec739c00"
CORPUS_REVISION = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
LANGUAGES = ("en", "hi", "bn")
REPOSITORIES = (("miracl/miracl", TOPICS_REVISION), ("miracl/miracl-corpus", CORPUS_REVISION))


class SourcePlanError(ValueError):
    """Official metadata does not identify a complete immutable source set."""


def build_plan(metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    assets = []
    for repository, revision in REPOSITORIES:
        value = metadata.get(repository)
        if not isinstance(value, Mapping) or value.get("sha") != revision:
            raise SourcePlanError(f"metadata revision mismatch for {repository}")
        siblings = value.get("siblings")
        if not isinstance(siblings, list):
            raise SourcePlanError("repository file metadata must be a list")
        seen = set()
        for record in siblings:
            if not isinstance(record, Mapping) or not isinstance(record.get("rfilename"), str):
                raise SourcePlanError("invalid repository file entry")
            path = record["rfilename"]
            if path in seen:
                raise SourcePlanError("duplicate repository file metadata")
            seen.add(path)
            if repository == "miracl/miracl":
                selected = re.fullmatch(r"miracl-v1\.0-(en|hi|bn)/(topics/topics|qrels/qrels)\.miracl-v1\.0-\1-(train|dev)\.tsv", path) is not None
            else:
                selected = re.fullmatch(r"miracl-corpus-v1\.0-(en|hi|bn)/docs-[0-9]+\.jsonl\.gz", path) is not None
            if path == "README.md":
                destination = "provenance/" + ("miracl-topics-README.md" if repository.endswith("/miracl") else "miracl-corpus-README.md")
            elif selected:
                destination = path
            else:
                continue
            size = record.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise SourcePlanError(f"missing positive byte size for {path}")
            asset = {"name": repository + ":" + path, "url": f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}", "path": destination, "size": size, "license": "Apache-2.0 (publisher dataset card; preserve underlying Wikipedia provenance)", "revision": revision}
            lfs = record.get("lfs")
            if isinstance(lfs, Mapping):
                checksum = lfs.get("sha256")
                if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None or lfs.get("size") != size:
                    raise SourcePlanError(f"missing/inconsistent content checksum for {path}")
                asset["sha256"] = checksum
            else:
                checksum = record.get("blobId")
                if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{40}", checksum) is None:
                    raise SourcePlanError(f"missing Git blob checksum for {path}")
                asset["git_blob_sha1"] = checksum
            assets.append(asset)
    paths = {asset["path"] for asset in assets}
    if len(paths) != len(assets):
        raise SourcePlanError("source assets collide on disk")
    for language in LANGUAGES:
        for split in ("train", "dev"):
            for kind in ("topics", "qrels"):
                expected = f"miracl-v1.0-{language}/{kind}/{kind}.miracl-v1.0-{language}-{split}.tsv"
                if expected not in paths:
                    raise SourcePlanError(f"missing official source {expected}")
        shards = sorted(int(re.search(r"docs-(\d+)\.jsonl\.gz$", path).group(1)) for path in paths if path.startswith(f"miracl-corpus-v1.0-{language}/"))
        if not shards or shards != list(range(len(shards))):
            raise SourcePlanError(f"missing/noncontiguous corpus shards for {language}")
    assets.sort(key=lambda asset: asset["path"])
    return {
        "schema_version": 1, "resource": "MIRACL v1.0 English/Hindi/Bengali original publisher files",
        "revisions": dict(REPOSITORIES), "languages": list(LANGUAGES), "assets": assets,
        "total_download_bytes": sum(asset["size"] for asset in assets),
        "corpus_policy": "all original shards form the sampling frame; retain every selected-query judged passage including nonrelevant labels",
        "evaluation_policy": "official dev is locked final evaluation; no testA/testB files are acquired",
        "provenance": ["https://github.com/project-miracl/miracl", "https://huggingface.co/datasets/miracl/miracl", "https://huggingface.co/datasets/miracl/miracl-corpus"],
    }


def fetch_metadata(*, opener: Any = None) -> dict[str, Mapping[str, Any]]:
    request = urllib.request.urlopen if opener is None else opener
    result = {}
    for repository, revision in REPOSITORIES:
        url = f"https://huggingface.co/api/datasets/{repository}/revision/{revision}?blobs=true"
        with request(url, timeout=30) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise SourcePlanError("repository metadata exceeds the bounded response limit")
        result[repository] = json.loads(payload)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.allow_network:
            raise SourcePlanError("planning requires explicit --allow-network; it does not download dataset files")
        if args.output.exists() or args.output.is_symlink():
            raise SourcePlanError("refusing to replace an existing source lock")
        plan = build_plan(fetch_metadata())
        _atomic_write_json(args.output, plan)
    except (ValueError, OSError, TypeError) as exc:
        print(f"plan_miracl: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"lock": str(args.output), "asset_count": len(plan["assets"]), "total_download_bytes": plan["total_download_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
