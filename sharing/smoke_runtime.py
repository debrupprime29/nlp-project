"""Check the genuine shared runtime, including real loopback HTTP requests.

This is an engineering smoke check, not a relevance benchmark. It never trains,
downloads models, modifies runtime files, or claims that a ranking is correct.
"""

from __future__ import annotations

import argparse
import gc
import html
import http.client
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from sharing.run_app import _KnownE5TokenizerWarning, build_arguments, get_runtime_layout
from sharing.runtime_bundle import (
    LANGUAGES, PROJECT_ROOT, BundleError, safe_relative_path, sha256_file, verify_runtime,
)


CASES = {
    "en": (
        ("native", "Which company introduced the 4004 microprocessor, and what was its original purpose?", ()),
        ("protected", "NASA did not introduce the Intel 4004 in 2020.", ("NASA", "not", "2020")),
    ),
    "hi": (
        ("native", "अंतरराष्ट्रीय न्यायालय में न्यायाधीश कितने वर्ष के लिए चुने जाते हैं?", ()),
        ("protected", "NASA ने 2020 में अंतरराष्ट्रीय न्यायालय के न्यायाधीश नहीं चुने थे।", ("NASA", "2020", "नहीं")),
    ),
    "bn": (
        ("native", "একই মৌলের সমস্থানিকগুলোর নিউট্রন সংখ্যায় কী পার্থক্য থাকে?", ()),
        ("protected", "NASA 2020 সালে সমস্থানিক আবিষ্কার করেছে কি না?", ("NASA", "2020", "না")),
    ),
}


class SmokeError(ValueError):
    """A runtime, response or cleanup check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_result(result: dict, application, language: str, query: str, protected: tuple[str, ...]) -> list[str]:
    _require(result.get("fixture_only") is False, "A genuine non-fixture result is required")
    _require(result.get("document_count") == 1500, "Expected the 1,500-passage CPU-pilot collection")
    _require(result.get("query") == query and result["views"]["original"] == query,
             "The exact original query must be preserved")
    _require(result["diagnostics"]["weight_source"] == "provider", "The learned gate must be active")
    views = [query, *result["views"]["native"]]
    if result["views"]["roman"] is not None:
        views.append(result["views"]["roman"])
    for span in protected:
        _require(all(span in view for view in views), f"Protected span changed: {span!r}")
    weights = result["active_weights"]
    rankings = result["branch_results"]
    _require(bool(weights) and all(_finite(value) and value >= 0 for value in weights.values()),
             "Active weights must be finite and nonnegative")
    _require(math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9), "Active weights must sum to one")
    _require(set(weights) == {branch for branch, hits in rankings.items() if hits},
             "Unavailable branches must be masked")
    for branch_hits in rankings.values():
        for rank, hit in enumerate(branch_hits, 1):
            _require(hit["rank"] == rank and _finite(hit["score"]), "Invalid branch rank or score")
            _require(hit["doc_id"] in application._documents, "Branch hit is outside the loaded corpus")
    hits = result["hits"]
    _require(len(hits) == 10, "Expected ten ranked passages")
    _require(len({hit["doc_id"] for hit in hits}) == 10, "Duplicate final passage IDs")
    previous_score = math.inf
    for rank, hit in enumerate(hits, 1):
        _require(hit["rank"] == rank and hit["language"] == language, "Wrong final rank or collection language")
        _require(_finite(hit["score"]) and 0 <= hit["score"] <= previous_score + 1e-12,
                 "Final scores must be finite, nonnegative and descending")
        previous_score = hit["score"]
        document = application._documents[hit["doc_id"]]
        for field in ("title", "text", "source"):
            _require(isinstance(hit[field], str) and bool(hit[field].strip()), f"Missing passage {field}")
            _require(hit[field] == getattr(document, field), f"Passage {field} differs from the loaded corpus")
        contributions = hit["contributions"]
        _require(bool(contributions) and set(contributions) <= set(weights), "Invalid contribution branches")
        _require(all(_finite(value) and value >= 0 for value in contributions.values()),
                 "Contributions must be finite and nonnegative")
        _require(math.isclose(sum(contributions.values()), hit["score"], rel_tol=1e-9, abs_tol=1e-12),
                 "Contributions must add up to the final score")
    return [hit["doc_id"] for hit in hits]


def _request(port: int, target: str) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        body = response.read()
        headers = {name.lower(): value for name, value in response.getheaders()}
        _require(int(headers["content-length"]) == len(body), "HTTP Content-Length differs from response body")
        _require(headers.get("cache-control") == "no-store", "Responses must not be cached")
        return response.status, headers, body
    finally:
        connection.close()


def _check_language(runtime: Path, language: str) -> dict:
    # Import only after main has configured offline operation and CPU thread limits.
    from app import application_from_arguments, build_argument_parser, make_server

    started = time.perf_counter()
    args = build_argument_parser().parse_args(build_arguments(runtime, language))
    application = application_from_arguments(args)
    loaded_seconds = time.perf_counter() - started
    case_reports = []
    for name, query, protected in CASES[language]:
        result = application.search(query)
        ids = _validate_result(result, application, language, query, protected)
        case_reports.append({"name": name, "query": query, "top_doc_ids": ids[:3],
                             "active_branches": list(result["active_weights"]), "status": "passed"})

    server = make_server(application, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    http_requests = 0
    thread_started = False
    try:
        thread.start()
        thread_started = True
        status, headers, body = _request(server.server_port, "/healthz")
        http_requests += 1
        _require(status == 200 and headers["content-type"].startswith("application/json"), "Health endpoint failed")
        _require(json.loads(body) == {"status": "ok", "document_count": 1500, "fixture_only": False},
                 "Health endpoint reported the wrong runtime")
        status, headers, body = _request(server.server_port, "/")
        http_requests += 1
        _require(status == 200 and headers["content-type"].startswith("text/html"), "Home page failed")
        _require(b"Local collection" in body and b"Adaptive gate" in body, "Home page is not the real gated app")
        for index, (_, query, protected) in enumerate(CASES[language]):
            encoded = urlencode({"q": query})
            status, headers, body = _request(server.server_port, "/api/search?" + encoded)
            http_requests += 1
            _require(status == 200 and headers["content-type"].startswith("application/json"), "Search API failed")
            ids = _validate_result(json.loads(body), application, language, query, protected)
            _require(ids[:3] == case_reports[index]["top_doc_ids"], "Direct and HTTP API rankings differ")
            status, headers, body = _request(server.server_port, "/search?" + encoded)
            http_requests += 1
            page = body.decode("utf-8")
            _require(status == 200 and headers["content-type"].startswith("text/html"), "Search HTML failed")
            _require("Ranked passages" in page and '<article class="passage"' in page,
                     "Search HTML does not contain ranked passages")
            _require(html.escape(query, quote=True) in page and html.escape(ids[0], quote=True) in page,
                     "Search HTML omitted the original input or first passage")
        status, _, body = _request(server.server_port, "/api/search?q=")
        http_requests += 1
        _require(status == 400 and bool(json.loads(body).get("error")), "Empty input must return HTTP 400")
        status, _, body = _request(server.server_port, "/api/missing")
        http_requests += 1
        _require(status == 404 and bool(json.loads(body).get("error")), "Missing route must return HTTP 404")
    finally:
        try:
            if thread_started:
                try:
                    server.shutdown()
                finally:
                    thread.join(timeout=10)
        finally:
            server.server_close()
    _require(not thread.is_alive(), "HTTP server did not stop")
    report = {"language": language, "status": "passed", "cases": case_reports,
              "distinct_query_cases": len(case_reports), "search_invocations": 6,
              "http_requests": http_requests, "load_seconds": loaded_seconds,
              "total_seconds": time.perf_counter() - started, "http_server_closed": True,
              "checks": ["genuine_trained_runtime", "1500_passage_collection", "learned_gate",
                         "original_and_protected_text", "masked_normalized_weights", "finite_rankings",
                         "source_passage_identity", "direct_api_html_consistency", "health_200",
                         "invalid_input_400", "missing_route_404", "loopback_server_cleanup"]}
    del application, server, thread, result
    gc.collect()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "artifacts/shared-runtime")
    parser.add_argument("--report", type=Path, help="write a new JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    # Assignment, rather than setdefault, also overrides a caller's unsafe settings.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    logger = logging.getLogger("transformers.tokenization_utils_base")
    warning_filter = _KnownE5TokenizerWarning(get_runtime_layout(args.runtime_dir)["checkpoint"])
    logger.addFilter(warning_filter)
    try:
        if args.report is not None and (args.report.exists() or args.report.is_symlink()):
            raise SmokeError(f"Refusing to overwrite report: {args.report}")
        print("Verifying genuine runtime and source checksums…", file=sys.stderr, flush=True)
        manifest = verify_runtime(args.runtime_dir)
        metadata = manifest["metadata"]
        _require(metadata.get("name") == "cpu-pilot-runtime-v1" and metadata.get("languages") == list(LANGUAGES),
                 "Expected the complete three-language CPU-pilot runtime")
        code = metadata.get("source_code_sha256")
        _require(isinstance(code, dict) and bool(code), "Missing source-code compatibility record")
        for name, expected in code.items():
            safe_relative_path(name)
            _require(sha256_file(PROJECT_ROOT / name) == expected, f"Source code differs from runtime: {name}")
        language_reports = []
        for language in LANGUAGES:
            print(f"Checking {language}: trained model, native/protected queries and real HTTP…", file=sys.stderr, flush=True)
            language_reports.append(_check_language(args.runtime_dir, language))
            gc.collect()
        report = {"schema_version": 1, "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "Engineering integration only; not semantic-ranking accuracy or a new benchmark",
                  "runtime_dir": str(args.runtime_dir.absolute()), "offline": True, "cpu_threads": 1,
                  "languages": language_reports, "distinct_query_cases": 6,
                  "search_invocations": sum(item["search_invocations"] for item in language_reports),
                  "http_requests": sum(item["http_requests"] for item in language_reports),
                  "all_http_servers_closed": True}
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        sys.stdout.write(rendered)
        return 0
    except (BundleError, SmokeError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Runtime smoke check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.removeFilter(warning_filter)


if __name__ == "__main__":
    raise SystemExit(main())
