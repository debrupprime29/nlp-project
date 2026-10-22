"""Dependency-free application, HTTP, input-boundary, and escaping tests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from app import AppError, SearchApplication, demo_application, make_server
from data.prepare_corpus import iter_jsonl
from nlp.language_id import LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator, build_query_views
from retrieval.bm25 import BM25Document, BM25Index
from retrieval.char_tfidf import CharDocument, CharTfidfIndex
from retrieval.fusion import BRANCHES, RetrievalPipeline
from training.train_gate import GateConfig, GateExample, train_gate


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ("en:earthquake", "Earthquake causes", "earthquake causes plate faults", "earthquake causes plate faults", "en", "fixture-en"),
    ("hi:earthquake", "भूकंप", "भूकंप क्यों आते हैं", "bhukamp kyu aate hain", "hi", "fixture-hi"),
    ("bn:earthquake", "ভূমিকম্প", "ভূমিকম্প কেন হয়", "bhumikompo keno hoy", "bn", "fixture-bn"),
    ("en:rain", "Rain", "clouds create rain", "clouds create rain", "en", "fixture-en"),
)


class ParsedPage(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def make_pipeline(records=DOCUMENTS, *, lid=None, transliterator=None):
    lexical = BM25Index([BM25Document(doc_id, title, text, language, source) for doc_id, title, text, _, language, source in records])
    character = CharTfidfIndex([CharDocument(doc_id, roman, language, source, "manual_fixture") for doc_id, _, _, roman, language, source in records])
    return RetrievalPipeline(lexical, character, language_identifier=lid, transliterator=transliterator)


def request_path(query, *, route="/api/search"):
    return route + "?" + urlencode({"q": query})


@contextmanager
def running_server(application):
    # Bind failures are deliberately failures, not silently skipped checks.
    with make_server(application, host="127.0.0.1", port=0) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield server, f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            worker.join(timeout=3)
            if worker.is_alive():
                raise AssertionError("Owned HTTP test server did not shut down")


class AppFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lid = TokenLanguageIdentifier.fit([LabeledTokenExample(**record) for record in iter_jsonl(ROOT / "examples/tiny/lid_train.jsonl")])
        cls.transliterator = LexiconTransliterator.from_jsonl(ROOT / "examples/tiny/xlit_lexicon.jsonl")

    def setUp(self):
        self.pipeline = make_pipeline(lid=self.lid, transliterator=self.transliterator)
        self.application = SearchApplication(self.pipeline)

    def assert_security_headers(self, headers):
        headers = {name.lower(): value for name, value in headers.items()}
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertIn("no-store", headers.get("cache-control", ""))
        self.assertIn("content-security-policy", headers)
        self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])
        self.assertIn("base-uri 'none'", headers["content-security-policy"])
        self.assertNotIn("access-control-allow-origin", headers)


class AppSearchTests(AppFixture):
    def test_actual_retrieval_returns_complete_passages_rankings_and_diagnostics(self):
        result = self.application.search("bhukamp kyu aate hain")
        self.assertEqual(result["query"], "bhukamp kyu aate hain")
        self.assertEqual(result["hits"][0]["doc_id"], "hi:earthquake")
        document = {record[0]: record for record in DOCUMENTS}
        self.assertEqual(len({hit["doc_id"] for hit in result["hits"]}), len(result["hits"]))
        self.assertEqual([hit["rank"] for hit in result["hits"]], list(range(1, len(result["hits"]) + 1)))
        self.assertLessEqual(len(result["hits"]), 10)
        for hit in result["hits"]:
            record = document[hit["doc_id"]]
            self.assertEqual((hit["title"], hit["text"], hit["language"], hit["source"]), (record[1], record[2], record[4], record[5]))
            self.assertTrue(math.isfinite(hit["score"]))
            self.assertAlmostEqual(math.fsum(hit["contributions"].values()), hit["score"])
        self.assertEqual(result["views"]["original"], result["query"])
        self.assertIn("भूकंप क्यों आते हैं", result["views"]["native"])
        self.assertEqual(set(result["branch_results"]), set(BRANCHES))
        self.assertAlmostEqual(math.fsum(result["active_weights"].values()), 1)
        self.assertIn("search_count", result["diagnostics"])

    def test_native_english_roman_and_noisy_queries_use_the_actual_pipeline(self):
        for query, doc_id in (
            ("earthquake causes", "en:earthquake"),
            ("भूकंप क्यों आते हैं", "hi:earthquake"),
            ("ভূমিকম্প কেন হয়", "bn:earthquake"),
            ("bhukamp kyu aate hain", "hi:earthquake"),
            ("bhumikompo keno hoy", "bn:earthquake"),
            ("bhukmp kyu aate hain", "hi:earthquake"),
        ):
            with self.subTest(query=query):
                result = self.application.search(query)
                self.assertEqual(result["hits"][0]["doc_id"], doc_id)
                expected = self.pipeline.search(query)
                self.assertEqual([hit["doc_id"] for hit in result["hits"]], [hit.doc_id for hit in expected.hits])

    def test_fixed_weights_and_learned_gate_are_used_and_identified(self):
        examples = []
        for index, query in enumerate(("earthquake causes", "bhukamp kyu aate hain")):
            result = self.pipeline.search(query)
            examples.append(GateExample(f"q-{index}", f"q-{index}", f"g-{index}", "fusion_train", build_query_views(query, self.lid, self.transliterator), result.branch_results, {"en:earthquake" if index == 0 else "hi:earthquake": 1}))
        gate = train_gate(examples, pipeline_manifest=self.pipeline.manifest, config=GateConfig(epochs=10)).gate
        learned = RetrievalPipeline(self.pipeline.bm25_index, self.pipeline.char_index, language_identifier=self.lid, transliterator=self.transliterator, weight_provider=gate)
        learned_result = SearchApplication(learned).search("earthquake causes")
        self.assertEqual(learned_result["diagnostics"]["weight_source"], "provider")
        fixed_weights = {branch: float(branch == "bm25_original") for branch in BRANCHES}
        fixed_result = SearchApplication(self.pipeline, weights=fixed_weights).search("earthquake causes")
        self.assertEqual(fixed_result["diagnostics"]["weight_source"], "explicit")
        self.assertEqual(fixed_result["active_weights"]["bm25_original"], 1)
        self.assertEqual(fixed_result["active_weights"].get("char_tfidf", 0), 0)

    def test_out_of_vocabulary_query_has_empty_results_and_no_nan(self):
        result = self.application.search("zzzzzzzzzzzzzzzzzz")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["active_weights"], {})
        self.assertEqual(result["diagnostics"]["candidate_union_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_programmatic_search_rejects_empty_invalid_control_and_overlength_input(self):
        for query in ("", "   ", None, [], b"earthquake", "earthquake\x00faults", "earthquake\r\nInjected: header", "x" * 4097):
            with self.subTest(query=query), self.assertRaises((AppError, TypeError)):
                self.application.search(query)


class AppHtmlAndRoutesTests(AppFixture):
    def test_landing_form_is_accessible_buildless_and_does_not_search(self):
        with patch.object(self.pipeline, "search", side_effect=AssertionError("landing must not search")):
            status, headers, body = self.application.handle("/")
        self.assertEqual(status, 200)
        self.assert_security_headers(headers)
        self.assertIn("text/html", headers.get("Content-Type", headers.get("content-type", "")))
        page = ParsedPage(body.decode("utf-8"))
        forms = [attrs for tag, attrs in page.elements if tag == "form"]
        self.assertTrue(any(attrs.get("action") == "/search" and attrs.get("method", "get").lower() == "get" for attrs in forms))
        inputs = [attrs for tag, attrs in page.elements if tag == "input" and attrs.get("name") == "q"]
        self.assertEqual(len(inputs), 1)
        self.assertTrue(any(tag == "label" and attrs.get("for") == inputs[0].get("id") for tag, attrs in page.elements))
        self.assertFalse(any(tag == "script" for tag, _ in page.elements))

    def test_json_endpoint_matches_service_and_html_displays_diagnostics(self):
        query = "bhukamp kyu aate hain"
        status, headers, body = self.application.handle(request_path(query))
        self.assertEqual(status, 200)
        self.assert_security_headers(headers)
        self.assertIn("application/json", headers.get("Content-Type", headers.get("content-type", "")))
        result = json.loads(body)
        self.assertEqual(result["hits"][0]["doc_id"], "hi:earthquake")
        self.assertEqual(result["views"]["original"], query)
        status, headers, body = self.application.handle(request_path(query, route="/search"))
        self.assertEqual(status, 200)
        self.assert_security_headers(headers)
        text = " ".join(ParsedPage(body.decode("utf-8")).text)
        for expected in ("भूकंप क्यों आते हैं", "fixture-hi", "hi", "bm25_original", "bm25_native", "char_tfidf"):
            self.assertIn(expected, text)

    def test_query_title_passage_and_source_are_escaped_without_executable_markup(self):
        title = '<img src=x onerror="alert(1)">'
        text = 'earthquake <script>alert("passage")</script>'
        source = '<svg onload="alert(2)">'
        application = SearchApplication(make_pipeline((("evil", title, text, "earthquake", "en", source),)))
        query = 'earthquake <script>alert("query")</script>'
        status, headers, body = application.handle(request_path(query, route="/search"))
        self.assertEqual(status, 200)
        self.assert_security_headers(headers)
        markup = body.decode("utf-8")
        page = ParsedPage(markup)
        self.assertFalse(any(tag in {"script", "img", "svg"} for tag, _ in page.elements))
        self.assertFalse(any(name.startswith("on") for _, attrs in page.elements for name in attrs))
        self.assertNotIn(title, markup)
        self.assertNotIn(source, markup)
        self.assertNotIn(text, markup)
        self.assertIn("&lt;script&gt;", markup)
        # JSON may contain these exact strings as data, never HTML execution.
        result = json.loads(application.handle(request_path("earthquake"))[2])
        self.assertEqual(result["hits"][0]["title"], title)
        self.assertEqual(result["hits"][0]["source"], source)

    def test_missing_blank_and_out_of_vocabulary_queries_have_explicit_states(self):
        for path in ("/search", "/search?q=", "/?q=%20%20"):
            with self.subTest(path=path):
                status, _, body = self.application.handle(path)
                self.assertEqual(status, 200)
                self.assertIn("query", body.decode("utf-8").lower())
        for path in ("/api/search", "/api/search?q=", "/api/search?q=%20%20"):
            with self.subTest(path=path):
                status, headers, body = self.application.handle(path)
                self.assertEqual(status, 400)
                self.assert_security_headers(headers)
                self.assertIn("error", json.loads(body))
        status, _, body = self.application.handle(request_path("zzzzzzzzzzzzzzzzzz", route="/search"))
        self.assertEqual(status, 200)
        self.assertIn("no", " ".join(ParsedPage(body.decode("utf-8")).text).lower())

    def test_duplicate_unknown_malformed_and_oversized_parameters_fail_safely(self):
        bad_paths = (
            "/api/search?q=earthquake&q=rain", "/api/search?q=earthquake&other=rain", "/api/search?q=%ZZ",
            "/api/search?q=%ff", "/api/search?q=%00", "/api/search?q=hello%0d%0aInjected%3Ayes",
            request_path("x" * 4097), request_path("a " * 257),
        )
        for path in bad_paths:
            with self.subTest(path=path[:120]):
                status, headers, body = self.application.handle(path)
                self.assertIn(status, (400, 414))
                self.assert_security_headers(headers)
                self.assertIn("error", json.loads(body))
                self.assertNotIn("Traceback", body.decode("utf-8"))

    def test_unknown_routes_and_traversal_never_read_local_files(self):
        for path in ("/PROJECT_SPEC.md", "/../PROJECT_SPEC.md", "/%2e%2e/PROJECT_SPEC.md", "/etc/passwd", "/favicon.ico", "//evil.example/search?q=x"):
            with self.subTest(path=path):
                status, headers, body = self.application.handle(path)
                self.assertIn(status, (400, 404))
                self.assert_security_headers(headers)
                self.assertNotIn(b"root:x:", body)
                self.assertNotIn(b"The course project", body)

    def test_health_endpoint_contains_no_query_or_local_artifact_paths(self):
        status, headers, body = self.application.handle("/healthz")
        self.assertEqual(status, 200)
        self.assert_security_headers(headers)
        payload = json.loads(body)
        self.assertTrue(payload)
        self.assertNotIn(str(ROOT), body.decode("utf-8"))


class AppHttpTests(AppFixture):
    def fetch(self, url, *, method="GET", headers=None):
        opener = build_opener(ProxyHandler({}))
        request = Request(url, method=method, headers=headers or {})
        try:
            response = opener.open(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, dict(response.headers), response.read()

    def test_real_loopback_get_head_and_health_requests(self):
        with running_server(self.application) as (_, base_url):
            status, headers, body = self.fetch(base_url + request_path("earthquake causes"))
            self.assertEqual(status, 200)
            self.assert_security_headers(headers)
            self.assertEqual(json.loads(body)["hits"][0]["doc_id"], "en:earthquake")
            status, headers, body = self.fetch(base_url + "/", method="HEAD")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"")
            self.assertGreater(int(headers["Content-Length"]), 0)
            self.assertEqual(self.fetch(base_url + "/healthz")[0], 200)

    def test_real_loopback_rejects_methods_and_foreign_host(self):
        with running_server(self.application) as (_, base_url):
            for method in ("POST", "PUT", "DELETE", "OPTIONS"):
                with self.subTest(method=method):
                    status, headers, _ = self.fetch(base_url + "/api/search?q=earthquake", method=method)
                    self.assertEqual(status, 405)
                    self.assertIn("GET", headers.get("Allow", ""))
                    self.assert_security_headers(headers)
            status, headers, _ = self.fetch(base_url + "/healthz", headers={"Host": "evil.example"})
            self.assertIn(status, (400, 403))
            self.assert_security_headers(headers)

    def test_nonloopback_binding_is_not_allowed_by_default(self):
        for host in ("0.0.0.0", "example.com", "192.168.1.1"):
            with self.subTest(host=host), self.assertRaises((AppError, ValueError)):
                make_server(self.application, host=host, port=0)


class AppCliTests(unittest.TestCase):
    def run_cli(self, arguments):
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.run([sys.executable, "-m", "app", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=10, env=environment)

    def test_offline_demo_uses_explicit_fixtures_and_never_opens_network(self):
        with patch("socket.socket.connect", side_effect=AssertionError("offline demo must not access network")):
            application = demo_application()
            result = application.search("bhukamp kyu aate hain")
        self.assertTrue(result["fixture_only"])
        self.assertTrue(result["hits"])
        status, _, body = application.handle("/")
        self.assertEqual(status, 200)
        self.assertIn("fixture", body.decode("utf-8").lower())

    def test_cli_demo_query_outputs_machine_readable_results_without_starting_server(self):
        completed = self.run_cli(["--demo", "--query", "bhukamp kyu aate hain"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["fixture_only"])
        self.assertTrue(result["hits"])
        self.assertEqual(result["query"], "bhukamp kyu aate hain")
        self.assertIn("text", result["hits"][0])

    def test_cli_missing_or_incompatible_configuration_is_clear_and_nonmutating(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-corpus.jsonl"
            for arguments in ([], ["--demo", "--corpus", str(path), "--query", "earthquake"], ["--corpus", str(path), "--query", "earthquake"], ["--demo", "--query", ""]):
                with self.subTest(arguments=arguments):
                    completed = self.run_cli(arguments)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertNotIn("Traceback", completed.stderr)
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
