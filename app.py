"""Local-only browser and JSON demo for the multilingual retrieval pipeline.

Run ``python -m app --demo`` for an explicitly synthetic, offline demonstration.
Use ``--corpus`` and persisted resources/indexes for a real collection. This is
a loopback research tool, not a production web service or generated-answer app.
"""

from __future__ import annotations

import argparse
import html
import ipaddress
import json
import re
import socket
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urlsplit

from data.prepare_corpus import CorpusPreparationError
from nlp.language_id import AUXILIARY_TRAIN, LanguageIdError, LabeledTokenExample, TokenLanguageIdentifier
from nlp.query_views import LexiconTransliterator, MAX_QUERY_CHARACTERS, QueryViewError
from nlp.text import tokenize_with_spans
from retrieval.bm25 import BM25Document, BM25Error, BM25Index, load_corpus as load_bm25_corpus
from retrieval.char_tfidf import CharDocument, CharTfidfError, CharTfidfIndex, load_corpus as load_char_corpus
from retrieval.dense import DenseConfig, DenseDocument, DenseError, DenseIndex, FixtureHashEncoder, _add_encoder_arguments, _encoder_from_arguments
from retrieval.fusion import BRANCHES, FusionConfig, FusionError, RetrievalPipeline, _parse_weights, _weights
from training.train_gate import GateError, LinearSoftmaxGate


SCHEMA_VERSION = 1
MAX_REQUEST_TARGET_BYTES = 65536
MAX_QUERY_TOKENS = 256
BRANCH_LABELS = {
    "bm25_original": "Original · BM25", "bm25_native": "Native views · BM25",
    "char_tfidf": "Roman view · Character TF-IDF", "dense": "Semantic · Dense",
}
_EXAMPLES = ("earthquake causes", "bhukamp kyu aate hain", "ভূমিকম্প কেন হয়", "bhumikompo keno hoy")
_SECURITY_HEADERS = {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class AppError(ValueError):
    """Invalid local demo configuration or search input."""


def _query(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise AppError("Enter a nonempty search query.")
    if len(text) > MAX_QUERY_CHARACTERS:
        raise AppError(f"Keep the query within {MAX_QUERY_CHARACTERS} characters.")
    if any(ord(character) < 32 or ord(character) == 127 or 0xD800 <= ord(character) <= 0xDFFF for character in text):
        raise AppError("Queries cannot contain control characters or invalid Unicode.")
    if len(tokenize_with_spans(text)) > MAX_QUERY_TOKENS:
        raise AppError(f"Keep the query within {MAX_QUERY_TOKENS} tokens.")
    return text


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


class SearchApplication:
    """Serve passage results without reading or accepting relevance labels."""

    def __init__(self, pipeline: RetrievalPipeline, *, weights: Mapping[str, float] | None = None) -> None:
        if not isinstance(pipeline, RetrievalPipeline):
            raise AppError("SearchApplication requires a RetrievalPipeline.")
        if weights is not None:
            try:
                _weights(weights, BRANCHES)
            except FusionError as exc:
                raise AppError(str(exc)) from exc
            if pipeline.weight_provider is not None:
                raise AppError("Choose either fixed weights or an adaptive gate, not both.")
        documents = getattr(pipeline.bm25_index, "_documents", None)
        if documents is None:
            raise AppError("The lexical index must retain passage text and metadata.")
        self._documents = {document.doc_id: document for document in documents}
        if len(self._documents) != len(documents):
            raise AppError("Passage document IDs must be unique.")
        self.pipeline = pipeline
        self.weights = dict(weights) if weights is not None else None

    @property
    def fixture_only(self) -> bool:
        return bool(self.pipeline.manifest["fixture_only"])

    def search(self, text: str) -> dict[str, Any]:
        text = _query(text)
        result = self.pipeline.search(text, self.weights)
        hits = []
        for hit in result.hits[:10]:
            document = self._documents.get(hit.doc_id)
            if document is None:
                raise AppError("The retriever returned a passage outside the loaded corpus.")
            hits.append({**asdict(hit), "title": document.title, "text": document.text, "language": document.language, "source": document.source})
        return {
            "schema_version": SCHEMA_VERSION, "query": text, "hits": hits,
            "views": dict(result.diagnostics["views"]), "active_weights": dict(result.active_weights),
            "branch_results": {branch: [asdict(hit) for hit in results] for branch, results in result.branch_results.items()},
            "diagnostics": dict(result.diagnostics), "fixture_only": self.fixture_only,
            "document_count": len(self._documents), "pipeline_fingerprint": self.pipeline.manifest["pipeline_fingerprint"],
            "notice": "Synthetic fixture: demonstrates software behavior, not retrieval quality." if self.fixture_only else "Passage retrieval over the loaded collection; results are not generated answers.",
        }

    def _response(self, status: int, body: bytes, content_type: str) -> tuple[int, dict[str, str], bytes]:
        return status, {**_SECURITY_HEADERS, "Content-Type": content_type, "Content-Length": str(len(body))}, body

    def handle(self, path: str) -> tuple[int, dict[str, str], bytes]:
        """Handle a GET request target, independent of sockets for testing."""
        api = isinstance(path, str) and path.startswith(("/api/", "/healthz"))
        query = ""
        try:
            if not isinstance(path, str) or len(path.encode("utf-8")) > MAX_REQUEST_TARGET_BYTES:
                raise AppError("The request URL is too long.")
            if any(ord(character) < 32 or ord(character) == 127 for character in path) or re.search(r"%(?![0-9a-fA-F]{2})", path):
                raise AppError("The request URL is malformed.")
            parts = urlsplit(path)
            if parts.scheme or parts.netloc or parts.fragment or not parts.path.startswith("/"):
                raise AppError("Use a local relative request URL.")
            if parts.path not in ("/", "/search", "/api/search", "/healthz"):
                body = _json({"error": "Not found."}) if api else self._render(error="This page does not exist.")
                return self._response(404, body, "application/json; charset=utf-8" if api else "text/html; charset=utf-8")
            parameters = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True, encoding="utf-8", errors="strict", max_num_fields=1)
            if set(parameters) - {"q"} or any(len(values) != 1 for values in parameters.values()):
                raise AppError("Supply only one q search parameter.")
            if parts.path == "/healthz":
                if parameters:
                    raise AppError("The health endpoint does not accept query parameters.")
                return self._response(200, _json({"status": "ok", "document_count": len(self._documents), "fixture_only": self.fixture_only}), "application/json; charset=utf-8")
            query = parameters.get("q", [""])[0]
            if parts.path == "/api/search":
                return self._response(200, _json(self.search(query)), "application/json; charset=utf-8")
            result = self.search(query) if query.strip() else None
            if not query.strip() and (len(query) > MAX_QUERY_CHARACTERS or any(ord(character) < 32 for character in query)):
                _query(query)
            return self._response(200, self._render(query=query, result=result), "text/html; charset=utf-8")
        except (AppError, ValueError, UnicodeError) as exc:
            message = str(exc) if isinstance(exc, AppError) else "The request or query is invalid. Check its encoding and parameters."
            body = _json({"error": message}) if api else self._render(query=query[:MAX_QUERY_CHARACTERS], error=message)
            return self._response(400, body, "application/json; charset=utf-8" if api else "text/html; charset=utf-8")
        except Exception:
            # Do not disclose checkpoint paths, tracebacks, or internal models
            # through a browser response. Command-line loading still reports
            # specific configuration failures before the server starts.
            message = "Search could not be completed with the loaded resources."
            body = _json({"error": message}) if api else self._render(query=query[:MAX_QUERY_CHARACTERS], error=message)
            return self._response(500, body, "application/json; charset=utf-8" if api else "text/html; charset=utf-8")

    def _render(self, *, query: str = "", result: Mapping[str, Any] | None = None, error: str | None = None) -> bytes:
        fixture = self.fixture_only
        badge = "Synthetic fixture · Offline demo" if fixture else "Local collection · Research demo"
        examples = "".join(f'<a href="/?{_escape(urlencode({"q": example}))}">{_escape(example)}</a>' for example in _EXAMPLES)
        mode = "Adaptive gate" if self.pipeline.weight_provider is not None else "Fixed weights" if self.weights is not None else "Equal-weight fusion"
        if error:
            content = f'<section class="empty error" role="alert"><span class="eyebrow">REQUEST NEEDS ATTENTION</span><h2>Let’s try that again.</h2><p>{_escape(error)}</p></section>'
        elif result is None:
            content = '<section class="empty"><span class="empty-icon" aria-hidden="true">⌕</span><span class="eyebrow">READY WHEN YOU ARE</span><h2>Your next search can cross scripts.</h2><p>Ask in English, Hindi, Bengali, or Romanized text. Explore ranked passages and see how each retrieval branch contributed.</p><div class="empty-note">Original wording is preserved. Search returns passages, not generated answers.</div></section>'
        elif not result["hits"]:
            content = '<section class="empty" role="status"><span class="eyebrow">NO MATCHES</span><h2>No passages matched this query.</h2><p>Try a shorter phrase, another script, or a query closer to the loaded collection.</p></section>'
        else:
            content = '<div class="results-heading" role="status"><h2>Ranked passages</h2><span>' + str(len(result["hits"])) + ' results · ' + f'{result["diagnostics"]["total_seconds"] * 1000:.1f}' + ' ms</span></div>'
            for hit in result["hits"]:
                parts = "".join(f'<span>{_escape(BRANCH_LABELS[branch])} <b>{value:.6f}</b></span>' for branch, value in hit["contributions"].items())
                title = hit["title"] or hit["doc_id"]
                content += f'<article class="passage" lang="{_escape(hit["language"])}"><div class="passage-top"><span class="rank">{hit["rank"]:02d}</span><span class="language">{_escape(hit["language"].upper())}</span><span class="source">{_escape(hit["source"])}</span><span class="score">RRF score {hit["score"]:.6f}</span></div><h3>{_escape(title)}</h3><p class="passage-text">{_escape(hit["text"])}</p><details><summary>Branch contributions · {_escape(hit["doc_id"])}</summary><div class="contributions">{parts}</div></details></article>'
        if result is not None:
            view_data = result["views"]
            native = view_data["native"] or ["No distinct native rewrite"]
            view_rows = (("Original", [view_data["original"]]), ("Native", native), ("Roman", [view_data["roman"] or "Unavailable"]))
            views = "".join(f'<div class="view-row"><dt>{label}</dt><dd>{"<br>".join(_escape(value) for value in values)}</dd></div>' for label, values in view_rows)
            weights = ""
            for branch in BRANCHES:
                active = branch in result["active_weights"]
                weight = result["active_weights"].get(branch, 0.0)
                weights += f'<div class="weight"><div><span>{_escape(BRANCH_LABELS[branch])}</span><b>{weight:.1%}' + ("" if active else " · inactive") + f'</b></div><meter min="0" max="1" value="{weight:.12f}">{weight:.1%}</meter></div>'
            debug = f'<details class="trace"><summary>Full diagnostic trace</summary><pre>{_escape(json.dumps({"branches": result["branch_results"], "diagnostics": result["diagnostics"]}, ensure_ascii=False, indent=2, allow_nan=False))}</pre></details>'
        else:
            views = '<p class="muted">Query views appear after your first search.</p>'
            weights = '<p class="muted">Active branches are masked per query; their weights always sum to one.</p>'
            debug = ""
        notice = "Fixture encoders and resources demonstrate software behavior, not multilingual semantic quality. Their rankings are not benchmark results." if fixture else "This local tool searches the supplied collection. Relevance labels are not used during search."
        document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Local multilingual passage retrieval with transparent query views and branch contributions."><title>Multiview · Passage search</title><style>{_CSS}</style></head>
<body><a class="skip" href="#results">Skip to results</a><div class="shell"><header><a class="brand" href="/"><span class="brand-mark" aria-hidden="true">mv</span><span>Multiview<span class="brand-sub">PASSAGE SEARCH</span></span></a><span class="badge">{_escape(badge)}</span></header>
<main><section class="search-panel" aria-labelledby="search-title"><div class="eyebrow">ENGLISH / हिंदी / বাংলা</div><h1 id="search-title">Find the passage. Keep the meaning.</h1><p class="intro">One query, complementary retrieval views, a transparent ranking.</p><form action="/search" method="get" role="search"><label for="query">Search your collection</label><div class="search-box"><span aria-hidden="true">⌕</span><input id="query" name="q" type="search" value="{_escape(query)}" maxlength="{MAX_QUERY_CHARACTERS}" placeholder="Try a question in any supported script…" autocomplete="off" aria-describedby="query-help"><button type="submit">Search <span aria-hidden="true">↗</span></button></div><p id="query-help" class="query-help">Original text stays intact. Up to 10 ranked passages.</p></form><div class="examples"><span>Try it</span>{examples}</div></section>
<div class="workspace"><section id="results" class="results" aria-label="Search results">{content}</section><aside aria-label="Retrieval details"><section class="side-card"><div class="side-heading"><h2>Inside this search</h2><span class="dot" aria-hidden="true"></span></div><dl class="collection"><div><dt>Collection</dt><dd>{len(self._documents):,} passages</dd></div><div><dt>Fusion</dt><dd>{_escape(mode)}</dd></div></dl><h3>Query views</h3><dl class="views">{views}</dl><h3>Branch weights</h3>{weights}</section>{debug}<p class="research-note">{_escape(notice)}</p></aside></div></main><footer><span>Built for robust multilingual retrieval.</span><span>Local research demo · No generated answers</span></footer></div></body></html>'''
        return document.encode("utf-8")


_CSS = """
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17342f;background:#f5f7f3;font-synthesis:none;line-height:1.5}*{box-sizing:border-box}body{margin:0}a{color:#23584e}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #c58142;outline-offset:4px}.shell{max-width:1280px;margin:auto;padding:0 42px}header{display:flex;justify-content:space-between;align-items:center;padding:29px 0;border-bottom:1px solid #dce4db}.brand{display:flex;align-items:center;gap:12px;text-decoration:none;font-size:21px;font-weight:720;letter-spacing:-.6px}.brand-mark{background:#204e42;color:#eff6e7;border-radius:13px;width:43px;height:43px;display:grid;place-items:center;font-size:20px;letter-spacing:-2px;padding-right:2px}.brand-sub{display:block;color:#6a7a6c;font-size:8px;letter-spacing:2.7px;font-weight:650}.badge{font-size:11px;letter-spacing:.2px;color:#59653c;background:#edf0dc;border:1px solid #dde4c3;border-radius:20px;padding:7px 12px}.search-panel{padding:48px 0 31px}.eyebrow{font-size:10px;font-weight:750;letter-spacing:1.9px;color:#687b5e}h1{font-weight:590;font-size:clamp(26px,3.1vw,39px);letter-spacing:-1.6px;line-height:1.2;margin:14px 0 10px}.intro{color:#718176;font-size:14px;margin:0 0 30px}form label{display:block;font-size:11px;font-weight:650;margin-bottom:9px}.search-box{display:flex;align-items:center;border:1px solid #bfcdbc;border-radius:12px;background:#fff;box-shadow:0 5px 16px #233d2510;padding:6px;max-width:100%;gap:10px}.search-box>span{padding-left:13px;font-size:25px;color:#789078}.search-box input{min-width:0;flex:1;padding:13px 5px;font:inherit;font-size:15px;border:0;background:transparent;color:#18382e}.search-box input:focus{outline:0}.search-box:focus-within{outline:2px solid #9eb29b;outline-offset:2px}button{font:inherit;background:#234f42;border:0;color:#fff;padding:13px 23px;border-radius:8px;font-size:13px;font-weight:620;cursor:pointer}button:hover{background:#173c30}button span{margin-left:15px;font-size:17px}.query-help{font-size:10px;color:#7b897b;margin:8px 1px 16px}.examples{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:10px}.examples>span{color:#879180;margin-right:4px}.examples a{text-decoration:none;border:1px solid #dce5d9;border-radius:6px;background:#fafcf8;padding:5px 9px;color:#5b7159}.examples a:hover{background:#e9efe4}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 315px;gap:27px;border-top:1px solid #dfe6dc;padding:28px 0 49px}.results{min-width:0}.empty{min-height:349px;background:#fcfdfb;border:1px dashed #d0ddcc;border-radius:13px;padding:47px 37px;display:flex;align-items:center;flex-direction:column;text-align:center;justify-content:center}.empty-icon{font-size:34px;color:#6a8864;margin-bottom:14px}.empty h2{font-size:20px;font-weight:550;letter-spacing:-.5px;max-width:410px;margin:12px 0 8px}.empty p{max-width:420px;font-size:13px;color:#7a8978;line-height:1.8;margin:0}.empty-note{font-size:10px;color:#95a08d;border-top:1px solid #e8ede4;margin-top:26px;padding-top:15px}.error{border-color:#e4bd9d;background:#fffaf5}.error .eyebrow{color:#a16a37}.results-heading{display:flex;align-items:center;justify-content:space-between;margin:1px 1px 18px;gap:10px}.results-heading h2{font-size:14px;font-weight:650;margin:0}.results-heading>span{font-size:10px;color:#7b8977}.passage{padding:21px 23px;background:white;border:1px solid #dce5d9;border-radius:11px;margin-bottom:14px;box-shadow:0 2px 4px #26492803}.passage-top{display:flex;gap:9px;align-items:center;font-size:10px;min-width:0}.rank{font-size:11px;color:#8a9b83;font-weight:550;margin-right:3px}.language{color:#4f6849;background:#eff3e9;border-radius:4px;padding:2px 5px;letter-spacing:.4px;font-size:8px}.source{color:#94a08d;overflow-wrap:anywhere}.score{margin-left:auto;color:#8b9886;font-size:9px;white-space:nowrap}.passage h3{font-size:17px;font-weight:600;line-height:1.5;margin:14px 0 8px;overflow-wrap:anywhere}.passage-text{font-size:13px;color:#5d715d;line-height:1.85;white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 16px}details{border-top:1px solid #edf0e8;padding-top:11px}summary{font-size:10px;color:#70846a;cursor:pointer;overflow-wrap:anywhere}.contributions{display:grid;gap:5px;font-size:10px;color:#74846d;padding-top:13px}.contributions span{display:flex;justify-content:space-between;gap:8px}.contributions b{font-weight:500}.side-card{background:#edf2e7;border:1px solid #dce6d3;border-radius:12px;padding:21px}.side-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}.side-heading h2{font-size:13px;margin:0;font-weight:650}.dot{width:6px;height:6px;background:#6a8b5d;border-radius:50%;box-shadow:0 0 0 4px #dfe9d7}.collection{display:grid;gap:7px;padding-bottom:18px;margin:0;border-bottom:1px solid #d9e2d0;font-size:10px}.collection>div{display:flex;justify-content:space-between;gap:8px}.collection dt{color:#7b8a73}.collection dd{margin:0;color:#4b6345;text-align:right}.side-card h3{font-size:10px;font-weight:700;letter-spacing:.5px;margin:21px 0 12px}.views{margin:0}.view-row{margin-bottom:13px}.view-row dt{font-size:8px;text-transform:uppercase;letter-spacing:1px;color:#8a9781;margin-bottom:4px}.view-row dd{font-size:11px;color:#536d4a;margin:0;overflow-wrap:anywhere}.muted{color:#8a977f;font-size:11px;line-height:1.8}.weight{margin:13px 0}.weight>div{display:flex;align-items:center;justify-content:space-between;font-size:9px;gap:8px}.weight b{font-size:8px;color:#7d8e71;font-weight:500}.weight meter{width:100%;height:5px;display:block;margin-top:7px;border:0;background:#dbe5d2;border-radius:5px}.weight meter::-webkit-meter-bar{background:#dbe5d2;border:0;border-radius:5px}.weight meter::-webkit-meter-optimum-value{background:#709365;border-radius:5px}.weight meter::-moz-meter-bar{background:#709365}.trace{padding:15px 3px;border:0}.trace pre{font-size:10px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;padding:13px;background:#eaf0e4;border-radius:7px;color:#5c7355}.research-note{font-size:10px;color:#8c9884;line-height:1.8;margin:13px 5px}footer{border-top:1px solid #dfe6dc;padding:22px 0 28px;display:flex;justify-content:space-between;gap:15px;font-size:9px;color:#96a08e}.skip{position:absolute;left:20px;top:-60px;background:white;padding:10px;z-index:1}.skip:focus{top:10px}@media(max-width:800px){.shell{padding:0 22px}.workspace{grid-template-columns:minmax(0,1fr);gap:21px}aside{display:grid;gap:10px}.search-panel{padding-top:32px}.side-card{padding:20px}.empty{min-height:290px;padding:35px 20px}.badge{font-size:9px;padding:6px 9px}.brand{font-size:18px}.brand-mark{width:37px;height:37px}.search-box input{font-size:13px}button{padding:12px 16px}button span{margin-left:5px}footer{flex-wrap:wrap}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
/* Metadata stays readable on compact screens; smaller labels are decorative. */
.intro,.passage-text,.empty p{font-size:14px}.query-help,.examples,.badge,.collection,.view-row dd,.weight>div,.muted,.research-note,.contributions,summary,.passage-top,.score,footer{font-size:12px}.weight b,.language,.view-row dt{font-size:11px}.side-heading h2,.results-heading h2{font-size:16px}.side-card h3,form label{font-size:13px}.trace pre{font-size:12px}.brand-sub{font-size:9px}.query-help,.research-note,.source,.score,.view-row dt,footer{color:#62715b}.collection dt,.muted{color:#68795f}.weight>div{align-items:flex-start}.weight b{text-align:right}.empty-note{font-size:12px;color:#68795f}
"""


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_server(application: SearchApplication, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Create a context-managed, loopback-only HTTP server; port zero is valid."""
    if not _loopback(host) or isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise AppError("Bind only to a loopback host and a port in [0, 65535].")

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalResearchDemo"
        sys_version = ""

        def _send(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if getattr(self, "command", None) != "HEAD":
                self.wfile.write(body)

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            # Parser-level failures and unrecognized HTTP verbs need the same
            # safe response contract as application-level request errors.
            status = 405 if code == 501 else code
            try:
                reason = HTTPStatus(status).phrase
            except ValueError:
                reason = "Invalid HTTP request"
            status, headers, body = application._response(status, _json({"error": reason}), "application/json; charset=utf-8")
            if status == 405:
                headers["Allow"] = "GET, HEAD"
            self.close_connection = True
            self._send(status, headers, body)

        def _host_valid(self) -> bool:
            values = self.headers.get_all("Host", [])
            if len(values) != 1:
                return False
            try:
                authority = urlsplit("//" + values[0])
                return not authority.username and not authority.password and not authority.path and not authority.query and not authority.fragment and authority.hostname is not None and _loopback(authority.hostname) and (authority.port or 80) == self.server.server_port
            except ValueError:
                return False

        def do_GET(self) -> None:
            if not self._host_valid():
                self._send(*application._response(400, _json({"error": "A matching loopback Host header is required."}), "application/json; charset=utf-8"))
                return
            self._send(*application.handle(self.path))

        do_HEAD = do_GET

        def _method_not_allowed(self) -> None:
            status, headers, body = application._response(405, _json({"error": "Only GET and HEAD are supported."}), "application/json; charset=utf-8")
            self._send(status, {**headers, "Allow": "GET, HEAD"}, body)
            self.close_connection = True

        do_POST = _method_not_allowed
        do_PUT = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_OPTIONS = _method_not_allowed
        do_TRACE = _method_not_allowed
        do_CONNECT = _method_not_allowed

        def log_message(self, format: str, *args: Any) -> None:
            # Search text and local paths must not become incidental access logs.
            return

    class LocalHTTPServer(HTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        allow_reuse_address = True

        def get_request(self) -> tuple[socket.socket, Any]:
            connection, address = super().get_request()
            connection.settimeout(10.0)
            return connection, address

    return LocalHTTPServer((host, port), Handler)


def demo_application(*, weights: Mapping[str, float] | None = None, config: FusionConfig = FusionConfig()) -> SearchApplication:
    """Inline, wheel-portable fixtures; no files, network, or pretrained models."""
    records = (
        ("demo:en:earthquake", "Why earthquakes happen", "An earthquake occurs when rocks suddenly slip along a fault in the Earth's crust. The released energy travels as seismic waves.", "en", "an earthquake occurs when rocks suddenly slip along a fault earth crust energy seismic waves"),
        ("demo:hi:earthquake", "भूकंप क्यों आते हैं", "भूकंप पृथ्वी की सतह के नीचे चट्टानों के अचानक खिसकने से आते हैं। इससे निकली ऊर्जा भूकंपीय तरंगों के रूप में फैलती है।", "hi", "bhukamp kyu aate hain prithvi ki satah ke niche chattano ke achanak khisakne se aate hain urja tarange"),
        ("demo:bn:earthquake", "ভূমিকম্প কেন হয়", "পৃথিবীর ভূত্বকের নিচে শিলা হঠাৎ সরে গেলে ভূমিকম্প হয়। এর ফলে শক্তি তরঙ্গের আকারে চারদিকে ছড়িয়ে পড়ে।", "bn", "bhumikompo keno hoy prithibir bhutwaker niche shila sore gele shokti toronger akare choriye pore"),
        ("demo:en:rain", "How rain forms", "Rain forms when water vapor in the atmosphere cools and condenses into droplets in clouds. Larger droplets fall toward the ground.", "en", "rain forms water vapor atmosphere cools condenses droplets clouds ground"),
        ("demo:hi:rain", "बारिश कैसे होती है", "वायुमंडल में जलवाष्प ठंडी होकर पानी की बूंदों में बदलती है। बादलों में बूंदें बड़ी होने पर बारिश के रूप में गिरती हैं।", "hi", "barish kaise hoti hai vayumandal jalvashp thandi pani bundo badalo barish"),
        ("demo:bn:rain", "বৃষ্টি কীভাবে হয়", "বাতাসের জলীয় বাষ্প ঠান্ডা হয়ে মেঘের মধ্যে ছোট ছোট জলবিন্দু তৈরি করে। জলবিন্দুগুলি বড় হয়ে বৃষ্টি হিসেবে মাটিতে পড়ে।", "bn", "brishti kibhabe hoy batasher joliyo bashpo thanda megh jolbindu brishti matite pore"),
    )
    source = "handwritten-synthetic-demo-v1"
    lexical = BM25Index([BM25Document(doc_id, title, text, language, source) for doc_id, title, text, language, _ in records])
    character = CharTfidfIndex([CharDocument(doc_id, roman, language, source, "manual_fixture") for doc_id, _, _, language, roman in records])
    dense = DenseIndex([DenseDocument(doc_id, title, text, language, source) for doc_id, title, text, language, _ in records], FixtureHashEncoder(64, allow_fixture=True))
    labeled = {"en": ("earthquake", "causes", "rain", "clouds"), "hi": ("bhukamp", "kyu", "aate", "hain"), "bn": ("bhumikompo", "keno", "hoy", "brishti"), "other": ("2026", "99", "пример", "مرحبا")}
    identifier = TokenLanguageIdentifier.fit([
        LabeledTokenExample(token, 0, len(token), language, AUXILIARY_TRAIN, f"demo-{language}-{token}", source)
        for language, tokens in labeled.items() for token in tokens
    ])
    pairs = (("hi", "भूकंप", "bhukamp"), ("hi", "क्यों", "kyu"), ("hi", "आते", "aate"), ("hi", "हैं", "hain"), ("bn", "ভূমিকম্প", "bhumikompo"), ("bn", "কেন", "keno"), ("bn", "হয়", "hoy"))
    transliterator = LexiconTransliterator([{"language": language, "native": native, "romanized": roman, "source": source, "revision": "v1", "count": 1} for language, native, roman in pairs])
    return SearchApplication(RetrievalPipeline(lexical, character, dense, identifier, transliterator, config), weights=weights)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="prepared passage JSONL")
    source.add_argument("--demo", action="store_true", help="explicit inline synthetic fixture; no downloads")
    parser.add_argument("--query", help="emit one JSON search response and exit without starting a server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lid-model", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--dense-index", type=Path)
    weighting = parser.add_mutually_exclusive_group()
    weighting.add_argument("--gate-model", type=Path)
    weighting.add_argument("--weights", help="JSON object or file with all four fixed branch weights")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-branch-top-k", type=int, default=100)
    parser.add_argument("--per-view-top-k", type=int, default=100)
    parser.add_argument("--k-view", type=float, default=60.0)
    parser.add_argument("--k", type=float, default=60.0)
    _add_encoder_arguments(parser)
    return parser


def application_from_arguments(args: argparse.Namespace) -> SearchApplication:
    if not 1 <= args.top_k <= 10:
        raise AppError("The demo displays between 1 and 10 passages per query.")
    config = FusionConfig(args.per_branch_top_k, args.per_view_top_k, args.top_k, args.k_view, args.k)
    config.validate()
    weights = _parse_weights(args.weights)
    if args.demo:
        if any((args.lid_model, args.lexicon, args.dense_index, args.gate_model, args.checkpoint, args.allow_download)):
            raise AppError("--demo uses only its inline synthetic resources; use --corpus for saved models/indexes.")
        return demo_application(weights=weights, config=config)
    dense = None
    if args.dense_index:
        dense = DenseIndex.load(args.dense_index, _encoder_from_arguments(args), expected_config=DenseConfig(max_length=args.max_length, batch_size=args.batch_size))
    pipeline = RetrievalPipeline(
        BM25Index(load_bm25_corpus(args.corpus)), CharTfidfIndex(load_char_corpus(args.corpus)), dense,
        TokenLanguageIdentifier.load_json(args.lid_model) if args.lid_model else None,
        LexiconTransliterator.from_jsonl(args.lexicon) if args.lexicon else None, config,
    )
    if args.gate_model:
        pipeline.weight_provider = LinearSoftmaxGate.load_json(args.gate_model, expected_pipeline_manifest=pipeline.manifest)
    return SearchApplication(pipeline, weights=weights)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if not _loopback(args.host) or not 0 <= args.port <= 65535:
            raise AppError("Use a loopback host and a port in [0, 65535].")
        application = application_from_arguments(args)
        if args.query is not None:
            sys.stdout.buffer.write(_json(application.search(args.query)))
            return 0
        with make_server(application, args.host, args.port) as server:
            address = f"[{args.host}]" if ":" in args.host else args.host
            print(f"Local passage search: http://{address}:{server.server_port}/", flush=True)
            if application.fixture_only:
                print("Synthetic fixture only: not a pretrained semantic retrieval benchmark.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (AppError, FusionError, GateError, BM25Error, CharTfidfError, DenseError, QueryViewError, LanguageIdError, CorpusPreparationError, OSError, TypeError, ValueError) as exc:
        print(f"app: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
