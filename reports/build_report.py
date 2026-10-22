#!/usr/bin/env python3
"""Build and validate a strict two-page report plus one references-only page.

Run from the checkout with:
    PYTHONPATH="$PWD/.venv/report-deps" python3 reports/build_report.py

Only reports/ outputs are written. Frozen experiment artifacts are read-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
PDF = OUT / "uncertainty_aware_retrieval_report.pdf"
STUDY = ROOT / "artifacts/study/cpu-pilot"
CHECKS = ROOT / "artifacts/checks"
WIDTH, HEIGHT = A4
MARGIN = 48
CONTENT_WIDTH = WIDTH - 2 * MARGIN
BOTTOM = 59
TOP = HEIGHT - 47
NAVY = colors.HexColor("#17334D")
TEAL = colors.HexColor("#147D83")
INK = colors.HexColor("#202F3C")
MUTED = colors.HexColor("#566675")
PALE = colors.HexColor("#EEF4F7")

SOURCES = {
    "miracl": "https://huggingface.co/datasets/miracl/miracl",
    "corpus": "https://huggingface.co/datasets/miracl/miracl-corpus",
    "dakshina": "https://github.com/google-research-datasets/dakshina",
    "dakshina_download": (
        "https://storage.googleapis.com/gresearch/dakshina/"
        "dakshina_dataset_v1.0.tar"
    ),
    "comi": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA",
    "e5": "https://huggingface.co/intfloat/multilingual-e5-small",
    "uroman": "https://github.com/isi-nlp/uroman",
}
PINS = {
    "miracl": "5be20db9509754dadad47689368639fcec739c00",
    "corpus": "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
    "comi": "0214cb358e59e36da60e01d3d59ddd24df897749",
    "e5": "fd1525a9fd15316a2d503bf26ab031a61d056e98",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> tuple[dict, dict]:
    """Do not regenerate or tune any research input while preparing the PDF."""
    lock = read_json(STUDY / "final-lock.json")
    assert len(lock["files_sha256"]) == 729
    for name, expected in lock["files_sha256"].items():
        assert digest(Path(name)) == expected, f"Changed frozen input: {name}"
    completion = read_json(CHECKS / "cpu-pilot-stage6-complete-2026-10-20.json")
    assert completion["all_pass"]
    for name, expected in completion["output_files_sha256"].items():
        assert digest(Path(name)) == expected, f"Changed result artifact: {name}"
    results = read_json(STUDY / "final-report/report.json")
    assert results["fixture_only"] is False
    assert results["actual_track_count"] == 27
    engineering = read_json(CHECKS / "engineering-final-2026-10-20.json")
    assert engineering["all_pass"] and engineering["tests_each_hash_seed"] == 484
    assert {run["hash_seed"] for run in engineering["runs"]} == {0, 917}
    assert all(run["skips"] == 0 for run in engineering["runs"])
    app = read_json(CHECKS / "cpu-pilot-real-app-2026-10-20.json")
    assert app["all_pass"] and app["actual_http_requests"] == 21
    assert app["query_cases"] == 6 and app["actual_search_invocations"] == 12
    return results, lock


def register_fonts() -> None:
    folder = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (
        ("ReportSans", "DejaVuSans.ttf"),
        ("ReportSans-Bold", "DejaVuSans-Bold.ttf"),
        ("ReportSans-Oblique", "DejaVuSans-Oblique.ttf"),
        ("ReportMono", "DejaVuSansMono.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily(
        "ReportSans", normal="ReportSans", bold="ReportSans-Bold",
        italic="ReportSans-Oblique", boldItalic="ReportSans-Bold",
    )


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body", fontName="ReportSans", fontSize=10.1, leading=13.65,
        textColor=INK, spaceAfter=7, alignment=TA_LEFT,
        allowWidows=0, allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=body, fontName="ReportSans-Bold", fontSize=20.4,
            leading=25, textColor=NAVY, spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, fontSize=10.5, leading=14,
            textColor=MUTED, spaceAfter=15,
        ),
        "kicker": ParagraphStyle(
            "kicker", parent=body, fontName="ReportSans-Bold", fontSize=8.1,
            leading=11, textColor=TEAL, spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "heading", parent=body, fontName="ReportSans-Bold", fontSize=12.1,
            leading=16, textColor=NAVY, spaceBefore=5, spaceAfter=7,
            keepWithNext=1,
        ),
        "small": ParagraphStyle(
            "small", parent=body, fontSize=9.1, leading=12, textColor=MUTED,
            spaceAfter=7,
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=9.0, leading=12, spaceAfter=0,
        ),
        "cell_header": ParagraphStyle(
            "cell_header", parent=body, fontName="ReportSans-Bold",
            fontSize=8.6, leading=11.5, textColor=colors.white, spaceAfter=0,
        ),
        "ref": ParagraphStyle(
            "ref", parent=body, fontSize=9.55, leading=12.8, spaceAfter=4,
        ),
        "pin": ParagraphStyle(
            "pin", parent=body, fontName="ReportMono", fontSize=8.0,
            leading=10.7, textColor=MUTED, spaceAfter=10,
        ),
    }


def para(text: str, s: dict, kind: str = "body") -> Paragraph:
    return Paragraph(text, s[kind])


def link(url: str, label: str | None = None) -> str:
    return f'<link href="{escape(url, {chr(34): "&quot;"})}" color="#147D83">{escape(label or url)}</link>'


def metric(results: dict, track: str, method: str, name="ndcg_at_10") -> float:
    return results["tracks"][track]["macro_languages"][method]["metrics"][name]


def split_table(s: dict) -> Table:
    headers = ["Encoder train", "Fusion train", "Validation", "Final (official dev)"]
    counts = ["24 groups", "16 groups", "16 groups*", "24 groups"]
    data = [[para(item, s, "cell_header") for item in headers],
            [para(item, s, "cell") for item in counts]]
    table = Table(data, colWidths=[CONTENT_WIDTH / 4] * 4, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), PALE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def results_table(results: dict, s: dict) -> tuple[Table, list[list[str]]]:
    rows = [
        ("Original-query BM25", "B1_bm25_original", "base"),
        ("Protected Roman character TF-IDF", "B2_protected_roman_view_character", "base"),
        ("Original-query dense E5", "B3_dense_original", "base"),
        ("Equal fusion", "B5_equal", "base"),
        ("Validation-tuned fixed fusion", "fixed_selected", "base"),
        ("Full learned gate", "trace_weights", "full"),
        ("Retrained: no uncertainty features", "trace_weights", "no_uncertainty"),
        ("Retrained: no agreement features", "trace_weights", "no_agreement"),
    ]
    plain = [["Method", "Frozen E5", "Adapted\nseed 13", "Adapted\nseed 29"]]
    for label, method, variant in rows:
        frozen = f"{metric(results, 'frozen', method):.6f}" if variant == "base" else "—"
        vals = [f"{metric(results, f'adapted-seed-{seed}/{variant}', method):.6f}"
                for seed in (13, 29)]
        plain.append([label, frozen, *vals])
    formatted = [[para(item.replace("\n", "<br/>"), s, "cell_header" if i == 0 else "cell")
                  for item in row] for i, row in enumerate(plain)]
    table = Table(formatted, colWidths=[257, 80, 80, CONTENT_WIDTH - 417], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CAD7E0")),
    ]))
    return table, plain


def page_one(s: dict) -> list:
    return [
        para("PROJECT REPORT  /  20 OCTOBER 2026", s, "kicker"),
        para("Uncertainty-Aware<br/>Multi-View Retrieval", s, "title"),
        para("English, Hindi and Bengali · Audited public-data CPU pilot", s, "subtitle"),
        para("1  Introduction", s, "heading"),
        para(
            "Search across Indian languages must accommodate script differences, Romanized "
            "spellings and language mixing. Exact matching can miss relevant passages; forced "
            "rewriting can damage names, numbers or negation. This project preserves original "
            "queries, constructs cautious alternatives and combines lexical and neural evidence. "
            "Its broader noisy-query motivation is distinct from the original-query pilot "
            "evaluated here.", s),
        para("2  Problem statement", s, "heading"),
        para(
            "Can query-dependent reliability weighting outperform equal and "
            "validation-tuned fixed fusion without harming original-query retrieval? The system "
            "must support English, Hindi and Bengali, expose uncertainty and retain passage "
            "provenance. Public sources did not supply the independently reviewed bilingual "
            "challenge required by the original design. The approved original-query pilot "
            "therefore excludes unreviewed noisy variants and consistency supervision.", s),
        para("3  Solution approach", s, "heading"),
        para(
            "<b>Data and isolation.</b> MIRACL queries, relevance judgments and Wikipedia "
            "passages [1, 2] form separate monolingual tracks, each with the same fixed "
            "1,500 passages across systems. All selected judged passages, including grade-zero "
            "ones, are retained and supplemented with deterministic distractors. Whole query "
            "groups preserve their existing assignments; official training supplies the three "
            "internal partitions below, and official development supplies the locked final set.", s),
        split_table(s),
        Spacer(1, 5),
        para(
            "Per language. *Hindi has 17 validation queries in 16 groups; total validation "
            "queries = 49. Final evaluation = 72 original queries/groups (24 per language).", s, "small"),
        para(
            "<b>Retrieval.</b> Protected views feed four possible branches: original-query "
            "BM25, distinct native-view BM25, Roman-view character 3–5-gram TF-IDF and dense "
            "E5. Dakshina training lexicons [3] and explicitly algorithmic uroman mappings [6] "
            "provide a resource baseline, not a validated IndicXlit model. Auxiliary language ID "
            "uses public COMI-LINGUA Hindi–English labels [4] and weak Bengali word labels; "
            "it has no Bengali gold test. Unavailable branches are masked and active weights "
            "renormalized in reciprocal-rank fusion (rank constant 60). Held-out native-to-Roman "
            "word matching is low: 15.68% Hindi and 21.72% Bengali.", s),
        para(
            "<b>Neural adaptation.</b> Multilingual E5-small [5] produces normalized "
            "384-dimensional embeddings. Frozen and adapted models use identical query/passage "
            "prefixes and 128-token right truncation of title plus body (encoding batch 8). "
            "Each seed (13, 29) receives one SGD epoch on 72 original training queries: "
            "learning rate 0.001, training batch 1 and one judged nonrelevant negative per query. "
            "There is no paired-variant or consistency loss.", s),
        para(
            "<b>Fusion and delivery.</b> A linear masked-softmax gate learns branch utility "
            "from query, language-ID, transliteration and agreement features on fusion training "
            "only. Validation selects gate settings and fixed weights. Both seeds are reported; "
            "seed 13/full gate is the predeclared demo, without final-set retuning. The app returns ten source passages, "
            "view diagnostics and contributions—not generated answers.", s),
    ]


def page_two(results: dict, s: dict) -> tuple[list, list[list[str]]]:
    table, plain = results_table(results, s)
    mrrs = [metric(results, f"adapted-seed-{seed}/full", "trace_weights", "mrr_at_10")
            for seed in (13, 29)]
    recalls = [metric(results, f"adapted-seed-{seed}/full", "trace_weights", "recall_at_100")
               for seed in (13, 29)]
    return [
        para("4  Experimental results", s, "heading"),
        para(
            "<b>Table 1. Final language-macro nDCG@10.</b> Mean of the three language "
            "means; higher is better. Lexical scores are unchanged across encoders; "
            "dashes indicate not run.", s),
        table,
        Spacer(1, 8),
        para(
            f"Full-gate <b>MRR@10 = {mrrs[0]:.6f} / {mrrs[1]:.6f}</b> and "
            f"<b>Recall@100 = {recalls[0]:.6f} / {recalls[1]:.6f}</b> for seeds 13 / 29. "
            "Serving returns ten hits; ranking replay evaluates recall at 100.", s),
        para(
            "<b>Findings.</b> Adaptation does not improve dense final nDCG in either seed. "
            "The full gate beats equal fusion but trails dense-only and tuned fixed fusion. "
            "Against dense-only, its query-level nDCG wins/ties/losses are 18/22/32 "
            "(seed 13) and 17/19/36 (seed 29). Removing agreement features improves point "
            "estimates; removing uncertainty features slightly lowers them. These retrained "
            "ablations are descriptive, not proof of significance or a causal explanation.", s),
        para(
            "<b>Availability matters.</b> Native-view BM25 is masked on all 72 final queries: "
            "no distinct safe native view is produced. Character retrieval is unavailable on "
            "all 24 Bengali queries and one Hindi query; missing character-only runs score "
            "zero rather than being dropped. Thus the experiment does not demonstrate a "
            "four-active-branch or reliable Bengali normalization benefit.", s),
        para(
            "<b>Engineering verification.</b> All 13 functional modules are integrated. "
            "The recorded final suite passes 484 tests under each of two hash seeds (0, 917), "
            "with zero skips and warnings treated as errors. Offline wheel build/install and "
            "real cached-E5 checks pass. The real-data app passes six query cases, twelve "
            "search invocations and 21 HTTP requests across all three languages, including "
            "API/HTML, health and invalid-input checks. Independent metric recomputation "
            "passes; 729 frozen input files remain intact [7].", s),
        para("5  Conclusion", s, "heading"),
        para(
            "The project delivers an auditable end-to-end multilingual passage retriever, "
            "reproducible training, controlled comparisons and an inspectable search app. "
            "The proposed retrieval-quality gains are <b>not demonstrated</b> by this pilot: "
            "frozen dense E5 is strongest by macro nDCG, and a simple tuned fixed fusion "
            "outperforms the full learned gate. Working software and successful tests do "
            "not establish research superiority.", s),
        para(
            "<b>Limits and next work.</b> These small, judged-passage-retaining collections "
            "are not full MIRACL or the prepared 50,000-passage-per-language experiment. "
            "E5 already has MIRACL-training exposure, so its baseline is not zero-shot. "
            "Seeds share evaluation queries; exact-duplicate grouping is not a semantic "
            "paraphrase audit. No natural noisy/code-mixed robustness, cross-lingual, "
            "human-challenge or consistency benefit is established. Larger experiments "
            "need more capacity and independently reviewed bilingual data; browser visual "
            "review is also outstanding.", s),
    ], plain


def reference(s: dict, number: int, title: str, description: str,
              url: str, pin: str | None = None, extra: str = "") -> list:
    parts = [para(f"<b>[{number}] {title}.</b> {description}", s, "ref"),
             para(link(url), s, "ref")]
    if extra:
        parts.append(para(extra, s, "ref"))
    if pin:
        parts.append(para(f"Revision: {pin}", s, "pin"))
    else:
        parts.append(Spacer(1, 9))
    return parts


def page_three(s: dict) -> list:
    parts = [
        para("REFERENCES ONLY", s, "kicker"),
        para("References and data access", s, "title"),
        para("Public dataset links, exact resource versions and execution evidence", s, "subtitle"),
    ]
    parts += reference(
        s, 1, "MIRACL queries and relevance judgments",
        "Project MIRACL. English, Hindi and Bengali original train/development queries "
        "and passage judgments. Dataset card declares Apache-2.0.",
        SOURCES["miracl"], PINS["miracl"],
        link(f'{SOURCES["miracl"]}/tree/{PINS["miracl"]}', "Browse/download the pinned data files"),
    )
    parts += reference(
        s, 2, "MIRACL passage corpus",
        "Project MIRACL. Wikipedia passage text, titles and IDs for the three monolingual "
        "collections. Card declares Apache-2.0; preserve underlying Wikipedia attribution.",
        SOURCES["corpus"], PINS["corpus"],
        link(f'{SOURCES["corpus"]}/tree/{PINS["corpus"]}', "Browse/download the pinned corpus files"),
    )
    parts += reference(
        s, 3, "Dakshina v1.0",
        "Roark et al. (2020), Google Research. Hindi/Bengali native–Roman word lexicons; "
        "training pairs supply attested mappings, not retrieval labels. CC-BY-SA-4.0.",
        SOURCES["dakshina"],
        extra=(link(SOURCES["dakshina_download"], "Official download: dakshina_dataset_v1.0.tar")
               + "<br/>Acquired lexicon members: GCS generation 1590618303711280; "
               "member checksums recorded locally, not a verified full-archive MD5."),
    )
    parts += reference(
        s, 4, "COMI-LINGUA",
        "LingoIITGN. Human Hindi–English token annotations used for auxiliary language-ID "
        "training/calibration/testing; not Bengali retrieval gold. CC-BY-4.0.",
        SOURCES["comi"], PINS["comi"],
        link(f'{SOURCES["comi"]}/tree/{PINS["comi"]}', "Browse/download the pinned data files"),
    )
    parts += reference(
        s, 5, "Multilingual E5-small",
        "intfloat. Pretrained 384-dimensional retrieval encoder and model card; prior "
        "training includes MIRACL. MIT model license.",
        SOURCES["e5"], PINS["e5"],
    )
    parts += reference(
        s, 6, "uroman 1.3.1.1",
        "Ulf Hermjakob, USC Information Sciences Institute. Algorithmic out-of-vocabulary "
        "Romanization; bundled USC permissive license and acknowledgment apply.",
        SOURCES["uroman"],
        extra=("Acknowledgment: This project uses the universal romanizer software 'uroman' "
               "written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020)."),
    )
    parts += [
        para("<b>[7] Local execution and audit artifacts (20 October 2026).</b> "
             "Paths relative to the project checkout; these are local evidence, not public dataset URLs.", s, "ref"),
        para("artifacts/study/cpu-pilot/final-report/report.json<br/>"
             "artifacts/checks/cpu-pilot-stage6-complete-2026-10-20.json<br/>"
             "artifacts/checks/engineering-final-2026-10-20.json<br/>"
             "artifacts/checks/cpu-pilot-real-app-2026-10-20.json<br/>"
             "artifacts/checks/cpu-pilot-final-independent-audit-2026-10-20.json", s, "pin"),
    ]
    return parts


def draw_page(pdf: canvas.Canvas, flowables: list, page: int) -> float:
    """Each fixed frame must fit its own page; never silently spill to references."""
    pdf.setStrokeColor(TEAL)
    pdf.setLineWidth(2)
    pdf.line(MARGIN, HEIGHT - 30, MARGIN + 36, HEIGHT - 30)
    frame = Frame(MARGIN, BOTTOM, CONTENT_WIDTH, TOP - BOTTOM,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    pending = list(flowables)
    frame.addFromList(pending, pdf)
    if pending:
        raise RuntimeError(f"Page {page} overflows: {len(pending)} flowables left")
    pdf.setStrokeColor(colors.HexColor("#D0DCE4"))
    pdf.setLineWidth(0.5)
    pdf.line(MARGIN, 44, WIDTH - MARGIN, 44)
    pdf.setFont("ReportSans", 7.7)
    pdf.setFillColor(MUTED)
    pdf.drawString(MARGIN, 30, "PUBLIC-DATA CPU PILOT")
    pdf.drawCentredString(WIDTH / 2, 30, f"{page} / 3")
    pdf.drawRightString(WIDTH - MARGIN, 30,
                        "REFERENCES ONLY" if page == 3 else "MAIN REPORT")
    used = TOP - frame._y
    pdf.showPage()
    return round(used, 2)


def validate_and_render(table: list[list[str]], input_hashes: dict,
                        used_heights: list[float]) -> dict:
    reader = PdfReader(PDF)
    assert len(reader.pages) == 3, "Exactly two main pages + one reference page required"
    texts = [page.extract_text() for page in reader.pages]
    for heading in ("1 Introduction", "2 Problem statement", "3 Solution approach"):
        assert heading in texts[0], f"Missing page-one section: {heading}"
    for heading in ("4 Experimental results", "5 Conclusion"):
        assert heading in texts[1], f"Missing page-two section: {heading}"
    assert "References and data access" in texts[2]
    assert not re.search(r"[1-5] +(Introduction|Problem statement|Solution approach|Experimental results|Conclusion)", texts[2])
    assert not any("References and data access" in text for text in texts[:2])
    for number in range(1, 8):
        assert f"[{number}]" in texts[2], f"Missing reference {number}"
    for row in table[1:]:
        assert row[0] in texts[1]
        assert all(value in texts[1] for value in row[1:])
    uris = []
    for page in reader.pages:
        for annotation in page.get("/Annots", []):
            item = annotation.get_object()
            if item.get("/A", {}).get("/URI"):
                uris.append(str(item["/A"]["/URI"]))
    for url in SOURCES.values():
        assert url in uris, f"Missing clickable source: {url}"
    for name, expected in input_hashes.items():
        assert digest(Path(name)) == expected, f"Report build modified input: {name}"

    rendered = []
    page_geometry = []
    doc = pymupdf.open(PDF)
    for index, page in enumerate(doc):
        boxes = [pymupdf.Rect(*block[:4]) for block in page.get_text("blocks")]
        assert boxes, f"Empty page {index + 1}"
        for rect in boxes:
            assert rect.x0 >= MARGIN - 1 and rect.x1 <= WIDTH - MARGIN + 1, (index, rect)
            assert rect.y0 >= 35 and rect.y1 <= HEIGHT - 20, (index, rect)
        # Main/reference text never enters the reserved footer strip.
        for line in page.get_text("dict")["blocks"]:
            for subline in line.get("lines", []):
                for span in subline["spans"]:
                    if span["size"] > 8:
                        assert span["bbox"][3] <= HEIGHT - BOTTOM + 1, (index, span)
        image_path = OUT / f"report_page_{index + 1}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False).save(str(image_path))
        rendered.append(str(image_path.relative_to(ROOT)))
        page_geometry.append({
            "page": index + 1, "size_points": [round(page.rect.width, 2), round(page.rect.height, 2)],
            "used_frame_height_points": used_heights[index],
            "word_count": len(page.get_text("words")),
        })
    doc.close()
    (OUT / "report_text.txt").write_text(
        "\n\n".join(f"=== PAGE {index + 1} ===\n{text}" for index, text in enumerate(texts)),
        encoding="utf-8",
    )
    return {
        "all_automated_checks_pass": True,
        "pdf": str(PDF.relative_to(ROOT)), "sha256": digest(PDF),
        "pages": 3, "main_pages": [1, 2], "references_only_page": 3,
        "frozen_input_files_verified_unchanged": 729,
        "result_artifact_files_verified_unchanged": 7,
        "clickable_source_links": sorted(set(uris)),
        "rendered_pages": rendered, "page_geometry": page_geometry,
        "metric_values": table,
        "visual_review": "Review the three rendered PNG pages before handoff.",
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results, lock = verify_inputs()
    completion = read_json(CHECKS / "cpu-pilot-stage6-complete-2026-10-20.json")
    baseline_hashes = dict(lock["files_sha256"])
    baseline_hashes.update(completion["output_files_sha256"])
    baseline_hashes[str(STUDY / "final-lock.json")] = digest(STUDY / "final-lock.json")
    register_fonts()
    s = styles()
    second, table = page_two(results, s)
    pdf = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1, invariant=1)
    pdf.setTitle("Uncertainty-Aware Multi-View Retrieval — Public-Data CPU Pilot")
    pdf.setAuthor("NLP Project")
    pdf.setSubject("Two-page project report plus one references-only page, with dataset links")
    pdf.setCreator("NLP Project report builder / ReportLab")
    used_heights = [draw_page(pdf, content, page)
                    for page, content in enumerate((page_one(s), second, page_three(s)), 1)]
    pdf.save()
    validation = validate_and_render(table, baseline_hashes, used_heights)
    (OUT / "report_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: validation[key] for key in (
        "pdf", "sha256", "pages", "all_automated_checks_pass", "page_geometry"
    )}, indent=2))


if __name__ == "__main__":
    main()
