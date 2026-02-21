import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.services.ocr import OCREngine
from app.services.pdf import image_bytes_to_numpy, pdf_pages_to_images

logger = logging.getLogger("ocr.parse")
router = APIRouter()

MAX_SIZE = settings.max_file_size_mb * 1024 * 1024

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Thread pool for running blocking OCR inference off the async event loop
_ocr_executor = ThreadPoolExecutor(max_workers=1)

# Heartbeat interval in seconds - must be well under proxy timeout (~100s)
_HEARTBEAT_INTERVAL = 5.0


def _get_engine() -> OCREngine:
    return OCREngine.get()


def _sse_event(data: dict[str, Any]) -> str:
    """Format a dict as an SSE `data:` line."""
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _run_in_thread_with_heartbeats(
    func: Any,
    *args: Any,
    status_label: str = "processing",
    loop: asyncio.AbstractEventLoop | None = None,
) -> Any:
    """Run a blocking *func* in a thread while yielding heartbeat dicts.

    This is an async generator.  It yields SSE-ready dicts while *func*
    executes, then yields a final ``{"_result": <return value>}`` dict so
    the caller can retrieve the actual result.
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    future = loop.run_in_executor(_ocr_executor, func, *args)
    t_start = time.time()

    while not future.done():
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=_HEARTBEAT_INTERVAL)
            # Future completed within the heartbeat window
            break
        except asyncio.TimeoutError:
            elapsed = round(time.time() - t_start, 1)
            yield {"status": status_label, "elapsed": elapsed}

    result = future.result()  # propagate exceptions if any
    yield {"_result": result}


async def _parse_sse_generator(
    content: bytes,
    content_type: str,
    filename: str,
    page: int,
) -> AsyncGenerator[str, None]:
    """Async generator that drives the parse pipeline and yields SSE events."""
    t0 = time.time()

    # -- 1. Upload acknowledged --
    yield _sse_event({"status": "uploading"})
    logger.info("File size: %.1f MB", len(content) / 1024 / 1024)

    # -- 2. PDF / image conversion --
    yield _sse_event({"status": "converting_pdf"})

    if content_type == "application/pdf":
        logger.info("Converting PDF to images (%d dpi)...", settings.pdf_dpi)
        loop = asyncio.get_running_loop()
        pages = await loop.run_in_executor(
            _ocr_executor, pdf_pages_to_images, content, settings.pdf_dpi,
        )
        logger.info("PDF has %d pages", len(pages))
        if page >= len(pages):
            yield _sse_event({"status": "error", "message": f"Page {page} out of range (0-{len(pages) - 1})"})
            return
        image = pages[page]
        total_pages = len(pages)
    elif content_type and content_type.startswith("image/"):
        image = image_bytes_to_numpy(content)
        total_pages = 1
    else:
        yield _sse_event({"status": "error", "message": "Unsupported file type. Upload PDF or image."})
        return

    logger.info("Image shape: %s", image.shape)

    # Save source image
    ts = time.strftime("%H%M%S")
    safe_name = (filename or "doc").replace(" ", "_")[:30]
    _save_source(image, ts, safe_name, page)

    # -- 3. Structure OCR (PP-StructureV3) with heartbeats --
    yield _sse_event({"status": "running_structure_ocr"})

    engine = _get_engine()

    structure_result: dict[str, Any] | None = None
    async for msg in _run_in_thread_with_heartbeats(
        engine._run_structure, image, status_label="running_structure_ocr",
    ):
        if "_result" in msg:
            structure_result = msg["_result"]
        else:
            yield _sse_event(msg)

    assert structure_result is not None

    # -- 4. VLM OCR (PaddleOCR-VL) with heartbeats --
    vlm_result: dict[str, Any] | None = None
    if engine.vlm:
        yield _sse_event({"status": "running_vlm"})
        async for msg in _run_in_thread_with_heartbeats(
            engine._run_vlm, image, status_label="running_vlm",
        ):
            if "_result" in msg:
                vlm_result = msg["_result"]
            else:
                yield _sse_event(msg)

    ocr_elapsed = round(time.time() - t0, 1)

    ocr_result: dict[str, Any] = {
        "structure": structure_result,
        "vlm": vlm_result,
        "elapsed_s": ocr_elapsed,
    }

    # Save OCR results
    _save_json(ocr_result["structure"], ts, safe_name, page, "structure")
    if ocr_result["vlm"]:
        _save_json(ocr_result["vlm"], ts, safe_name, page, "vlm")

    # Save markdown
    md = ocr_result["structure"].get("markdown", "")
    if md:
        md_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page}_structure.md"
        md_path.write_text(md, encoding="utf-8")
        logger.info("Structure markdown saved to: %s", md_path)
        logger.info("--- MARKDOWN (first 3000 chars) ---\n%s\n--- END ---", md[:3000])

    vlm_md = ""
    if ocr_result["vlm"]:
        vlm_md = ocr_result["vlm"].get("markdown", "")
        if vlm_md:
            vlm_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page}_vlm.md"
            vlm_path.write_text(vlm_md, encoding="utf-8")
            logger.info("VLM markdown saved to: %s", vlm_path)

    # -- 5. Render HTML --
    yield _sse_event({"status": "rendering"})

    rendered_html = _render_from_structure(ocr_result)
    html_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page}_rendered.html"
    html_path.write_text(rendered_html, encoding="utf-8")
    logger.info("Rendered HTML saved to: %s", html_path)

    elapsed = time.time() - t0
    logger.info("=== Parse complete in %.1fs ===", elapsed)

    # -- 6. Final complete event with full payload --
    yield _sse_event({
        "status": "complete",
        "result": {
            "page": page,
            "total_pages": total_pages,
            "structure_markdown": md,
            "vlm_markdown": vlm_md,
            "rendered_html": rendered_html,
            "ocr_elapsed_s": ocr_result["elapsed_s"],
            "total_elapsed_s": round(elapsed, 1),
        },
    })


@router.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    page: Annotated[int, Form()] = 0,
) -> StreamingResponse:
    """Parse a document using dual OCR engine, streaming progress via SSE.

    Returns ``text/event-stream`` with progress heartbeats so the
    connection survives RunPod / Cloudflare proxy timeouts.
    """
    if not file.content_type:
        raise HTTPException(400, "Missing content type")

    logger.info("=== Parse request: %s (%s) page=%d ===", file.filename, file.content_type, page)

    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(413, f"File too large. Max {settings.max_file_size_mb}MB")

    content_type = file.content_type
    filename = file.filename or "document"

    return StreamingResponse(
        _parse_sse_generator(content, content_type, filename, page),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


def _render_from_structure(ocr_result: dict) -> str:
    """Build the form-replica HTML from OCR output.

    Produces professional HTML styled like Korean government tax forms
    (매입처별세금계산서합계표).  Uses structure tables, text blocks, and
    the raw markdown from PP-StructureV3.
    """
    structure = ocr_result.get("structure", {})
    md: str = structure.get("markdown", "")
    tables: list[dict] = structure.get("tables", [])
    text_blocks: list[dict] = structure.get("text_blocks", [])

    # Collect usable table HTML fragments
    table_htmls: list[str] = [t["html"] for t in tables if t.get("html")]

    # ── Classify text blocks by vertical position ──────────────
    # Sort by the top-edge (box[1]) so we can split header-area text from
    # the rest.  Blocks in the upper 25 % of the page are treated as
    # header / title / period / submitter info.
    sorted_blocks = sorted(text_blocks, key=lambda b: b.get("box", [0, 0, 0, 0])[1])

    if sorted_blocks:
        max_y = max(b.get("box", [0, 0, 0, 0])[3] for b in sorted_blocks) or 1
        header_cutoff = max_y * 0.25
    else:
        header_cutoff = 0.0

    header_blocks: list[dict] = []
    body_blocks: list[dict] = []
    for blk in sorted_blocks:
        top_y = blk.get("box", [0, 0, 0, 0])[1]
        if top_y <= header_cutoff:
            header_blocks.append(blk)
        else:
            body_blocks.append(blk)

    # ── Try to identify a document title ───────────────────────
    # The tallest / most-confident block near the top is likely the title.
    doc_title = ""
    info_blocks: list[dict] = []
    if header_blocks:
        # Pick the block whose bounding box has the largest height as title
        def _box_height(b: dict) -> float:
            box = b.get("box", [0, 0, 0, 0])
            return abs(box[3] - box[1])

        title_candidate = max(header_blocks, key=_box_height)
        doc_title = title_candidate.get("text", "").strip()
        info_blocks = [b for b in header_blocks if b is not title_candidate]

    # ── Start building HTML ────────────────────────────────────
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>OCR Result</title>",
        "<style>",
        RESULT_CSS,
        "</style>",
        "</head>",
        "<body>",
        '<div class="gov-page">',
    ]

    # ── Document header ────────────────────────────────────────
    parts.append('<div class="doc-header">')
    if doc_title:
        parts.append(f'<div class="doc-title">{_esc(doc_title)}</div>')
    else:
        parts.append('<div class="doc-title">OCR Document</div>')
    parts.append("</div>")  # .doc-header

    # ── Info rows from header text blocks ──────────────────────
    if info_blocks:
        parts.append('<div class="info-grid">')
        for blk in info_blocks:
            txt = (blk.get("text") or "").strip()
            if not txt:
                continue
            # Try to split on common Korean delimiters (: , 】 등)
            label, value = _split_label_value(txt)
            parts.append('<div class="info-row">')
            parts.append(f'<div class="info-label">{_esc(label)}</div>')
            parts.append(f'<div class="info-value">{_esc(value)}</div>')
            parts.append("</div>")
        parts.append("</div>")  # .info-grid

    # ── Tables ─────────────────────────────────────────────────
    if table_htmls:
        for i, raw_html in enumerate(table_htmls):
            styled = _enhance_table_html(raw_html)
            label = f"Table {i + 1}" if len(table_htmls) > 1 else ""
            parts.append('<div class="table-section">')
            if label:
                parts.append(f'<div class="section-label">{label}</div>')
            parts.append(styled)
            parts.append("</div>")
    elif md:
        # Fallback: render markdown as preformatted text inside the page
        parts.append('<div class="table-section">')
        parts.append(f"<pre>{_esc(md)}</pre>")
        parts.append("</div>")

    # ── Non-header text blocks below the tables ────────────────
    if body_blocks:
        remaining_texts = [
            (b.get("text") or "").strip() for b in body_blocks
        ]
        remaining_texts = [t for t in remaining_texts if t]
        if remaining_texts:
            parts.append('<div class="info-grid" style="margin-top:10px">')
            for txt in remaining_texts:
                label, value = _split_label_value(txt)
                parts.append('<div class="info-row">')
                parts.append(f'<div class="info-label">{_esc(label)}</div>')
                parts.append(f'<div class="info-value">{_esc(value)}</div>')
                parts.append("</div>")
            parts.append("</div>")

    # ── Collapsible raw markdown reference ─────────────────────
    if md:
        parts.append('<div class="raw-reference">')
        parts.append("<details>")
        parts.append("<summary>Raw OCR Markdown (click to expand)</summary>")
        parts.append(f"<pre>{_esc(md)}</pre>")
        parts.append("</details>")
        parts.append("</div>")

    parts.append("</div>")  # .gov-page
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Pattern that matches strings composed entirely of digits, commas, dots,
# minus signs, percent, and optional whitespace — i.e. numeric content.
_NUMERIC_RE = re.compile(r"^[\s\d,.\-\u2013\u2014%()]+$")

# Delimiters commonly used in Korean form labels
_LABEL_SPLIT_RE = re.compile(r"[:：】\]\)]\s*")


def _split_label_value(text: str) -> tuple[str, str]:
    """Best-effort split of a text line into (label, value).

    Tries common Korean form delimiters.  Falls back to returning
    the whole string as the label with an empty value.
    """
    m = _LABEL_SPLIT_RE.search(text)
    if m:
        label = text[: m.start()].strip().lstrip("【[(")
        value = text[m.end() :].strip()
        if label:
            return label, value
    return text, ""


def _enhance_table_html(raw_html: str) -> str:
    """Post-process a PaddleOCR ``<table>`` fragment.

    * Adds ``class="num"`` to ``<td>`` cells whose content looks numeric
      so they get monospace + right-aligned styling.
    * Adds ``class="txt"`` to cells with Korean / Latin text so they
      are left-aligned.
    * Leaves header cells (``<th>``) untouched.
    """
    def _classify_cell(match: re.Match[str]) -> str:
        tag = match.group(1)       # opening attrs portion (colspan etc.)
        content = match.group(2)   # inner text
        close = match.group(3)     # </td>

        stripped = content.strip()
        if not stripped:
            return match.group(0)

        if _NUMERIC_RE.match(stripped):
            cls = "num"
        else:
            cls = "txt"

        # Inject class; preserve existing attributes
        if "class=" in tag:
            return match.group(0)  # already has a class, don't override
        # Insert class right after "<td"
        new_tag = tag.replace("<td", f'<td class="{cls}"', 1)
        return f"{new_tag}{content}{close}"

    return re.sub(
        r"(<td[^>]*>)(.*?)(</td>)",
        _classify_cell,
        raw_html,
        flags=re.DOTALL,
    )


RESULT_CSS = """
/* ── Reset & base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Malgun Gothic', '맑은 고딕', 'Apple SD Gothic Neo',
                 'Noto Sans KR', sans-serif;
    font-size: 11px;
    line-height: 1.45;
    color: #111;
    background: #f4f4f0;
    padding: 30px 10px;
}

/* ── Page container (mimics printed A4 form) ──────────────── */
.gov-page {
    max-width: 860px;
    margin: 0 auto;
    background: #fff;
    border: 2px solid #000;
    padding: 28px 30px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,.12);
}

/* ── Document header block ────────────────────────────────── */
.doc-header {
    text-align: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #000;
}
.doc-header .doc-id {
    font-size: 9px;
    color: #555;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.doc-header .doc-title {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 4px 0;
}
.doc-header .doc-subtitle {
    font-size: 10px;
    color: #333;
}

/* ── Text-block info rows (period, submitter, etc.) ───────── */
.info-grid {
    display: table;
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 12px;
    border: 1px solid #000;
}
.info-row {
    display: table-row;
}
.info-label, .info-value {
    display: table-cell;
    border: 1px solid #000;
    padding: 3px 7px;
    vertical-align: middle;
    font-size: 10px;
}
.info-label {
    background: #f0f0ee;
    font-weight: 600;
    width: 120px;
    text-align: center;
    white-space: nowrap;
}
.info-value {
    text-align: left;
}

/* ── Table styling (core form tables) ─────────────────────── */
.table-section {
    margin-bottom: 18px;
}
.table-section .section-label {
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 4px;
    padding-left: 2px;
    color: #222;
}

table {
    width: 100%;
    border-collapse: collapse;
    table-layout: auto;
}

/* Outer border of each table is heavier */
.table-section table {
    border: 2px solid #000;
}

th, td {
    border: 1px solid #000;
    padding: 2px 5px;
    text-align: center;
    vertical-align: middle;
    font-size: 10px;
    line-height: 1.35;
    word-break: keep-all;
}

th {
    background: #f0f0ee;
    font-weight: 600;
    font-size: 10px;
}

/* Numeric cells: monospace, right-aligned */
td.num, .num {
    font-family: 'Consolas', 'D2Coding', 'Nanum Gothic Coding', monospace;
    text-align: right;
    letter-spacing: -0.3px;
}

/* Left-align cells that are clearly text */
td.txt, .txt {
    text-align: left;
}

/* Zebra striping – very subtle */
tbody tr:nth-child(even) td {
    background: #fafaf8;
}

/* Compact header rows sometimes have sub-headers */
thead th {
    background: #ebebea;
}

/* ── Collapsible raw-markdown reference ───────────────────── */
.raw-reference {
    margin-top: 22px;
    border-top: 1px dashed #aaa;
    padding-top: 10px;
}
.raw-reference summary {
    font-size: 11px;
    font-weight: 600;
    color: #555;
    cursor: pointer;
    padding: 4px 0;
    user-select: none;
}
.raw-reference summary:hover {
    color: #000;
}
.raw-reference pre {
    font-family: 'D2Coding', 'Consolas', 'Courier New', monospace;
    font-size: 10px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-all;
    background: #fafaf8;
    border: 1px solid #ddd;
    padding: 12px 14px;
    margin-top: 6px;
    max-height: 500px;
    overflow-y: auto;
}

/* ── Print-friendly ───────────────────────────────────────── */
@media print {
    body { background: #fff; padding: 0; }
    .gov-page { border: none; box-shadow: none; padding: 0; }
    .raw-reference { display: none; }
}
"""


def _save_source(image: np.ndarray, ts: str, name: str, page: int) -> None:
    from PIL import Image

    path = OUTPUT_DIR / f"{ts}_{name}_p{page}_source.png"
    Image.fromarray(image).save(path)
    logger.info("Source image saved to: %s", path)


def _save_json(data: dict, ts: str, name: str, page: int, label: str) -> None:
    path = OUTPUT_DIR / f"{ts}_{name}_p{page}_{label}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("%s JSON saved to: %s", label.capitalize(), path)
