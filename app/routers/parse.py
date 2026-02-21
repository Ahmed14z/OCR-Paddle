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
from app.services.llm_correct import correct_ocr_text
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


async def _process_single_page(
    image: np.ndarray,
    engine: OCREngine,
    ts: str,
    safe_name: str,
    page_idx: int,
    t0: float,
) -> AsyncGenerator[str | dict[str, Any], None]:
    """Process a single page image through the OCR pipeline.

    Yields SSE event strings for progress, and a final dict with key
    ``"_page_result"`` containing the per-page output.
    """
    logger.info("Image shape: %s", image.shape)
    _save_source(image, ts, safe_name, page_idx)

    # -- Structure OCR --
    yield _sse_event({"status": "running_structure_ocr"})
    structure_result: dict[str, Any] | None = None
    async for msg in _run_in_thread_with_heartbeats(
        engine._run_structure, image, status_label="running_structure_ocr",
    ):
        if "_result" in msg:
            structure_result = msg["_result"]
        else:
            yield _sse_event(msg)
    assert structure_result is not None

    # -- VLM OCR --
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
    _save_json(ocr_result["structure"], ts, safe_name, page_idx, "structure")
    if ocr_result["vlm"]:
        _save_json(ocr_result["vlm"], ts, safe_name, page_idx, "vlm")

    md = ocr_result["structure"].get("markdown", "")
    if md:
        md_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page_idx}_structure.md"
        md_path.write_text(md, encoding="utf-8")
        logger.info("Structure markdown saved to: %s", md_path)
        logger.info("--- MARKDOWN (first 3000 chars) ---\n%s\n--- END ---", md[:3000])

    vlm_md = ""
    if ocr_result["vlm"]:
        vlm_md = ocr_result["vlm"].get("markdown", "")
        if vlm_md:
            vlm_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page_idx}_vlm.md"
            vlm_path.write_text(vlm_md, encoding="utf-8")
            logger.info("VLM markdown saved to: %s", vlm_path)

    # -- LLM post-correction for Korean character errors --
    primary_md = vlm_md or md
    corrected_md = primary_md
    if primary_md and settings.openrouter_api_key:
        yield _sse_event({"status": "correcting_korean"})
        corrected_md = await correct_ocr_text(primary_md)
        if corrected_md != primary_md:
            corrected_path = OUTPUT_DIR / f"{ts}_{safe_name}_p{page_idx}_corrected.md"
            corrected_path.write_text(corrected_md, encoding="utf-8")
            logger.info("LLM-corrected markdown saved to: %s", corrected_path)

    struct_tables = ocr_result["structure"].get("tables", [])
    text_blocks = ocr_result["structure"].get("text_blocks", [])

    yield {
        "_page_result": {
            "md": md,
            "vlm_md": vlm_md,
            "corrected_md": corrected_md,
            "struct_tables": struct_tables,
            "text_blocks": text_blocks,
            "ocr_elapsed_s": ocr_result["elapsed_s"],
        }
    }


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
        pdf_pages = await loop.run_in_executor(
            _ocr_executor, pdf_pages_to_images, content, settings.pdf_dpi,
        )
        total_pages = len(pdf_pages)
        logger.info("PDF has %d pages", total_pages)

        if page == -1:
            pages_to_process = list(range(total_pages))
        elif page >= total_pages:
            yield _sse_event({"status": "error", "message": f"Page {page + 1} out of range (1-{total_pages})"})
            return
        else:
            pages_to_process = [page]
    elif content_type and content_type.startswith("image/"):
        pdf_pages = [image_bytes_to_numpy(content)]
        total_pages = 1
        pages_to_process = [0]
    else:
        yield _sse_event({"status": "error", "message": "Unsupported file type. Upload PDF or image."})
        return

    ts = time.strftime("%H%M%S")
    safe_name = (filename or "doc").replace(" ", "_")[:30]
    engine = _get_engine()

    all_md: list[str] = []
    all_vlm_md: list[str] = []
    all_corrected_md: list[str] = []
    all_struct_tables: list[list[Any]] = []
    all_text_blocks: list[list[dict]] = []
    last_ocr_elapsed = 0.0

    for idx, page_idx in enumerate(pages_to_process):
        if len(pages_to_process) > 1:
            yield _sse_event({
                "status": "processing",
                "message": f"Processing page {idx + 1} of {len(pages_to_process)}...",
            })

        image = pdf_pages[page_idx]
        async for event in _process_single_page(image, engine, ts, safe_name, page_idx, t0):
            if isinstance(event, dict) and "_page_result" in event:
                pr = event["_page_result"]
                all_md.append(pr["md"])
                all_vlm_md.append(pr["vlm_md"])
                all_corrected_md.append(pr["corrected_md"])
                all_struct_tables.append(pr["struct_tables"])
                all_text_blocks.append(pr["text_blocks"])
                last_ocr_elapsed = pr["ocr_elapsed_s"]
            else:
                yield event  # type: ignore[misc]

    # -- Render HTML --
    yield _sse_event({"status": "rendering"})

    combined_corrected = "\n\n".join(filter(None, all_corrected_md))
    combined_struct_tables = [t for tables in all_struct_tables for t in tables]
    combined_text_blocks = [b for blocks in all_text_blocks for b in blocks]
    rendered_html = _render_merged(combined_corrected, combined_struct_tables, combined_text_blocks)

    page_label = "all" if page == -1 else f"p{page}"
    html_path = OUTPUT_DIR / f"{ts}_{safe_name}_{page_label}_rendered.html"
    html_path.write_text(rendered_html, encoding="utf-8")
    logger.info("Rendered HTML saved to: %s", html_path)

    elapsed = time.time() - t0
    logger.info("=== Parse complete in %.1fs ===", elapsed)

    combined_md = "\n\n".join(filter(None, all_md))
    combined_vlm = "\n\n".join(filter(None, all_vlm_md))

    # -- Final complete event with full payload --
    yield _sse_event({
        "status": "complete",
        "result": {
            "page": page,
            "total_pages": total_pages,
            "structure_markdown": combined_md,
            "vlm_markdown": combined_vlm,
            "corrected_markdown": combined_corrected,
            "rendered_html": rendered_html,
            "ocr_elapsed_s": last_ocr_elapsed,
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


def _render_merged(
    merged_md: str,
    struct_tables: list[dict] | None = None,
    text_blocks: list[dict] | None = None,
) -> str:
    """Build styled HTML from merged markdown (headings + inline HTML tables).

    If *struct_tables* are provided, any VLM table that is just a header stub
    (single ``<tr>``) gets replaced by the matching Structure table HTML which
    preserves empty data rows.

    If *text_blocks* are provided, text that the VLM missed (form headers,
    footers, management numbers) is added at the top and bottom.
    """
    if not merged_md:
        body_html = '<p style="text-align:center;color:#888">No OCR output</p>'
    else:
        body_html = _vlm_markdown_to_html(merged_md, struct_tables, text_blocks)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Result</title>
<style>
{RESULT_CSS}
</style>
</head>
<body>
<div class="gov-page">
{body_html}
</div>
</body>
</html>"""


def _vlm_markdown_to_html(
    md: str,
    struct_tables: list[dict] | None = None,
    text_blocks: list[dict] | None = None,
) -> str:
    """Convert VLM markdown (with inline HTML tables) to styled HTML body content.

    VLM output is markdown with headings (###), paragraphs, and raw HTML
    <table> blocks. We convert headings to styled divs and pass tables
    through our enhancer for numeric cell styling.

    If a VLM table is a stub (only 1-2 ``<tr>`` rows), and *struct_tables*
    has a matching table with more rows, we swap in the Structure version
    so empty data rows are preserved.

    If *text_blocks* are provided, any text the VLM missed (detected by
    PP-StructureV3's overall OCR but not present in VLM output) is added
    at the appropriate position based on vertical coordinates.
    """
    lines = md.split("\n")
    parts: list[str] = []
    vlm_table_idx = 0

    # Sort structure tables by vertical position for matching
    sorted_struct: list[dict] = []
    if struct_tables:
        indexed = []
        for st in struct_tables:
            boxes = st.get("cell_boxes", [])
            y_center = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes) if boxes else float("inf")
            indexed.append((y_center, st))
        indexed.sort(key=lambda x: x[0])
        sorted_struct = [s for _, s in indexed]

    # Collect VLM text content to detect what's missing
    vlm_text_content = md.lower()

    # Find text blocks that VLM missed — text that appears in Structure
    # but NOT in VLM output. Split into header (top 15%) and footer (bottom 15%)
    header_texts: list[str] = []
    footer_texts: list[str] = []
    if text_blocks:
        sorted_blocks = sorted(text_blocks, key=lambda b: b.get("box", [0, 0, 0, 0])[1])
        if sorted_blocks:
            max_y = max(b.get("box", [0, 0, 0, 0])[3] for b in sorted_blocks) or 1
            for blk in sorted_blocks:
                txt = (blk.get("text") or "").strip()
                if not txt or len(txt) < 3:
                    continue
                # Skip text that's already in VLM output
                if txt.lower() in vlm_text_content or txt[:8].lower() in vlm_text_content:
                    continue
                y_top = blk.get("box", [0, 0, 0, 0])[1]
                if y_top < max_y * 0.15:
                    header_texts.append(txt)
                elif y_top > max_y * 0.85:
                    footer_texts.append(txt)

    # Add header texts the VLM missed (form ID, regulation reference, etc.)
    if header_texts:
        for txt in header_texts:
            parts.append(f'<p class="form-header-text">{_esc(txt)}</p>')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Headings → styled sections
        if stripped.startswith("# "):
            text = stripped.lstrip("# ").strip()
            parts.append(f'<div class="doc-header"><div class="doc-title">{_esc(text)}</div></div>')
        elif stripped.startswith("### "):
            text = stripped.lstrip("# ").strip()
            parts.append(f'<div class="section-label">{_esc(text)}</div>')
        elif stripped.startswith("## "):
            text = stripped.lstrip("# ").strip()
            parts.append(f'<div class="section-label" style="font-size:13px">{_esc(text)}</div>')

        # Raw HTML tables — check if stub, possibly replace
        elif "<table" in stripped:
            table_html = stripped
            tr_count = stripped.lower().count("<tr")

            # If VLM table is a stub (≤2 rows) and we have a matching
            # Structure table with more rows, use the Structure version
            if tr_count <= 2 and vlm_table_idx < len(sorted_struct):
                struct_html = sorted_struct[vlm_table_idx].get("html", "")
                struct_tr_count = struct_html.lower().count("<tr")
                if struct_tr_count > tr_count:
                    logger.info(
                        "Table %d: VLM has %d rows (stub), Structure has %d — using Structure HTML",
                        vlm_table_idx, tr_count, struct_tr_count,
                    )
                    table_html = struct_html

            parts.append('<div class="table-section">')
            parts.append(_enhance_table_html(table_html))
            parts.append("</div>")
            vlm_table_idx += 1

        # Footnotes / small text
        elif stripped.startswith("*"):
            parts.append(f'<p class="footnote">{_esc(stripped)}</p>')

        # Regular text paragraphs
        else:
            parts.append(f"<p>{_esc(stripped)}</p>")

    # Add footer texts the VLM missed (management number, print info, etc.)
    if footer_texts:
        parts.append('<div class="doc-footer">')
        for txt in footer_texts:
            parts.append(f'<p class="footer-text">{_esc(txt)}</p>')
        parts.append("</div>")

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
    max-width: none;
    margin: 0 auto;
    background: #fff;
    border: 2px solid #000;
    padding: 28px 30px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,.12);
    overflow-x: auto;
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

/* ── Form header/footer text (regulation refs, print info) ── */
.form-header-text {
    font-size: 9px;
    color: #444;
    margin: 2px 0;
    line-height: 1.3;
}

.doc-footer {
    margin-top: 20px;
    padding-top: 8px;
    border-top: 1px solid #ccc;
}

.footer-text {
    font-size: 9px;
    color: #555;
    margin: 2px 0;
    line-height: 1.3;
}

/* ── Footnotes / small text ───────────────────────────────── */
.footnote {
    font-size: 9px;
    color: #555;
    margin: 6px 0;
    line-height: 1.4;
}

p {
    font-size: 11px;
    margin: 4px 0;
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
