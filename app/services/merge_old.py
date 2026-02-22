"""Merge PP-StructureV3 (accurate Korean text) with PaddleOCR-VL (correct table structure)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger("ocr.merge")

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_NUMERIC_RE = re.compile(r"^[\s\d,.\-\u2013\u2014%()]+$")


# ── Data types ────────────────────────────────────────────────


@dataclass
class CellInfo:
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass
class VLMTable:
    raw_html: str
    flat_cells: list[CellInfo]


@dataclass
class StructTable:
    html: str
    cell_boxes: list[list[float]]
    texts: list[str]
    scores: list[float]
    boxes: list[list[float]]
    y_center: float
    cell_to_texts: dict[int, list[int]]


# ── Public API ────────────────────────────────────────────────


def merge_ocr_results(
    structure: dict[str, Any],
    vlm: dict[str, Any] | None,
) -> str:
    """Merge Structure + VLM into a single markdown string.

    Uses VLM markdown as the structural skeleton, substituting Korean
    text from OCRv5 where it is more accurate.  Falls back to structure
    markdown when VLM is unavailable.
    """
    vlm_md = (vlm or {}).get("markdown", "")
    struct_md = structure.get("markdown", "")

    if not vlm_md:
        return struct_md

    # Parse VLM markdown into lines, extract table HTML blocks
    vlm_tables = _extract_tables_from_markdown(vlm_md)
    struct_tables = _prepare_struct_tables(structure)

    if not struct_tables or not vlm_tables:
        return vlm_md

    # Match tables by order + text similarity
    pairs = _match_tables(vlm_tables, struct_tables)

    # Merge each matched pair
    merged_html: dict[int, str] = {}
    for vlm_idx, struct_idx in pairs:
        if struct_idx is not None:
            merged = _merge_table(vlm_tables[vlm_idx], struct_tables[struct_idx])
            merged_html[vlm_idx] = merged

    # Rebuild markdown, replacing VLM table HTML with merged HTML
    return _rebuild_markdown(vlm_md, vlm_tables, merged_html)


# ── VLM parsing ──────────────────────────────────────────────


def _extract_tables_from_markdown(md: str) -> list[VLMTable]:
    """Find all inline HTML <table> blocks in the VLM markdown."""
    tables: list[VLMTable] = []
    for line in md.split("\n"):
        stripped = line.strip()
        if "<table" in stripped:
            cells = _parse_html_table_cells(stripped)
            tables.append(VLMTable(raw_html=stripped, flat_cells=cells))
    return tables


def _parse_html_table_cells(html: str) -> list[CellInfo]:
    """Parse HTML table into a flat list of cells in document order."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    cells: list[CellInfo] = []
    for tag in table.find_all(["td", "th"]):
        cells.append(CellInfo(
            text=tag.get_text(strip=True),
            rowspan=int(tag.get("rowspan", 1)),
            colspan=int(tag.get("colspan", 1)),
            is_header=tag.name == "th",
        ))
    return cells


# ── Structure table preparation ──────────────────────────────


def _prepare_struct_tables(structure: dict[str, Any]) -> list[StructTable]:
    """Convert raw structure output into sorted StructTable objects."""
    tables: list[StructTable] = []

    for raw in structure.get("tables", []):
        cell_boxes = raw.get("cell_boxes", [])
        texts = raw.get("texts", [])
        scores = raw.get("scores", [])
        text_boxes = raw.get("boxes", [])

        if cell_boxes:
            y_center = sum((b[1] + b[3]) / 2 for b in cell_boxes) / len(cell_boxes)
        else:
            y_center = float("inf")

        cell_to_texts = _match_texts_to_cells(cell_boxes, text_boxes)

        tables.append(StructTable(
            html=raw.get("html", ""),
            cell_boxes=cell_boxes,
            texts=texts,
            scores=scores,
            boxes=text_boxes,
            y_center=y_center,
            cell_to_texts=cell_to_texts,
        ))

    tables.sort(key=lambda t: t.y_center)
    return tables


def _match_texts_to_cells(
    cell_boxes: list[list[float]],
    text_boxes: list[list[float]],
) -> dict[int, list[int]]:
    """Assign each text detection to its best-matching cell (IoU + distance)."""
    if not cell_boxes or not text_boxes:
        return {}

    matched: dict[int, list[int]] = {}

    for ti, tbox in enumerate(text_boxes):
        best_ci, best_score = -1, (float("inf"), float("inf"))
        for ci, cbox in enumerate(cell_boxes):
            iou = _iou(cbox, tbox)
            dist = _l1_dist(cbox, tbox)
            score = (1.0 - iou, dist)
            if score < best_score:
                best_score = score
                best_ci = ci
        if best_ci >= 0:
            matched.setdefault(best_ci, []).append(ti)

    # Sort texts within each cell by reading order
    for ci in matched:
        matched[ci].sort(key=lambda ti: (text_boxes[ti][1], text_boxes[ti][0]))

    return matched


# ── Table matching ───────────────────────────────────────────


def _match_tables(
    vlm_tables: list[VLMTable],
    struct_tables: list[StructTable],
) -> list[tuple[int, int | None]]:
    """Match VLM tables to structure tables by text overlap."""
    pairs: list[tuple[int, int | None]] = []
    si = 0

    for vi, vt in enumerate(vlm_tables):
        vlm_texts = {c.text.strip() for c in vt.flat_cells if c.text.strip()}
        best_match, best_score = None, 0.0

        for cand_si in range(si, min(si + 4, len(struct_tables))):
            st = struct_tables[cand_si]
            struct_texts = {t.strip() for t in st.texts if t.strip()}

            if not vlm_texts and not struct_texts:
                score = 0.5
            elif not vlm_texts or not struct_texts:
                score = 0.1
            else:
                inter = len(vlm_texts & struct_texts)
                union = len(vlm_texts | struct_texts)
                score = inter / union if union else 0.0

            if score > best_score:
                best_score = score
                best_match = cand_si

        if best_match is not None and best_score > 0.05:
            pairs.append((vi, best_match))
            si = best_match + 1
        else:
            pairs.append((vi, None))

    return pairs


# ── Core merge ───────────────────────────────────────────────


def _merge_table(vlm_table: VLMTable, struct_table: StructTable) -> str:
    """Merge a single VLM table with its matched structure table.

    Keeps VLM HTML structure, substitutes cell text from OCRv5 where
    the Korean text is more accurate.
    """
    # Get structure cells in reading order with their texts
    struct_cells = _struct_cells_in_order(struct_table)

    if not struct_cells:
        return vlm_table.raw_html

    # Greedy forward alignment by text similarity
    replacements: dict[int, str] = {}
    si = 0

    for vi, vc in enumerate(vlm_table.flat_cells):
        vtext = vc.text.strip()
        if not vtext:
            # Skip empty VLM cells, advance struct pointer past empties too
            while si < len(struct_cells) and not struct_cells[si]["text"].strip():
                si += 1
            continue

        # Search forward for best matching struct cell
        best_si, best_sim = None, 0.0
        for cand in range(si, min(si + 10, len(struct_cells))):
            sim = _text_sim(vtext, struct_cells[cand]["text"])
            if sim > best_sim:
                best_sim = sim
                best_si = cand

        if best_si is not None and best_sim > 0.15:
            stext = struct_cells[best_si]["text"].strip()
            conf = struct_cells[best_si]["confidence"]
            chosen = _choose_text(vtext, stext, conf)
            if chosen != vtext:
                replacements[vi] = chosen
            si = best_si + 1

    if not replacements:
        return vlm_table.raw_html

    return _apply_replacements(vlm_table.raw_html, replacements)


def _struct_cells_in_order(st: StructTable) -> list[dict[str, Any]]:
    """Get structure cells in reading order (top-to-bottom, left-to-right)."""
    if not st.cell_boxes:
        return []

    indexed = list(enumerate(st.cell_boxes))
    indexed.sort(key=lambda ib: ib[1][1])  # sort by y_top

    # Cluster into rows by Y overlap
    rows: list[list[tuple[int, list[float]]]] = []
    cur_row: list[tuple[int, list[float]]] = [indexed[0]]
    cur_y = (indexed[0][1][1] + indexed[0][1][3]) / 2

    for ci, box in indexed[1:]:
        y_mid = (box[1] + box[3]) / 2
        h = box[3] - box[1]
        if abs(y_mid - cur_y) < max(h * 0.5, 15):
            cur_row.append((ci, box))
            cur_y = sum((b[1] + b[3]) / 2 for _, b in cur_row) / len(cur_row)
        else:
            rows.append(cur_row)
            cur_row = [(ci, box)]
            cur_y = y_mid
    rows.append(cur_row)

    # Sort each row left-to-right, build output
    result: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda ib: ib[1][0])
        for ci, box in row:
            tis = st.cell_to_texts.get(ci, [])
            texts = [st.texts[ti] for ti in tis]
            scores = [st.scores[ti] for ti in tis]
            result.append({
                "text": " ".join(t for t in texts if t),
                "confidence": sum(scores) / len(scores) if scores else 0.0,
            })

    return result


# ── Text selection ───────────────────────────────────────────


def _choose_text(vlm_text: str, struct_text: str, confidence: float) -> str:
    """Pick the best text between VLM and Structure."""
    if not struct_text:
        return vlm_text
    if not vlm_text:
        return struct_text

    vlm_numeric = bool(_NUMERIC_RE.match(vlm_text))
    struct_numeric = bool(_NUMERIC_RE.match(struct_text))

    # Pure numbers: prefer VLM (better digit grouping)
    if vlm_numeric and struct_numeric:
        return vlm_text

    # Korean text: prefer Structure (OCRv5 is more accurate)
    if _HANGUL_RE.search(vlm_text) or _HANGUL_RE.search(struct_text):
        return struct_text if confidence >= 0.5 else vlm_text

    # Default: prefer Structure if confident
    return struct_text if confidence >= 0.5 else vlm_text


def _text_sim(a: str, b: str) -> float:
    """Character-bigram Jaccard similarity."""
    a = re.sub(r"[\s|,.\[\](){}]", "", a)
    b = re.sub(r"[\s|,.\[\](){}]", "", b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def bg(s: str) -> set[str]:
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    sa, sb = bg(a), bg(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ── HTML rebuilding ──────────────────────────────────────────


def _apply_replacements(html: str, replacements: dict[int, str]) -> str:
    """Replace cell text in HTML table, preserving all attributes."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html

    for idx, tag in enumerate(table.find_all(["td", "th"])):
        if idx in replacements:
            tag.string = replacements[idx]

    return str(soup)


def _rebuild_markdown(
    vlm_md: str,
    vlm_tables: list[VLMTable],
    merged_html: dict[int, str],
) -> str:
    """Replace VLM table HTML in the original markdown with merged HTML."""
    if not merged_html:
        return vlm_md

    result = vlm_md
    for vi, vt in enumerate(vlm_tables):
        if vi in merged_html:
            result = result.replace(vt.raw_html, merged_html[vi], 1)

    return result


# ── Geometry helpers ─────────────────────────────────────────


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _l1_dist(a: list[float], b: list[float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])
