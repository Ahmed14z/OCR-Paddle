"""Structured dual-cell table alignment for OCR engine fusion.

Builds a UnifiedTable representation where each cell carries readings from
both PP-StructureV3 (accurate Korean text) and PaddleOCR-VL (correct layout),
runs deterministic pre-filtering, and provides utilities for LLM formatting
and HTML rebuilding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from bs4 import BeautifulSoup

logger = logging.getLogger("ocr.table_align")

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_NUMERIC_RE = re.compile(r"^[\s\d,.\-\u2013\u2014%()]+$")
_GARBAGE_CHARS_RE = re.compile(r"[|\[\]]")


# ── Type definitions ─────────────────────────────────────────


class UnifiedCell(TypedDict):
    """One table cell with both OCR engines' readings."""

    id: str
    row: int
    col: int
    rowspan: int
    colspan: int
    is_header: bool
    struct_text: str
    struct_conf: float
    vlm_text: str
    chosen_text: str
    needs_llm: bool


class UnifiedTable(TypedDict):
    """Complete dual-cell representation of one table from both engines."""

    table_idx: int
    n_rows: int
    n_cols: int
    cells: list[UnifiedCell]
    vlm_html: str


@dataclass
class CellInfo:
    """Parsed cell from an HTML table."""

    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


class StructCellData(TypedDict):
    """Per-cell data extracted from PP-StructureV3 output."""

    cell_idx: int
    cell_box: list[float]
    text: str
    confidence: float
    text_indices: list[int]


# ── VLM table parsing ────────────────────────────────────────


def parse_vlm_tables(vlm_markdown: str) -> list[tuple[str, list[CellInfo]]]:
    """Extract inline HTML tables from VLM markdown.

    Returns list of (raw_html, parsed_cells) tuples, one per table.
    """
    if not vlm_markdown:
        return []

    tables: list[tuple[str, list[CellInfo]]] = []
    for line in vlm_markdown.split("\n"):
        stripped = line.strip()
        if "<table" in stripped:
            cells = _parse_html_table_cells(stripped)
            tables.append((stripped, cells))

    return tables


def _parse_html_table_cells(html: str) -> list[CellInfo]:
    """Parse HTML table into a flat list of cells in document order."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    cells: list[CellInfo] = []
    for tag in table.find_all(["td", "th"]):
        cells.append(
            CellInfo(
                text=tag.get_text(strip=True),
                rowspan=int(tag.get("rowspan", 1)),
                colspan=int(tag.get("colspan", 1)),
                is_header=tag.name == "th",
            )
        )
    return cells


# ── Structure table parsing ──────────────────────────────────


def parse_struct_table(table_data: dict[str, Any]) -> list[StructCellData]:
    """Convert Structure's table output into per-cell data with text assignments.

    Uses IoU + L1 distance matching to assign text detections to cells,
    then sorts cells into reading order (top-to-bottom, left-to-right).
    """
    cell_boxes = table_data.get("cell_boxes", [])
    texts = table_data.get("texts", [])
    scores = table_data.get("scores", [])
    text_boxes = table_data.get("boxes", [])

    if not cell_boxes:
        return []

    cell_to_texts = _match_texts_to_cells(cell_boxes, text_boxes)

    cells: list[StructCellData] = []
    for ci, cbox in enumerate(cell_boxes):
        tis = cell_to_texts.get(ci, [])
        matched_texts = [texts[ti] for ti in tis if ti < len(texts)]
        matched_scores = [scores[ti] for ti in tis if ti < len(scores)]
        cells.append(
            StructCellData(
                cell_idx=ci,
                cell_box=cbox,
                text=" ".join(t for t in matched_texts if t),
                confidence=(
                    sum(matched_scores) / len(matched_scores)
                    if matched_scores
                    else 0.0
                ),
                text_indices=tis,
            )
        )

    # Sort into reading order: cluster by Y into rows, sort each row by X
    cells = _sort_cells_reading_order(cells)
    return cells


def _sort_cells_reading_order(cells: list[StructCellData]) -> list[StructCellData]:
    """Sort struct cells top-to-bottom, left-to-right within rows."""
    if not cells:
        return cells

    # Sort by y_center first
    indexed = sorted(cells, key=lambda c: (c["cell_box"][1] + c["cell_box"][3]) / 2)

    # Cluster into rows by Y overlap
    rows: list[list[StructCellData]] = []
    cur_row: list[StructCellData] = [indexed[0]]
    cur_y = (indexed[0]["cell_box"][1] + indexed[0]["cell_box"][3]) / 2

    for cell in indexed[1:]:
        y_mid = (cell["cell_box"][1] + cell["cell_box"][3]) / 2
        h = cell["cell_box"][3] - cell["cell_box"][1]
        threshold = max(h * 0.5, 15)
        if abs(y_mid - cur_y) < threshold:
            cur_row.append(cell)
            cur_y = sum(
                (c["cell_box"][1] + c["cell_box"][3]) / 2 for c in cur_row
            ) / len(cur_row)
        else:
            rows.append(cur_row)
            cur_row = [cell]
            cur_y = y_mid
    rows.append(cur_row)

    # Sort each row left-to-right
    result: list[StructCellData] = []
    for row in rows:
        row.sort(key=lambda c: c["cell_box"][0])
        result.extend(row)

    return result


# ── Cell ID assignment ───────────────────────────────────────


def assign_cell_ids(cells: list[CellInfo]) -> list[tuple[str, int, int]]:
    """Assign EASE-style cell IDs by walking the logical grid.

    Tracks a 2D occupancy grid accounting for rowspan/colspan.
    Returns (cell_id, row, col) per cell in same order as input.
    """
    n_rows, n_cols = _compute_grid_dimensions(cells)
    occupied: set[tuple[int, int]] = set()
    assignments: list[tuple[str, int, int]] = []

    current_row = 0
    current_col = 0

    for cell in cells:
        # Find next unoccupied position
        while (current_row, current_col) in occupied:
            current_col += 1
            if current_col >= n_cols:
                current_col = 0
                current_row += 1

        row, col = current_row, current_col

        # Mark occupied positions for this cell's span
        for dr in range(cell.rowspan):
            for dc in range(cell.colspan):
                occupied.add((row + dr, col + dc))

        cell_id = f"{_col_letter(col)}{row}"
        assignments.append((cell_id, row, col))

        # Advance column pointer
        current_col += cell.colspan

    return assignments


def _compute_grid_dimensions(cells: list[CellInfo]) -> tuple[int, int]:
    """Compute (n_rows, n_cols) of the logical grid from cell spans."""
    if not cells:
        return (0, 0)

    occupied: set[tuple[int, int]] = set()
    current_row = 0
    current_col = 0
    max_row = 0
    max_col = 0

    # First pass: estimate n_cols from first row
    # (we need an upper bound to know when to wrap)
    # Use a generous estimate: sum of colspans in all cells
    est_cols = sum(c.colspan for c in cells)

    for cell in cells:
        while (current_row, current_col) in occupied:
            current_col += 1
            if current_col >= est_cols:
                current_col = 0
                current_row += 1

        row, col = current_row, current_col
        for dr in range(cell.rowspan):
            for dc in range(cell.colspan):
                occupied.add((row + dr, col + dc))
                max_row = max(max_row, row + dr)
                max_col = max(max_col, col + dc)

        current_col += cell.colspan

    return (max_row + 1, max_col + 1)


def _col_letter(col: int) -> str:
    """Convert 0-based column index to Excel-style letter(s): 0->A, 25->Z, 26->AA."""
    result = ""
    c = col
    while True:
        result = chr(ord("A") + c % 26) + result
        c = c // 26 - 1
        if c < 0:
            break
    return result


# ── Table matching ───────────────────────────────────────────


def match_tables(
    vlm_tables: list[tuple[str, list[CellInfo]]],
    struct_tables: list[dict[str, Any]],
) -> list[tuple[int, int | None]]:
    """Match VLM tables to Structure tables by text overlap.

    Uses Jaccard similarity of cell text sets with forward scanning.
    """
    pairs: list[tuple[int, int | None]] = []
    si = 0

    for vi, (_, vlm_cells) in enumerate(vlm_tables):
        vlm_texts = {c.text.strip() for c in vlm_cells if c.text.strip()}
        best_match: int | None = None
        best_score = 0.0

        for cand_si in range(si, min(si + 4, len(struct_tables))):
            st = struct_tables[cand_si]
            struct_texts = {
                t.strip() for t in st.get("texts", []) if t.strip()
            }

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


# ── Build unified table ──────────────────────────────────────


def build_unified_table(
    vlm_html: str,
    vlm_cells: list[CellInfo],
    struct_table_data: dict[str, Any] | None,
    table_idx: int,
) -> UnifiedTable:
    """Build a UnifiedTable from VLM and Structure data for one table.

    Uses VLM HTML for structure (rowspan/colspan), populates struct_text
    from Structure's cell_to_texts mapping, runs pre-filtering.
    """
    # Step 1: Assign IDs from VLM structure
    ids = assign_cell_ids(vlm_cells)
    n_rows, n_cols = _compute_grid_dimensions(vlm_cells)

    # Step 2: Parse Structure data if available
    struct_cells: list[StructCellData] = []
    if struct_table_data is not None:
        struct_cells = parse_struct_table(struct_table_data)

    # Step 3: Align VLM cells to Structure cells via text similarity
    alignment = _align_cells(vlm_cells, struct_cells)

    # Step 4: Build UnifiedCell list
    cells: list[UnifiedCell] = []
    for i, vc in enumerate(vlm_cells):
        cell_id, row, col = ids[i]
        matched_struct = alignment.get(i)

        cells.append(
            UnifiedCell(
                id=cell_id,
                row=row,
                col=col,
                rowspan=vc.rowspan,
                colspan=vc.colspan,
                is_header=vc.is_header,
                struct_text=matched_struct["text"] if matched_struct else "",
                struct_conf=(
                    matched_struct["confidence"] if matched_struct else 0.0
                ),
                vlm_text=vc.text,
                chosen_text="",
                needs_llm=False,
            )
        )

    # Step 5: Pre-filter
    pre_filter_cells(cells)

    logger.info(
        "Table %d: %dx%d, %d cells, %d need LLM review",
        table_idx,
        n_rows,
        n_cols,
        len(cells),
        sum(1 for c in cells if c["needs_llm"]),
    )

    return UnifiedTable(
        table_idx=table_idx,
        n_rows=n_rows,
        n_cols=n_cols,
        cells=cells,
        vlm_html=vlm_html,
    )


def _align_cells(
    vlm_cells: list[CellInfo],
    struct_cells: list[StructCellData],
) -> dict[int, StructCellData]:
    """Align VLM cells to Structure cells using greedy text similarity.

    Returns dict mapping vlm_cell_index -> matched StructCellData.
    """
    if not struct_cells:
        return {}

    alignment: dict[int, StructCellData] = {}
    si = 0

    for vi, vc in enumerate(vlm_cells):
        vtext = vc.text.strip()
        if not vtext:
            # Skip empty VLM cells, advance past empty struct cells
            while si < len(struct_cells) and not struct_cells[si]["text"].strip():
                si += 1
            continue

        best_si: int | None = None
        best_sim = 0.0
        for cand in range(si, min(si + 10, len(struct_cells))):
            sim = _text_sim(vtext, struct_cells[cand]["text"])
            if sim > best_sim:
                best_sim = sim
                best_si = cand

        if best_si is not None and best_sim > 0.15:
            alignment[vi] = struct_cells[best_si]
            si = best_si + 1

    return alignment


# ── Pre-filtering ────────────────────────────────────────────


def pre_filter_cells(cells: list[UnifiedCell]) -> list[UnifiedCell]:
    """Set chosen_text and needs_llm for each cell using deterministic rules.

    Rules (first match wins):
    1. Both empty -> chosen="", needs_llm=False
    2. Both agree -> chosen=struct_text, needs_llm=False
    3. One empty -> chosen=non-empty, needs_llm=False
    4. Both numeric and match after cleanup -> chosen=vlm_text, needs_llm=False
    5. Structure has garbage chars (|,[,]) -> chosen=vlm_text, needs_llm=True
    6. High confidence + high similarity -> chosen=struct_text, needs_llm=False
    7. Low confidence -> chosen=vlm_text, needs_llm=True
    8. Korean text disagrees -> chosen=struct_text, needs_llm=True
    9. Default -> chosen=struct_text if conf>=0.5 else vlm_text, needs_llm=True
    """
    n_auto = 0
    n_llm = 0

    for cell in cells:
        st = cell["struct_text"].strip()
        vt = cell["vlm_text"].strip()
        conf = cell["struct_conf"]

        # Normalize for comparison
        st_norm = _normalize(st)
        vt_norm = _normalize(vt)

        # Rule 1: Both empty
        if not st and not vt:
            cell["chosen_text"] = ""
            cell["needs_llm"] = False
            n_auto += 1
            continue

        # Rule 2: Both agree
        if st_norm == vt_norm and st_norm:
            cell["chosen_text"] = st
            cell["needs_llm"] = False
            n_auto += 1
            continue

        # Rule 3: One empty
        if not st:
            cell["chosen_text"] = vt
            cell["needs_llm"] = False
            n_auto += 1
            continue
        if not vt:
            cell["chosen_text"] = st
            cell["needs_llm"] = False
            n_auto += 1
            continue

        # Rule 4: Both numeric, match after cleanup
        st_clean = _clean_numeric(st)
        vt_clean = _clean_numeric(vt)
        if (
            _NUMERIC_RE.match(st_clean)
            and _NUMERIC_RE.match(vt_clean)
            and _strip_all_spaces(st_clean) == _strip_all_spaces(vt_clean)
        ):
            cell["chosen_text"] = vt  # VLM has better digit grouping
            cell["needs_llm"] = False
            n_auto += 1
            continue

        # Rule 5: Structure has garbage chars
        if _GARBAGE_CHARS_RE.search(st):
            cell["chosen_text"] = vt
            cell["needs_llm"] = True
            n_llm += 1
            continue

        # Rule 6: High confidence + high similarity
        if conf >= 0.9 and _text_sim(st, vt) >= 0.8:
            cell["chosen_text"] = st
            cell["needs_llm"] = False
            n_auto += 1
            continue

        # Rule 7: Low confidence
        if conf < 0.85:
            cell["chosen_text"] = vt
            cell["needs_llm"] = True
            n_llm += 1
            continue

        # Rule 8: Korean text disagrees
        if _HANGUL_RE.search(st) or _HANGUL_RE.search(vt):
            cell["chosen_text"] = st  # OCRv5 generally better for Korean
            cell["needs_llm"] = True
            n_llm += 1
            continue

        # Rule 9: Default
        cell["chosen_text"] = st if conf >= 0.5 else vt
        cell["needs_llm"] = True
        n_llm += 1

    logger.info(
        "Pre-filter: %d auto-resolved, %d need LLM review (of %d total)",
        n_auto,
        n_llm,
        len(cells),
    )
    return cells


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", s.lower().strip())


def _clean_numeric(s: str) -> str:
    """Remove garbage chars from a potentially numeric string."""
    return _GARBAGE_CHARS_RE.sub("", s)


def _strip_all_spaces(s: str) -> str:
    """Remove all whitespace for numeric comparison."""
    return re.sub(r"\s", "", s)


# ── LLM formatting ──────────────────────────────────────────


def format_cells_for_llm(table: UnifiedTable) -> str:
    """Build compact text representation of cells needing LLM review.

    Format per cell:
        Cell {idx} [{row},{col}] {th|td}: S="{struct}" c={conf:.2f}  V="{vlm}"

    Returns empty string if no cells need review.
    """
    ambiguous = [
        (i, c) for i, c in enumerate(table["cells"]) if c["needs_llm"]
    ]

    if not ambiguous:
        return ""

    lines: list[str] = [
        f"Table {table['table_idx']}: {table['n_rows']}x{table['n_cols']}, "
        f"{len(ambiguous)} cells need review:"
    ]

    for idx, cell in ambiguous:
        tag = "th" if cell["is_header"] else "td"
        st = _truncate(cell["struct_text"], 100)
        vt = _truncate(cell["vlm_text"], 100)
        lines.append(
            f'Cell {idx} [{cell["row"]},{cell["col"]}] {tag}: '
            f'S="{st}" c={cell["struct_conf"]:.2f}  V="{vt}"'
        )

    return "\n".join(lines)


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if too long."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ── Apply LLM corrections ───────────────────────────────────


def apply_llm_corrections(
    table: UnifiedTable,
    corrections: list[dict[str, Any]],
) -> UnifiedTable:
    """Apply validated LLM patches to the unified table.

    Each correction: {"id": int, "text": str}
    Validates bounds, needs_llm flag, non-empty text, length sanity.
    """
    cells = table["cells"]
    n_applied = 0
    n_skipped = 0

    for corr in corrections:
        cell_id = corr.get("id")
        text = corr.get("text")

        if not isinstance(cell_id, int) or not isinstance(text, str):
            logger.warning("Invalid correction format: %s", corr)
            n_skipped += 1
            continue

        if cell_id < 0 or cell_id >= len(cells):
            logger.warning("Correction id %d out of range [0, %d)", cell_id, len(cells))
            n_skipped += 1
            continue

        cell = cells[cell_id]
        if not cell["needs_llm"]:
            logger.warning(
                "Correction for cell %d (%s) but it was auto-resolved, skipping",
                cell_id,
                cell["id"],
            )
            n_skipped += 1
            continue

        if not text.strip():
            logger.warning("Empty correction text for cell %d, skipping", cell_id)
            n_skipped += 1
            continue

        # Length sanity: correction shouldn't be absurdly longer than both sources
        max_src_len = max(len(cell["struct_text"]), len(cell["vlm_text"]), 1)
        if len(text) > max_src_len * 3:
            logger.warning(
                "Correction for cell %d too long (%d vs max source %d), skipping",
                cell_id,
                len(text),
                max_src_len,
            )
            n_skipped += 1
            continue

        cell["chosen_text"] = text
        cell["needs_llm"] = False
        n_applied += 1

    logger.info(
        "Table %d: applied %d/%d LLM corrections (%d skipped)",
        table["table_idx"],
        n_applied,
        len(corrections),
        n_skipped,
    )
    return table


# ── HTML rebuild ─────────────────────────────────────────────


def rebuild_table_html(table: UnifiedTable) -> str:
    """Rebuild the VLM HTML table with corrected cell texts.

    Replaces each cell's text content with chosen_text while
    preserving all HTML attributes (rowspan, colspan, class, style).
    """
    try:
        soup = BeautifulSoup(table["vlm_html"], "html.parser")
        html_table = soup.find("table")
        if not html_table:
            return table["vlm_html"]

        tags = html_table.find_all(["td", "th"])
        cells = table["cells"]

        for i, tag in enumerate(tags):
            if i < len(cells):
                tag.string = cells[i]["chosen_text"]

        return str(soup)
    except Exception as e:
        logger.error("Failed to rebuild table HTML: %s", e)
        return table["vlm_html"]


# ── Geometry helpers ─────────────────────────────────────────


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
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
    """L1 distance between two [x1, y1, x2, y2] box corners."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])


def _match_texts_to_cells(
    cell_boxes: list[list[float]],
    text_boxes: list[list[float]],
) -> dict[int, list[int]]:
    """Assign each text detection to its best-matching cell by IoU + distance."""
    if not cell_boxes or not text_boxes:
        return {}

    matched: dict[int, list[int]] = {}

    for ti, tbox in enumerate(text_boxes):
        best_ci = -1
        best_score: tuple[float, float] = (float("inf"), float("inf"))
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
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    sa, sb = bg(a), bg(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
