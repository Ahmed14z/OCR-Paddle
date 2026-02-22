"""LLM-based post-correction for OCR output via OpenRouter."""

import json as json_mod
import logging
import re
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("ocr.llm_correct")

_SYSTEM_PROMPT = """\
You are an OCR error corrector for Korean documents. You receive OCR-extracted text \
(markdown with inline HTML tables) that may contain character-level errors from a \
vision model.

Your job:
1. Fix Korean character errors. Common OCR confusions include characters that differ \
by one jamo component (e.g. 십억↔심익, 합계↔함계, 사업자↔사업지, 처별↔처벌, \
십↔심, 합↔함). Use surrounding context to determine the correct character.
2. Do NOT change numbers, table structure (HTML tags, rowspan, colspan), or layout.
3. Do NOT add, remove, or reorder any content.
4. Do NOT add explanations. Return ONLY the corrected text in the exact same format.
5. Preserve all HTML tags, attributes, and markdown formatting exactly as-is.
"""


async def correct_ocr_text(ocr_markdown: str) -> str:
    """Send OCR markdown to an LLM for Korean character error correction.

    Returns the corrected markdown, or the original if LLM is unavailable.
    """
    if not settings.openrouter_api_key:
        logger.warning("No OPENROUTER_API_KEY set — skipping LLM correction")
        return ocr_markdown

    if not ocr_markdown.strip():
        return ocr_markdown

    key_preview = settings.openrouter_api_key[:12] + "..." if len(settings.openrouter_api_key) > 12 else settings.openrouter_api_key
    logger.info(
        "Sending OCR text to LLM (model=%s, key=%s, %d chars)...",
        settings.openrouter_model, key_preview, len(ocr_markdown),
    )

    try:
        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": ocr_markdown},
            ],
            "temperature": 0.0,
            "max_tokens": min(len(ocr_markdown) * 2, 16000),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code != 200:
                logger.error("LLM API error %d: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        corrected = data["choices"][0]["message"]["content"]

        if not corrected or not corrected.strip():
            logger.warning("LLM returned empty response — using original")
            return ocr_markdown

        logger.info("LLM correction done (%d → %d chars)", len(ocr_markdown), len(corrected))
        return corrected

    except Exception as e:
        logger.error("LLM correction failed: %s — using original", e)
        return ocr_markdown


# ── Table cell arbitration ───────────────────────────────────

_TABLE_SYSTEM_PROMPT = """\
You correct OCR errors in Korean tax/financial tables. You receive cell data \
from two OCR engines and must pick or synthesize the correct text for each cell.

## Engines
- S (Structure/OCRv5): Accurate Korean character recognition but sometimes \
garbles multi-digit numbers with stray |, [, ] characters or extra spaces.
- V (VLM/Vision-Language): Clean number formatting and layout, but confuses \
similar Korean characters.

## Common Korean OCR Confusions (fix these)
These characters differ by one jamo and are frequently confused:
- 합 ↔ 함 (ㅂ vs ㅁ final): 합계→함계, 합산→함산
- 십 ↔ 심 (ㅂ vs ㅁ final): 십억→심억/심익
- 별 ↔ 벌 (ㄹ presence): 처별→처벌, 매입처별→매입처벌
- 자 ↔ 지 (ㅏ vs ㅣ vowel): 사업자→사업지
- 계 ↔ 게 (ㅖ vs ㅔ vowel): 합계→합게
- 액 ↔ 엑 (ㅐ vs ㅔ vowel): 금액→금엑, 세액→세엑
- 세 ↔ 새 (ㅔ vs ㅐ vowel)
- 처 ↔ 쳐 (palatalization)

## Rules
1. For numbers: prefer V (cleaner digit grouping). Remove stray |, [, ] from S.
2. For Korean text: prefer S if confidence >= 0.85, else use V. Apply jamo fixes.
3. If both are clearly wrong, synthesize the best text from context.
4. Keep cell text concise — no explanations or extra whitespace.

## Output
Return ONLY a JSON array of patches for cells you want to correct:
[{"id": <cell_index>, "text": "<corrected text>"}]

- id: the Cell index number from the input
- text: the final corrected text
- Only include cells where you are making a change
- If no corrections needed, return: []
- Do NOT wrap in markdown code blocks
- Do NOT add any text before or after the JSON array\
"""


async def correct_table_cells(
    table_prompt: str,
    table_idx: int,
) -> list[dict[str, Any]]:
    """Send formatted cell data to LLM for table arbitration.

    Returns list of {"id": int, "text": str} correction dicts.
    Returns empty list on any failure (graceful degradation).
    """
    if not settings.openrouter_api_key:
        logger.warning("No OPENROUTER_API_KEY — skipping table correction")
        return []

    if not table_prompt.strip():
        return []

    logger.info(
        "Table %d: sending %d chars to LLM (model=%s)...",
        table_idx,
        len(table_prompt),
        settings.table_correction_model,
    )
    t0 = time.time()

    try:
        payload: dict[str, Any] = {
            "model": settings.table_correction_model,
            "messages": [
                {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                {"role": "user", "content": table_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max(512, len(table_prompt)),
        }

        async with httpx.AsyncClient(
            timeout=float(settings.table_correction_timeout),
        ) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code != 200:
                logger.error(
                    "Table %d: LLM API error %d: %s",
                    table_idx,
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        response_text: str = data["choices"][0]["message"]["content"]
        corrections = parse_llm_table_response(response_text)

        elapsed = time.time() - t0
        logger.info(
            "Table %d: LLM returned %d corrections in %.1fs",
            table_idx,
            len(corrections),
            elapsed,
        )
        return corrections

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            "Table %d: LLM correction failed after %.1fs: %s",
            table_idx,
            elapsed,
            e,
        )
        return []


def parse_llm_table_response(response_text: str) -> list[dict[str, Any]]:
    """Extract and validate JSON correction array from LLM response.

    Handles markdown code blocks, leading/trailing text, malformed items.
    Returns validated list of {"id": int, "text": str} dicts.
    """
    if not response_text or not response_text.strip():
        return []

    text = response_text.strip()

    # Strip markdown code block wrapper if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Find outermost [ ... ] pair
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("No JSON array found in LLM table response")
        return []

    json_str = text[start : end + 1]

    try:
        parsed = json_mod.loads(json_str)
    except json_mod.JSONDecodeError as e:
        logger.warning("Failed to parse LLM table response JSON: %s", e)
        return []

    if not isinstance(parsed, list):
        logger.warning("LLM table response is not a JSON array")
        return []

    valid: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        cell_id = item.get("id")
        cell_text = item.get("text")
        if not isinstance(cell_id, int):
            continue
        if not isinstance(cell_text, str):
            continue
        valid.append({"id": cell_id, "text": cell_text})

    return valid
