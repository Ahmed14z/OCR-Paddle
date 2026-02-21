"""LLM-based post-correction for OCR output via OpenRouter."""

import logging
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

    logger.info("Sending OCR text to LLM for Korean correction (%d chars)...", len(ocr_markdown))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": ocr_markdown},
                    ],
                    "temperature": 0.0,
                    "max_tokens": len(ocr_markdown) * 2,
                },
            )
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
