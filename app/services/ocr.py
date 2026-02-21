"""Dual OCR engine: PP-StructureV3 (tables/digits) + PaddleOCR-VL (layout/semantics)."""

import logging
import os
import time
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger("ocr.engine")

# Suppress paddle model source check on startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


class OCREngine:
    """Singleton dual OCR engine."""

    _instance: "OCREngine | None" = None

    @classmethod
    def get(cls) -> "OCREngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        from paddleocr import PPStructureV3

        logger.info("Initializing PP-StructureV3 (lang=%s, device=%s)...", settings.ocr_lang, settings.ocr_device)
        t0 = time.time()

        self.structure = PPStructureV3(
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            use_table_recognition=True,
            use_doc_orientation_classify=True,
            use_region_detection=True,
            use_doc_unwarping=False,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            use_textline_orientation=False,
            device=settings.ocr_device,
        )
        logger.info("PP-StructureV3 ready in %.1fs", time.time() - t0)

        # PaddleOCR-VL will be initialized here when available
        self.vlm = None
        self._try_init_vlm()

    def _try_init_vlm(self) -> None:
        """Try to initialize PaddleOCR-VL-1.5 (optional, GPU-heavy)."""
        try:
            from paddleocr import PaddleOCRVL

            logger.info("Initializing PaddleOCR-VL-1.5...")
            t0 = time.time()
            self.vlm = PaddleOCRVL(device=settings.ocr_device, use_queues=False)
            logger.info("PaddleOCR-VL-1.5 ready in %.1fs", time.time() - t0)
        except (ImportError, Exception) as e:
            logger.warning("PaddleOCR-VL not available: %s — using PP-StructureV3 only", e)
            self.vlm = None

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        """Run dual OCR on an image and return merged results."""
        t0 = time.time()

        # Engine 1: PP-StructureV3 (deterministic tables + digits)
        structure_result = self._run_structure(image)

        # Engine 2: PaddleOCR-VL (semantic layout — optional)
        vlm_result = self._run_vlm(image) if self.vlm else None

        elapsed = time.time() - t0
        logger.info("Total OCR done in %.1fs", elapsed)

        return {
            "structure": structure_result,
            "vlm": vlm_result,
            "elapsed_s": round(elapsed, 1),
        }

    def _run_structure(self, image: np.ndarray) -> dict[str, Any]:
        """Run PP-StructureV3 and parse its output."""
        logger.info("Running PP-StructureV3...")
        t0 = time.time()

        results = list(self.structure.predict(image))

        tables: list[dict[str, Any]] = []
        text_blocks: list[dict[str, Any]] = []
        markdown = ""

        for res in results:
            raw = res.json
            # PP-StructureV3 nests everything under a "res" key
            inner = raw.get("res", raw)

            # Extract markdown from the result object
            md_data = getattr(res, "markdown", None)
            if isinstance(md_data, dict):
                markdown = md_data.get("markdown_texts", "")
            elif isinstance(md_data, str):
                markdown = md_data
            else:
                markdown = ""

            # Extract tables
            for table in inner.get("table_res_list", []):
                table_entry: dict[str, Any] = {
                    "html": table.get("pred_html", ""),
                    "cell_boxes": table.get("cell_box_list", []),
                }
                ocr_pred = table.get("table_ocr_pred", {})
                if ocr_pred:
                    table_entry["texts"] = ocr_pred.get("rec_texts", [])
                    table_entry["scores"] = ocr_pred.get("rec_scores", [])
                    table_entry["boxes"] = ocr_pred.get("rec_boxes", [])
                tables.append(table_entry)

            # Extract text blocks from overall OCR
            ocr_res = inner.get("overall_ocr_res", {})
            for text, score, box in zip(
                ocr_res.get("rec_texts", []),
                ocr_res.get("rec_scores", []),
                ocr_res.get("rec_boxes", []),
            ):
                text_blocks.append({"text": text, "confidence": score, "box": box})

        elapsed = time.time() - t0
        logger.info(
            "PP-StructureV3: %.1fs | %d tables | %d text blocks",
            elapsed, len(tables), len(text_blocks),
        )
        return {
            "markdown": markdown,
            "tables": tables,
            "text_blocks": text_blocks,
        }

    def _run_vlm(self, image: np.ndarray) -> dict[str, Any] | None:
        """Run PaddleOCR-VL-1.5 and return markdown output."""
        if not self.vlm:
            return None

        logger.info("Running PaddleOCR-VL-1.5...")
        t0 = time.time()

        try:
            # Save to temp file — VL's internal preprocessing has numpy 2.x
            # scalar conversion issues when receiving arrays directly
            import tempfile
            from PIL import Image

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
                Image.fromarray(image).save(f, format="PNG")

            try:
                results = list(self.vlm.predict(tmp_path, use_queues=False))
            finally:
                os.unlink(tmp_path)

            markdown = ""
            for res in results:
                md_data = getattr(res, "markdown", None)
                if isinstance(md_data, dict):
                    markdown = md_data.get("markdown_texts", "")
                elif isinstance(md_data, str):
                    markdown = md_data
                else:
                    markdown = ""
            elapsed = time.time() - t0
            logger.info("PaddleOCR-VL: %.1fs", elapsed)
            return {"markdown": markdown}
        except Exception as e:
            logger.error("PaddleOCR-VL failed: %s", e)
            return None
