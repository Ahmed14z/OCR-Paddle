import numpy as np
import fitz  # pymupdf
from PIL import Image
import io


def pdf_pages_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[np.ndarray]:
    """Convert each page of a PDF to a numpy RGB array."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[np.ndarray] = []

    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(np.array(img))

    doc.close()
    return images


def image_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to numpy RGB array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)
