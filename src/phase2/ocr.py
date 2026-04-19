import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LAYER 1: PREPROCESSING  (heavily improved)
# ─────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skewed/tilted scans before OCR."""
    coords = np.column_stack(np.where(image < 200))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for handwritten answer sheets.

    Fixes vs original:
    - Deskew:            Original had no skew correction at all
    - CLAHE:             Better contrast than simple grayscale conversion
    - Denoise:           Removes scan noise before thresholding
    - Adaptive thresh:   Global threshold (150) fails on uneven lighting;
                         adaptive handles shadows, ink bleed, worn paper
    - Morph cleanup:     Closes small gaps left by thresholding
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 1 — Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2 — Deskew (missing in original)
    gray = deskew(gray)

    # Step 3 — CLAHE (contrast-limited adaptive histogram equalization)
    # Much better than raw grayscale for faded or low-contrast scans
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 4 — Denoise (must happen BEFORE thresholding)
    # Original had no denoising; scan artifacts corrupt thresholding
    denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21)

    # Step 5 — Adaptive thresholding (replaces fixed threshold of 150)
    # ADAPTIVE_THRESH_GAUSSIAN_C handles uneven lighting across the page
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,   # neighbourhood size — tune for your scan DPI
        C=10            # constant subtracted from mean — tune for ink density
    )

    # Step 6 — Morphological cleanup
    # Closes tiny gaps in characters caused by thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


# ─────────────────────────────────────────────
# LAYER 2A: OCR ENGINE — Tesseract (current)
# ─────────────────────────────────────────────

def extract_text_tesseract(image_path: str) -> str:
    """
    Tesseract-based extraction.

    Fixes vs original:
    - PSM 6:    Assumes a uniform block of text (better for answer sheets)
    - OEM 3:    Uses LSTM engine, not legacy mode
    - Whitelist: Helps with numeric/symbol-heavy answers
    - PIL:      Tesseract handles PIL images more reliably than raw arrays
    """
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)

    config = (
        "--psm 6 "       # Uniform block of text
        "--oem 3 "       # LSTM OCR engine (best accuracy)
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789.,;:?!()/\\-+= "
    )

    text = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


# ─────────────────────────────────────────────
# LAYER 2B: OCR ENGINE — TrOCR (upgrade path)
# ─────────────────────────────────────────────
# Requires: pip install transformers torch paddlepaddle paddleocr

def extract_text_trocr(image_path: str) -> str:
    """
    TrOCR-based extraction with PaddleOCR for line detection.

    Switch to this when Tesseract accuracy is insufficient.
    First run will download ~1.3 GB of model weights.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from paddleocr import PaddleOCR
    import torch

    # ── One-time model load (cache after first call in production) ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    ).to(device)
    model.eval()

    detector = PaddleOCR(use_angle_cls=True, lang="en",
                         rec=False, show_log=False)

    # ── Preprocessing ──
    processed = preprocess_image(image_path)

    # ── Line detection (PaddleOCR handles layout) ──
    result = detector.ocr(processed, rec=False)
    if not result or result[0] is None:
        logger.warning("No text regions detected.")
        return ""

    boxes = sorted(result[0], key=lambda b: b[0][1])  # top → bottom
    h, w = processed.shape[:2]

    # ── Per-line recognition (TrOCR) ──
    recognized_lines = []
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        x1 = max(0, pts[:, 0].min() - 4)
        x2 = min(w, pts[:, 0].max() + 4)
        y1 = max(0, pts[:, 1].min() - 4)
        y2 = min(h, pts[:, 1].max() + 4)

        crop = Image.fromarray(processed[y1:y2, x1:x2]).convert("RGB")
        pixel_values = processor(
            images=crop, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )
        line = processor.batch_decode(ids, skip_special_tokens=True)[0]
        recognized_lines.append(line.strip())

    return "\n".join(recognized_lines)


# ─────────────────────────────────────────────
# LAYER 3: POST-PROCESSING
# ─────────────────────────────────────────────

def post_process(text: str) -> str:
    """Normalize whitespace and remove garbage characters."""
    import re
    text = re.sub(r' {2,}', ' ', text)          # collapse multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)       # collapse blank lines
    text = re.sub(r'[^\x20-\x7E\n]', '', text)  # strip non-ASCII artifacts
    return text.strip()


# ─────────────────────────────────────────────
# PUBLIC API  (drop-in replacement)
# ─────────────────────────────────────────────

def extract_text_from_image(
    image_path: str,
    engine: str = "tesseract"   # swap to "trocr" when ready
) -> str:
    """
    Main entry point.

    Args:
        image_path: Path to scanned answer sheet.
        engine:     "tesseract" (default, no extra deps)
                    "trocr"     (higher accuracy, GPU recommended)
    """
    logger.info(f"Processing: {image_path} | engine={engine}")

    if engine == "tesseract":
        raw = extract_text_tesseract(image_path)
    elif engine == "trocr":
        raw = extract_text_trocr(image_path)
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'tesseract' or 'trocr'.")

    return post_process(raw)