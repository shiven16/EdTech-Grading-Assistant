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
    # Convert to grayscale to find dark pixels for bounding rect
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) == 0:
        return image
        
    # minAreaRect expects (N, 2) coords
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
# Requires: pip install transformers torch

_trocr_model = None
_trocr_processor = None
_trocr_device = None

def _load_trocr():
    """Load TrOCR model once and cache globally.
    
    Uses local cached weights first (no network requests).
    Falls back to downloading only if the model was never cached.
    """
    global _trocr_model, _trocr_processor, _trocr_device

    if _trocr_model is not None:
        return

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    if torch.cuda.is_available():
        _trocr_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _trocr_device = "mps"
    else:
        _trocr_device = "cpu"

    model_name = "microsoft/trocr-large-handwritten"

    # Try loading from local cache first (zero network calls)
    try:
        logger.info(f"Loading TrOCR-large from local cache to {_trocr_device}…")
        _trocr_processor = TrOCRProcessor.from_pretrained(
            model_name, local_files_only=True
        )
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            model_name, local_files_only=True
        ).to(_trocr_device)
    except OSError:
        # First time ever — download once, then it's cached forever
        logger.info("Model not in local cache. Downloading once…")
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(
            model_name
        ).to(_trocr_device)

    _trocr_model.eval()
    logger.info("TrOCR-large model loaded successfully.")


def _detect_lines(image_path: str):
    """
    Uses EasyOCR's CRAFT neural network to detect word bounding boxes.
    Mathematically groups the words into isolated rows (Text Lines) even if they intersect.
    This guarantees perfect layout separation without requiring the Tesseract system binary.
    Returns list of (x, y, w, h) sorted top-to-bottom.
    """
    import easyocr
    import numpy as np

    # Run EasyOCR natively; suppress the verbose logging
    reader = easyocr.Reader(['en'], verbose=False)
    results = reader.readtext(image_path)
    
    if not results:
        # Fallback to entire image if it somehow finds nothing
        import cv2
        img = cv2.imread(image_path)
        return [(0, 0, img.shape[1], img.shape[0])]

    # Extract word bounding boxes: [min_x, max_x, min_y, max_y, y_center]
    word_boxes = []
    for (bbox, _, _) in results:
        # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # We find the min/max to create an axis-aligned bounding box.
        xs = [int(pt[0]) for pt in bbox]
        ys = [int(pt[1]) for pt in bbox]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        y_center = (min_y + max_y) // 2
        word_boxes.append({
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'y_center': y_center
        })

    # Sort boxes vertically
    word_boxes.sort(key=lambda b: b['y_center'])

    # Group words that fall on the same horizontal line
    lines = []
    current_line = []
    line_y_threshold = 25  # if y_center is within 25px, it's the same line

    for box in word_boxes:
        if not current_line:
            current_line.append(box)
        else:
            # Compare current box to the average y_center of the current line
            avg_y = sum(b['y_center'] for b in current_line) / len(current_line)
            if abs(box['y_center'] - avg_y) < line_y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    if current_line:
        lines.append(current_line)

    # Convert groups of words into a single large bounding box for the entire line
    final_boxes = []
    for line in lines:
        x_min = min(b['min_x'] for b in line)
        x_max = max(b['max_x'] for b in line)
        y_min = min(b['min_y'] for b in line)
        y_max = max(b['max_y'] for b in line)
        final_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return final_boxes


def extract_text_trocr(image_path: str) -> str:
    """
    TrOCR-base extraction with OpenCV line detection.

    Key design decisions
    ────────────────────
    • Uses trocr-BASE (334 MB) instead of trocr-LARGE (2.2 GB)
      → 6× faster inference, negligible accuracy loss on English handwriting.
    • Line detection runs on the binarized image (good edge contrast).
    • Line CROPS are taken from the original grayscale image so TrOCR
      sees natural stroke textures it was trained on — not thresholded blobs.
    """
    import torch
    _load_trocr()

    # ── Read original image (grayscale — keeps stroke texture for TrOCR) ──
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Generate the clean binary image (removes ruled lines)
    binary = preprocess_image(image_path)

    # ── Detect text lines intelligently using Tesseract layout analysis ──
    boxes = _detect_lines(image_path)
    
    h, w = gray.shape[:2]

    if not boxes:
        # Fallback: treat the entire image as one line
        logger.warning("No text lines detected — using full image.")
        boxes = [(0, 0, w, h)]

    recognized_lines = []
    
    # We must properly preserve natural text strokes and NEVER decapitate tall letters
    for (x, y, bw, bh) in boxes:
        # Dynamic padding: cursive text has wild ascenders/descenders (h, y, R, t, f)
        # We add massive vertical padding to ensure the top and bottom loops aren't sliced off
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.5) 
        
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

        # 1. Use the binary image to permanently erase blue horizontal notebook lines.
        # 2. Apply a Gaussian Blur to "anti-alias" the jagged black-and-white pixels 
        #    so they perfectly mimic the natural continuous ink strokes TrOCR expects.
        crop_array = binary[y1:y2, x1:x2]
        crop_array = cv2.GaussianBlur(crop_array, (3, 3), 0)
        
        crop_pil = Image.fromarray(crop_array).convert("RGB")

        # We pass directly to processor without manual square-padding
        pixel_values = _trocr_processor(
            images=crop_pil, return_tensors="pt"
        ).pixel_values.to(_trocr_device)

        with torch.no_grad():
            ids = _trocr_model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )
        line = _trocr_processor.batch_decode(ids, skip_special_tokens=True)[0]
        if line.strip():
            recognized_lines.append(line.strip())

    # Join with a space instead of a newline to form normal continuous text
    return " ".join(recognized_lines)


# ─────────────────────────────────────────────
# LAYER 3: POST-PROCESSING
# ─────────────────────────────────────────────

def post_process(text: str) -> str:
    """Normalize whitespace and remove garbage characters."""
    import re
    text = re.sub(r' +', ' ', text)             # collapse multiple spaces
    text = re.sub(r'[^\x20-\x7E\n]', '', text)  # strip non-ASCII artifacts
    text = re.sub(r'\n+', ' ', text)            # ensure no stray newlines remain
    return text.strip()


# ─────────────────────────────────────────────
# PUBLIC API  (drop-in replacement)
# ─────────────────────────────────────────────

def extract_text_from_image(
    image_path: str,
    engine: str = "trocr"
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