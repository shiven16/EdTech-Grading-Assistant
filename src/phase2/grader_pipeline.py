"""
Phase 2 Grading Pipeline
OCR (Tesseract/TrOCR) → BERT Regression → Score

Usage:
    from src.phase2.grader_pipeline import grade_with_bert, is_ready
"""

from pathlib import Path
import sys

# ── Ensure BERT_method is importable ─────────────────────────────────────────
BERT_DIR = Path(__file__).resolve().parent / "BERT_method"
sys.path.insert(0, str(BERT_DIR))

from ocr import extract_text_from_image
import inference as bert_inference


def is_ready() -> bool:
    """Return True if the trained BERT model weights exist."""
    return bert_inference.is_model_ready()


def grade_with_bert(
    image_path: str,
    question: str,
    reference_answer: str,
    max_marks: float = 10.0,
    ocr_engine: str = "trocr",
) -> dict:
    """
    Grade a handwritten answer sheet using OCR + BERT.

    Args:
        image_path:       Path to uploaded image file.
        question:         Exam question text.
        reference_answer: Reference / model answer.
        max_marks:        Max marks awarded for this question.
        ocr_engine:       "tesseract" or "trocr".

    Returns:
        {
            "extracted_text": str,
            "score":          float,
            "max_marks":      float,
            "percentage":     float,
            "model_ready":    bool,
        }
    """
    if not is_ready():
        raise RuntimeError(
            "BERT model is not trained yet. "
            "Run: python src/phase2/BERT_method/train.py"
        )

    # Step 1 – OCR
    extracted_text = extract_text_from_image(image_path, engine=ocr_engine)

    # Step 2 – BERT scoring
    result = bert_inference.predict(
        question=question,
        ref_answer=reference_answer,
        student_answer=extracted_text,
        max_marks=max_marks,
    )

    return {
        "extracted_text": extracted_text,
        "score":          result["score"],
        "max_marks":      result["max_marks"],
        "percentage":     result["percentage"],
        "model_ready":    True,
    }


def grade_text_with_bert(
    student_answer: str,
    question: str,
    reference_answer: str,
    max_marks: float = 10.0,
) -> dict:
    """
    Grade a *typed* answer (no image) using BERT only.

    Useful for testing or when OCR is not needed.
    """
    if not is_ready():
        raise RuntimeError("BERT model is not trained yet.")

    result = bert_inference.predict(
        question=question,
        ref_answer=reference_answer,
        student_answer=student_answer,
        max_marks=max_marks,
    )
    return {
        "extracted_text": student_answer,
        "score":          result["score"],
        "max_marks":      result["max_marks"],
        "percentage":     result["percentage"],
        "model_ready":    True,
    }
