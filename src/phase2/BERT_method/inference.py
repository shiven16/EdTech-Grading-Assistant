"""
BERT Grader — Inference Module
Can be used standalone or imported by grader_pipeline.py
"""

import sys
from pathlib import Path

# ── Make sibling modules importable regardless of CWD ────────────────────────
BERT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BERT_DIR))

import torch
import config

# ── Lazy globals (populated on first call to predict()) ──────────────────────
_model     = None
_tokenizer = None
_device    = None


def is_model_ready() -> bool:
    """Return True if the trained weights file exists."""
    return config.MODEL_PATH.exists()


def _load_model():
    """Load model + tokenizer into globals (idempotent)."""
    global _model, _tokenizer, _device

    if _model is not None:
        return  # already loaded

    if not is_model_ready():
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_PATH}.\n"
            "Run training first: python src/phase2/BERT_method/train.py"
        )

    from transformers import AutoTokenizer
    from model import GradingModel

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    _model     = GradingModel()
    _model.load_state_dict(torch.load(config.MODEL_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()


def predict(question: str, ref_answer: str, student_answer: str, max_marks: float = 10.0) -> dict:
    """
    Grade a single student answer.

    Args:
        question:       The exam question text.
        ref_answer:     The reference / model answer.
        student_answer: The student's answer (from OCR or typed).
        max_marks:      Maximum marks for this question (default 10).

    Returns:
        dict with keys: score (float), max_marks (float), percentage (float)
    """
    _load_model()

    # ── Fallback exactly matching logic (ignore spaces/punctuation/case) ──
    import re
    def normalize_text(t: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', t).lower()
        
    s_norm = normalize_text(student_answer)
    r_norm = normalize_text(ref_answer)
    
    if s_norm == r_norm and s_norm != "":
        return {
            "score":      max_marks,
            "max_marks":  max_marks,
            "percentage": 100.0,
        }

    text = f"{question} [SEP] {ref_answer} [SEP] {student_answer}"

    encoding = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_LEN
    )

    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        output = _model(input_ids, attention_mask)

    normalised = float(output.item())
    normalised = max(0.0, min(1.0, normalised))   # clamp to [0,1]
    score      = round(normalised * max_marks, 2)

    return {
        "score":      score,
        "max_marks":  max_marks,
        "percentage": round(normalised * 100, 1),
    }


if __name__ == "__main__":
    if not is_model_ready():
        print(f"Model not found at {config.MODEL_PATH}")
        print("Run: python src/phase2/BERT_method/train.py")
    else:
        q = "What is evaporation?"
        r = "Evaporation is when liquid turns into gas due to heat."
        s = "Water turns into vapor when heated."
        result = predict(q, r, s)
        print(f"Score: {result['score']} / {result['max_marks']}  ({result['percentage']}%)")