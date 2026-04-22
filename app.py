import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# ── Phase 2 imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "phase2"))
from grader_pipeline import grade_with_bert, grade_text_with_bert, is_ready

# ── Phase 1 import (kept intact) ──────────────────────────────────────────────
try:
    from src.grader import grade_answer as grade_answer_phase1
    from src.ocr import extract_text_from_image as ocr_phase1
    _PHASE1_AVAILABLE = True
except ImportError:
    _PHASE1_AVAILABLE = False


app = FastAPI(title="EdTech Grading Assistant — Phase 2")

os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BERT grading (image upload → OCR → BERT)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/grade/bert")
async def grade_bert(
    image: UploadFile = File(...),
    question: str = Form(...),
    reference_answer: str = Form(...),
    max_marks: float = Form(10.0),
    ocr_engine: str = Form("trocr"),
):
    """Grade a scanned handwritten answer using OCR + BERT."""
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="BERT model not trained yet. Run: python src/phase2/BERT_method/train.py"
        )

    # Save uploaded image to a temp location
    suffix = Path(image.filename).suffix or ".jpg"
    tmp_path = f"uploads/tmp_{image.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        result = grade_with_bert(
            image_path=tmp_path,
            question=question,
            reference_answer=reference_answer,
            max_marks=max_marks,
            ocr_engine=ocr_engine,
        )
        return JSONResponse({"success": True, **result})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/grade/bert/text")
async def grade_bert_text(
    student_answer: str = Form(...),
    question: str = Form(...),
    reference_answer: str = Form(...),
    max_marks: float = Form(10.0),
):
    """Grade a typed (no image) answer using BERT only."""
    if not is_ready():
        raise HTTPException(status_code=503, detail="BERT model not trained yet.")

    try:
        result = grade_text_with_bert(
            student_answer=student_answer,
            question=question,
            reference_answer=reference_answer,
            max_marks=max_marks,
        )
        return JSONResponse({"success": True, **result})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/status")
async def status():
    """Health-check — tells frontend whether BERT model is ready."""
    return JSONResponse({
        "bert_ready": is_ready(),
        "phase1_available": _PHASE1_AVAILABLE,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — keyword/similarity grading (kept intact)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/grade")
async def grade_submission(
    image: UploadFile = File(...),
    concepts: str = Form(...),
):
    """Phase 1: keyword-based grading."""
    if not _PHASE1_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 1 grader not installed.")

    tmp_path = f"uploads/tmp_{image.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        concept_list   = json.loads(concepts)
        student_answer = ocr_phase1(tmp_path)
        result         = grade_answer_phase1(concept_list, student_answer)

        return JSONResponse({
            "success":        True,
            "extracted_text": student_answer,
            "result":         result,
            "concepts":       concept_list,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
