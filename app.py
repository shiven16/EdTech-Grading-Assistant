import os
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import shutil

from src.grader import grade_answer
from src.ocr import extract_text_from_image

app = FastAPI(title="EdTech Grading Assistant")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/grade")
async def grade_submission(
    image: UploadFile = File(...),
    concepts: str = Form(...)
):
    try:
        temp_image_path = f"temp_{image.filename}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        concept_list = json.loads(concepts)
        
        student_answer = extract_text_from_image(temp_image_path)
        
        result = grade_answer(concept_list, student_answer)
        
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        return JSONResponse({
            "success": True,
            "extracted_text": student_answer,
            "result": result,
            "concepts": concept_list
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
