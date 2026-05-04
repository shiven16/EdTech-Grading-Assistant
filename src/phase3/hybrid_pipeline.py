import sys
import pickle
from pathlib import Path

# Fix paths
PHASE3_DIR = Path(__file__).resolve().parent
SRC_DIR = PHASE3_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR / "phase1"))
sys.path.insert(0, str(SRC_DIR / "phase2"))
sys.path.insert(0, str(SRC_DIR / "phase2" / "BERT_method"))

# Phase 1 imports
try:
    from src.phase1.keyword_baseline import concept_match_score
    from src.phase1.tfidf_advanced import concept_similarity_score
    from src.phase1.preprocess import clean_text
except ImportError:
    from keyword_baseline import concept_match_score
    from tfidf_advanced import concept_similarity_score
    from preprocess import clean_text

# Phase 2 imports
from src.phase2.ocr import extract_text_from_image
import src.phase2.BERT_method.inference as bert_inference

_meta_model = None

def is_ready() -> bool:
    """Check if BERT and Meta-model are both ready."""
    meta_path = PROJECT_ROOT / "meta_model.pkl"
    return bert_inference.is_model_ready() and meta_path.exists()

def _load_meta_model():
    global _meta_model
    if _meta_model is not None:
        return
    
    meta_path = PROJECT_ROOT / "meta_model.pkl"
    if not meta_path.exists():
        raise FileNotFoundError("meta_model.pkl not found! Train the meta regressor first.")
        
    with open(meta_path, 'rb') as f:
        _meta_model = pickle.load(f)

def extract_concepts_from_reference(reference_answer: str):
    cleaned = clean_text(reference_answer)
    return [word for word in cleaned.split() if len(word) > 2]

def get_hybrid_score(question: str, reference_answer: str, student_answer: str, max_marks: float = 10.0):
    """
    Computes Phase 1 and Phase 2 scores, then runs them through the Meta-Regressor.
    """
    _load_meta_model()
    
    # 1. Phase 1 features
    concepts = extract_concepts_from_reference(reference_answer)
    if not concepts:
        concepts = [reference_answer]
        
    k_score, matched_k, _ = concept_match_score(concepts, student_answer)
    sim_score = concept_similarity_score(concepts, student_answer)
    
    # 2. Phase 2 features
    bert_res = bert_inference.predict(
        question=question, 
        ref_answer=reference_answer, 
        student_answer=student_answer, 
        max_marks=1.0 # Standardize to [0, 1] for the meta-regressor
    )
    bert_score = bert_res['score']
    
    # 3. Meta-Regressor prediction
    # Feature order must match training: ['phase1_keyword', 'phase1_tfidf', 'phase2_bert']
    import pandas as pd
    features = pd.DataFrame([{
        'phase1_keyword': k_score,
        'phase1_tfidf': sim_score,
        'phase2_bert': bert_score
    }])
    
    pred_score_normalized = _meta_model.predict(features)[0]
    
    # Clamp to [0, 1] just in case
    pred_score_normalized = max(0.0, min(1.0, pred_score_normalized))
    
    # Scale to max_marks
    final_score = round(pred_score_normalized * max_marks, 2)
    
    return {
        "score": final_score,
        "max_marks": max_marks,
        "percentage": round(pred_score_normalized * 100, 1),
        "phase1_keyword_score": round(k_score, 2),
        "phase1_tfidf_score": round(sim_score, 2),
        "phase2_bert_score": round(bert_score, 2),
    }

def grade_with_hybrid(
    image_path: str,
    question: str,
    reference_answer: str,
    max_marks: float = 10.0,
    ocr_engine: str = "trocr",
) -> dict:
    if not is_ready():
        raise RuntimeError("Hybrid model is not fully trained.")

    # Step 1 – OCR
    extracted_text = extract_text_from_image(image_path, engine=ocr_engine)

    # Step 2 – Hybrid Scoring
    result = get_hybrid_score(
        question=question,
        reference_answer=reference_answer,
        student_answer=extracted_text,
        max_marks=max_marks,
    )

    return {
        "extracted_text": extracted_text,
        **result,
        "model_ready": True,
    }

def grade_text_with_hybrid(
    student_answer: str,
    question: str,
    reference_answer: str,
    max_marks: float = 10.0,
) -> dict:
    if not is_ready():
        raise RuntimeError("Hybrid model is not fully trained.")

    result = get_hybrid_score(
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        max_marks=max_marks,
    )
    
    return {
        "extracted_text": student_answer,
        **result,
        "model_ready": True,
    }
