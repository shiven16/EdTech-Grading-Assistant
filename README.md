# Automated EdTech Grading Assistant

An AI-powered grading system that evaluates student short-answers using a **Phase 3 Stacking Ensemble Meta-Regressor**. The pipeline synthesizes sparse lexical heuristics (TF-IDF & Keyword Matching) with dense semantic embeddings (BERT) to mathematically assign accurate grades, completely replacing manual grading.

---

## Overview & Architecture Evolution

Grading handwritten subjective answers is notoriously difficult due to OCR noise and semantic generalisation. This project has evolved through three distinct phases:

### Phase 1: The Classical Baseline
- **OCR:** Tesseract (Heuristic/Rule-based)
- **Grading:** TF-IDF Cosine Similarity & Keyword Overlap
- **Limitation:** Tesseract failed catastrophically on handwriting, causing the downstream models to score 0.0. Lexical grading was too rigid.

### Phase 2: The Neural Upgrade
- **OCR:** TrOCR (Transformer-based Optical Character Recognition) by Microsoft Research
- **Grading:** Fine-tuned BERT-base-uncased with a regression head
- **Dataset:** SciEntsBank dataset (4,969 training samples)
- **Limitation:** Deep neural networks suffer from high variance, occasionally hallucinating correctness when the sentence structure matches the reference answer but the meaning contradicts it.

### Phase 3: The Hybrid Meta-Regressor (Current)
- **OCR:** CRAFT + TrOCR
- **Grading:** Stacking Ensemble Meta-Regressor (Scikit-Learn OLS Linear Regression)
- **Mechanism:** The model takes 3 engineered features `[Keyword Score, TF-IDF Score, BERT Score]` and computes the optimal weightage to minimise error.
- **Future Work:** **The full exam sheet grading pipeline (segmenting a multi-question, full-page exam sheet into localized answer crops) is planned but NOT implemented yet.** Currently, the system evaluates individual cropped answers.

---

## Dataset

We use the **SciEntsBank** dataset sourced from HuggingFace (`nkazi/SciEntsBank`), replacing the ASAP-SAS dataset from Phase 1.
- **Train:** 4,969 samples
- **Test:** 540 samples
- **Structure:** Question, Reference Answer, Student Answer, Label (0 to 3 ordinal scale).

---

## Project Structure

```text
edtech-grader/
│
├── data/
│   └── phase2/              # SciEntsBank Train/Test datasets
│
├── src/
│   ├── phase1/              # Lexical: Preprocessing, TF-IDF, Keyword matching
│   ├── phase2/              # Semantic: TrOCR, BERT_method training scripts
│   └── phase3/              # Hybrid: Meta-Regressor training and unified pipeline
│
├── static/                  # Frontend UI (HTML, CSS, JS)
├── app.py                   # FastAPI Backend Server
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd edtech-grader
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: PyTorch is required for TrOCR and BERT. Ensure it matches your hardware architecture, e.g., Apple Silicon MPS or CUDA).*

### 4. NLTK Stopwords
```bash
python -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('stopwords')"
```

---

## Running the Project

### Start the FastAPI Server
```bash
python3 app.py
```
- Open your browser and navigate to `http://localhost:8000`.
- Use the web interface to upload an image of a handwritten answer, or test using the raw text input.

### Training the Models (Optional)
If you wish to retrain the underlying models from scratch:

**1. Retrain Phase 2 BERT:**
```bash
python3 src/phase2/BERT_method/train.py
```

**2. Retrain Phase 3 Meta-Regressor:**
```bash
python3 src/phase3/train_meta.py
```
*(This extracts features and trains the OLS Regressor, saving it as `meta_model.pkl`)*

---

## Model Evaluation Metrics

Our Phase 3 ensemble drastically outperforms isolated Phase 2 configurations:
- **Phase 3 MAE:** 0.1665
- **Phase 3 MSE:** 0.0495

**Learned Feature Weights:**
- Keyword: `0.02`
- TF-IDF: `0.23`
- BERT: `1.09`
- Intercept: `-0.16`