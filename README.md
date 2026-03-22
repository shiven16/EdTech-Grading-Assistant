# Automated EdTech Grading Assistant

An AI-powered grading system that evaluates student answers using a hybrid pipeline combining classical OCR (Tesseract) and machine learning techniques (keyword matching + TF-IDF similarity).

---

## Overview

Grading handwritten or subjective answers is time-consuming and inconsistent. This project builds an automated grading pipeline that:

- Extracts text from handwritten images using OCR  
- Cleans and preprocesses the extracted text  
- Compares it with rubric-defined concepts  
- Generates a score based on concept coverage and similarity  

---

## System Pipeline

Image (optional)  
↓  
Tesseract OCR (classical)  
↓  
Text Cleaning & Preprocessing  
↓  
Keyword Matching (Baseline ML)  
↓  
TF-IDF + Cosine Similarity (Advanced ML)  
↓  
Final Score + Feedback  

---

## Features

- Classical OCR using Tesseract (baseline-compliant)
- Concept-based grading using rubrics
- TF-IDF similarity for flexible matching
- Robust to OCR noise
- Modular and extensible architecture
- Supports both image and text input

---

## Project Structure

edtech-grader/
│
├── data/
│   ├── raw_docs/          # Original DOCX files (dataset)
│   ├── processed/         # Parsed JSON datasets
│   └── Edtech_grading_file.jpeg  # Sample handwritten answer
│
├── src/
│   ├── doc_parser.ipynb
│   ├── ocr.py
│   ├── preprocess.py
│   ├── keyword_baseline.py
│   ├── tfidf_advanced.py
│   ├── utils.py
│   └── grader.py
│
├── main.py
├── requirements.txt
└── README.md

---

## Dataset

Dataset source:  
https://www.kaggle.com/competitions/asap-sas/data

### What we use:
- ReadMe .docx files
- Extract:
  - Questions
  - "Possible Correct Responses" (concepts)

### Important:
- We do NOT train on student answers in this phase
- We do NOT use labeled scores
- We use rubric-based concept extraction only

---

## Setup Instructions

### 1. Clone the repository

git clone <your-repo-url>
cd edtech-grader

---

### 2. Create virtual environment

python3 -m venv venv
source venv/bin/activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Prepare dataset

Download dataset from Kaggle and place .docx files inside:
data/raw_docs/

---

### 5. Run parser

Open and run:
src/doc_parser.ipynb

This generates:
data/processed/valid_dataset.json

---

## Running the Project

python main.py

---

## Grading Methodology

### Baseline ML (Keyword Matching)

- Extract concepts from rubric  
- Match words with student answer  
- Score based on matched concepts  

### Advanced ML (TF-IDF)

- Convert text into vectors  
- Compare concepts with answer chunks  
- Use cosine similarity  

### Final Score

Final Score = 0.4 × Keyword Score + 0.6 × TF-IDF Score

---