import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent paths
PHASE3_DIR = Path(__file__).resolve().parent
SRC_DIR = PHASE3_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "phase1"))
sys.path.insert(0, str(SRC_DIR / "phase2"))
sys.path.insert(0, str(SRC_DIR / "phase2" / "BERT_method"))

# Phase 1 imports
try:
    from src.phase1.keyword_baseline import concept_match_score
    from src.phase1.tfidf_advanced import concept_similarity_score
    from src.phase1.preprocess import clean_text
except ImportError:
    # Try importing directly if sys.path allows
    from keyword_baseline import concept_match_score
    from tfidf_advanced import concept_similarity_score
    from preprocess import clean_text

# Phase 2 imports
from src.phase2.BERT_method.utils import label_to_score
from src.phase2.BERT_method import inference as bert_inference


def extract_concepts_from_reference(reference_answer):
    """
    Since Phase 1 relies on 'concepts' (keywords), we treat the cleaned 
    words of the reference answer as the concepts.
    """
    cleaned = clean_text(reference_answer)
    return [word for word in cleaned.split() if len(word) > 2]


def build_meta_dataset(csv_path, max_samples=None):
    """
    Reads the dataset, gets Phase 1 and Phase 2 scores for each row,
    and returns a pandas DataFrame with features and labels.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
        
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    phase1_keyword_scores = []
    phase1_tfidf_scores = []
    phase2_bert_scores = []
    labels = []
    
    print("Extracting features (This may take a while as BERT runs inference)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row['question'])
        ref = str(row['reference_answer'])
        ans = str(row['student_answer'])
        raw_label = row['label']
        
        # Ground truth score [0, 1]
        target_score = label_to_score(raw_label)
        labels.append(target_score)
        
        # Phase 1: Keyword Matching and TF-IDF
        concepts = extract_concepts_from_reference(ref)
        if not concepts:
            concepts = [ref]
            
        k_score, _, _ = concept_match_score(concepts, ans)
        sim_score = concept_similarity_score(concepts, ans)
        
        phase1_keyword_scores.append(k_score)
        phase1_tfidf_scores.append(sim_score)
        
        # Phase 2: BERT Model
        # BERT predicts score on [0, 1] scale (assuming max_marks=1.0 for simplicity)
        bert_res = bert_inference.predict(question=q, ref_answer=ref, student_answer=ans, max_marks=1.0)
        phase2_bert_scores.append(bert_res['score'])

    feature_df = pd.DataFrame({
        'phase1_keyword': phase1_keyword_scores,
        'phase1_tfidf': phase1_tfidf_scores,
        'phase2_bert': phase2_bert_scores,
        'target_score': labels
    })
    
    return feature_df


def train_meta_regressor():
    train_csv = PROJECT_ROOT / "data" / "phase2" / "train" / "train.csv"
    
    # We can cache the extracted features to save time during multiple runs
    cache_path = PHASE3_DIR / "meta_features_cache.csv"
    
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        # Load a subset if you want it to be faster, e.g., max_samples=500
        df = build_meta_dataset(train_csv, max_samples=None) 
        df.to_csv(cache_path, index=False)
        print(f"Saved feature cache to {cache_path}")

    # Prepare training data
    X = df[['phase1_keyword', 'phase1_tfidf', 'phase2_bert']]
    y = df['target_score']
    
    print("\nTraining Meta-Regressor (LinearRegression)...")
    meta_model = LinearRegression()
    meta_model.fit(X, y)
    
    print("\nModel Coefficients:")
    print(f" Phase 1 Keyword Weight: {meta_model.coef_[0]:.4f}")
    print(f" Phase 1 TF-IDF Weight:  {meta_model.coef_[1]:.4f}")
    print(f" Phase 2 BERT Weight:    {meta_model.coef_[2]:.4f}")
    print(f" Intercept:              {meta_model.intercept_:.4f}")
    
    # Evaluate
    preds = meta_model.predict(X)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    
    print("\nTraining Set Performance:")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    
    # Save the meta-model
    model_out_path = PROJECT_ROOT / "meta_model.pkl"
    with open(model_out_path, 'wb') as f:
        pickle.dump(meta_model, f)
        
    print(f"\nMeta-Regressor saved to {model_out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="Max samples to process for training")
    args = parser.parse_args()

    # Ensure BERT model is ready
    if not bert_inference.is_model_ready():
        print("Error: BERT model weights not found! Phase 3 needs the trained Phase 2 model.")
        sys.exit(1)
        
    train_csv = PROJECT_ROOT / "data" / "phase2" / "train" / "train.csv"
    cache_path = PHASE3_DIR / f"meta_features_cache_{args.samples or 'all'}.csv"
    
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = build_meta_dataset(train_csv, max_samples=args.samples) 
        df.to_csv(cache_path, index=False)
        print(f"Saved feature cache to {cache_path}")

    X = df[['phase1_keyword', 'phase1_tfidf', 'phase2_bert']]
    y = df['target_score']
    
    print("\nTraining Meta-Regressor (LinearRegression)...")
    meta_model = LinearRegression()
    meta_model.fit(X, y)
    
    print("\nModel Coefficients:")
    print(f" Phase 1 Keyword Weight: {meta_model.coef_[0]:.4f}")
    print(f" Phase 1 TF-IDF Weight:  {meta_model.coef_[1]:.4f}")
    print(f" Phase 2 BERT Weight:    {meta_model.coef_[2]:.4f}")
    print(f" Intercept:              {meta_model.intercept_:.4f}")
    
    preds = meta_model.predict(X)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    
    print("\nTraining Set Performance:")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    
    model_out_path = PROJECT_ROOT / "meta_model.pkl"
    with open(model_out_path, 'wb') as f:
        pickle.dump(meta_model, f)
        
    print(f"\nMeta-Regressor saved to {model_out_path}")
