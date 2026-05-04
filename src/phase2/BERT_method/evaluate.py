import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import SciDataset
from model import GradingModel
import config

def evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, "../../../data/phase2/test/test.csv")
    
    print("Loading test dataset...")
    test_data  = load_dataset("csv", data_files=test_path)["train"]
    test_dataset  = SciDataset(test_data)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    print("Loading model architecture...")
    model = GradingModel()
    
    model_path = os.path.join(base_dir, "../../../model.pth")
        
    if not os.path.exists(model_path):
        print("Error: Could not find model.pth. Make sure to train the model first.")
        return

    # Check for Mac GPU acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Loading weights from {os.path.basename(model_path)}...")
    # Load weights properly mapping to the detected device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Evaluating on {device}...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            
            outputs = model(input_ids, attention_mask)
            preds = outputs.cpu().squeeze().numpy()
            
            # Handle edge case where a batch has exactly 1 element
            if preds.ndim == 0:
                preds = np.expand_dims(preds, axis=0)
                
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    
    print(f"\n--- Evaluation Results ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    evaluate()
