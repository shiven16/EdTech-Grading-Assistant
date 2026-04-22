"""
BERT Grading Model Trainer
Run from project root:
    python src/phase2/BERT_method/train.py
"""

import sys
from pathlib import Path

# ── Make sure sibling modules (model, dataset, utils, config) are importable ──
BERT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BERT_DIR))

import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

import config
from dataset import SciDataset
from model import GradingModel


# ── Optional: QWK metric ──────────────────────────────────────────────────────
def _qwk(y_true, y_pred, min_rating=0, max_rating=3):
    """Quadratic Weighted Kappa (rounded predictions)."""
    try:
        from sklearn.metrics import cohen_kappa_score
        y_pred_clipped = [max(min_rating, min(max_rating, round(v))) for v in y_pred]
        return cohen_kappa_score(y_true, y_pred_clipped, weights="quadratic",
                                 labels=list(range(min_rating, max_rating + 1)))
    except Exception:
        return float("nan")


def evaluate(model, loader, loss_fn, device):
    """Return (avg_loss, rmse, qwk) on a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device).unsqueeze(1)

            out  = model(ids, mask)
            loss = loss_fn(out, lbls)
            total_loss += loss.item()

            # Convert normalised score → label for QWK
            scores = out.squeeze(1).cpu().tolist()
            labels = lbls.squeeze(1).cpu().tolist()
            all_preds.extend(scores)
            all_labels.extend(labels)

    avg_loss = total_loss / len(loader)
    rmse     = math.sqrt(avg_loss)          # MSE loss → RMSE

    # QWK expects original label scale (0-3)
    from utils import score_to_label
    pred_labels  = [score_to_label(s) for s in all_preds]
    true_labels  = [score_to_label(s) for s in all_labels]
    qwk          = _qwk(true_labels, pred_labels)

    return avg_loss, rmse, qwk


def train():
    print(f"\n{'='*55}")
    print("  BERT Grader — Training")
    print(f"  Train: {config.DATA_TRAIN}")
    print(f"  Test : {config.DATA_TEST}")
    print(f"  Output: {config.MODEL_PATH}")
    print(f"{'='*55}\n")

    # ── Load data ─────────────────────────────────────────
    train_data = load_dataset("csv", data_files=str(config.DATA_TRAIN))["train"]
    test_data  = load_dataset("csv", data_files=str(config.DATA_TEST))["train"]
    print(f"Train rows: {len(train_data)} | Test rows: {len(test_data)}")

    train_dataset = SciDataset(train_data)
    test_dataset  = SciDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model, optimiser, loss ────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model     = GradingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    loss_fn   = torch.nn.MSELoss()

    # ── Training loop ─────────────────────────────────────
    best_rmse = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [train]", leave=False)
        for batch in pbar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            out  = model(ids, mask)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        test_loss, rmse, qwk = evaluate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test MSE: {test_loss:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"QWK: {qwk:.4f}"
        )

        # Save best checkpoint
        if rmse < best_rmse:
            best_rmse = rmse
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  ✓ Best model saved (RMSE={best_rmse:.4f})")

    print(f"\nTraining complete. Best RMSE: {best_rmse:.4f}")
    print(f"Model saved → {config.MODEL_PATH}\n")


if __name__ == "__main__":
    train()