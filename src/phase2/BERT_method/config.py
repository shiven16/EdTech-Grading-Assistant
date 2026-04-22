from pathlib import Path

# ── Model ────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 256
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5

# ── Paths (always relative to project root) ───────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # 3 levels up from BERT_method/
DATA_TRAIN   = PROJECT_ROOT / "data" / "phase2" / "train" / "train.csv"
DATA_TEST    = PROJECT_ROOT / "data" / "phase2" / "test"  / "test.csv"
MODEL_DIR    = PROJECT_ROOT / "models" / "bert_grader"
MODEL_PATH   = MODEL_DIR / "model.pth"

# Label range in the dataset (0 = correct, 3 = incorrect → normalise to [0,1])
LABEL_MIN = 0
LABEL_MAX = 3