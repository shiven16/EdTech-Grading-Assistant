"""
Label ↔ Score conversion helpers.

The SciEntsBank dataset labels are:
  0 → correct answer         (best)
  1 → partially correct       
  2 → contradictory          
  3 → irrelevant / incorrect  (worst)

We normalise so that 0 (correct) → 1.0 and 3 (incorrect) → 0.0.
This makes the BERT output directly proportional to "how good the answer is".
"""

from config import LABEL_MIN, LABEL_MAX


def label_to_score(label: int) -> float:
    """Convert dataset label (0–3) to a normalised score in [0, 1].
    label=0 (correct)   → 1.0
    label=3 (incorrect) → 0.0
    """
    label = int(label)
    return 1.0 - (label - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)


def score_to_label(score: float) -> int:
    """Inverse of label_to_score. Rounds to nearest integer label."""
    label_float = (1.0 - score) * (LABEL_MAX - LABEL_MIN) + LABEL_MIN
    return int(round(max(LABEL_MIN, min(LABEL_MAX, label_float))))