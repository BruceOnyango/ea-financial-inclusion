"""Central configuration: paths, display labels, and model ordering.

Single source of truth shared by the package and the dashboard, mirroring the
constants used in the training notebook so both stay in lockstep.
"""
from __future__ import annotations

from pathlib import Path

# Project root = <root>/src/fia/config.py -> parents[2] is <root>
BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"
DATA_PATH = BASE_DIR / "data" / "financial_inclusion.csv"

# Canonical model display order (matches the notebook).
MODEL_ORDER = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "ANN (MLP)",
    "SVM (RBF)",
]

# Metric key -> human label for tables and charts.
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}

# Target-class display names.
POSITIVE_LABEL = "Banked"
NEGATIVE_LABEL = "Not banked"


def ordered_models(names) -> list[str]:
    """Sort model names by MODEL_ORDER; append any unknown names alphabetically."""
    known = [m for m in MODEL_ORDER if m in names]
    extra = sorted(n for n in names if n not in MODEL_ORDER)
    return known + extra