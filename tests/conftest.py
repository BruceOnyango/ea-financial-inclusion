"""Shared fixtures: a tiny real pipeline + a minimal results dict.

Building a genuine (but small) sklearn pipeline keeps the tests honest — we exercise
real predict_proba behaviour — while staying fast and fully offline.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture
def raw_frame():
    """A handful of rows with the two feature types the app cares about."""
    return pd.DataFrame(
        {
            "country": ["Kenya", "Uganda", "Kenya", "Rwanda"],
            "age_of_respondent": [30, 55, 24, 40],
            "y": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def fake_bundle(raw_frame):
    """A real fitted pipeline over {country, age}, bundled like the notebook's."""
    X = raw_frame[["country", "age_of_respondent"]]
    y = raw_frame["y"]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age_of_respondent"]),
            ("nom", OneHotEncoder(handle_unknown="ignore"), ["country"]),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression())]).fit(X, y)
    return {
        "pipeline": pipe,
        "threshold": 0.5,
        "winner": "Logistic Regression",
        "feature_columns": ["country", "age_of_respondent"],
    }


@pytest.fixture
def model_file(fake_bundle, tmp_path):
    """Persist the bundle to a temp joblib and return the path."""
    p = tmp_path / "model.joblib"
    joblib.dump(fake_bundle, p)
    return p


@pytest.fixture
def results_dict():
    """Minimal results.json payload with every key the dashboard reads."""
    return {
        "meta": {
            "winner": "SVM (RBF)",
            "winner_threshold": 0.707,
            "n_rows": 23524,
            "positive_rate": 0.141,
            "cv_folds": 2,
            "countries": ["Kenya", "Rwanda", "Tanzania", "Uganda"],
        },
        "class_balance": {"No": 20212, "Yes": 3312},
        "eda_banked_rate": {"country": {"Kenya": 20.0, "Rwanda": 10.0}},
        "test_metrics": {
            "SVM (RBF)": {
                "accuracy": 0.845,
                "balanced_accuracy": 0.759,
                "precision": 0.464,
                "recall": 0.640,
                "f1": 0.538,
                "roc_auc": 0.858,
            },
            "KNN": {
                "accuracy": 0.851,
                "balanced_accuracy": 0.705,
                "precision": 0.473,
                "recall": 0.502,
                "f1": 0.487,
                "roc_auc": 0.820,
            },
        },
        "thresholds": {"SVM (RBF)": 0.707, "KNN": 0.818},
        "confusion_matrices": {"SVM (RBF)": [[5000, 1062], [358, 638]]},
        "roc_curves": {
            "SVM (RBF)": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.858}
        },
        "form_options": {
            "country": {
                "type": "categorical",
                "choices": ["Kenya", "Rwanda", "Tanzania", "Uganda"],
            },
            "age_of_respondent": {
                "type": "numeric",
                "min": 16,
                "max": 100,
                "default": 38,
            },
        },
    }


@pytest.fixture
def results_file(results_dict, tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results_dict))
    return p
