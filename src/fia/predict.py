"""Turn one form submission into a prediction using the loaded model bundle.

Pure logic given a bundle: build a one-row frame in the exact training column order,
score it, apply the tuned threshold, and return a small result dict for the UI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def predict_one(bundle: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Predict banked/not-banked for a single respondent.

    Parameters
    ----------
    bundle : the loaded model bundle (pipeline, threshold, feature_columns).
    record : {feature_name: value} for every training feature.

    Returns
    -------
    dict with probability, threshold, integer label and display string.
    """
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in record]
    if missing:
        raise ValueError(f"Missing feature(s) for prediction: {missing}")

    # Reindex to the exact training order — extra keys ignored, order guaranteed.
    row = pd.DataFrame([{c: record[c] for c in cols}], columns=cols)

    proba = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    label = int(proba >= threshold)
    return {
        "probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "label": label,
        "label_text": "Banked" if label else "Not banked",
    }
