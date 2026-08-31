"""Load and validate the two deployment artifacts.

Thin I/O boundary: read model.joblib and results.json from disk, fail loudly with a
clear message if either is missing or malformed. No statistics computed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from fia import config

_REQUIRED_BUNDLE_KEYS = {"pipeline", "threshold", "winner", "feature_columns"}
_REQUIRED_RESULTS_KEYS = {
    "meta",
    "test_metrics",
    "confusion_matrices",
    "roc_curves",
    "form_options",
}


def load_model(path: Path | None = None) -> dict[str, Any]:
    """Load the model bundle and check it carries the keys the app relies on."""
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run the training notebook and copy model.joblib into artifacts/."
        )
    bundle = joblib.load(path)
    missing = _REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def load_results(path: Path | None = None) -> dict[str, Any]:
    """Load results.json and check the top-level structure."""
    path = path or config.RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Results artifact not found at {path}. "
            "Run the training notebook and copy results.json into artifacts/."
        )
    with open(path, encoding="utf-8") as fh:
        results = json.load(fh)
    missing = _REQUIRED_RESULTS_KEYS - set(results)
    if missing:
        raise ValueError(f"results.json is missing keys: {sorted(missing)}")
    return results
