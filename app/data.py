"""Cached access to the deployment artifacts for the view layer.

Loads model.joblib and results.json once each (LRU-cached) so callbacks don't touch
disk on every interaction. All prediction goes through predict_record, keeping the
app layer free of any model handling.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fia import artifacts, predict


@lru_cache(maxsize=1)
def get_results() -> dict[str, Any]:
    """Load results.json once."""
    return artifacts.load_results()


@lru_cache(maxsize=1)
def get_model() -> dict[str, Any]:
    """Load the model bundle once."""
    return artifacts.load_model()


def predict_record(record: dict[str, Any]) -> dict[str, Any]:
    """Score a single respondent through the cached model bundle."""
    return predict.predict_one(get_model(), record)


def clear_cache() -> None:
    """Reset the LRU caches (used by tests and after an artifact refresh)."""
    get_results.cache_clear()
    get_model.cache_clear()
