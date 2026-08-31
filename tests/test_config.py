"""Tests for fia.config — paths, labels, and model ordering."""

import fia.config as config


def test_paths_hang_off_base_dir():
    assert config.ARTIFACTS_DIR == config.BASE_DIR / "artifacts"
    assert config.MODEL_PATH.name == "model.joblib"
    assert config.RESULTS_PATH.name == "results.json"
    assert config.DATA_PATH.name == "financial_inclusion.csv"


def test_base_dir_is_project_root():
    # config.py lives at <root>/src/fia/config.py
    assert (config.BASE_DIR / "src" / "fia" / "config.py").exists()


def test_metric_labels_cover_reported_metrics():
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        assert key in config.METRIC_LABELS


def test_ordered_models_orders_known_and_appends_unknown():
    out = config.ordered_models(["SVM (RBF)", "Mystery", "KNN"])
    assert out == ["KNN", "SVM (RBF)", "Mystery"]


def test_ordered_models_handles_empty():
    assert config.ordered_models([]) == []


def test_class_labels_are_distinct():
    assert config.POSITIVE_LABEL != config.NEGATIVE_LABEL
