"""Tests for fia.artifacts — loading and validation of both artifacts."""

import json

import joblib
import pytest

from fia import artifacts


def test_load_model_ok(model_file):
    bundle = artifacts.load_model(model_file)
    assert bundle["winner"] == "Logistic Regression"
    assert set(bundle) >= {"pipeline", "threshold", "feature_columns"}


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        artifacts.load_model(tmp_path / "nope.joblib")


def test_load_model_missing_keys(tmp_path):
    bad = tmp_path / "bad.joblib"
    joblib.dump({"pipeline": 1}, bad)  # missing threshold/winner/columns
    with pytest.raises(ValueError, match="missing keys"):
        artifacts.load_model(bad)


def test_load_results_ok(results_file):
    res = artifacts.load_results(results_file)
    assert res["meta"]["winner"] == "SVM (RBF)"
    assert "test_metrics" in res


def test_load_results_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Results artifact not found"):
        artifacts.load_results(tmp_path / "nope.json")


def test_load_results_missing_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"meta": {}}))  # missing the rest
    with pytest.raises(ValueError, match="missing keys"):
        artifacts.load_results(bad)


def test_default_paths_are_used(monkeypatch, model_file, results_file):
    # When no path is passed, the config defaults are read.
    monkeypatch.setattr(artifacts.config, "MODEL_PATH", model_file)
    monkeypatch.setattr(artifacts.config, "RESULTS_PATH", results_file)
    assert artifacts.load_model()["winner"] == "Logistic Regression"
    assert artifacts.load_results()["meta"]["winner"] == "SVM (RBF)"
