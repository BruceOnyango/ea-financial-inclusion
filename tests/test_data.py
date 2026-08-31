"""Tests for app.data — cached artifact access and the prediction entry point."""

import app.data as data


def test_get_results_is_cached(monkeypatch, results_dict):
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return results_dict

    monkeypatch.setattr(data.artifacts, "load_results", fake_load)
    data.clear_cache()
    first, second = data.get_results(), data.get_results()
    assert first is second  # same cached object
    assert calls["n"] == 1  # loaded exactly once


def test_get_model_is_cached(monkeypatch, fake_bundle):
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return fake_bundle

    monkeypatch.setattr(data.artifacts, "load_model", fake_load)
    data.clear_cache()
    assert data.get_model() is fake_bundle
    data.get_model()
    assert calls["n"] == 1


def test_predict_record_uses_cached_model(monkeypatch, fake_bundle):
    monkeypatch.setattr(data.artifacts, "load_model", lambda: fake_bundle)
    data.clear_cache()
    out = data.predict_record({"country": "Kenya", "age_of_respondent": 30})
    assert set(out) == {"probability", "threshold", "label", "label_text"}


def test_clear_cache_forces_reload(monkeypatch, results_dict):
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return dict(results_dict)

    monkeypatch.setattr(data.artifacts, "load_results", fake_load)
    data.clear_cache()
    data.get_results()
    data.clear_cache()
    data.get_results()
    assert calls["n"] == 2  # reloaded after the cache was cleared
