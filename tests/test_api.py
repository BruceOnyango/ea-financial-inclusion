"""Tests for the FastAPI prediction service — fully offline via fake_bundle."""

from fastapi.testclient import TestClient

from api import main


def _payload():
    return {
        "country": "Kenya",
        "location_type": "Rural",
        "cellphone_access": "Yes",
        "household_size": 4,
        "age_of_respondent": 32,
        "gender_of_respondent": "Female",
        "relationship_with_head": "Head of Household",
        "marital_status": "Married/Living together",
        "education_level": "Primary education",
        "job_type": "Self employed",
    }


def test_health(monkeypatch, fake_bundle):
    monkeypatch.setattr(main, "get_bundle", lambda: fake_bundle)
    r = TestClient(main.create_api()).get("/financial-inclusion/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "Logistic Regression"}


def test_predict_ok(monkeypatch, fake_bundle):
    monkeypatch.setattr(main, "get_bundle", lambda: fake_bundle)
    r = TestClient(main.create_api()).post(
        "/financial-inclusion/api/predict", json=_payload()
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "probability",
        "probability_pct",
        "banked",
        "verdict",
        "recommendation",
    }
    assert isinstance(body["banked"], bool)
    assert body["probability_pct"] == round(body["probability"] * 100, 1)


def test_predict_validation_error(monkeypatch, fake_bundle):
    monkeypatch.setattr(main, "get_bundle", lambda: fake_bundle)
    r = TestClient(main.create_api()).post(
        "/financial-inclusion/api/predict", json={"country": "Kenya"}
    )
    assert r.status_code == 422  # missing required fields


def test_recommendation_both_branches():
    assert "unbanked" in main.recommendation(False).lower()
    assert "already banked" in main.recommendation(True).lower()


def test_normalize_maps_long_labels():
    out = main.normalize(
        {"job_type": "Employed Government", "country": "Kenya", "household_size": 4}
    )
    assert out["job_type"] == "Formally employed Government"  # aliased
    assert out["country"] == "Kenya"  # passed through
    assert out["household_size"] == 4  # non-string untouched


def test_screen_returns_verdict(monkeypatch, fake_bundle):
    monkeypatch.setattr(main, "get_bundle", lambda: fake_bundle)
    out = main.screen(_payload())
    assert out["verdict"] in ("Banked", "Not banked")


def test_get_bundle_is_cached(monkeypatch, fake_bundle):
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return fake_bundle

    monkeypatch.setattr(main.artifacts, "load_model", fake_load)
    main.get_bundle.cache_clear()
    assert main.get_bundle() is fake_bundle
    main.get_bundle()
    assert calls["n"] == 1
