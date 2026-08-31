"""Tests for fia.predict — single-record prediction logic."""

import pytest

from fia import predict


def _record():
    return {"country": "Kenya", "age_of_respondent": 30}


def test_predict_one_shape(fake_bundle):
    out = predict.predict_one(fake_bundle, _record())
    assert set(out) == {"probability", "threshold", "label", "label_text"}
    assert 0.0 <= out["probability"] <= 1.0
    assert out["label"] in (0, 1)


def test_label_matches_threshold(fake_bundle):
    out = predict.predict_one(fake_bundle, _record())
    assert out["label"] == int(out["probability"] >= out["threshold"])
    assert out["label_text"] == ("Banked" if out["label"] else "Not banked")


def test_extra_keys_are_ignored(fake_bundle):
    out = predict.predict_one(fake_bundle, {**_record(), "junk": 999})
    assert 0.0 <= out["probability"] <= 1.0


def test_missing_feature_raises(fake_bundle):
    with pytest.raises(ValueError, match="Missing feature"):
        predict.predict_one(fake_bundle, {"country": "Kenya"})  # no age


def test_threshold_boundary_is_inclusive(fake_bundle):
    # Force threshold to 0 -> everything is predicted banked (proba >= 0 always).
    bundle = {**fake_bundle, "threshold": 0.0}
    assert predict.predict_one(bundle, _record())["label"] == 1
