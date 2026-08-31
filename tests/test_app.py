"""Tests for app.app — layout builders and callback bodies, fully offline."""

import plotly.graph_objects as go
import pytest
from dash import Dash, html
from dash.exceptions import PreventUpdate

import app.app as appmod


def _collect_ids(component):
    """Recursively gather every component id in a layout tree."""
    ids = []
    cid = getattr(component, "id", None)
    if cid is not None:
        ids.append(cid)
    for child in getattr(component, "children", None) or []:
        if hasattr(child, "children") or getattr(child, "id", None) is not None:
            ids.extend(_collect_ids(child))
    return ids


def test_pct_formats():
    assert appmod._pct(0.141) == "14.1%"


def test_kpi_cards_has_four(results_dict):
    cards = appmod.kpi_cards(results_dict)
    assert len(cards) == 4
    assert all(isinstance(c, html.Div) for c in cards)


def test_build_layout_contains_key_ids(results_dict):
    layout = appmod.build_layout(results_dict)
    ids = _collect_ids(layout)
    for expected in [
        "metric-select",
        "metric-graph",
        "roc-graph",
        "cm-select",
        "cm-graph",
        "eda-select",
        "eda-graph",
        "predict-btn",
        "prediction-result",
    ]:
        assert expected in ids
    # both a categorical and a numeric feature field were built
    assert {"type": "feat", "name": "country"} in ids
    assert {"type": "feat", "name": "age_of_respondent"} in ids


def test_cb_metric_returns_figure(monkeypatch, results_dict):
    monkeypatch.setattr(appmod.data, "get_results", lambda: results_dict)
    assert isinstance(appmod.cb_metric("roc_auc"), go.Figure)


def test_cb_confusion_returns_figure(monkeypatch, results_dict):
    monkeypatch.setattr(appmod.data, "get_results", lambda: results_dict)
    assert isinstance(appmod.cb_confusion("SVM (RBF)"), go.Figure)


def test_cb_eda_returns_figure(monkeypatch, results_dict):
    monkeypatch.setattr(appmod.data, "get_results", lambda: results_dict)
    assert isinstance(appmod.cb_eda("country"), go.Figure)


def test_render_prediction_banked():
    out = appmod.render_prediction({"label": 1, "probability": 0.83, "threshold": 0.71})
    assert "banked" in out.className and "not-banked" not in out.className


def test_render_prediction_unbanked():
    out = appmod.render_prediction({"label": 0, "probability": 0.20, "threshold": 0.71})
    assert "not-banked" in out.className


def test_cb_predict_click(monkeypatch):
    monkeypatch.setattr(
        appmod.data,
        "predict_record",
        lambda rec: {"label": 1, "probability": 0.9, "threshold": 0.7},
    )
    ids = [
        {"type": "feat", "name": "country"},
        {"type": "feat", "name": "age_of_respondent"},
    ]
    out, cls = appmod.cb_predict(1, ["Kenya", 30], ids)
    assert "banked" in out.className
    assert cls == "result-shown"


def test_cb_predict_no_click_prevents_update():
    with pytest.raises(PreventUpdate):
        appmod.cb_predict(0, [], [])


def test_create_app_builds(monkeypatch, results_dict):
    monkeypatch.setattr(appmod.data, "get_results", lambda: results_dict)
    app = appmod.create_app()
    assert isinstance(app, Dash)
    assert app.layout is not None
