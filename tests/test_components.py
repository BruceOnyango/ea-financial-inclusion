"""Tests for app.components — pure figure builders over results.json data."""

import plotly.graph_objects as go

from app import components


def test_model_comparison_bar_orders_best_last(results_dict):
    fig = components.model_comparison_bar(results_dict["test_metrics"], "roc_auc")
    assert isinstance(fig, go.Figure)
    # y is sorted ascending, so the highest-AUC model sits last (top of the bar chart).
    ys = list(fig.data[0].y)
    assert ys[-1] == "SVM (RBF)"  # 0.858 is the max in the fixture


def test_model_comparison_uses_requested_metric(results_dict):
    fig = components.model_comparison_bar(results_dict["test_metrics"], "recall")
    xs = list(fig.data[0].x)
    assert max(xs) == 0.640  # SVM recall in the fixture


def test_confusion_matrix_annotates_all_cells(results_dict):
    cm = results_dict["confusion_matrices"]["SVM (RBF)"]
    fig = components.confusion_matrix_heatmap(cm, "SVM (RBF)")
    # 4 cells -> 4 annotations, and the TP count appears.
    assert len(fig.layout.annotations) == 4
    texts = {a.text for a in fig.layout.annotations}
    assert "638" in texts


def test_roc_overlay_has_curve_plus_chance_line(results_dict):
    fig = components.roc_overlay(results_dict["roc_curves"])
    names = [t.name for t in fig.data]
    assert any("SVM (RBF)" in n for n in names)
    assert len(fig.data) == len(results_dict["roc_curves"]) + 1  # +chance diagonal


def test_banked_rate_bar_sorted_and_percent(results_dict):
    fig = components.banked_rate_bar(
        results_dict["eda_banked_rate"]["country"], "country"
    )
    xs = list(fig.data[0].x)
    assert xs == sorted(xs)  # ascending
    assert "20.0%" in list(fig.data[0].text)
