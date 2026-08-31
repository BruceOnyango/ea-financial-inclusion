"""Pure Plotly figure builders.

Each function takes plain data from results.json and returns a styled figure. No file
I/O, no model calls, no statistics — the numbers arrive already computed. This is what
lets the tests assert exact values: the figures only ever re-plot what the notebook made.
"""

from __future__ import annotations

import plotly.graph_objects as go

from app import theme
from fia import config


def model_comparison_bar(test_metrics: dict, metric: str = "roc_auc") -> go.Figure:
    """Horizontal bars of one metric across models, best at the top."""
    models = config.ordered_models(test_metrics.keys())
    pairs = sorted(((m, test_metrics[m][metric]) for m in models), key=lambda t: t[1])
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colours = [theme.TEAL if n == names[-1] else theme.NAVY for n in names]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    label = config.METRIC_LABELS.get(metric, metric)
    fig.update_xaxes(range=[0, 1])
    return theme.style_figure(fig, f"Model comparison — {label}", height=320)


def confusion_matrix_heatmap(cm: list[list[int]], model: str) -> go.Figure:
    """2x2 confusion matrix as an annotated heatmap."""
    labels = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL]
    z = cm
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"Pred: {l}" for l in labels],
            y=[f"True: {l}" for l in labels],
            colorscale=[[0, theme.PAPER], [1, theme.TEAL]],
            showscale=False,
        )
    )
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{val:,}",
                showarrow=False,
                font=dict(color=theme.INK, size=15),
            )
    fig.update_yaxes(autorange="reversed")
    return theme.style_figure(fig, f"Confusion matrix — {model}", height=320)


def roc_overlay(roc_curves: dict) -> go.Figure:
    """All models' ROC curves on one axis, plus the chance diagonal."""
    fig = go.Figure()
    for name in config.ordered_models(roc_curves.keys()):
        c = roc_curves[name]
        fig.add_trace(
            go.Scatter(
                x=c["fpr"],
                y=c["tpr"],
                mode="lines",
                name=f"{name} ({c['auc']:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dot", color=theme.MUTED),
            name="Chance",
            showlegend=False,
        )
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return theme.style_figure(fig, "ROC curves (held-out test)", height=380)


def banked_rate_bar(rates: dict, driver: str) -> go.Figure:
    """Banked-rate-by-category bars for one EDA driver."""
    items = sorted(rates.items(), key=lambda t: t[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker_color=theme.TEAL,
            text=[f"{v:.1f}%" for _, v in items],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(title="% with a bank account")
    return theme.style_figure(fig, f"Banked rate by {driver}", height=320)
