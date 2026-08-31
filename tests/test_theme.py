"""Tests for app.theme — tokens and the figure styler."""

import plotly.graph_objects as go

from app import theme


def test_palette_tokens_are_hex():
    for token in [theme.INK, theme.TEAL, theme.CLAY, theme.PAPER, theme.CARD]:
        assert token.startswith("#") and len(token) == 7


def test_series_has_five_colours():
    # One colour per model in the comparison.
    assert len(theme.SERIES) == 5


def test_base_layout_carries_title_and_height():
    lay = theme.base_layout("Hello", height=300)
    assert lay["title"]["text"] == "Hello"
    assert lay["height"] == 300
    assert lay["colorway"] == theme.SERIES


def test_style_figure_applies_layout():
    fig = theme.style_figure(go.Figure(), "T", height=280)
    assert fig.layout.height == 280
    assert fig.layout.paper_bgcolor.lower() == theme.CARD.lower()
    assert fig.layout.title.text == "T"
