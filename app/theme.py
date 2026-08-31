"""Design tokens and a single Plotly template.

One palette, one font stack, one figure template — imported everywhere so the whole
dashboard shares a visual language instead of defaulting to Plotly's stock look.
"""

from __future__ import annotations

import plotly.graph_objects as go

# --- Palette -------------------------------------------------------------------
# Editorial, finance-sector restraint: ink navy, one teal accent, warm neutrals.
INK = "#1a2b3c"  # near-black navy — text and axes
NAVY = "#22384f"  # panel headers
TEAL = "#2a9d8f"  # primary accent (the "banked" / positive signal)
CLAY = "#c8553d"  # secondary accent (the "not banked" / miss signal)
SAND = "#e9e4d8"  # warm neutral fill
PAPER = "#faf8f4"  # page background — off-white, not stark
CARD = "#ffffff"  # card surface
MUTED = "#6b7783"  # secondary text
GRID = "#e6e1d6"  # gridlines — barely there

# Ordered palette for multi-series charts (the five models).
SERIES = ["#22384f", "#2a9d8f", "#c8553d", "#e9b44c", "#7d8ca3"]

FONT = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"


def base_layout(title: str = "", height: int = 360) -> dict:
    """Shared layout kwargs — apply to every figure for a consistent look."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT), x=0.01),
        font=dict(family=FONT, color=INK, size=12),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        height=height,
        margin=dict(l=60, r=24, t=48, b=44),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        colorway=SERIES,
    )


def style_figure(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    """Apply the base layout and strip Plotly's floating mode bar chrome."""
    fig.update_layout(**base_layout(title, height))
    fig.update_layout(
        modebar_remove=["lasso", "select", "autoScale", "zoomIn", "zoomOut"]
    )
    return fig
