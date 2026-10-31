"""Plotly chart helpers with a consistent, colour-blind-checked palette."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Categorical order validated for adjacent-pair CVD separation (dataviz reference palette).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#c2410c"], [0.25, "#f0a07a"], [0.5, "#e7e6e2"], [0.75, "#86b6ef"], [1.0, "#184f95"]]
NEUTRAL = "#8a8985"
REGION_COLORS = {"North America": PALETTE[0], "Europe": PALETTE[1], "Japan": PALETTE[2], "Other": PALETTE[3]}

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE


def style(fig: go.Figure, height: int | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13),
        title=dict(font=dict(size=15), y=0.98, yanchor="top", x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=90 if legend else 56, b=10),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def color_map(names) -> dict:
    """Stable entity -> colour mapping (colour follows the model, never its rank)."""
    names = list(dict.fromkeys(names))
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def metric_bars(frame: pd.DataFrame, model_col: str, metrics: list[str], title: str, yaxis: str) -> go.Figure:
    long = frame.melt(id_vars=[model_col], value_vars=metrics, var_name="Metric", value_name="Value").dropna()
    fig = px.bar(long, x=model_col, y="Value", color="Metric", barmode="group", title=title,
                 labels={"Value": yaxis, model_col: ""}, text_auto=".3f")
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickangle=-30)
    return style(fig)


def actual_vs_predicted(df: pd.DataFrame, title: str, log_axes: bool) -> go.Figure:
    hover = [c for c in ["Name", "Platform", "Year", "Genre"] if c in df]
    fig = px.scatter(df, x="actual", y="predicted", hover_data=hover, opacity=0.55, title=title,
                     labels={"actual": "Actual global sales (M copies)", "predicted": "Predicted global sales (M copies)"})
    fig.update_traces(marker=dict(size=8, color=PALETTE[0], line=dict(width=1, color="white")))
    lo = max(min(df["actual"].min(), df["predicted"].min()), 0.005 if log_axes else 0)
    hi = max(df["actual"].max(), df["predicted"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (y = x)",
                             line=dict(color=NEUTRAL, dash="dash", width=2)))
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return style(fig, height=460)


def histogram(values, title: str, xlabel: str, nbins: int = 60, color: str = PALETTE[0], log_x: bool = False) -> go.Figure:
    v = pd.Series(values).dropna()
    if log_x:
        v = v[v > 0]
        fig = go.Figure(go.Histogram(x=np.log10(v), nbinsx=nbins, marker_color=color, marker_line_color="white", marker_line_width=1))
        ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks], title=xlabel + " (log scale)")
    else:
        fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color=color, marker_line_color="white", marker_line_width=1))
        fig.update_xaxes(title=xlabel)
    fig.update_layout(title=title, bargap=0.02)
    fig.update_yaxes(title="Number of titles")
    return style(fig, legend=False)
