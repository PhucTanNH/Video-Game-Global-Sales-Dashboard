"""Section 2 - Exploratory data analysis (descriptive associations only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vgsales_core.config import GENERATION_ORDER, ORIGINAL_NUM, REGION_COLUMNS, TARGET
from vgsales_core.plots import DIVERGING, PALETTE, REGION_COLORS, SEQUENTIAL, style

from .state import csv_download, empty_state, show_chart, show_table

GROUPS = {
    "Genre": "Genre",
    "Platform": "Platform",
    "Publisher tier": "Publisher_Tier",
    "Console generation": "ConsoleGen",
    "Publisher (top 20 by titles)": "Publisher",
}


def group_summary(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    g = frame.groupby(col, observed=True)[TARGET]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "Median": g.median(),
            "Q1": g.quantile(0.25),
            "Q3": g.quantile(0.75),
            "Mean": g.mean(),
            "Total (M)": g.sum(),
        }
    )
    out["IQR"] = out["Q3"] - out["Q1"]
    return out.reset_index().sort_values("Median", ascending=False)


def render(filtered: pd.DataFrame):
    st.header("Exploratory data analysis")
    st.info(
        "Everything in this section is **descriptive**: associations in the recorded data, not model results and not causal effects. "
        "Sidebar filters apply here.",
        icon=":material/query_stats:",
    )
    if filtered.empty:
        empty_state("No titles match the current sidebar filters. Widen the filters to explore the data.")
        return

    # ---------------- Group comparison ----------------
    st.subheader("How do sales differ across groups?")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    group_label = c1.selectbox("Group by", list(GROUPS), key="eda_group")
    log_y = c2.toggle("Log-scale sales axis", value=True, key="eda_log")
    min_n = c3.number_input("Hide groups with fewer than n titles", 1, 500, 10, key="eda_minn")
    col = GROUPS[group_label]
    frame = filtered
    if col == "Publisher":
        top = filtered["Publisher"].value_counts().head(20).index
        frame = filtered[filtered["Publisher"].isin(top)]
    summary = group_summary(frame, col)
    summary = summary[summary["n"] >= min_n]
    if summary.empty:
        empty_state("No group reaches the minimum sample size. Lower the threshold or widen the filters.")
    else:
        order = summary[col].tolist()
        plot_frame = frame[frame[col].isin(order)]
        fig = px.box(plot_frame, x=col, y=TARGET, category_orders={col: order}, points=False,
                     labels={TARGET: "Global sales (M copies)", col: group_label},
                     title=f"Global sales by {group_label.lower()} (box = IQR, line = median, whiskers = min to max; sorted by median)")
        fig.update_traces(marker_color=PALETTE[0], line_color=PALETTE[0], fillcolor="rgba(42,120,214,0.25)")
        if log_y:
            fig.update_yaxes(type="log")
        fig.update_xaxes(tickangle=-35)
        show_chart(style(fig, height=430, legend=False))
        show_table(
            summary.style.format({"Median": "{:.3f}", "Q1": "{:.3f}", "Q3": "{:.3f}", "IQR": "{:.3f}", "Mean": "{:.3f}", "Total (M)": "{:,.1f}"}),
            hide_index=True,
        )
        csv_download(summary, "Download group summary (CSV)", f"group_summary_{col}.csv", "dl_grp")
        small = (summary["n"] < 30).sum()
        if small:
            st.caption(f"{small} group(s) have fewer than 30 titles; their medians are unstable.")

    # ---------------- Regional composition ----------------
    st.subheader("Where are the sales made?")
    region_group = st.selectbox("Regional composition by", ["Genre", "Console generation", "Publisher tier", "Platform"], key="eda_region")
    rcol = GROUPS[region_group]
    reg = filtered.groupby(rcol, observed=True)[list(REGION_COLUMNS)].sum().rename(columns=REGION_COLUMNS)
    counts = filtered.groupby(rcol, observed=True).size()
    reg = reg[reg.sum(axis=1) > 0]
    if reg.empty:
        empty_state("No regional sales for the current filter.")
    else:
        share = reg.div(reg.sum(axis=1), axis=0)
        overall = filtered[list(REGION_COLUMNS)].sum().rename(REGION_COLUMNS)
        overall_share = overall / overall.sum()
        share.loc["All titles"] = overall_share
        counts.loc["All titles"] = len(filtered)
        long = share.reset_index().melt(id_vars=rcol, var_name="Region", value_name="Share")
        long["n"] = long[rcol].map(counts)
        long["Units (M)"] = long.apply(lambda r: (reg.loc[r[rcol], r["Region"]] if r[rcol] in reg.index else overall[r["Region"]]), axis=1)
        fig = px.bar(long, y=rcol, x="Share", color="Region", orientation="h", color_discrete_map=REGION_COLORS,
                     hover_data={"n": True, "Units (M)": ":.1f", "Share": ":.1%"},
                     title="Share of units sold by region (sum of regional sales / sum of global sales)")
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of units")
        fig.update_yaxes(title="", categoryorder="array", categoryarray=list(share.index)[::-1])
        fig.update_traces(marker_line_color="white", marker_line_width=2)
        show_chart(style(fig, height=max(320, 28 * len(share) + 120)))
        jp = share["Japan"].drop("All titles").sort_values(ascending=False)
        st.caption(
            f"Japan share is highest for **{jp.index[0]}** ({jp.iloc[0]:.0%}) vs {overall_share['Japan']:.0%} overall in the current filter. "
            "Shares are of total units, so a few blockbusters can dominate a group."
        )

    # ---------------- Year trends ----------------
    st.subheader("How did releases and sales change over time?")
    yearly = filtered.groupby("Year").agg(n=(TARGET, "size"), median=(TARGET, "median"), mean=(TARGET, "mean"), total=(TARGET, "sum")).reset_index()
    t1, t2 = st.columns(2)
    with t1:
        fig = px.bar(yearly, x="Year", y="n", title="Titles released per year", labels={"n": "Titles"})
        fig.update_traces(marker_color=PALETTE[0], marker_line_width=0)
        show_chart(style(fig, height=330, legend=False))
    with t2:
        stat = st.radio("Sales statistic", ["Median", "Mean", "Total"], horizontal=True, key="eda_trend_stat")
        fig = px.line(yearly, x="Year", y=stat.lower(), markers=True, hover_data={"n": True},
                      title=f"{stat} global sales per title by release year" if stat != "Total" else "Total global sales by release year",
                      labels={stat.lower(): "Global sales (M copies)"})
        fig.update_traces(line=dict(color=PALETTE[1], width=2), marker=dict(size=8))
        fig.update_yaxes(rangemode="tozero")
        show_chart(style(fig, height=330, legend=False))
    st.caption("Years with few titles (e.g. 2017, 2020) produce unstable statistics. The fall after 2008 partly reflects the data source's coverage, not only the market.")

    # ---------------- Correlations ----------------
    st.subheader("Which numeric variables move together?")
    method = st.radio("Correlation", ["Spearman (rank, robust to skew)", "Pearson (linear)"], horizontal=True, key="eda_corr")
    num_cols = [c for c in ORIGINAL_NUM if c in filtered] + [TARGET]
    corr = filtered[num_cols].corr(method="spearman" if method.startswith("Spearman") else "pearson")
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING,
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="r")))
    fig.update_layout(title=f"{method.split(' ')[0]} correlation of the saved model's numeric inputs and the target (n = {len(filtered):,})")
    fig.update_xaxes(tickangle=-45)
    show_chart(style(fig, height=640, legend=False))
    st.caption("Correlation with Global_Sales is a descriptive association. Strong mutual correlation between inputs also means permutation importance can be shared or distorted between them.")

    # ---------------- Genre x generation ----------------
    st.subheader("Do genre sales patterns differ across console generations?")
    h1, h2 = st.columns(2)
    stat = h1.radio("Cell value", ["Median", "Mean", "Title count"], horizontal=True, key="eda_heat_stat")
    hide = h2.number_input("Blank out cells with fewer than n titles", 1, 200, 10, key="eda_heat_min")
    counts = filtered.pivot_table(index="Genre", columns="ConsoleGen", values=TARGET, aggfunc="size").fillna(0)
    agg = {"Median": "median", "Mean": "mean", "Title count": "size"}[stat]
    pivot = filtered.pivot_table(index="Genre", columns="ConsoleGen", values=TARGET, aggfunc=agg)
    cols = [c for c in GENERATION_ORDER if c in pivot.columns]
    pivot, counts = pivot[cols], counts.reindex(columns=cols).fillna(0)
    masked = pivot.where(counts >= hide)
    fig = go.Figure(
        go.Heatmap(
            z=masked.values, x=cols, y=masked.index, colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            customdata=counts.reindex(masked.index).values,
            text=np.where(np.isnan(masked.values), "", np.vectorize(lambda v: f"{v:.2f}" if stat != "Title count" else f"{v:.0f}")(np.nan_to_num(masked.values))),
            texttemplate="%{text}", hovertemplate="%{y} / %{x}<br>value: %{z:.3f}<br>n = %{customdata:.0f}<extra></extra>",
            colorbar=dict(title=stat if stat == "Title count" else f"{stat} (M)"),
        )
    )
    fig.update_layout(title=f"{stat} global sales by genre x console generation (blank: n < {hide})" if stat != "Title count" else "Titles by genre x console generation")
    show_chart(style(fig, height=520, legend=False))
    st.caption("'Other' groups platforms outside generations 5-8 (e.g. 2600, NES, SNES, Genesis, PC).")
