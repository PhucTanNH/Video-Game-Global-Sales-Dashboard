"""Section 4 - Error analysis by group (out-of-sample predictions only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vgsales_core.config import GENERATION_ORDER
from vgsales_core.evaluation import group_errors, year_group_label
from vgsales_core.plots import NEUTRAL, PALETTE, color_map, style

from .state import attach_context, csv_download, empty_state, get_saved_csv, prediction_sources, show_chart, show_table

DIMENSIONS = {"Genre": "Genre", "Platform": "Platform", "Console generation": "ConsoleGen", "Release-year group": "Year group",
              "Publisher tier": "Publisher_Tier"}


def render():
    st.header("Error analysis")
    st.caption("Where do the models err most? Computed from out-of-sample predictions only (retrospective holdout or CV validation years).")
    sources = prediction_sources()
    if not sources:
        empty_state("No predictions available. The saved model could not be loaded and no session experiment has been run.")
        return
    labels = list(sources)
    holdout_labels = [l for l in labels if sources[l]["set"] == "holdout"]
    picked = st.multiselect("Prediction sets to compare (same evaluation set recommended)", labels, default=holdout_labels[:1] or labels[:1], key="err_src")
    if not picked:
        empty_state("Select at least one prediction set.")
        return
    sets = {sources[l]["set"] for l in picked}
    if len(sets) > 1:
        st.warning("You mixed holdout and CV out-of-fold predictions; they cover different years and are not directly comparable.")
    c1, c2, c3 = st.columns(3)
    dim_label = c1.selectbox("Group by", list(DIMENSIONS), key="err_dim")
    metric = c2.radio("Metric", ["MAE", "RMSE", "Median abs. error", "RMSLE"], horizontal=True, key="err_metric")
    min_n = c3.number_input("Small-sample threshold (n)", 5, 500, 30, key="err_minn")
    dim = DIMENSIONS[dim_label]

    tables = []
    for lab in picked:
        frame = attach_context(sources[lab]["frame"])
        year_col = "valid_year" if "valid_year" in frame else "Year"
        frame["Year group"] = frame[year_col].map(year_group_label)
        g = group_errors(frame, dim, min_n=int(min_n))
        g.insert(0, "Prediction set", lab)
        g.insert(1, "Model", sources[lab]["model"])
        tables.append(g)
    table = pd.concat(tables, ignore_index=True)
    small_col = f"Small sample (n < {int(min_n)})"

    order = table.groupby(dim)[metric].mean().sort_values(ascending=False).index.tolist()
    if dim == "ConsoleGen":
        order = [g for g in GENERATION_ORDER if g in order]
    if dim == "Year group":
        order = sorted(order)
    cmap = color_map(table["Model"].unique())
    plot = table.assign(Reliability=np.where(table[small_col], f"n < {int(min_n)} (unreliable)", "adequate n"))
    fig = px.bar(plot, x=dim, y=metric, color="Model", barmode="group", pattern_shape="Reliability",
                 pattern_shape_map={"adequate n": "", f"n < {int(min_n)} (unreliable)": "/"},
                 hover_data={"n": True, "MAE": ":.3f", "RMSE": ":.3f", "Median abs. error": ":.3f", "Median actual": ":.3f"},
                 category_orders={dim: order}, color_discrete_map=cmap,
                 title=f"{metric} by {dim_label.lower()} ({'log1p scale' if metric == 'RMSLE' else 'M copies'}; hatched = fewer than {int(min_n)} titles)",
                 labels={metric: metric if metric == "RMSLE" else f"{metric} (M copies)", dim: dim_label})
    overall = table.groupby("Model").apply(lambda g: np.average(g["MAE"], weights=g["n"])) if metric == "MAE" else None
    if overall is not None and len(overall) == 1:
        fig.add_hline(y=float(overall.iloc[0]), line_dash="dash", line_color=NEUTRAL, annotation_text="overall MAE")
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(rangemode="tozero")
    show_chart(style(fig, height=440))

    fig = px.bar(table[table["Prediction set"] == picked[0]], x=dim, y="n", category_orders={dim: order},
                 title=f"Number of evaluated titles per group ({sources[picked[0]]['model']})", labels={"n": "Titles", dim: dim_label})
    fig.update_traces(marker_color=PALETTE[0])
    fig.add_hline(y=int(min_n), line_dash="dot", line_color=NEUTRAL, annotation_text=f"n = {int(min_n)}")
    fig.update_xaxes(tickangle=-35)
    show_chart(style(fig, height=300, legend=False))

    show_cols = ["Model", dim, "n", "MAE", "RMSE", "Median abs. error", "RMSLE", "Mean signed error", "Median actual", "MAE / overall MAE", small_col]
    show_table(table[show_cols].style.format({c: "{:.3f}" for c in ["MAE", "RMSE", "Median abs. error", "RMSLE", "Mean signed error", "Median actual", "MAE / overall MAE"]}),
               hide_index=True)
    csv_download(table, "Download group errors (CSV)", f"errors_by_{dim}.csv", "dl_err")

    flagged = table[(~table[small_col]) & (table["MAE / overall MAE"] >= 1.5)]
    if not flagged.empty:
        lines = [f"- **{r[dim]}** ({r['Model']}): MAE {r['MAE']:.3f}M = {r['MAE / overall MAE']:.1f}x overall, n = {r['n']}" for _, r in flagged.iterrows()]
        st.markdown("**Groups with relatively large errors (MAE >= 1.5x overall, adequate n):**\n" + "\n".join(lines))
        st.caption("High-MAE groups usually also have higher typical sales (see 'Median actual'): absolute errors scale with sales, so compare the RMSLE (relative error) column before concluding a group is 'harder'.")
    small = table[table[small_col]]
    if not small.empty:
        st.warning(f"{small[dim].nunique()} group(s) have fewer than {int(min_n)} evaluated titles; do not draw conclusions from them.")

    # Consistency check with the saved CSV
    saved_csv = get_saved_csv("error_by_genre.csv")
    saved_label = next((l for l in picked if not sources[l]["session"]), None)
    if saved_csv is not None and saved_label and dim == "Genre":
        mine = table[table["Prediction set"] == saved_label].set_index("Genre")
        ref = saved_csv.set_index("Genre")
        joined = ref.join(mine[["MAE", "n"]], how="outer")
        ok = np.allclose(joined["mean"], joined["MAE"], atol=1e-6) and (joined["count"] == joined["n"]).all()
        if ok:
            st.success("Saved-model genre MAE recomputed here matches models/error_by_genre.csv.", icon=":material/check_circle:")
        else:
            st.warning("Recomputed genre MAE differs from models/error_by_genre.csv.")
