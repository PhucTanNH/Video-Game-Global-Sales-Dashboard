"""Section 1 - Dataset overview."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vgsales_core.config import LEAKAGE_REGISTRY, ORIGINAL_CAT, ORIGINAL_NUM, STATUS_LABELS, TARGET
from vgsales_core.plots import histogram

from .state import csv_download, empty_state, get_meta, get_saved_csv, get_saved_holdout, get_saved_model, show_chart, show_table


def sales_summary(values: pd.Series) -> pd.DataFrame:
    v = values.dropna()
    q = v.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    rows = [
        ("Titles (n)", len(v)),
        ("Mean", v.mean()),
        ("Standard deviation", v.std()),
        ("Median", v.median()),
        ("10th percentile", q[0.1]),
        ("25th percentile (Q1)", q[0.25]),
        ("75th percentile (Q3)", q[0.75]),
        ("90th percentile", q[0.9]),
        ("99th percentile", q[0.99]),
        ("Maximum", v.max()),
        ("Skewness", v.skew()),
        ("Share of titles below 0.1M", (v < 0.1).mean()),
        ("Share of total sales from top 1% of titles", v[v >= q[0.99]].sum() / v.sum() if v.sum() else np.nan),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Global sales (M copies) / value"])


def render(data: pd.DataFrame, filtered: pd.DataFrame):
    st.header("Dataset overview")
    st.caption("Descriptive information about the dataset and the saved model. Sidebar filters apply to the distribution below.")

    c = st.columns(5)
    c[0].metric("Titles (rows)", f"{len(data):,}")
    c[1].metric("Release years", f"{data['Year'].min()}-{data['Year'].max()}")
    c[2].metric("Platforms", data["Platform"].nunique())
    c[3].metric("Publishers", data["Publisher"].nunique())
    c[4].metric("Missing values", int(data.isna().sum().sum()))
    st.caption(
        "One row = one title on one platform (multi-platform titles appear once per platform). "
        "Global_Sales are cumulative lifetime units in millions as recorded by VGChartz (Kaggle snapshot), not first-year sales. "
        "Only 4 titles are dated after 2016 (3 in 2017, 1 in 2020)."
    )

    st.subheader("Saved model card")
    meta = get_meta()
    model, err = get_saved_model()
    if meta is None:
        st.error("models/model_meta.json is missing or unreadable; the saved model card cannot be shown.")
    else:
        saved = get_saved_holdout()
        test_csv = get_saved_csv("test_results.csv")
        cols = st.columns([1.2, 1])
        with cols[0]:
            st.markdown(
                f"""
| Item | Value |
|---|---|
| Model | {meta['model']} (v{meta.get('model_version', '?')}) |
| Target | `{meta['target']}`, trained on {meta.get('target_transform', 'n/a')} |
| Inputs | {len(meta['features_cat'])} categorical + {len(meta['features_num'])} numeric = {len(meta['features_cat']) + len(meta['features_num'])} (Original-27) |
| Training period | {meta['train_years'][0]}-{meta['train_years'][1]} ({meta['train_rows']:,} rows) |
| Retrospective holdout | {meta['test_years'][0]}-{meta['test_years'][1]} ({meta['test_rows']:,} rows) |
| Selection | {meta.get('cv_strategy', 'n/a')} |
"""
            )
        with cols[1]:
            if err:
                st.error(err)
            elif saved:
                m = saved["metrics"]
                st.metric("Holdout RMSE (recomputed now)", f"{m['RMSE']:.3f} M")
                st.metric("Holdout MAE (recomputed now)", f"{m['MAE']:.3f} M")
                st.metric("Holdout R² (recomputed now)", f"{m['R2']:.3f}")
                stored = meta["metrics_test"]
                ok = all(abs(m[k] - stored[k]) < 1e-6 for k in ["RMSE", "MAE", "R2"])
                if ok:
                    st.success("Recomputed holdout metrics match model_meta.json and test_results.csv.", icon=":material/check_circle:")
                else:
                    st.warning("Recomputed holdout metrics differ from model_meta.json. Check the library versions.")
        st.warning(
            "The 2013+ holdout was already used while the report was developed, so these scores are **retrospective**, "
            "not an independent test. The saved model's inputs also include columns this dashboard flags as leaky "
            "(see Leakage Audit).",
            icon=":material/warning:",
        )
        if test_csv is None:
            st.caption("models/test_results.csv not found.")

    st.subheader("Global sales distribution")
    if filtered.empty:
        empty_state("No titles match the current sidebar filters. Widen the filters to see the distribution.")
    else:
        log_x = st.toggle("Log-scale x axis", value=True, key="ov_log")
        left, right = st.columns([1.6, 1])
        with left:
            show_chart(histogram(filtered[TARGET], f"Distribution of global sales (n = {len(filtered):,})", "Global sales (M copies)", log_x=log_x))
        with right:
            summary = sales_summary(filtered[TARGET])
            show_table(summary.style.format({"Global sales (M copies) / value": "{:,.3f}"}), hide_index=True)
        st.caption("Sales are strongly right-skewed (median far below mean), so group comparisons in this dashboard use medians and IQRs alongside means.")

    st.subheader("Data dictionary and predictor roles")
    rows = []
    for col in data.columns:
        status, note = LEAKAGE_REGISTRY.get(col, ("uncertain" if col.endswith(("Prev3Y_AvgSales", "Prev3Y_Count", "Prev1Y_AvgSales", "Prev1Y_Count", "Momentum", "vs_Market_Avg", "MarketShare_Prev3Y", "TotalSales", "MaxSales")) else "descriptive", "Not used by any feature configuration; not individually audited."))
        rows.append(
            {
                "Column": col,
                "Type": str(data[col].dtype),
                "Missing": int(data[col].isna().sum()),
                "Unique": int(data[col].nunique()),
                "Role / leakage status": STATUS_LABELS.get(status, status),
                "In saved model": col in ORIGINAL_CAT + ORIGINAL_NUM,
                "Note": note,
            }
        )
    dictionary = pd.DataFrame(rows)
    show_table(dictionary, hide_index=True, height=360)
    csv_download(dictionary, "Download data dictionary (CSV)", "data_dictionary.csv", "dl_dict")
    st.caption(
        "Platform_Cumulative_Titles / _Sales are generated at load time (strictly earlier years) and are not stored in the Excel file. "
        "Rank, critic-score and user-score columns are not present in this dataset."
    )
