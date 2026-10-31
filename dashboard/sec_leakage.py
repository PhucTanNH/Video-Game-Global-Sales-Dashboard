"""Section 5 - Data leakage audit."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from vgsales_core.config import ABSENT_LEAKY_COLUMNS, FORBIDDEN_PREDICTORS, LEAKAGE_REGISTRY, STATUS_LABELS, TARGET
from vgsales_core.evaluation import describe_protocol
from vgsales_core.features import FEATURE_CONFIG_KEYS, STRICT_CAT, STRICT_NUM, assert_no_forbidden, get_feature_config
from vgsales_core.models import MODEL_NAMES, availability, default_params
from vgsales_core.plots import PALETTE, style

from .sec_models import comparison_groups, display_frame, highlight_best, protocol_widgets, run_experiments, session_results_frame
from .state import csv_download, empty_state, experiments, get_audit, get_data, show_chart, show_table

STATUS_COLORS = {"forbidden": "#e34948", "leaky": "#eb6834", "uncertain": "#eda100", "pre-release*": "#2a78d6", "pre-release": "#1baf7a"}


def render():
    st.header("Data leakage audit")
    st.markdown(
        "A chronological split alone does not make a model leakage-free: each **feature** must also be computable from information "
        "available at the forecast origin (before release). This section audits the inputs by recomputing them from raw rows."
    )
    st.markdown(
        """
| Leakage type | What it means here | How the dashboard handles it |
|---|---|---|
| **Target leakage** | Inputs derived from the outcome (regional sales, shares, log sales, rank) | Blocked by a guard (`assert_no_forbidden`) on every feature configuration |
| **Post-origin leakage** | Inputs using information from after release (full-period publisher ranking, same-year sales order, lifetime sales of earlier titles) | Audited below; Strict and Rebuilt configurations remove or reduce it |
| **Split leakage** | Training and validation sharing a release year | Session CV validates on whole release years; the saved CV used row-based folds |
| **Selection leakage** | Choosing models or settings on the test period | Selection by CV only; the reused 2013+ period is labelled *retrospective* |
"""
    )
    audit, franchise = get_audit()
    data = get_data()

    st.subheader("1. Variable roles")
    rows = []
    strict = set(STRICT_CAT) | set(STRICT_NUM)
    original = set(get_feature_config("original").features)
    for col in list(data.columns):
        status, note = LEAKAGE_REGISTRY.get(col, (None, None))
        if status is None:
            continue
        rows.append({"Column": col, "Status": STATUS_LABELS[status], "In Original-27": col in original, "In Strict": col in strict, "Evidence / reason": note})
    for col in ABSENT_LEAKY_COLUMNS:
        rows.append({"Column": col, "Status": "Not in dataset", "In Original-27": False, "In Strict": False, "Evidence / reason": "Discussed in the report; absent from this Excel file, so it cannot leak here."})
    roles = pd.DataFrame(rows)
    pick = st.multiselect("Show statuses", sorted(roles["Status"].unique()), default=sorted(roles["Status"].unique()), key="leak_status")
    show_table(roles[roles["Status"].isin(pick)], hide_index=True, height=380)
    csv_download(roles, "Download variable roles (CSV)", "leakage_variable_roles.csv", "dl_roles")
    st.caption("'Prior-year window (verified)' columns are strictly prior-year, but they use the *lifetime* sales of earlier titles as recorded years later. "
               "At a real forecast origin those totals were still growing, so some post-origin information remains. Rebuilt features with a temporal gap reduce, but do not remove, this.")

    st.subheader("2. Recomputation evidence")
    c = st.columns(4)
    c[0].metric("Rows in shared franchise-years", f"{franchise['rows_sharing_franchise_year']:,}")
    c[1].metric("Sales-ordered franchise-years", f"{franchise['share_negative_rank_corr']:.0%}",
                help="Share of franchise-years where Prev_Franchise_TitleCount rises as sales fall (rank correlation < 0).")
    c[2].metric("Rows: max uses same-year titles", f"{franchise['rows_maxsales_differs']:,}")
    c[3].metric("Rows: inflated franchise count", f"{franchise['rows_count_inflated']:,}")
    st.markdown(
        "Within a release year, the file's franchise history was accumulated in **descending sales order**. A lower-selling port therefore "
        "'inherits' the sales of the best-selling version released the same year. For example, *Saints Row 2* (PS3, 2008) has "
        "Prev_Franchise_MaxSales = 3.03, which is the X360 version's sales from the same year."
    )
    a = audit.copy()
    a["Status label"] = a["Status"].map(STATUS_LABELS)
    fig = px.bar(a.sort_values("Agreement with strict rebuild (gap 0)"), x="Agreement with strict rebuild (gap 0)", y="Feature", orientation="h",
                 color="Status", color_discrete_map=STATUS_COLORS, hover_data=["Note"],
                 title="Agreement between each Original-27 column and its strict prior-year rebuild (share of rows equal)")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.02])
    show_chart(style(fig, height=760))
    st.caption("100% agreement confirms the column is computable from strictly earlier years. Lower values mean the file uses extra information "
               "(leaky) or an undocumented rule (uncertain). Publisher_Tier: the file ranks publishers by title count over 1980-2020 (top 10 = Major, 11-50 = Mid).")

    st.subheader("3. Feature configurations")
    cfg_rows = []
    for key in FEATURE_CONFIG_KEYS:
        cfg = get_feature_config(key)
        try:
            assert_no_forbidden(cfg.features)
            guard = "passed"
        except ValueError as exc:
            guard = str(exc)
        flagged = [f for f in cfg.features if LEAKAGE_REGISTRY.get(f, ("",))[0] in ("leaky", "uncertain")] if cfg.gap is None else \
                  [f for f in cfg.features if f in ("Is_Sequel", "Num_Platforms", "Is_Multiplatform")]
        cfg_rows.append({"Configuration": cfg.label, "Inputs": len(cfg.features), "Forbidden-column guard": guard,
                         "Leaky / uncertain inputs": ", ".join(flagged) or "none", "Description": cfg.description})
    show_table(pd.DataFrame(cfg_rows), hide_index=True)
    st.caption("The report's 'Strict-13' list was not available to this dashboard, so the Strict configuration here is defined from the verified columns and labelled accordingly. "
               f"Forbidden columns ({', '.join(FORBIDDEN_PREDICTORS)}) never enter any configuration.")

    st.subheader("4. Compare feature configurations")
    st.markdown("Run the same model under several configurations with the same year-ahead protocol. Scores are expected to **drop** as leaked information is removed; "
                "that drop estimates how much of the original score depended on it.")
    cutoff, n_folds, seed = protocol_widgets("leak")
    available = [m for m in MODEL_NAMES if availability(m)[0] and m != "Stacking"]
    c1, c2 = st.columns([1, 2])
    model = c1.selectbox("Model (default settings)", available, index=available.index("XGBoost") if "XGBoost" in available else 0, key="leak_model")
    keys = c2.multiselect("Configurations", FEATURE_CONFIG_KEYS, default=FEATURE_CONFIG_KEYS, format_func=lambda k: get_feature_config(k).label, key="leak_cfgs")
    if st.button("Run configuration comparison", type="primary", disabled=not keys, key="leak_run"):
        jobs = [(model, default_params(model), k) for k in keys] + [("Median baseline", {}, keys[0])]
        run_experiments(jobs, cutoff, n_folds, seed)

    session = session_results_frame()
    if session.empty:
        empty_state("No runs yet.")
        return
    ok = session[(session["Status"] == "ok") & (session["Model"].isin([model, "Median baseline"]))]
    proto = describe_protocol(cutoff, n_folds, data["Year"].to_numpy())
    ok = ok[(ok["Evaluation protocol"] == proto) & (ok["Seed"].eq(seed) | ok["_baseline"])]
    labels = [get_feature_config(k).label for k in keys] + ["Baseline (no inputs)"]
    ok = ok[ok["Feature configuration"].isin(labels)]
    # Keep the default-parameter run for each configuration.
    ok = ok[ok["Hyperparameters"].ne("") & ~ok["Hyperparameters"].str.startswith("User-set")]
    if ok.empty:
        empty_state("No matching runs for this model / protocol yet. Press the button above.")
        return
    st.caption(f"Protocol for all rows: {proto}")
    tbl = ok.assign(Model=ok["Model"] + " - " + ok["Feature configuration"])
    show_table(highlight_best(display_frame(tbl, "CV RMSE mean")), hide_index=True)
    long = ok.melt(id_vars=["Feature configuration"], value_vars=["CV RMSE mean", "Holdout RMSE", "CV RMSLE mean", "Holdout RMSLE"],
                   var_name="Score", value_name="Value")
    fig = px.bar(long, x="Feature configuration", y="Value", color="Score", barmode="group", text_auto=".3f",
                 title=f"{model}: error by feature configuration (lower is better)", labels={"Value": "Error", "Feature configuration": ""})
    fig.update_yaxes(rangemode="tozero")
    show_chart(style(fig, height=430))
    st.caption("RMSE is in M copies; RMSLE is on the log1p scale. Differences between configurations of a few thousandths are within seed-to-seed variation; check the CV std column.")
