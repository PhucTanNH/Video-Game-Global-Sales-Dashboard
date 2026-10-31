"""Section 6 - Pre-release sales prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vgsales_core.config import SAVED_TRAIN_END_YEAR, TARGET
from vgsales_core.evaluation import conformal_interval, conformal_quantile, empirical_coverage
from vgsales_core.features import get_feature_config
from vgsales_core.predict import NEW_FRANCHISE, NEW_PUBLISHER, build_query, distribution_warnings, features_for_query

from .state import SAVED_LABEL, experiments, get_data, get_features, get_meta, get_saved_holdout, get_saved_model, show_table

MIN_YEAR, MAX_YEAR = 1981, 2025


def _model_options():
    opts = {}
    model, err = get_saved_model()
    if model is not None and get_meta():
        opts[SAVED_LABEL] = {"model": model, "feature_key": "original", "cutoff": SAVED_TRAIN_END_YEAR, "session": False}
    for key, r in experiments().items():
        if r.status == "ok" and not r.is_baseline and r.fitted is not None:
            opts[f"Session: {r.model} | {r.feature_label} | trained {get_data()['Year'].min()}-{r.cutoff} [{key}]"] = {
                "model": r.fitted, "feature_key": r.feature_config, "cutoff": r.cutoff, "session": True, "result": r}
    return opts, err


def render():
    st.header("Sales prediction (pre-release)")
    st.markdown("Estimate lifetime global sales for a hypothetical title using **only information available before its release**: "
                "the attributes you enter plus history from titles released in earlier years. No regional sales, ranks or review scores are used.")
    data = get_data()
    opts, err = _model_options()
    if not opts:
        st.error(f"No model available. {err} Run a session experiment in Model Comparison to train one.")
        return
    label = st.selectbox("Model", list(opts), key="pred_model")
    chosen = opts[label]
    cfg = get_feature_config(chosen["feature_key"])
    st.caption(f"Inputs: {cfg.label}. Training period ends {chosen['cutoff']}. "
               + ("Saved pretrained artefact." if not chosen["session"] else "Model retrained in this session (kept in memory only)."))

    franchises = data["Franchise"].value_counts()
    franchises = franchises[franchises >= 1].index.sort_values().tolist()
    publishers = sorted(data["Publisher"].unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.number_input("Release year", MIN_YEAR, MAX_YEAR, 2015, step=1, key="pred_year")
        genre = st.selectbox("Genre", sorted(data["Genre"].unique()), key="pred_genre")
        platform = st.selectbox("Platform", sorted(data["Platform"].unique()), index=sorted(data["Platform"].unique()).index("PS3"), key="pred_platform")
    with c2:
        publisher = st.selectbox("Publisher", [NEW_PUBLISHER] + publishers, index=1 + publishers.index("Electronic Arts") if "Electronic Arts" in publishers else 0, key="pred_pub")
        franchise = st.selectbox("Franchise", [NEW_FRANCHISE] + franchises, key="pred_franchise",
                                 help="Franchise history (best earlier title, number of earlier titles) is computed from earlier release years only.")
    with c3:
        prior = data[(data["Franchise"] == franchise) & (data["Year"] < year)] if franchise != NEW_FRANCHISE else data.iloc[0:0]
        is_sequel = st.checkbox("Sequel / later entry in a series", value=len(prior) > 0, key=f"pred_seq_{franchise}_{year}")
        num_platforms = st.slider("Announced launch platforms", 1, 10, 1, key="pred_np",
                                  help="Number of platforms the title is announced for. Counts platforms released in the same year or earlier.")
        if franchise != NEW_FRANCHISE:
            st.caption(f"Earlier franchise rows: {len(prior)}; best earlier sales: {prior[TARGET].max() if len(prior) else 0:.2f}M")

    # ---- validation ----
    errors, warnings = [], []
    query, prof = build_query(data, year=int(year), genre=genre, platform=platform, publisher=publisher, franchise=franchise,
                              is_sequel=is_sequel, num_platforms=num_platforms)
    if platform != "PC" and year < prof["Platform_Launch_Year"]:
        errors.append(f"{platform} launched in {prof['Platform_Launch_Year']}; a release in {year} is impossible.")
    if year > prof["last_year"]:
        warnings.append(f"The dataset has no {platform} titles after {prof['last_year']}; platform history for {year} is extrapolated.")
    if year > 2016:
        warnings.append("Release years after 2016 have almost no data (4 titles), so every history-based input is weak or zero.")
    if year > data["Year"].max() + 1:
        warnings.append(f"History windows for {year} contain no titles at all (data end in {data['Year'].max()}); the prediction is mostly driven by static attributes.")
    if franchise != NEW_FRANCHISE and len(prior) == 0:
        warnings.append(f"'{franchise}' has no titles before {year}; it is treated like a new IP for history features.")

    X_row, notes = features_for_query(data, query, chosen["feature_key"])
    for c in cfg.cat:
        if pd.isna(X_row[c].iloc[0]):
            X_row[c] = "Unknown"
    X_row[list(cfg.num)] = X_row[list(cfg.num)].astype(float).fillna(0.0)
    X_all = get_features(chosen["feature_key"])
    train_mask = data["Year"] <= chosen["cutoff"]
    warnings += distribution_warnings(X_all[train_mask], X_row, cfg.cat, cfg.num)
    if not chosen["session"]:
        warnings.append("The saved model was trained on the file's leaky franchise and publisher-tier columns; here they are computed strictly from earlier years, "
                        "which is honest for a new title but can differ from the training distribution (typically lower franchise counts).")

    for e in errors:
        st.error(e)
    with st.expander(f"Input checks ({len(warnings)} warning(s))", expanded=bool(warnings) and len(warnings) <= 4):
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("All inputs are within the model's training distribution.")

    st.markdown("**Prediction interval** (optional)")
    i1, i2 = st.columns(2)
    want_interval = i1.toggle("Show split-conformal interval", value=True, key="pred_int")
    level = i2.select_slider("Nominal coverage", [0.5, 0.8, 0.9], value=0.8, format_func=lambda v: f"{v:.0%}", key="pred_level")

    if st.button("Predict sales", type="primary", disabled=bool(errors), key="pred_go"):
        try:
            pred = float(np.clip(chosen["model"].predict(X_row[cfg.features])[0], 0, None))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            return
        st.subheader("Result")
        r1, r2 = st.columns([1, 2])
        r1.metric("Predicted lifetime global sales", f"{pred:.2f} M copies")
        tier = "Blockbuster (>= 5M)" if pred >= 5 else "Hit (1-5M)" if pred >= 1 else "Moderate (0.5-1M)" if pred >= 0.5 else "Low (< 0.5M)"
        r1.caption(f"Band: {tier}")
        with r2:
            if want_interval:
                if chosen["session"]:
                    res = chosen["result"]
                    calib = res.cv_predictions
                    q = conformal_quantile(calib["actual"], calib["predicted"], 1 - level)
                    lo, hi = conformal_interval([pred], q)
                    cov = empirical_coverage(res.holdout_predictions["actual"], res.holdout_predictions["predicted"], q) if res.holdout_predictions is not None else float("nan")
                    st.metric(f"{level:.0%} split-conformal interval", f"{lo[0]:.2f} - {hi[0]:.2f} M")
                    st.caption(f"Calibrated on {len(calib):,} out-of-fold CV predictions (validation years <= {res.cutoff}); log-scale half-width {q:.3f}. "
                               f"Coverage actually achieved on the {res.cutoff + 1}+ holdout: **{cov:.1%}** (nominal {level:.0%}).")
                else:
                    saved = get_saved_holdout()
                    if saved is None:
                        st.caption("No calibration data available.")
                    else:
                        fr = saved["predictions"]
                        q = conformal_quantile(fr["actual"], fr["predicted"], 1 - level)
                        lo, hi = conformal_interval([pred], q)
                        st.metric(f"{level:.0%} split-conformal interval", f"{lo[0]:.2f} - {hi[0]:.2f} M")
                        st.caption(f"Calibrated on the saved model's {len(fr):,} holdout residuals (2013+, log scale, half-width {q:.3f}). "
                                   "This is the only out-of-sample set for the saved model, so its coverage cannot be checked on separate data.")
                st.caption("Coverage holds on average over titles *if* future titles resemble the calibration titles (exchangeability). "
                           "Market drift over time can break this. The interval is multiplicative (log scale), so it is asymmetric in millions.")
        st.info("This is a statistical estimate from historical patterns, not a guaranteed outcome. Sales also depend on marketing, quality, "
                "reviews and timing, none of which are available before release.", icon=":material/info:")
        with st.expander("Inputs passed to the model"):
            shown = X_row[cfg.features].T.reset_index()
            shown.columns = ["Feature", "Value"]
            shown["Value"] = shown["Value"].astype(str)
            visible = f"titles released in or before {int(year) - 1 - (cfg.gap or 0)}"
            shown["How it was obtained"] = shown["Feature"].map(notes).fillna(f"computed from {visible}")
            for c in ["Genre", "Platform", "Year", "Is_Sequel"]:
                shown.loc[shown["Feature"] == c, "How it was obtained"] = "user input"
            for c in ["Platform_Manufacturer", "ConsoleGen", "Platform_Lifecycle_Stage", "Years_Since_Platform_Launch"]:
                shown.loc[shown["Feature"] == c, "How it was obtained"] = "static platform attribute"
            show_table(shown, hide_index=True)
