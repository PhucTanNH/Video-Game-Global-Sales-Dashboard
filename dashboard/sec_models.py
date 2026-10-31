"""Section 3 - Model comparison: saved artefacts, session experiments, diagnostics, importance."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vgsales_core.config import ALLOWED_CUTOFFS, DEFAULT_SEED, SAVED_TRAIN_END_YEAR, TARGET
from vgsales_core.evaluation import (
    METRIC_DIRECTION,
    METRIC_HELP,
    ExperimentResult,
    describe_protocol,
    permutation_importance_frame,
    result_row,
    run_single,
)
from vgsales_core.features import FEATURE_CONFIG_KEYS, get_feature_config
from vgsales_core.models import (
    BASELINE_NAMES,
    MODEL_NAMES,
    MODEL_SPECS,
    MedianBaseline,
    PublisherGenreMedianBaseline,
    availability,
    build_estimator,
    default_params,
    tuning_status,
)
from vgsales_core.plots import NEUTRAL, PALETTE, actual_vs_predicted, color_map, histogram, metric_bars, style

from .state import (
    SAVED_LABEL,
    attach_context,
    csv_download,
    empty_state,
    experiment_key,
    experiments,
    get_data,
    get_features,
    get_meta,
    get_saved_csv,
    get_saved_holdout,
    get_saved_model,
    prediction_sources,
    show_chart,
    show_table,
)

# Rough single-fit cost on ~14k rows with 2 threads (measured in this project), seconds.
FIT_COST = {"Linear Regression": 0.5, "Ridge": 0.2, "kNN": 0.5, "Random Forest": 11, "XGBoost": 1.5, "LightGBM": 2,
            "CatBoost": 8, "Median baseline": 0.01, "Publisher-Genre median baseline": 0.1}
SAVED_NAME_MAP = {"XGBoost (tuned)": "XGBoost", "LightGBM (tuned)": "LightGBM", "Stacking (XGB + LGBM + CatBoost + RF)": "Stacking",
                  "Median (baseline)": "Median baseline"}
METRICS = ["RMSE", "MAE", "R2", "RMSLE"]
PRETTY = {"R2": "R²"}


def pretty(col: str) -> str:
    for k, v in PRETTY.items():
        col = col.replace(k, v)
    return col


# ---------------------------------------------------------------------------
# Experiment settings and execution (shared with the leakage section)
# ---------------------------------------------------------------------------
def protocol_widgets(prefix: str) -> tuple[int, int, int]:
    data = get_data()
    c1, c2, c3 = st.columns(3)
    cutoff = c1.selectbox(
        "Development cut-off year", ALLOWED_CUTOFFS, index=ALLOWED_CUTOFFS.index(SAVED_TRAIN_END_YEAR), key=f"{prefix}_cutoff",
        help="Years <= cut-off are used for CV and the final fit; later years form the retrospective holdout. 2012 = saved split.",
    )
    n_folds = c2.slider("Year-ahead CV folds", 2, 6, 3, key=f"{prefix}_folds",
                        help="Each fold validates on one release year, training on all earlier years.")
    seed = c3.number_input("Random seed", 0, 2**31 - 1, DEFAULT_SEED, step=1, key=f"{prefix}_seed")
    st.caption(describe_protocol(cutoff, n_folds, data["Year"].to_numpy()))
    if cutoff != SAVED_TRAIN_END_YEAR:
        st.warning(f"Cut-off {cutoff} differs from the saved split (2012): results are not comparable with the saved artefacts.")
    return int(cutoff), int(n_folds), int(seed)


def hyperparameter_widgets(name: str) -> dict:
    spec = MODEL_SPECS[name]
    params = {}
    if not spec.params:
        st.caption("No hyperparameters.")
    for p in spec.params:
        key = f"hp_{name}_{p.name}"
        if p.kind == "int":
            params[p.name] = int(st.number_input(p.name, int(p.min), int(p.max), int(p.default), step=int(p.step or 1), key=key, help=p.help or None))
        elif p.kind == "float":
            if p.log:
                params[p.name] = float(st.number_input(p.name, float(p.min), float(p.max), float(p.default), format="%.4f", key=key, help=p.help or None))
            else:
                params[p.name] = float(st.number_input(p.name, float(p.min), float(p.max), float(p.default), step=float(p.step or 0.01), format="%.4f", key=key, help=p.help or None))
        elif p.kind == "select":
            params[p.name] = st.selectbox(p.name, p.options, index=p.options.index(p.default), key=key)
        elif p.kind == "multi":
            params[p.name] = tuple(st.multiselect(p.name, p.options, default=list(p.default), key=key))
    if spec.note:
        st.caption(spec.note)
    st.caption(f"Status: {tuning_status(name, params)}")
    return params


def all_current_params() -> dict:
    """Current hyperparameters of every model (widgets if rendered, else defaults)."""
    out = {}
    for name in MODEL_NAMES:
        p = default_params(name)
        for k in p:
            key = f"hp_{name}_{k}"
            if key in st.session_state:
                p[k] = st.session_state[key]
                if isinstance(p[k], list):
                    p[k] = tuple(p[k])
        out[name] = p
    return out


def estimate_seconds(jobs, n_folds: int) -> float:
    total = 0.0
    for name, params, _ in jobs:
        if name == "Stacking":
            base = sum(FIT_COST.get(b, 5) for b in params.get("base_models", ()))
            per_fit = base * (int(params.get("inner_splits", 3)) + 1)
        else:
            per_fit = FIT_COST.get(name, 5)
        total += per_fit * (n_folds + 1)
    return total


def run_experiments(jobs, cutoff: int, n_folds: int, seed: int) -> list[str]:
    """Run (model, params, feature_key) jobs; results go to session state. Returns keys."""
    data = get_data()
    y = data[TARGET]
    store = experiments()
    params_all = all_current_params()
    keys = []
    bar = st.progress(0.0, text="Preparing experiment...")
    with st.status(f"Running {len(jobs)} job(s)...", expanded=True) as status:
        for i, (name, params, fkey) in enumerate(jobs):
            is_base = name in BASELINE_NAMES
            eff_key = "baseline" if is_base else fkey
            stack_ctx = {b: params_all[b] for b in params.get("base_models", ())} if name == "Stacking" else {}
            key = experiment_key(name, {**params, "_bases": stack_ctx}, eff_key, cutoff, n_folds, seed if not is_base else 0)
            keys.append(key)
            if key in store and store[key].status == "ok":
                status.write(f"Reused cached result: {name} ({eff_key})")
                bar.progress((i + 1) / len(jobs), text=f"Reused {name}")
                continue

            def progress(msg, i=i):
                bar.progress(min((i + 0.5) / len(jobs), 1.0), text=msg)

            t0 = time.perf_counter()
            try:
                if is_base:
                    est = MedianBaseline() if name == "Median baseline" else PublisherGenreMedianBaseline(min_count=3)
                    X = data[["Publisher", "Genre"]]
                    label = "Baseline (no inputs)" if name == "Median baseline" else "Baseline (Publisher, Genre medians)"
                    status_txt = "Fixed rule"
                else:
                    cfg = get_feature_config(fkey)
                    est = build_estimator(name, params, cfg, seed=seed, n_jobs=2, all_params=params_all)
                    X = get_features(fkey)
                    label = cfg.label
                    status_txt = tuning_status(name, params)
                res = run_single(name, est, X, y, data, cutoff, n_folds, feature_config=eff_key, feature_label=label,
                                 params=params, tuning_status=status_txt, seed=seed, progress=progress, is_baseline=is_base)
            except Exception as exc:  # noqa: BLE001 - construction failure (e.g. missing dependency)
                res = ExperimentResult(model=name, feature_config=eff_key, feature_label=fkey, params=params,
                                       tuning_status="-", protocol=describe_protocol(cutoff, n_folds, data["Year"].to_numpy()),
                                       cutoff=cutoff, seed=seed, status="failed", error=f"{type(exc).__name__}: {exc}", is_baseline=is_base)
            store[key] = res
            dt = time.perf_counter() - t0
            if res.status == "ok":
                m = res.holdout_metrics
                cv = res.cv_summary()
                status.write(f"Finished {name} on {res.feature_label} in {dt:.1f}s; CV RMSE {cv.get('CV RMSE mean', float('nan')):.3f}, holdout RMSE {m.get('RMSE', float('nan')):.3f}")
            else:
                status.write(f"FAILED {name}: {res.error}")
            bar.progress((i + 1) / len(jobs), text=f"Done {name}")
        failed = sum(store[k].status != "ok" for k in keys)
        status.update(label=f"Experiment finished: {len(keys) - failed} succeeded, {failed} failed", state="error" if failed else "complete", expanded=bool(failed))
    return keys


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------
def saved_results_frame() -> pd.DataFrame | None:
    test = get_saved_csv("test_results.csv")
    cv = get_saved_csv("cv_results.csv")
    meta = get_meta() or {}
    if test is None:
        return None
    test = test.rename(columns={"RMSE (M)": "Holdout RMSE", "MAE (M)": "Holdout MAE", "R2": "Holdout R2"})
    if cv is not None:
        cv = cv.rename(columns={
            "RMSE_CV_Mean": "CV RMSE mean", "RMSE_CV_Std": "CV RMSE std", "MAE_CV_Mean": "CV MAE mean",
            "MAE_CV_Std": "CV MAE std", "R2_CV_Mean": "CV R2 mean", "R2_CV_Std": "CV R2 std"})
        frame = test.merge(cv, on="Model", how="left")
    else:
        frame = test
    frame["Holdout RMSLE"] = np.nan
    frame["Holdout inference time (s)"] = np.nan
    saved = get_saved_holdout()
    deployed = meta.get("model")
    if saved and deployed in set(frame["Model"]):
        idx = frame["Model"] == deployed
        frame.loc[idx, "Holdout RMSLE"] = saved["metrics"]["RMSLE"]
        frame.loc[idx, "Holdout inference time (s)"] = saved["inference_time_s"]
    frame["Source"] = "Saved artefact (train_models.py)"
    frame["Feature configuration"] = np.where(frame["Model"].str.contains("Median"), "Baseline (no inputs)", "Original-27")
    def _status(m):
        spec = MODEL_SPECS.get(SAVED_NAME_MAP.get(m, m))
        return spec.default_status if spec else "Fixed rule"

    frame["Hyperparameters"] = frame["Model"].map(_status)
    frame["Evaluation protocol"] = "Row-based TimeSeriesSplit(5) on 1980-2012 (folds can split a release year); holdout 2013-2020"
    frame["Holdout period"] = "2013-2020 (retrospective)"
    frame["Deployed"] = frame["Model"] == deployed
    frame["Final fit time (s)"] = np.nan
    return frame


def session_results_frame() -> pd.DataFrame:
    rows = []
    for key, res in experiments().items():
        r = result_row(res)
        r["Run ID"] = key
        r["Group"] = (res.feature_config, res.cutoff, len(res.fold_scores) if res.fold_scores else None)
        r["_cutoff"] = res.cutoff
        r["_folds"] = res.protocol
        r["_baseline"] = res.is_baseline
        r["_feature"] = res.feature_config
        rows.append(r)
    return pd.DataFrame(rows)


def comparison_groups(session: pd.DataFrame) -> dict:
    """Comparable groups: same feature configuration + protocol; baselines join every group with the same protocol."""
    groups = {}
    if session.empty:
        return groups
    ok = session[session["Status"] == "ok"]
    models = ok[~ok["_baseline"]]
    bases = ok[ok["_baseline"]]
    for (feat, proto), g in models.groupby(["Feature configuration", "Evaluation protocol"], sort=False):
        b = bases[bases["Evaluation protocol"] == proto]
        groups[f"{feat} | {proto}"] = pd.concat([g, b], ignore_index=True)
    for proto, b in bases.groupby("Evaluation protocol", sort=False):
        if not any(proto in k for k in groups):
            groups[f"Baselines only | {proto}"] = b.reset_index(drop=True)
    return groups


def metric_of(col: str) -> str | None:
    """'Holdout RMSE' -> 'RMSE', 'CV R² mean' -> 'R2'; std / non-metric columns -> None."""
    if "std" in col:
        return None
    core = col.replace("Holdout ", "").replace("CV ", "").replace(" mean", "").replace("R²", "R2").strip()
    return core if core in METRICS else None


def highlight_best(frame: pd.DataFrame):
    def best(col: pd.Series):
        metric = metric_of(col.name)
        vals = pd.to_numeric(col, errors="coerce")
        if metric is None or vals.dropna().empty:
            return [""] * len(col)
        target = vals.min() if METRIC_DIRECTION[metric] == "min" else vals.max()
        return ["background-color: rgba(27,175,122,0.28); font-weight: 600" if (pd.notna(v) and np.isclose(v, target)) else "" for v in vals]

    metric_cols = [c for c in frame.columns if metric_of(c)]
    fmt = {c: "{:.4f}" for c in frame.columns if frame[c].dtype.kind == "f"}
    return frame.style.apply(best, subset=metric_cols).format(fmt, na_rep="n/a")


DISPLAY_COLS = ["Model", "Holdout RMSE", "Holdout MAE", "Holdout R2", "Holdout RMSLE", "CV RMSE mean", "CV RMSE std", "CV MAE mean",
                "CV MAE std", "CV R2 mean", "CV R2 std", "CV RMSLE mean", "CV RMSLE std", "Final fit time (s)", "Holdout inference time (s)",
                "CV time (s)", "Holdout period", "Feature configuration", "Hyperparameters", "Evaluation protocol", "Seed", "Run ID"]


def display_frame(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLS if c in frame]
    out = frame[cols].copy()
    if sort_by in out:
        metric = next((m for m in METRICS if sort_by.endswith(m) or f"CV {m} mean" == sort_by), "RMSE")
        out = out.sort_values(sort_by, ascending=METRIC_DIRECTION[metric] == "min", na_position="last")
    return out.rename(columns=pretty).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_run():
    st.markdown("Select models, set the protocol, and press **Run experiment**. Nothing is retrained until you press the button. "
                "Results stay available in this browser session; identical configurations are reused, not retrained.")
    cutoff, n_folds, seed = protocol_widgets("exp")
    c1, c2 = st.columns([2, 1])
    available = []
    for name in MODEL_NAMES:
        ok, why = availability(name)
        if ok:
            available.append(name)
        else:
            st.warning(f"{name} disabled: {why}")
    chosen = c1.multiselect("Models", available, default=[m for m in ["Ridge", "XGBoost", "LightGBM"] if m in available], key="exp_models")
    feature_key = c2.selectbox("Feature configuration", FEATURE_CONFIG_KEYS, format_func=lambda k: get_feature_config(k).label, key="exp_feat")
    st.caption(get_feature_config(feature_key).description)
    b1, b2 = st.columns(2)
    use_median = b1.checkbox("Median baseline", value=True, key="exp_b_med")
    use_pg = b2.checkbox("Publisher-Genre median baseline", value=True, key="exp_b_pg",
                         help="Median training sales of the same publisher and genre (>= 3 titles), falling back to the genre median, then the global median.")

    params_by_model = {}
    if chosen:
        st.markdown("**Hyperparameters** (defaults = settings used for the saved artefacts)")
        cols = st.columns(min(3, len(chosen)))
        for i, name in enumerate(chosen):
            with cols[i % len(cols)].expander(name, expanded=False):
                params_by_model[name] = hyperparameter_widgets(name)
        if "Stacking" in chosen:
            with st.expander("Base learners used inside Stacking"):
                st.caption("Stacking uses each base learner's settings from its own panel; open them here if they are not selected above.")
                for base in params_by_model["Stacking"].get("base_models", ()):
                    if base not in chosen:
                        with st.container(border=True):
                            st.markdown(f"**{base}**")
                            hyperparameter_widgets(base)

    jobs = [(m, params_by_model.get(m, default_params(m)), feature_key) for m in chosen]
    if use_median:
        jobs.append(("Median baseline", {}, feature_key))
    if use_pg:
        jobs.append(("Publisher-Genre median baseline", {"min_count": 3}, feature_key))
    est = estimate_seconds(jobs, n_folds)
    st.caption(f"Estimated run time: ~{est / 60:.1f} min on 2 CPU threads ({len(jobs)} job(s), {n_folds} CV folds + 1 final fit each)."
               + (" Stacking is the slowest option." if "Stacking" in chosen else ""))
    st.info("Select configurations by **CV** scores only. Choosing hyperparameters by looking at the holdout turns it into a tuning set (selection leakage).", icon=":material/rule:")
    run = st.button("Run experiment", type="primary", disabled=not jobs, key="exp_run")
    clear = st.button("Clear session results", key="exp_clear")
    if clear:
        st.session_state["experiments"] = {}
        st.session_state.pop("perm_results", None)
        st.toast("Session results cleared.")
    if run:
        run_experiments(jobs, cutoff, n_folds, seed)
    if not jobs:
        empty_state("Choose at least one model or baseline.")


def tab_results():
    st.markdown("Scores are only ranked **within** a block that shares the same data, feature configuration and protocol. "
                "Green cells mark the best value per metric in that block (lower RMSE/MAE/RMSLE, higher R²). "
                "Cells shown as **None** mean the metric was not recorded or is not applicable for that run.")
    with st.expander("Metric definitions"):
        for k, v in METRIC_HELP.items():
            st.markdown(f"- **{pretty(k)}**: {v}")
        st.markdown("- **CV mean / std**: mean and sample standard deviation across validation folds.\n"
                    "- **Final fit time**: refit on the full development period. **Inference time**: predicting the whole holdout.")
    sort_by = st.selectbox("Default sort", ["CV RMSE mean", "Holdout RMSE", "Holdout MAE", "Holdout R2", "Holdout RMSLE", "CV MAE mean", "CV R2 mean"],
                           format_func=pretty, key="res_sort")

    st.subheader("A. Saved artefacts (pretrained, from models/*.csv)")
    saved = saved_results_frame()
    if saved is None:
        st.warning("models/test_results.csv is missing; saved results unavailable.")
    else:
        show_table(highlight_best(display_frame(saved, sort_by)), hide_index=True)
        st.caption("Protocol: row-based TimeSeriesSplit(5) on 1980-2012 for selection; the deployed model (XGBoost tuned) was chosen by lowest CV RMSE. "
                   "Holdout RMSLE and timing were not saved for these runs: RMSLE and inference time are recomputed only for the deployed model; cells shown as None are not available. "
                   "Per-fold CV scores were not saved, only mean and std.")
        st.warning("Saved results use the Original-27 inputs, which include leaky columns, and a holdout reused during report development.", icon=":material/warning:")
        csv_download(saved, "Download saved results (CSV)", "saved_model_results.csv", "dl_saved")

    st.subheader("B. Session experiments (retrained in this session)")
    session = session_results_frame()
    if session.empty:
        empty_state("No session experiments yet. Use the **Run experiment** tab.")
        return
    failed = session[session["Status"] != "ok"]
    if not failed.empty:
        st.error("Some runs failed and are excluded from comparisons:")
        show_table(failed[["Model", "Feature configuration", "Error"]], hide_index=True)
    groups = comparison_groups(session)
    if len(groups) > 1:
        st.warning(f"{len(groups)} incompatible comparison blocks: models are ranked only inside each block.", icon=":material/call_split:")
    for i, (label, frame) in enumerate(groups.items()):
        st.markdown(f"**Block {i + 1}: {label.split(' | ')[0]}**")
        st.caption(label.split(" | ", 1)[1])
        disp = display_frame(frame, sort_by)
        show_table(highlight_best(disp), hide_index=True)
        models_only = frame[~frame["_baseline"]].dropna(subset=["CV RMSE mean"])
        if not models_only.empty:
            pick = models_only.loc[models_only["CV RMSE mean"].idxmin()]
            st.caption(f"CV-selected model in this block (lowest CV RMSE): **{pick['Model']}**. The holdout columns are reported for that choice, not used to make it.")
    export = session.drop(columns=[c for c in session.columns if c.startswith("_") or c == "Group"])
    csv_download(export, "Download all session results (CSV)", "session_experiment_results.csv", "dl_session")
    fold_rows = []
    for key, res in experiments().items():
        for f in res.fold_scores:
            fold_rows.append({"Run ID": key, "Model": res.model, "Feature configuration": res.feature_label, **f})
    if fold_rows:
        csv_download(pd.DataFrame(fold_rows), "Download per-fold CV scores (CSV)", "session_cv_folds.csv", "dl_folds")


def _group_choice(key: str):
    saved = saved_results_frame()
    session = session_results_frame()
    options = {}
    if saved is not None:
        options["Saved artefacts (Original-27, row-based CV, holdout 2013-2020)"] = ("saved", saved)
    for label, frame in comparison_groups(session).items():
        options[f"Session: {label}"] = ("session", frame)
    if not options:
        return None, None, None
    choice = st.selectbox("Comparison block", list(options), key=key)
    kind, frame = options[choice]
    return choice, kind, frame


def tab_charts():
    choice, kind, frame = _group_choice("chart_group")
    if frame is None:
        empty_state("No results available.")
        return
    all_models = frame["Model"].tolist()
    picked = st.multiselect("Models shown", all_models, default=all_models, key=f"chart_models_{choice}")
    frame = frame[frame["Model"].isin(picked)]
    if frame.empty:
        empty_state("Select at least one model.")
        return
    cmap = color_map(all_models)
    basis = st.radio("Scores", ["Retrospective holdout", "CV mean"], horizontal=True, key="chart_basis")
    pre = "Holdout " if basis.startswith("Retro") else "CV "
    suf = "" if pre == "Holdout " else " mean"
    rm, ma, r2 = f"{pre}RMSE{suf}", f"{pre}MAE{suf}", f"{pre}R2{suf}"

    c1, c2 = st.columns(2)
    with c1:
        f = frame.rename(columns={rm: "RMSE", ma: "MAE"})
        show_chart(metric_bars(f, "Model", ["RMSE", "MAE"], f"RMSE and MAE ({basis.lower()}; lower is better)", "Error (M copies)"))
    with c2:
        fig = px.bar(frame, x="Model", y=r2, color="Model", color_discrete_map=cmap, text_auto=".3f",
                     title=f"R² ({basis.lower()}; higher is better)", labels={r2: "R²", "Model": ""})
        fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=-30)
        show_chart(style(fig, legend=False))
    st.caption("Axes start at zero so small differences are not exaggerated. The grey line marks R² = 0 (no better than predicting the evaluation-set mean). R² can be strongly negative for models whose large errors exceed the variance of the evaluation set.")

    # CV variability
    st.markdown("**Cross-validation variability**")
    metric = st.selectbox("CV metric", METRICS, format_func=pretty, key="chart_cv_metric")
    if kind == "session":
        rows = []
        for key, res in experiments().items():
            if key in set(frame["Run ID"]):
                for fs in res.fold_scores:
                    rows.append({"Model": res.model, "Validation year": fs["valid_year"], metric: fs[metric], "n_valid": fs["n_valid"]})
        folds = pd.DataFrame(rows)
        if folds.empty:
            empty_state("No fold scores.")
        else:
            fig = px.line(folds, x="Validation year", y=metric, color="Model", markers=True, color_discrete_map=cmap, hover_data=["n_valid"],
                          title=f"{pretty(metric)} per year-ahead validation fold", labels={metric: pretty(metric)})
            fig.update_traces(line_width=2, marker_size=8)
            fig.update_xaxes(dtick=1)
            if metric != "R2":
                fig.update_yaxes(rangemode="tozero")
            show_chart(style(fig))
    else:
        mean_col, std_col = f"CV {metric} mean", f"CV {metric} std"
        if mean_col not in frame or frame[mean_col].isna().all():
            st.warning(f"CV {pretty(metric)} was not saved for the saved artefacts.")
        else:
            f = frame.dropna(subset=[mean_col])
            fig = px.bar(f, x="Model", y=mean_col, error_y=std_col, color="Model", color_discrete_map=cmap,
                         title=f"Saved CV {pretty(metric)}: mean ± 1 std over 5 row-based folds (per-fold values not saved)",
                         labels={mean_col: pretty(metric), "Model": ""})
            fig.update_xaxes(tickangle=-30)
            show_chart(style(fig, legend=False))

    # Relative to median baseline
    st.markdown("**Improvement over the median baseline**")
    base = frame[frame["Model"].str.contains("Median baseline|Median \\(baseline\\)", regex=True)]
    if base.empty:
        st.warning("No median baseline in this block; include it in the experiment to see relative improvements.")
    else:
        b = base.iloc[0]
        rel = frame[frame["Model"] != b["Model"]].copy()
        long = []
        for m, mlabel in [(rm, "RMSE"), (ma, "MAE")]:
            if pd.notna(b[m]) and b[m] > 0:
                for _, r in rel.iterrows():
                    if pd.notna(r[m]):
                        long.append({"Model": r["Model"], "Metric": mlabel, "Change vs median baseline (%)": 100 * (r[m] - b[m]) / b[m]})
        long = pd.DataFrame(long)
        if long.empty:
            st.warning("The baseline has no score on this basis.")
            long = pd.DataFrame(columns=["Model", "Metric", "Change vs median baseline (%)"])
        fig = px.bar(long, y="Model", x="Change vs median baseline (%)", color="Metric", barmode="group", orientation="h",
                     title=f"Error relative to the median baseline ({basis.lower()}); negative = better than baseline", text_auto=".1f")
        fig.add_vline(x=0, line_color=NEUTRAL)
        show_chart(style(fig, height=max(320, 50 * len(rel) + 120)))

    # Runtime
    st.markdown("**Runtime**")
    tcols = [c for c in ["Final fit time (s)", "Holdout inference time (s)", "CV time (s)"] if c in frame and frame[c].notna().any()]
    if kind == "saved":
        saved = get_saved_holdout()
        extra = f" The deployed model's holdout inference time, measured live in this app, is {saved['inference_time_s']:.3f}s." if saved else ""
        st.warning("Training and inference times were not recorded when the saved artefacts were produced; run a session experiment to measure them." + extra)
    elif not tcols:
        st.warning("Training and inference times were not recorded for the saved artefacts; run a session experiment to measure them.")
    else:
        long = frame.melt(id_vars="Model", value_vars=tcols, var_name="Measure", value_name="Seconds").dropna()
        logy = st.toggle("Log-scale time axis", value=True, key="chart_logt")
        fig = px.bar(long, x="Model", y="Seconds", color="Measure", barmode="group", title="Measured wall-clock time (seconds, 2 CPU threads)", text_auto=".2f")
        if logy:
            fig.update_yaxes(type="log")
        fig.update_xaxes(tickangle=-30)
        show_chart(style(fig))


def tab_diagnostics():
    sources = prediction_sources()
    if not sources:
        empty_state("No out-of-sample predictions available yet (saved model failed to load and no session runs).")
        return
    label = st.selectbox("Prediction set", list(sources), key="diag_src")
    src = sources[label]
    frame = attach_context(src["frame"])
    st.caption("Only out-of-sample predictions are shown: holdout predictions come from a model fitted on years <= cut-off; "
               "CV predictions come from fold models that never saw the validation year. No training-set predictions are used.")
    m = {"RMSE": np.sqrt(np.mean(frame["signed_error"] ** 2)), "MAE": frame["abs_error"].mean(),
         "Median abs. error": frame["abs_error"].median(), "n": len(frame)}
    c = st.columns(4)
    c[0].metric("RMSE (M)", f"{m['RMSE']:.3f}")
    c[1].metric("MAE (M)", f"{m['MAE']:.3f}")
    c[2].metric("Median |error| (M)", f"{m['Median abs. error']:.3f}")
    c[3].metric("Titles", f"{m['n']:,}")

    log_axes = st.toggle("Log-log axes (recommended for skewed sales)", value=True, key="diag_log")
    show_chart(actual_vs_predicted(frame, f"Actual vs predicted - {src['model']}", log_axes))
    c1, c2 = st.columns(2)
    with c1:
        clip = float(frame["signed_error"].abs().quantile(0.99))
        resid = frame["signed_error"].clip(-clip, clip)
        show_chart(histogram(resid, "Residual distribution (predicted - actual; clipped at 99th pct.)", "Residual (M copies)", color=PALETTE[0]))
    with c2:
        fig = px.scatter(frame, x="predicted", y="signed_error", opacity=0.5, hover_data=["Name", "Platform", "Year"],
                         title="Residual vs predicted", labels={"predicted": "Predicted (M copies)", "signed_error": "Residual: predicted - actual (M)"})
        fig.update_traces(marker=dict(size=8, color=PALETTE[0], line=dict(width=1, color="white")))
        fig.add_hline(y=0, line_color=NEUTRAL, line_dash="dash")
        show_chart(style(fig, legend=False))
    c3, c4 = st.columns(2)
    with c3:
        show_chart(histogram(frame["abs_error"], "Absolute-error distribution", "|Predicted - actual| (M copies)", color=PALETTE[1], log_x=True))
    with c4:
        e = np.sort(frame["abs_error"].to_numpy())
        fig = go.Figure(go.Scatter(x=e, y=np.arange(1, len(e) + 1) / len(e), mode="lines", line=dict(color=PALETTE[1], width=2)))
        fig.update_layout(title="Cumulative share of titles within an absolute error")
        fig.update_xaxes(type="log", title="Absolute error (M copies, log)")
        fig.update_yaxes(title="Share of titles", tickformat=".0%", range=[0, 1])
        show_chart(style(fig, legend=False))
    st.caption("Errors are concentrated in a few blockbusters: under-predicting hits dominates RMSE, while most titles have small absolute errors.")

    st.markdown("**Largest prediction errors**")
    n_top = st.slider("Rows", 5, 100, 20, key="diag_top")
    top = frame.sort_values("abs_error", ascending=False).head(n_top)
    cols = ["Name", "Platform", "Year", "Genre", "Publisher", "actual", "predicted", "signed_error", "abs_error"] + (["valid_year"] if "valid_year" in top else [])
    show_table(top[cols].rename(columns={"actual": "Actual (M)", "predicted": "Predicted (M)", "signed_error": "Signed error (M)", "abs_error": "Abs. error (M)"})
               .style.format({"Actual (M)": "{:.2f}", "Predicted (M)": "{:.2f}", "Signed error (M)": "{:+.2f}", "Abs. error (M)": "{:.2f}"}), hide_index=True)
    csv_download(frame, "Download this prediction set (CSV)", "predictions_selected.csv", "dl_pred_one")
    all_frames = []
    for lab, s in sources.items():
        all_frames.append(attach_context(s["frame"]).assign(prediction_set=lab, model=s["model"], evaluation=s["set"]))
    csv_download(pd.concat(all_frames, ignore_index=True), "Download all held-out predictions (CSV)", "heldout_predictions_all.csv", "dl_pred_all")


def tab_importance():
    st.markdown(
        "**Permutation importance** = increase in RMSE (M copies) on held-out titles when one input column is randomly shuffled, "
        "breaking its link with the target while leaving the model unchanged. Larger = the model relies more on that column *for prediction*. "
        "It is computed on raw input columns, so a categorical column counts as one feature."
    )
    st.warning("Correlated inputs (e.g. the Prev3Y averages and counts, Is_Multiplatform vs Num_Platforms) can share or mask importance. "
               "Predictive importance is not evidence of a causal effect. Coefficients, tree split-gain and permutation importance measure different things and are not compared here.",
               icon=":material/warning:")
    top_n = st.slider("Top features", 5, 27, 12, key="imp_top")

    st.subheader("Saved model (permutation_importance.csv)")
    imp = get_saved_csv("permutation_importance.csv")
    if imp is None:
        st.warning("models/permutation_importance.csv not found.")
    else:
        f = imp.sort_values("importance_mean", ascending=False).head(top_n).iloc[::-1]
        fig = px.bar(f, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
                     title="Saved XGBoost (tuned): permutation importance on the 2013+ holdout (5 repeats, seed 42)",
                     labels={"importance_mean": "Increase in RMSE (M copies)", "feature": ""})
        fig.update_traces(marker_color=PALETTE[0])
        show_chart(style(fig, height=max(320, 26 * len(f) + 120), legend=False))
        st.caption("The two strongest inputs (Prev_Franchise_MaxSales, Prev_Franchise_TitleCount) are flagged as leaky in the Leakage Audit, "
                   "so part of this importance reflects same-year information.")

    st.subheader("Session models (computed on demand)")
    fitted = {k: r for k, r in experiments().items() if r.status == "ok" and not r.is_baseline and r.fitted is not None and r.holdout_rows}
    if not fitted:
        empty_state("Run a session experiment first; permutation importance is then computed on that run's retrospective holdout.")
        return
    labels = {k: f"{r.model} | {r.feature_label} | cutoff {r.cutoff} [{k}]" for k, r in fitted.items()}
    picked = st.multiselect("Models", list(labels), format_func=labels.get, default=list(labels)[:2], key="imp_models")
    c1, c2, c3 = st.columns(3)
    repeats = c1.slider("Repeats", 1, 20, 5, key="imp_rep")
    seed = c2.number_input("Seed", 0, 2**31 - 1, DEFAULT_SEED, key="imp_seed")
    max_rows = c3.selectbox("Rows used", [500, 1000, 2000, None], index=3, format_func=lambda v: "All holdout rows" if v is None else f"Random {v}", key="imp_rows")
    if st.button("Compute permutation importance", key="imp_run", disabled=not picked):
        data = get_data()
        store = st.session_state.setdefault("perm_results", {})
        bar = st.progress(0.0)
        for i, k in enumerate(picked):
            r = fitted[k]
            X = get_features(r.feature_config)
            hold = np.where(data["Year"].to_numpy() > r.cutoff)[0]
            try:
                with st.spinner(f"Permuting inputs for {r.model}..."):
                    store[(k, repeats, seed, max_rows)] = permutation_importance_frame(r.fitted, X.iloc[hold], data[TARGET].iloc[hold],
                                                                                       n_repeats=repeats, seed=int(seed), max_rows=max_rows)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{r.model}: {type(exc).__name__}: {exc}")
            bar.progress((i + 1) / len(picked))
    store = st.session_state.get("perm_results", {})
    have = [(k, v) for (k, rep, sd, mr), v in store.items() if k in picked and rep == repeats and sd == seed and mr == max_rows]
    if not have:
        st.caption("Press the button to compute.")
        return
    feats = {fitted[k].feature_config for k, _ in have}
    if len(feats) > 1:
        st.warning("Selected models use different feature configurations; their importances are shown separately, not side by side.")
    for feat in feats:
        rows = []
        for k, v in have:
            if fitted[k].feature_config == feat:
                rows.append(v.assign(Model=f"{fitted[k].model} [{k}]"))
        long = pd.concat(rows)
        order = long.groupby("feature")["importance_mean"].max().sort_values(ascending=False).head(top_n).index[::-1]
        long = long[long["feature"].isin(order)]
        fig = px.bar(long, x="importance_mean", y="feature", color="Model", barmode="group", orientation="h", error_x="importance_std",
                     category_orders={"feature": list(order)}, labels={"importance_mean": "Increase in RMSE (M copies)", "feature": ""},
                     title=f"Permutation importance on the retrospective holdout - {get_feature_config(feat).label} ({repeats} repeats, seed {seed})")
        show_chart(style(fig, height=max(360, 22 * len(order) * len(rows) + 120)))
        csv_download(long, "Download importance (CSV)", f"permutation_importance_{feat}.csv", f"dl_imp_{feat}")


def render():
    st.header("Model comparison")
    model, err = get_saved_model()
    st.markdown(
        "Two kinds of results are kept strictly apart: **saved artefacts** (pretrained by `train_models.py`, loaded from `models/`) "
        "and **session experiments** (retrained here with year-ahead CV)."
    )
    if err:
        st.error(err)
    tabs = st.tabs(["Run experiment", "Results tables", "Charts", "Out-of-sample diagnostics", "Feature importance"])
    with tabs[0]:
        tab_run()
    with tabs[1]:
        tab_results()
    with tabs[2]:
        tab_charts()
    with tabs[3]:
        tab_diagnostics()
    with tabs[4]:
        tab_importance()
