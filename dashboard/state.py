"""Cached loaders, shared session state and small UI helpers."""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from vgsales_core.config import MODEL_DIR, SAVED_META_PATH, SAVED_MODEL_PATH, SAVED_TRAIN_END_YEAR, TARGET
from vgsales_core.data import load_prepared
from vgsales_core.evaluation import regression_metrics
from vgsales_core.features import (
    audit_original_features,
    build_feature_matrix,
    franchise_within_year_diagnostic,
)

warnings.filterwarnings("ignore", category=UserWarning)

SAVED_LABEL = "Saved artifact: XGBoost (tuned) - best_model.joblib"


# ---------------------------------------------------------------------------
# Rendering helpers (tolerate older Streamlit versions)
# ---------------------------------------------------------------------------
def show_chart(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def show_table(data, **kw):
    try:
        st.dataframe(data, width="stretch", **kw)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kw)


def csv_download(frame: pd.DataFrame, label: str, file_name: str, key: str):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv", key=key)


def empty_state(message: str):
    st.info(message, icon=":material/info:")


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_data() -> pd.DataFrame:
    return load_prepared()


@st.cache_data(show_spinner="Building feature matrix...")
def get_features(key: str) -> pd.DataFrame:
    return build_feature_matrix(get_data(), key)


@st.cache_data(show_spinner=False)
def get_meta() -> dict | None:
    try:
        with SAVED_META_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner="Loading saved model...")
def get_saved_model():
    """Return (model, error message)."""
    if not SAVED_MODEL_PATH.exists():
        return None, f"{SAVED_MODEL_PATH.name} not found in models/."
    try:
        return joblib.load(SAVED_MODEL_PATH), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not load the saved model ({type(exc).__name__}: {exc}). Check the xgboost / scikit-learn versions in requirements.txt."


@st.cache_data(show_spinner=False)
def get_saved_csv(name: str) -> pd.DataFrame | None:
    path = MODEL_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner="Scoring the saved model on the retrospective holdout...")
def get_saved_holdout() -> dict | None:
    """Out-of-sample predictions of the saved model (trained on <= 2012) for 2013+."""
    model, err = get_saved_model()
    meta = get_meta()
    if model is None or meta is None:
        return None
    data = get_data()
    feats = meta["features_cat"] + meta["features_num"]
    X = get_features("original")[feats]
    hold = np.where(data["Year"].to_numpy() > SAVED_TRAIN_END_YEAR)[0]
    t0 = time.perf_counter()
    pred = model.predict(X.iloc[hold])
    infer = time.perf_counter() - t0
    frame = pd.DataFrame({"row": hold, "Year": data["Year"].to_numpy()[hold], "actual": data[TARGET].to_numpy()[hold], "predicted": pred})
    return {"predictions": frame, "metrics": regression_metrics(frame["actual"], frame["predicted"]), "inference_time_s": infer}


@st.cache_data(show_spinner="Auditing feature construction...")
def get_audit():
    data = get_data()
    return audit_original_features(data), franchise_within_year_diagnostic(data)


# ---------------------------------------------------------------------------
# Session-state experiment store
# ---------------------------------------------------------------------------
def experiments() -> dict:
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = {}
    return st.session_state["experiments"]


def experiment_key(model: str, params: dict, feature_key: str, cutoff: int, n_folds: int, seed: int) -> str:
    blob = json.dumps([model, params, feature_key, cutoff, n_folds, seed], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def attach_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Join identifying columns (Name, Platform, ...) onto a prediction frame."""
    data = get_data()
    cols = ["Name", "Platform", "Year", "Genre", "Publisher", "ConsoleGen", "Publisher_Tier"]
    ctx = data.iloc[frame["row"].to_numpy()][cols].reset_index(drop=True)
    out = frame.drop(columns=[c for c in ["Year"] if c in frame]).reset_index(drop=True)
    out = pd.concat([ctx, out], axis=1)
    out["signed_error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["signed_error"].abs()
    return out


def prediction_sources() -> dict:
    """All out-of-sample prediction sets available: label -> dict(kind, frame, ...)."""
    sources = {}
    saved = get_saved_holdout()
    if saved:
        sources[f"{SAVED_LABEL} | retrospective holdout 2013+"] = {
            "model": "XGBoost (tuned) [saved]", "set": "holdout", "frame": saved["predictions"], "feature": "Original-27", "session": False}
    for key, res in experiments().items():
        if res.status != "ok":
            continue
        tag = f"{res.model} | {res.feature_label} | cutoff {res.cutoff} | seed {res.seed} [{key}]"
        if res.holdout_predictions is not None:
            sources[f"Session: {tag} | retrospective holdout {res.cutoff + 1}+"] = {
                "model": res.model, "set": "holdout", "frame": res.holdout_predictions, "feature": res.feature_label, "session": True, "key": key}
        if res.cv_predictions is not None:
            sources[f"Session: {tag} | CV out-of-fold (validation years)"] = {
                "model": res.model, "set": "cv", "frame": res.cv_predictions, "feature": res.feature_label, "session": True, "key": key}
    return sources
