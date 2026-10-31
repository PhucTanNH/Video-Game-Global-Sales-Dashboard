# ============================================================
# DASHBOARD STREAMLIT - Video Game Global Sales Prediction (ADY201m)
# Run: streamlit run app.py
# ============================================================
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_VERSION = "3.0.0"
PREVIOUS_VERSION = "2.0.0"

st.set_page_config(page_title=f"Video Game Sales Research Dashboard v{APP_VERSION}", layout="wide", page_icon="🎮")

st.markdown(
    """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  [data-testid="stCaptionContainer"] {line-height: 1.45;}
</style>
""",
    unsafe_allow_html=True,
)

from dashboard import sec_eda, sec_errors, sec_leakage, sec_models, sec_overview, sec_predict  # noqa: E402
from dashboard.state import get_data, get_meta  # noqa: E402

try:
    df = get_data()
except FileNotFoundError as exc:
    st.error(f"{exc} Place ADY201m_enhanced_v2.xlsx next to app.py.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the dataset: {type(exc).__name__}: {exc}")
    st.stop()

SECTIONS = {
    "1. Dataset Overview": sec_overview,
    "2. Exploratory Data Analysis": sec_eda,
    "3. Model Comparison": sec_models,
    "4. Error Analysis": sec_errors,
    "5. Leakage Audit": sec_leakage,
    "6. Sales Prediction": sec_predict,
}

# ---------- Sidebar ----------
st.sidebar.title("🎮 Video Game Sales")
st.sidebar.caption(f"Research dashboard v{APP_VERSION} (previous: v{PREVIOUS_VERSION}, git tag)")
section = st.sidebar.radio("Section", list(SECTIONS), key="nav")

filtered = df
if section in ("1. Dataset Overview", "2. Exploratory Data Analysis"):
    st.sidebar.markdown("### Descriptive filters")
    st.sidebar.caption("Apply to Overview and EDA only. Model results always use the full evaluation sets.")
    genres = st.sidebar.multiselect("Genre", sorted(df["Genre"].unique()), key="f_genre")
    platforms = st.sidebar.multiselect("Platform", sorted(df["Platform"].unique()), key="f_platform")
    publishers = st.sidebar.multiselect("Publisher", sorted(df["Publisher"].unique()), key="f_pub")
    gens = st.sidebar.multiselect("Console generation", ["Other", "Gen_5", "Gen_6", "Gen_7", "Gen_8"], key="f_gen")
    years = st.sidebar.slider("Release years", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())), key="f_years")
    mask = df["Year"].between(*years)
    if genres:
        mask &= df["Genre"].isin(genres)
    if platforms:
        mask &= df["Platform"].isin(platforms)
    if publishers:
        mask &= df["Publisher"].isin(publishers)
    if gens:
        mask &= df["ConsoleGen"].isin(gens)
    filtered = df[mask]
    st.sidebar.markdown(f"**Titles shown:** {len(filtered):,} / {len(df):,}")

with st.sidebar.expander(f"What's new in v{APP_VERSION}"):
    st.markdown(
        "- Six research sections with a consistent layout.\n"
        "- Interactive experiments: 8 models + 2 baselines, year-ahead CV, configurable settings.\n"
        "- Saved artefacts and session runs kept strictly apart.\n"
        "- Out-of-sample diagnostics, error analysis, permutation importance.\n"
        "- Leakage audit with recomputation evidence and leakage-reduced feature sets.\n"
        "- Pre-release prediction with strict history features and a split-conformal interval."
    )

# ---------- Header ----------
st.title("Video Game Global Sales Prediction")
meta = get_meta()
if meta:
    st.caption(
        f"ADY201m | Target: {meta['target']} (lifetime units, millions) | Saved model: {meta['model']} on Original-27 inputs, "
        f"trained {meta['train_years'][0]}-{meta['train_years'][1]} | Holdout {meta['test_years'][0]}-{meta['test_years'][1]} is retrospective (reused during report development)."
    )

module = SECTIONS[section]
if section in ("1. Dataset Overview",):
    module.render(df, filtered)
elif section == "2. Exploratory Data Analysis":
    module.render(filtered)
else:
    module.render()

st.markdown("---")
st.caption("ADY201m | Video Game Global Sales Prediction | Data: VGChartz via Kaggle (16,327 titles, 1980-2020) | "
           "Chronological evaluation; results labelled by feature configuration and protocol.")
