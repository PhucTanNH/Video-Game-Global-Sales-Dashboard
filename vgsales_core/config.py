"""Project paths, feature schemas and the leakage registry.

Every path is resolved relative to the project directory, so the dashboard runs
from any working directory.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "ADY201m_enhanced_v2.xlsx"
MODEL_DIR = ROOT / "models"
SAVED_MODEL_PATH = MODEL_DIR / "best_model.joblib"
SAVED_META_PATH = MODEL_DIR / "model_meta.json"

TARGET = "Global_Sales"
DEFAULT_SEED = 42
# Split used by the saved artefacts (train_models.py): train <= 2012, holdout >= 2013.
SAVED_TRAIN_END_YEAR = 2012
# Cut-offs offered in the UI. Each keeps at least three full release years for
# the retrospective holdout; 2012 is the saved split.
ALLOWED_CUTOFFS = (2010, 2011, 2012)

# --------------------------------------------------------------------------
# Feature schema of the saved model (identical to train_models.py / model_meta.json)
# --------------------------------------------------------------------------
ORIGINAL_CAT = [
    "Genre",
    "Platform",
    "Publisher_Tier",
    "Platform_Manufacturer",
    "Platform_Lifecycle_Stage",
    "ConsoleGen",
]
ORIGINAL_NUM = [
    "Year",
    "Is_Sequel",
    "Is_Multiplatform",
    "Num_Platforms",
    "Prev_Franchise_MaxSales",
    "Prev_Franchise_TitleCount",
    "Pub_Prev3Y_AvgSales",
    "Pub_Prev3Y_Count",
    "Genre_Prev3Y_AvgSales",
    "Genre_Prev3Y_Count",
    "Platform_Prev3Y_AvgSales",
    "Platform_Prev3Y_Count",
    "Years_Since_Platform_Launch",
    "Genre_vs_Market_Avg",
    "Publisher_MarketShare_Prev3Y",
    "Genre_Trend_Momentum",
    "PubGenre_Specialization",
    "GenrePlatform_Fit",
    "SameGenre_Releases_PrevYear",
    "Platform_Cumulative_Titles",
    "Platform_Cumulative_Sales",
]

# Columns that must never be used as predictors: the target itself, values derived
# from it, or post-release regional outcomes.
FORBIDDEN_PREDICTORS = {
    "Global_Sales": "Target variable.",
    "Log_Global_Sales": "Monotone transform of the target.",
    "NA_Sales": "Regional component of the target (post-release outcome).",
    "EU_Sales": "Regional component of the target (post-release outcome).",
    "JP_Sales": "Regional component of the target (post-release outcome).",
    "Other_Sales": "Regional component of the target (post-release outcome).",
    "NA_Share": "Computed from regional sales / Global_Sales.",
    "EU_Share": "Computed from regional sales / Global_Sales.",
    "JP_Share": "Computed from regional sales / Global_Sales.",
    "Other_Share": "Computed from regional sales / Global_Sales.",
    "Dominant_Region": "Argmax of the regional sales of the same title.",
}
# Columns the report discusses that are absent from this dataset.
ABSENT_LEAKY_COLUMNS = ["Rank", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]

# Leakage status of every column, verified by recomputation in
# vgsales_core.features.audit_original_features (numbers shown in the dashboard).
# status: forbidden | leaky | uncertain | pre-release | identifier | descriptive
LEAKAGE_REGISTRY = {
    # identifiers / descriptive-only
    "Name": ("identifier", "Title string; not used as a model input."),
    "Publisher": ("identifier", "Used only to build publisher-history features and the publisher-genre baseline."),
    "Franchise": ("identifier", "Used only to build franchise-history features."),
    "Decade": ("descriptive", "Coarse version of Year."),
    "Genre_Broad": ("descriptive", "Coarse grouping of Genre."),
    # static pre-release attributes
    "Year": ("pre-release", "Release year is fixed at the forecast origin."),
    "Genre": ("pre-release", "Known before release."),
    "Platform": ("pre-release", "Known before release."),
    "ConsoleGen": ("pre-release", "Static platform attribute."),
    "Platform_Manufacturer": ("pre-release", "Static platform attribute."),
    "Platform_Family": ("pre-release", "Static platform attribute."),
    "Platform_Launch_Year": ("pre-release", "External hardware launch year (e.g. 2600 = 1977), not derived from sales."),
    "Years_Since_Platform_Launch": ("pre-release", "Release year minus hardware launch year."),
    "Platform_Lifecycle_Stage": ("pre-release", "Binned Years_Since_Platform_Launch (PC = 'PC/Unknown')."),
    "Platform_Age_Sq": ("pre-release", "Square of platform age."),
    "Genre_x_ConsoleGen": ("pre-release", "Interaction of static attributes."),
    "Manufacturer_x_Gen": ("pre-release", "Interaction of static attributes."),
    "GenreBroad_x_Gen": ("pre-release", "Interaction of static attributes."),
    "Is_Sequel": ("uncertain", "Appears to be derived from the title (484 sequels have no earlier franchise row). The derivation code is not in the repository."),
    "Is_Remaster_Collection": ("uncertain", "Title-derived flag; derivation code not in the repository."),
    # verified strictly-prior-year windows (but use lifetime sales of earlier titles)
    "Pub_Prev3Y_AvgSales": ("pre-release*", "Verified: years Y-3..Y-1 only (100% match on recomputation)."),
    "Pub_Prev3Y_Count": ("pre-release*", "Verified: years Y-3..Y-1 only."),
    "Genre_Prev3Y_AvgSales": ("pre-release*", "Verified: years Y-3..Y-1 only."),
    "Genre_Prev3Y_Count": ("pre-release*", "Verified: years Y-3..Y-1 only."),
    "Platform_Prev3Y_AvgSales": ("pre-release*", "Verified: years Y-3..Y-1 only."),
    "Platform_Prev3Y_Count": ("pre-release*", "Verified: years Y-3..Y-1 only."),
    "SameGenre_Releases_PrevYear": ("pre-release*", "Verified: count of same-genre releases in Y-1."),
    "Genre_vs_Market_Avg": ("pre-release*", "Genre Prev3Y avg / market Prev3Y avg (99.8% match)."),
    "Genre_Trend_Momentum": ("pre-release*", "Genre Prev1Y avg / Prev3Y avg (99.9% match)."),
    "Publisher_MarketShare_Prev3Y": ("uncertain", "Publisher Prev3Y total / market Prev3Y total for 93% of rows; the remaining rows follow an undocumented rule."),
    "PubGenre_Specialization": ("uncertain", "File value = PubGenre_Prev3Y_AvgSales / Pub_Prev3Y_AvgSales (88%), but PubGenre_Prev3Y_AvgSales itself is NOT the prior-window mean (30% agreement; counts do match). Definition undocumented."),
    "GenrePlatform_Fit": ("uncertain", "File value = GenrePlatform_Prev3Y_AvgSales / Genre_Prev3Y_AvgSales (94%), but GenrePlatform_Prev3Y_AvgSales is NOT the prior-window mean (9% agreement; counts do match). Definition undocumented."),
    "Platform_Cumulative_Titles": ("pre-release*", "Built in this project from strictly earlier release years."),
    "Platform_Cumulative_Sales": ("pre-release*", "Built in this project from strictly earlier release years."),
    # leaky inputs used by the saved model
    "Prev_Franchise_MaxSales": ("leaky", "Includes SAME-YEAR franchise releases ordered by descending sales (within-year Spearman = -1 for 97% of franchise-years), so a title can inherit the sales of its own port."),
    "Prev_Franchise_TitleCount": ("leaky", "Counts same-year franchise rows in descending-sales order: the count itself encodes within-year sales rank."),
    "Publisher_Tier": ("leaky", "Top-10 / 11-50 publishers ranked by title count over the WHOLE 1980-2020 dataset (post-origin information)."),
    "Is_Major_Publisher": ("leaky", "Same as Publisher_Tier == 'Major'."),
    "Num_Platforms": ("uncertain", "Verified: distinct platforms of the same title released in years <= Y (100% match), so it includes same-year ports. Pre-release only if those platforms were announced before launch."),
    "Is_Multiplatform": ("uncertain", "Num_Platforms > 1 (same caveat)."),
}
for _col, _why in FORBIDDEN_PREDICTORS.items():
    LEAKAGE_REGISTRY[_col] = ("forbidden", _why)

STATUS_LABELS = {
    "forbidden": "Forbidden (target / target-derived)",
    "leaky": "Leaky (post-origin or within-year information)",
    "uncertain": "Uncertain (derivation not fully verifiable)",
    "pre-release*": "Prior-year window (verified), uses lifetime sales of earlier titles",
    "pre-release": "Pre-release attribute",
    "identifier": "Identifier / helper (not a predictor)",
    "descriptive": "Descriptive grouping (not used)",
}

GENERATION_ORDER = ["Other", "Gen_5", "Gen_6", "Gen_7", "Gen_8"]
REGION_COLUMNS = {"NA_Sales": "North America", "EU_Sales": "Europe", "JP_Sales": "Japan", "Other_Sales": "Other"}
