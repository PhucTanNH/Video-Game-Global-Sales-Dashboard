"""Feature construction for a single hypothetical pre-release title."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET
from .features import assert_no_forbidden, get_feature_config, rebuild_history_features

NEW_PUBLISHER = "(New / unlisted publisher)"
NEW_FRANCHISE = "(New IP - no earlier franchise titles)"


def platform_profile(data: pd.DataFrame, platform: str) -> dict:
    rows = data[data["Platform"] == platform]
    if rows.empty:
        raise ValueError(f"Unknown platform: {platform}")
    return {
        "Platform_Manufacturer": rows["Platform_Manufacturer"].mode().iloc[0],
        "ConsoleGen": rows["ConsoleGen"].mode().iloc[0],
        "Platform_Launch_Year": int(rows["Platform_Launch_Year"].mode().iloc[0]),
        "first_year": int(rows["Year"].min()),
        "last_year": int(rows["Year"].max()),
    }


def lifecycle(platform: str, launch_year: int, year: int) -> tuple[str, int]:
    """Reproduce the dataset's rule: <=2 Early, 3-5 Mid, >=6 Late; PC -> ('PC/Unknown', -1)."""
    if platform == "PC" or launch_year <= 0:
        return "PC/Unknown", -1
    ys = int(year - launch_year)
    return ("Early" if ys <= 2 else "Mid" if ys <= 5 else "Late"), ys


def build_query(data: pd.DataFrame, *, year: int, genre: str, platform: str, publisher: str, franchise: str,
                is_sequel: bool, num_platforms: int) -> tuple[pd.DataFrame, dict]:
    prof = platform_profile(data, platform)
    stage, ys = lifecycle(platform, prof["Platform_Launch_Year"], year)
    query = pd.DataFrame([{
        "Name": "__hypothetical_title__",
        "Platform": platform,
        "Year": int(year),
        "Genre": genre,
        "Publisher": "__new_publisher__" if publisher == NEW_PUBLISHER else publisher,
        "Franchise": "__new_franchise__" if franchise == NEW_FRANCHISE else franchise,
        "Platform_Manufacturer": prof["Platform_Manufacturer"],
        "ConsoleGen": prof["ConsoleGen"],
        "Platform_Lifecycle_Stage": stage,
        "Years_Since_Platform_Launch": ys,
        "Is_Sequel": int(bool(is_sequel)),
        "Num_Platforms": int(num_platforms),
    }])
    return query, prof


def _latest_prior_value(data: pd.DataFrame, column: str, mask: pd.Series, year: int) -> float:
    hist = data.loc[mask & (data["Year"] < year), ["Year", column]]
    if len(hist):
        return float(hist.loc[hist["Year"] == hist["Year"].max(), column].median())
    base = data.loc[data["Year"] < year, column]
    return float(base.median()) if len(base) else 0.0


def features_for_query(data: pd.DataFrame, query: pd.DataFrame, feature_key: str) -> tuple[pd.DataFrame, dict]:
    """Compute the model inputs for the query using only titles released before its year.

    For the Original-27 / Strict schemas the strict gap-0 rebuild is used (it reproduces every
    verified column exactly). The two columns whose file definition is undocumented
    (PubGenre_Specialization, GenrePlatform_Fit) are looked up from the most recent prior
    release year so that their scale matches what the saved model saw in training.
    """
    cfg = get_feature_config(feature_key)
    assert_no_forbidden(cfg.features)
    gap = cfg.gap if cfg.gap is not None else 0
    feats = rebuild_history_features(data, query, gap)
    notes = {}
    if cfg.gap is None:
        y = int(query["Year"].iloc[0])
        g, p, pub = query["Genre"].iloc[0], query["Platform"].iloc[0], query["Publisher"].iloc[0]
        feats["PubGenre_Specialization"] = _latest_prior_value(data, "PubGenre_Specialization", (data["Publisher"] == pub) & (data["Genre"] == g), y)
        feats["GenrePlatform_Fit"] = _latest_prior_value(data, "GenrePlatform_Fit", (data["Genre"] == g) & (data["Platform"] == p), y)
        notes["PubGenre_Specialization"] = "file value from the latest earlier year for this publisher-genre (definition undocumented)"
        notes["GenrePlatform_Fit"] = "file value from the latest earlier year for this genre-platform (definition undocumented)"
        notes["Prev_Franchise_MaxSales"] = "strict: franchise titles from earlier years only (the file's same-year version is leaky)"
        notes["Prev_Franchise_TitleCount"] = "strict: franchise titles from earlier years only"
        notes["Publisher_Tier"] = "rank of the publisher by titles released before this year (file uses 1980-2020 totals)"
    notes["Num_Platforms"] = "user input: announced launch platforms"
    return feats[cfg.features], notes


def _fmt(v) -> str:
    v = float(v)
    return f"{int(v)}" if v.is_integer() else f"{v:.4g}"


def distribution_warnings(X_train: pd.DataFrame, row: pd.DataFrame, cat_cols, num_cols, low=0.005, high=0.995) -> list[str]:
    out = []
    for c in cat_cols:
        v = row[c].iloc[0]
        seen = set(X_train[c].astype(str))
        if str(v) not in seen:
            out.append(f"**{c} = {v}** never occurs in the model's training period; the encoder treats it as an unseen category.")
    for c in num_cols:
        v = float(row[c].iloc[0])
        col = X_train[c].astype(float)
        lo, hi = col.quantile(low), col.quantile(high)
        if v < col.min() or v > col.max():
            out.append(f"**{c} = {_fmt(v)}** is outside the training range [{_fmt(col.min())}, {_fmt(col.max())}].")
        elif v < lo or v > hi:
            out.append(f"**{c} = {_fmt(v)}** lies in the extreme 0.5% tail of the training data [{_fmt(lo)}, {_fmt(hi)}].")
    return out
