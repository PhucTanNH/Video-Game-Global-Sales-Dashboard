"""Data loading and the shared preparation step used by the dashboard and training."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, ORIGINAL_CAT, TARGET


def add_platform_cumulative_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative platform history using strictly earlier release years.

    Identical to the function previously duplicated in app.py and train_models.py.
    """
    platform_year = (
        data.groupby(["Platform", "Year"], as_index=False, sort=False)
        .agg(
            current_year_titles=("Platform", "size"),
            current_year_sales=(TARGET, "sum"),
        )
        .sort_values(["Platform", "Year"], kind="mergesort")
    )
    platform_year["Platform_Cumulative_Titles"] = (
        platform_year.groupby("Platform", sort=False)["current_year_titles"].cumsum()
        - platform_year["current_year_titles"]
    )
    platform_year["Platform_Cumulative_Sales"] = (
        platform_year.groupby("Platform", sort=False)["current_year_sales"].cumsum()
        - platform_year["current_year_sales"]
    )
    return data.merge(
        platform_year[["Platform", "Year", "Platform_Cumulative_Titles", "Platform_Cumulative_Sales"]],
        on=["Platform", "Year"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name} (expected in the project directory).")
    return pd.read_excel(path)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the preparation in train_models.py (same rows, same order).

    Rows are sorted chronologically with a stable sort so that the saved split
    (train <= 2012: 14,237 rows; holdout >= 2013: 2,090 rows) is reproduced.
    """
    data = add_platform_cumulative_features(raw.copy())
    data[ORIGINAL_CAT] = data[ORIGINAL_CAT].fillna("Unknown").astype(str)
    data = data.dropna(subset=[TARGET, "Year"])
    data["Year"] = data["Year"].astype(int)
    data = data.sort_values("Year", kind="mergesort").reset_index(drop=True)
    return data


def load_prepared(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return prepare(load_raw(path))
