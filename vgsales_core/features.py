"""Feature configurations, leakage-reduced feature rebuilding and the leakage audit.

Three kinds of feature configuration are offered:

* ``original``  - the 27 inputs of the saved model, taken from the Excel file as-is.
* ``strict``    - only the original inputs whose prior-year construction was verified
                  by recomputation (no leaky or uncertain columns). Defined by this
                  dashboard; it is not necessarily the report's "Strict-13" list.
* ``rebuilt_gN`` - the same 27-column schema recomputed from raw rows with an
                  information gap of N years: every history statistic for a title
                  released in year Y only uses titles released in years <= Y-1-N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    FORBIDDEN_PREDICTORS,
    LEAKAGE_REGISTRY,
    ORIGINAL_CAT,
    ORIGINAL_NUM,
    TARGET,
)

YEARS = np.arange(1970, 2041)
TIER_MAJOR_RANK = 10  # rule recovered from the data: top-10 publishers by title count
TIER_MID_RANK = 50    # ranks 11-50 -> Mid, everything else -> Small


@dataclass(frozen=True)
class FeatureConfig:
    key: str
    label: str
    cat: tuple
    num: tuple
    description: str
    gap: int | None = None  # None -> columns taken from the Excel file

    @property
    def features(self) -> list[str]:
        return list(self.cat) + list(self.num)


STRICT_CAT = ("Genre", "Platform", "Platform_Manufacturer", "Platform_Lifecycle_Stage", "ConsoleGen")
STRICT_NUM = (
    "Year",
    "Years_Since_Platform_Launch",
    "Pub_Prev3Y_AvgSales",
    "Pub_Prev3Y_Count",
    "Genre_Prev3Y_AvgSales",
    "Genre_Prev3Y_Count",
    "Platform_Prev3Y_AvgSales",
    "Platform_Prev3Y_Count",
    "Genre_vs_Market_Avg",
    "Genre_Trend_Momentum",
    "SameGenre_Releases_PrevYear",
    "Platform_Cumulative_Titles",
    "Platform_Cumulative_Sales",
)


def get_feature_config(key: str) -> FeatureConfig:
    if key == "original":
        return FeatureConfig(
            "original",
            "Original-27 (saved model inputs)",
            tuple(ORIGINAL_CAT),
            tuple(ORIGINAL_NUM),
            "The saved model's 27 inputs from the Excel file. Contains columns flagged as leaky "
            "(Prev_Franchise_*, Publisher_Tier) and uncertain (Num_Platforms, Is_Multiplatform, Is_Sequel, "
            "PubGenre_Specialization, GenrePlatform_Fit, Publisher_MarketShare_Prev3Y).",
        )
    if key == "strict":
        return FeatureConfig(
            "strict",
            f"Strict-{len(STRICT_CAT) + len(STRICT_NUM)} (verified prior-year, dashboard-defined)",
            STRICT_CAT,
            STRICT_NUM,
            "Only original inputs whose prior-year construction was verified by recomputation; "
            "leaky and uncertain columns removed. Defined by this dashboard, not copied from the report.",
        )
    if key.startswith("rebuilt_g"):
        gap = int(key.removeprefix("rebuilt_g"))
        return FeatureConfig(
            key,
            f"Rebuilt-27 (gap {gap} yr)",
            tuple(ORIGINAL_CAT),
            tuple(ORIGINAL_NUM),
            f"Same 27-column schema recomputed from raw rows; history for release year Y uses only titles "
            f"released in years <= Y-{1 + gap}. Is_Sequel is kept from the file (uncertain); Num_Platforms "
            "counts platforms released in years <= Y (assumes announced platforms are known at the origin). "
            "PubGenre_Specialization / GenrePlatform_Fit use documented ratios of prior-window means, so their "
            "values differ from the file's undocumented versions. Publisher_Tier is re-ranked from title counts "
            "visible at the origin.",
            gap=gap,
        )
    raise KeyError(key)


FEATURE_CONFIG_KEYS = ["original", "strict", "rebuilt_g0", "rebuilt_g1", "rebuilt_g2"]


def assert_no_forbidden(features) -> None:
    """Raise if any forbidden (target-derived) column is about to be used as a predictor."""
    bad = [f for f in features if f in FORBIDDEN_PREDICTORS or f == TARGET]
    if bad:
        raise ValueError(f"Forbidden target-derived predictors requested: {bad}")


# ---------------------------------------------------------------------------
# Vectorised history lookups
# ---------------------------------------------------------------------------
def _key_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if keys:
        return frame[keys].astype(str)
    return pd.DataFrame({"_all": np.zeros(len(frame), dtype=int).astype(str)}, index=frame.index)


def _pivot(hist: pd.DataFrame, keys: list[str], how: str) -> pd.DataFrame:
    """Key x year table of per-year sum / count / max of the target."""
    kf = _key_frame(hist, keys)
    tmp = kf.copy()
    tmp["Year"] = hist["Year"].to_numpy()
    tmp["_y"] = hist[TARGET].to_numpy()
    grouped = tmp.groupby(list(kf.columns) + ["Year"])["_y"]
    table = {"sum": grouped.sum, "count": grouped.size, "max": grouped.max}[how]()
    pivot = table.unstack("Year").reindex(columns=YEARS).fillna(0.0)
    return pivot


class _Cum:
    """Cumulative-by-year table supporting window lookups for arbitrary query rows."""

    def __init__(self, pivot: pd.DataFrame, op: str = "sum"):
        values = pivot.to_numpy(dtype=float)
        cum = np.maximum.accumulate(values, axis=1) if op == "max" else np.cumsum(values, axis=1)
        # Extra all-zero row for keys that never appear in the history.
        self.cum = np.vstack([cum, np.zeros((1, cum.shape[1]))])
        self.index = pivot.index

    def _positions(self, query_keys: pd.DataFrame) -> np.ndarray:
        if isinstance(self.index, pd.MultiIndex):
            pos = self.index.get_indexer(pd.MultiIndex.from_frame(query_keys))
        else:
            pos = self.index.get_indexer(query_keys.iloc[:, 0])
        pos = np.where(pos < 0, len(self.cum) - 1, pos)
        return pos

    def upto(self, query_keys: pd.DataFrame, year: np.ndarray) -> np.ndarray:
        """Cumulative value over all years <= year (0 if year precedes the table)."""
        pos = self._positions(query_keys)
        yi = np.asarray(year, dtype=int) - YEARS[0]
        valid = yi >= 0
        yi = np.clip(yi, 0, len(YEARS) - 1)
        out = self.cum[pos, yi]
        return np.where(valid, out, 0.0)

    def window(self, query_keys: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Sum over years lo..hi inclusive."""
        return self.upto(query_keys, hi) - self.upto(query_keys, np.asarray(lo) - 1)


def _safe_div(a, b, fill):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, fill, dtype=float)
    np.divide(a, b, out=out, where=b > 0)
    return out


def rebuild_history_features(hist: pd.DataFrame, query: pd.DataFrame, gap: int) -> pd.DataFrame:
    """Recompute the 27-column schema for ``query`` rows from ``hist`` with an information gap.

    ``query`` needs: Name, Platform, Year, Genre, Publisher, Franchise, Platform_Manufacturer,
    ConsoleGen, Platform_Lifecycle_Stage, Years_Since_Platform_Launch, Is_Sequel.
    If ``Num_Platforms`` is present in ``query`` with a value > 0 it is used as given
    (prediction interface); otherwise it is rebuilt from ``hist``.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    year = query["Year"].to_numpy(dtype=int)
    last = year - 1 - gap          # last release year whose titles are visible
    first3 = last - 2              # three-year window: last-2 .. last
    out = pd.DataFrame(index=query.index)
    for col in ["Genre", "Platform", "Platform_Manufacturer", "Platform_Lifecycle_Stage", "ConsoleGen"]:
        out[col] = query[col].astype(str).to_numpy()
    out["Year"] = year
    out["Years_Since_Platform_Launch"] = query["Years_Since_Platform_Launch"].to_numpy()
    out["Is_Sequel"] = query["Is_Sequel"].to_numpy()

    def window_avg_count(keys):
        s = _Cum(_pivot(hist, keys, "sum")).window(_key_frame(query, keys), first3, last)
        c = _Cum(_pivot(hist, keys, "count")).window(_key_frame(query, keys), first3, last)
        return _safe_div(s, c, 0.0), c, s

    pub_avg, pub_cnt, pub_tot = window_avg_count(["Publisher"])
    gen_avg, gen_cnt, _ = window_avg_count(["Genre"])
    plat_avg, plat_cnt, _ = window_avg_count(["Platform"])
    mkt_avg, _, mkt_tot = window_avg_count([])
    pg_avg, _, _ = window_avg_count(["Publisher", "Genre"])
    gp_avg, _, _ = window_avg_count(["Genre", "Platform"])

    gen_s = _Cum(_pivot(hist, ["Genre"], "sum"))
    gen_c = _Cum(_pivot(hist, ["Genre"], "count"))
    gq = _key_frame(query, ["Genre"])
    gen_prev1_avg = _safe_div(gen_s.window(gq, last, last), gen_c.window(gq, last, last), 0.0)

    out["Pub_Prev3Y_AvgSales"] = pub_avg
    out["Pub_Prev3Y_Count"] = pub_cnt
    out["Genre_Prev3Y_AvgSales"] = gen_avg
    out["Genre_Prev3Y_Count"] = gen_cnt
    out["Platform_Prev3Y_AvgSales"] = plat_avg
    out["Platform_Prev3Y_Count"] = plat_cnt
    out["Genre_vs_Market_Avg"] = _safe_div(gen_avg, mkt_avg, 0.0)
    out["Publisher_MarketShare_Prev3Y"] = _safe_div(pub_tot, mkt_tot, 0.0)
    out["Genre_Trend_Momentum"] = _safe_div(gen_prev1_avg, gen_avg, 1.0)
    out["PubGenre_Specialization"] = _safe_div(pg_avg, pub_avg, 1.0)
    out["GenrePlatform_Fit"] = _safe_div(gp_avg, gen_avg, 0.0)
    out["SameGenre_Releases_PrevYear"] = gen_c.window(gq, last, last)

    pq = _key_frame(query, ["Platform"])
    out["Platform_Cumulative_Titles"] = _Cum(_pivot(hist, ["Platform"], "count")).upto(pq, last)
    out["Platform_Cumulative_Sales"] = _Cum(_pivot(hist, ["Platform"], "sum")).upto(pq, last)

    fq = _key_frame(query, ["Franchise"])
    out["Prev_Franchise_MaxSales"] = _Cum(_pivot(hist, ["Franchise"], "max"), op="max").upto(fq, last)
    out["Prev_Franchise_TitleCount"] = _Cum(_pivot(hist, ["Franchise"], "count")).upto(fq, last)

    # Publisher tier from cumulative title counts visible at the origin.
    pub_count_pivot = _pivot(hist, ["Publisher"], "count")
    cum_counts = pub_count_pivot.cumsum(axis=1)
    ranks = cum_counts.where(cum_counts > 0).rank(axis=0, ascending=False, method="min").fillna(np.inf)
    rank_table = np.vstack([ranks.to_numpy(), np.full((1, ranks.shape[1]), np.inf)])
    pos = pub_count_pivot.index.get_indexer(query["Publisher"].astype(str))
    pos = np.where(pos < 0, len(rank_table) - 1, pos)
    yi = last - YEARS[0]
    rank = np.where(yi >= 0, rank_table[pos, np.clip(yi, 0, len(YEARS) - 1)], np.inf)
    out["Publisher_Tier"] = np.where(rank <= TIER_MAJOR_RANK, "Major", np.where(rank <= TIER_MID_RANK, "Mid", "Small"))

    # Platforms of the same title released in years <= Y.
    given = query["Num_Platforms"].to_numpy() if "Num_Platforms" in query else None
    first_release = hist.groupby(["Name", "Platform"], as_index=False)["Year"].min()
    first_release[TARGET] = 1.0  # count via sum
    nq = _key_frame(query, ["Name"])
    rebuilt_np = _Cum(_pivot(first_release, ["Name"], "count")).upto(nq, year)
    rebuilt_np = np.maximum(rebuilt_np, 1.0)
    if given is not None:
        given = pd.to_numeric(pd.Series(given), errors="coerce").to_numpy()
        rebuilt_np = np.where(np.isfinite(given) & (given > 0), given, rebuilt_np)
    out["Num_Platforms"] = rebuilt_np
    out["Is_Multiplatform"] = (out["Num_Platforms"] > 1).astype(int)

    return out[ORIGINAL_CAT + ORIGINAL_NUM]


def build_feature_matrix(data: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return the predictor matrix for a configuration (row-aligned with ``data``)."""
    cfg = get_feature_config(key)
    assert_no_forbidden(cfg.features)
    if cfg.gap is None:
        X = data[cfg.features].copy()
    else:
        X = rebuild_history_features(data, data, cfg.gap)[cfg.features]
    for col in cfg.cat:
        X[col] = X[col].fillna("Unknown").astype(str)
    return X


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit_original_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compare each original numeric/categorical input with its gap-0 strict rebuild.

    A low agreement rate means the file's column uses information that the strict
    prior-year rule would not allow (or follows an undocumented rule).
    """
    rebuilt = rebuild_history_features(data, data, gap=0)
    rows = []
    for col in ORIGINAL_CAT + ORIGINAL_NUM:
        a = data[col]
        b = rebuilt[col]
        if a.dtype == object or str(a.dtype).startswith(("str", "string", "category")):
            agree = (a.astype(str).to_numpy() == b.astype(str).to_numpy()).mean()
        else:
            agree = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), atol=1e-3, rtol=1e-3).mean()
        status, note = LEAKAGE_REGISTRY.get(col, ("uncertain", "Not documented."))
        rows.append({"Feature": col, "Status": status, "Agreement with strict rebuild (gap 0)": float(agree), "Note": note})
    return pd.DataFrame(rows)


def franchise_within_year_diagnostic(data: pd.DataFrame) -> dict:
    """Quantify how the file's Prev_Franchise_* columns use same-year releases."""
    same = data.groupby(["Franchise", "Year"])[TARGET].transform("size") > 1
    sub = data.loc[same, ["Franchise", "Year", TARGET, "Prev_Franchise_TitleCount"]]
    sub = sub.assign(
        r_sales=sub.groupby(["Franchise", "Year"])[TARGET].rank(method="average"),
        r_cnt=sub.groupby(["Franchise", "Year"])["Prev_Franchise_TitleCount"].rank(method="average"),
    )

    def _corr(g):
        if g["r_sales"].nunique() < 2 or g["r_cnt"].nunique() < 2:
            return np.nan
        return np.corrcoef(g["r_sales"], g["r_cnt"])[0, 1]

    corrs = sub.groupby(["Franchise", "Year"])[["r_sales", "r_cnt"]].apply(_corr).dropna()
    strict = rebuild_history_features(data, data, gap=0)
    return {
        "rows_sharing_franchise_year": int(same.sum()),
        "franchise_years_tested": int(len(corrs)),
        "share_negative_rank_corr": float((corrs < 0).mean()) if len(corrs) else float("nan"),
        "median_rank_corr": float(corrs.median()) if len(corrs) else float("nan"),
        "rows_maxsales_differs": int((~np.isclose(data["Prev_Franchise_MaxSales"], strict["Prev_Franchise_MaxSales"])).sum()),
        "rows_count_inflated": int((data["Prev_Franchise_TitleCount"].to_numpy() > strict["Prev_Franchise_TitleCount"].to_numpy()).sum()),
        "n_rows": int(len(data)),
    }
