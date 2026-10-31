"""Fast unit tests for the shared core (run: python -m pytest -q tests)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vgsales_core import models as M  # noqa: E402
from vgsales_core.config import FORBIDDEN_PREDICTORS, SAVED_META_PATH, SAVED_MODEL_PATH, TARGET  # noqa: E402
from vgsales_core.data import load_prepared  # noqa: E402
from vgsales_core.evaluation import (  # noqa: E402
    conformal_quantile,
    empirical_coverage,
    regression_metrics,
    run_single,
    year_ahead_folds,
)
from vgsales_core.features import (  # noqa: E402
    FEATURE_CONFIG_KEYS,
    assert_no_forbidden,
    audit_original_features,
    build_feature_matrix,
    get_feature_config,
    rebuild_history_features,
)
from vgsales_core.predict import NEW_FRANCHISE, NEW_PUBLISHER, build_query, distribution_warnings, features_for_query  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_prepared()


def test_saved_split_is_reproduced(data):
    assert len(data) == 16327
    assert (data["Year"] <= 2012).sum() == 14237
    assert (data["Year"] > 2012).sum() == 2090
    assert data["Year"].is_monotonic_increasing


def test_saved_model_reproduces_metadata(data):
    pytest.importorskip("xgboost")
    import joblib

    meta = json.loads(SAVED_META_PATH.read_text(encoding="utf-8"))
    model = joblib.load(SAVED_MODEL_PATH)
    feats = meta["features_cat"] + meta["features_num"]
    assert feats == get_feature_config("original").features
    hold = data[data["Year"] > 2012]
    m = regression_metrics(hold[TARGET], model.predict(hold[feats]))
    for k in ["RMSE", "MAE", "R2"]:
        assert abs(m[k] - meta["metrics_test"][k]) < 1e-6


def test_verified_columns_match_strict_rebuild(data):
    audit = audit_original_features(data).set_index("Feature")["Agreement with strict rebuild (gap 0)"]
    exact = ["Pub_Prev3Y_AvgSales", "Pub_Prev3Y_Count", "Genre_Prev3Y_AvgSales", "Genre_Prev3Y_Count", "Platform_Prev3Y_AvgSales",
             "Platform_Prev3Y_Count", "SameGenre_Releases_PrevYear", "Platform_Cumulative_Titles", "Platform_Cumulative_Sales", "Num_Platforms"]
    assert (audit[exact] == 1.0).all()
    assert audit["Genre_vs_Market_Avg"] > 0.99 and audit["Genre_Trend_Momentum"] > 0.99
    # Leaky columns must disagree with their strict versions.
    assert audit["Prev_Franchise_TitleCount"] < 0.9
    assert audit["Prev_Franchise_MaxSales"] < 0.9
    assert audit["Publisher_Tier"] < 0.9


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_rebuilt_features_ignore_invisible_years(data, gap):
    """Changing the sales of any title released in years > Y-1-gap must not change features for year Y."""
    year = 2010
    perturbed = data.copy()
    invisible = perturbed["Year"] >= year - gap
    perturbed.loc[invisible, TARGET] = perturbed.loc[invisible, TARGET] * 7.3 + 5
    query = data[data["Year"] == year]
    a = rebuild_history_features(data, query, gap)
    b = rebuild_history_features(perturbed, query, gap)
    pd.testing.assert_frame_equal(a, b)


def test_no_forbidden_predictors_in_any_configuration(data):
    for key in FEATURE_CONFIG_KEYS:
        cfg = get_feature_config(key)
        assert not set(cfg.features) & set(FORBIDDEN_PREDICTORS)
        X = build_feature_matrix(data, key)
        assert not set(X.columns) & set(FORBIDDEN_PREDICTORS)
        assert X.isna().sum().sum() == 0
    with pytest.raises(ValueError):
        assert_no_forbidden(["Genre", "NA_Sales"])
    with pytest.raises(ValueError):
        assert_no_forbidden(["Global_Sales"])


def test_year_ahead_folds_never_share_a_year(data):
    years = data["Year"].to_numpy()
    folds = list(year_ahead_folds(years, 2012, 4))
    assert [v for v, _, _ in folds] == [2009, 2010, 2011, 2012]
    for v, tr, va in folds:
        assert years[tr].max() < v
        assert set(years[va]) == {v}


def test_failed_model_is_captured_not_raised(data):
    class Broken(M.MedianBaseline):
        def fit(self, X, y):
            raise RuntimeError("deliberate failure")

    X = build_feature_matrix(data, "strict")
    res = run_single("Broken", Broken(), X, data[TARGET], data, 2012, 2, feature_config="strict", feature_label="s",
                     params={}, tuning_status="-", seed=0)
    assert res.status == "failed" and "deliberate failure" in res.error and res.fitted is None


def test_every_model_builds_and_fits_small(data):
    sub = data[(data["Year"] >= 2005) & (data["Year"] <= 2008)].reset_index(drop=True)
    cfg = get_feature_config("original")
    X = build_feature_matrix(sub, "original")
    for name in M.MODEL_NAMES:
        ok, _ = M.availability(name)
        if not ok:
            continue
        params = M.default_params(name)
        if name in ("Random Forest",):
            params["n_estimators"] = 20
        if name == "Stacking":
            params = {**params, "base_models": ("Ridge", "kNN"), "inner_splits": 2}
        est = M.build_estimator(name, params, cfg, seed=1, n_jobs=1)
        pred = est.fit(X[sub["Year"] <= 2007], sub.loc[sub["Year"] <= 2007, TARGET]).predict(X[sub["Year"] == 2008])
        assert np.isfinite(pred).all(), name


def test_missing_dependency_disables_model(monkeypatch):
    monkeypatch.setattr(M, "has_package", lambda name: name != "catboost")
    ok, why = M.availability("CatBoost")
    assert not ok and "catboost" in why
    with pytest.raises(ImportError):
        M.build_estimator("CatBoost", None, get_feature_config("original"))


def test_publisher_genre_baseline_fallbacks():
    X = pd.DataFrame({"Publisher": ["A"] * 3 + ["B"], "Genre": ["Action"] * 3 + ["Puzzle"]})
    y = pd.Series([1.0, 2.0, 3.0, 9.0])
    b = M.PublisherGenreMedianBaseline(min_count=3).fit(X, y)
    q = pd.DataFrame({"Publisher": ["A", "B", "Z", "Z"], "Genre": ["Action", "Puzzle", "Puzzle", "Unseen"]})
    assert list(b.predict(q)) == [2.0, 9.0, 9.0, 2.5]


def test_prediction_features_for_unseen_inputs(data):
    q, prof = build_query(data, year=2015, genre="Shooter", platform="PS4", publisher=NEW_PUBLISHER, franchise=NEW_FRANCHISE,
                          is_sequel=False, num_platforms=3)
    for key in FEATURE_CONFIG_KEYS:
        X, _ = features_for_query(data, q, key)
        assert list(X.columns) == get_feature_config(key).features
        assert X.isna().sum().sum() == 0
    X, _ = features_for_query(data, q, "original")
    assert X["Pub_Prev3Y_Count"].iloc[0] == 0 and X["Prev_Franchise_TitleCount"].iloc[0] == 0
    assert X["Publisher_Tier"].iloc[0] == "Small" and X["Num_Platforms"].iloc[0] == 3
    train = build_feature_matrix(data, "original")[data["Year"] <= 2012]
    warns = distribution_warnings(train, X, get_feature_config("original").cat, get_feature_config("original").num)
    assert any("PS4" in w for w in warns)  # PS4 never appears before 2013


def test_prediction_uses_only_earlier_years(data):
    q, _ = build_query(data, year=2008, genre="Action", platform="X360", publisher="Take-Two Interactive",
                       franchise="Saints Row", is_sequel=True, num_platforms=2)
    X, _ = features_for_query(data, q, "original")
    # Only Saints Row (2006, 2.19M) precedes 2008; same-year Saints Row 2 sales must not be used.
    assert X["Prev_Franchise_TitleCount"].iloc[0] == 1
    assert abs(X["Prev_Franchise_MaxSales"].iloc[0] - 2.19) < 1e-9


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    truth = np.exp(rng.normal(1.5, 0.5, 6000))
    pred = np.expm1(np.log1p(truth) + rng.normal(0, 0.3, truth.size))
    q = conformal_quantile(truth[:3000], pred[:3000], alpha=0.2)
    cov = empirical_coverage(truth[3000:], pred[3000:], q)
    assert 0.77 < cov < 0.83
