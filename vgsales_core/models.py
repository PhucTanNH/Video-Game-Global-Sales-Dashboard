"""Model registry: hyperparameter specs, dependency checks and pipeline construction.

Every model is wrapped as
    TransformedTargetRegressor(log1p / expm1)  ->  Pipeline(preprocess, model)
so preprocessing, categorical encoding and the target transform are fitted only on
the training rows of each fold.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import FeatureConfig, assert_no_forbidden


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class ParamSpec:
    name: str
    kind: str  # int | float | select
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple = ()
    help: str = ""
    log: bool = False


@dataclass
class ModelSpec:
    name: str
    params: list = field(default_factory=list)
    dependency: str | None = None
    default_status: str = "Fixed defaults (train_models.py settings)"
    note: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "Linear Regression": ModelSpec("Linear Regression", [], default_status="No hyperparameters"),
    "Ridge": ModelSpec("Ridge", [ParamSpec("alpha", "float", 1.0, 0.001, 1000.0, None, help="L2 penalty strength", log=True)]),
    "kNN": ModelSpec(
        "kNN",
        [
            ParamSpec("n_neighbors", "int", 15, 1, 200, 1, help="Neighbours averaged per prediction"),
            ParamSpec("weights", "select", "distance", options=("distance", "uniform")),
        ],
    ),
    "Random Forest": ModelSpec(
        "Random Forest",
        [
            ParamSpec("n_estimators", "int", 200, 20, 1000, 10),
            ParamSpec("max_depth", "int", 10, 2, 40, 1),
            ParamSpec("min_samples_leaf", "int", 1, 1, 100, 1),
        ],
    ),
    "XGBoost": ModelSpec(
        "XGBoost",
        [
            ParamSpec("n_estimators", "int", 414, 50, 2000, 1),
            ParamSpec("max_depth", "int", 4, 1, 12, 1),
            ParamSpec("learning_rate", "float", 0.0541, 0.005, 0.5, 0.0001),
            ParamSpec("subsample", "float", 0.6064, 0.3, 1.0, 0.0001),
            ParamSpec("colsample_bytree", "float", 0.9769, 0.3, 1.0, 0.0001),
        ],
        dependency="xgboost",
        default_status="Report-tuned configuration (search not reproduced here)",
    ),
    "LightGBM": ModelSpec(
        "LightGBM",
        [
            ParamSpec("n_estimators", "int", 494, 50, 2000, 1),
            ParamSpec("num_leaves", "int", 63, 4, 512, 1),
            ParamSpec("learning_rate", "float", 0.0482, 0.005, 0.5, 0.0001),
            ParamSpec("min_child_samples", "int", 53, 1, 300, 1),
        ],
        dependency="lightgbm",
        default_status="Report-tuned configuration (search not reproduced here)",
    ),
    "CatBoost": ModelSpec(
        "CatBoost",
        [
            ParamSpec("iterations", "int", 500, 50, 3000, 10),
            ParamSpec("depth", "int", 6, 2, 10, 1),
            ParamSpec("learning_rate", "float", 0.05, 0.005, 0.5, 0.0001),
        ],
        dependency="catboost",
        note="Uses CatBoost's native categorical handling instead of one-hot encoding.",
    ),
    "Stacking": ModelSpec(
        "Stacking",
        [
            ParamSpec("base_models", "multi", ("XGBoost", "LightGBM", "CatBoost", "Random Forest"),
                      options=("XGBoost", "LightGBM", "CatBoost", "Random Forest", "Ridge", "kNN")),
            ParamSpec("meta_alpha", "float", 1.0, 0.001, 1000.0, None, help="Ridge meta-learner penalty", log=True),
            ParamSpec("inner_splits", "int", 3, 2, 5, 1, help="Chronological year-block folds used to build out-of-fold meta-features"),
        ],
        default_status="Fixed defaults; base learners use their own settings from this panel",
        note="Meta-features come from chronological out-of-fold predictions (year blocks), never from in-sample fits.",
    ),
}
MODEL_NAMES = list(MODEL_SPECS)
BASELINE_NAMES = ["Median baseline", "Publisher-Genre median baseline"]


def default_params(name: str) -> dict:
    return {p.name: p.default for p in MODEL_SPECS[name].params}


def availability(name: str) -> tuple[bool, str]:
    spec = MODEL_SPECS[name]
    if spec.dependency and not has_package(spec.dependency):
        return False, f"Package '{spec.dependency}' is not installed (pip install {spec.dependency})."
    if name == "Stacking":
        missing = [m for m in default_params("Stacking")["base_models"] if MODEL_SPECS[m].dependency and not has_package(MODEL_SPECS[m].dependency)]
        if missing:
            return True, f"Base learners unavailable and will be skipped: {missing}"
    return True, ""


def tuning_status(name: str, params: dict) -> str:
    defaults = default_params(name)
    norm = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v
    changed = [k for k, v in params.items() if norm(v) != norm(defaults.get(k))]
    if changed:
        return "User-set (manually adjusted: " + ", ".join(changed) + "; not search-tuned)"
    return MODEL_SPECS[name].default_status


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _fill_categoricals(X: pd.DataFrame, cat_cols=()):
    X = X.copy()
    for c in cat_cols:
        if c in X:
            X[c] = X[c].astype(object).where(X[c].notna(), "Unknown").astype(str)
    return X


def make_preprocessor(cfg: FeatureConfig, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("categorical", categorical, list(cfg.cat)), ("numeric", Pipeline(numeric_steps), list(cfg.num))],
        remainder="drop",
    )


class YearBlockStackingRegressor(RegressorMixin, BaseEstimator):
    """Stacking whose meta-features come from chronological out-of-fold predictions.

    Inner folds are contiguous blocks of release *years* (a release year is never split
    between the inner training and validation part). Falls back to a row-based
    TimeSeriesSplit when no Year column is present.
    """

    def __init__(self, estimators, final_estimator=None, n_splits=3, year_column="Year"):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.year_column = year_column

    def _splits(self, X):
        if hasattr(X, "columns") and self.year_column in X.columns:
            years = np.asarray(X[self.year_column])
            uniq = np.unique(years)
            if len(uniq) > self.n_splits:
                blocks = np.array_split(uniq, self.n_splits + 1)
                for i in range(1, len(blocks)):
                    valid_years = blocks[i]
                    train_idx = np.where(years < valid_years.min())[0]
                    valid_idx = np.where(np.isin(years, valid_years))[0]
                    if len(train_idx) and len(valid_idx):
                        yield train_idx, valid_idx
                return
        yield from TimeSeriesSplit(n_splits=self.n_splits).split(X)

    def fit(self, X, y):
        target = np.asarray(y, dtype=float)
        oof = np.full((len(target), len(self.estimators)), np.nan)
        for train_idx, valid_idx in self._splits(X):
            for j, (_, est) in enumerate(self.estimators):
                model = clone(est).fit(X.iloc[train_idx], target[train_idx])
                oof[valid_idx, j] = model.predict(X.iloc[valid_idx])
        rows = np.isfinite(oof).all(axis=1)
        if rows.sum() < 10:
            raise ValueError("Too few out-of-fold rows to fit the stacking meta-learner.")
        self.final_estimator_ = clone(self.final_estimator or Ridge()).fit(oof[rows], target[rows])
        self.estimators_ = [(n, clone(e).fit(X, target)) for n, e in self.estimators]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        base = np.column_stack([e.predict(X) for _, e in self.estimators_])
        return self.final_estimator_.predict(base)


def _core_model(name: str, params: dict, cfg: FeatureConfig, seed: int, n_jobs: int):
    """Return the unwrapped (log-target scale) pipeline for one model."""
    if name == "Linear Regression":
        return Pipeline([("preprocess", make_preprocessor(cfg, True)), ("model", LinearRegression())])
    if name == "Ridge":
        return Pipeline([("preprocess", make_preprocessor(cfg, True)), ("model", Ridge(alpha=float(params["alpha"])))])
    if name == "kNN":
        return Pipeline(
            [
                ("preprocess", make_preprocessor(cfg, True)),
                ("model", KNeighborsRegressor(n_neighbors=int(params["n_neighbors"]), weights=params["weights"])),
            ]
        )
    if name == "Random Forest":
        return Pipeline(
            [
                ("preprocess", make_preprocessor(cfg, False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    if name == "XGBoost":
        from xgboost import XGBRegressor

        return Pipeline(
            [
                ("preprocess", make_preprocessor(cfg, False)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=n_jobs,
                        verbosity=0,
                    ),
                ),
            ]
        )
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return Pipeline(
            [
                ("preprocess", make_preprocessor(cfg, False)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=int(params["n_estimators"]),
                        num_leaves=int(params["num_leaves"]),
                        learning_rate=float(params["learning_rate"]),
                        min_child_samples=int(params["min_child_samples"]),
                        random_state=seed,
                        n_jobs=n_jobs,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if name == "CatBoost":
        from catboost import CatBoostRegressor

        return Pipeline(
            [
                ("fill", FunctionTransformer(_fill_categoricals, kw_args={"cat_cols": list(cfg.cat)})),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=int(params["iterations"]),
                        depth=int(params["depth"]),
                        learning_rate=float(params["learning_rate"]),
                        loss_function="RMSE",
                        cat_features=tuple(cfg.cat),
                        random_seed=seed,
                        thread_count=n_jobs,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    raise KeyError(name)


def build_estimator(name: str, params: dict | None, cfg: FeatureConfig, seed: int = 42, n_jobs: int = 2,
                    all_params: dict | None = None):
    """Full estimator (log1p target transform + pipeline) for a model name."""
    assert_no_forbidden(cfg.features)
    params = {**default_params(name), **(params or {})}
    ok, why = availability(name)
    if not ok:
        raise ImportError(why)
    if name == "Stacking":
        all_params = all_params or {}
        bases = []
        for base in params["base_models"]:
            dep = MODEL_SPECS[base].dependency
            if dep and not has_package(dep):
                continue
            base_params = {**default_params(base), **all_params.get(base, {})}
            bases.append((base, _core_model(base, base_params, cfg, seed, n_jobs)))
        if len(bases) < 2:
            raise ValueError("Stacking needs at least two available base learners.")
        core = YearBlockStackingRegressor(
            estimators=bases, final_estimator=Ridge(alpha=float(params["meta_alpha"])), n_splits=int(params["inner_splits"])
        )
    else:
        core = _core_model(name, params, cfg, seed, n_jobs)
    return TransformedTargetRegressor(regressor=core, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


# ---------------------------------------------------------------------------
# Baselines (use only training-period targets)
# ---------------------------------------------------------------------------
class MedianBaseline(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class PublisherGenreMedianBaseline(RegressorMixin, BaseEstimator):
    """Median of training sales for the (Publisher, Genre) pair.

    Falls back to the genre median, then the global median, when the pair has fewer
    than ``min_count`` training titles. Requires Publisher and Genre columns.
    """

    def __init__(self, min_count: int = 3):
        self.min_count = min_count

    def fit(self, X, y):
        frame = X[["Publisher", "Genre"]].astype(str).assign(_y=np.asarray(y, dtype=float))
        pg = frame.groupby(["Publisher", "Genre"])["_y"].agg(["median", "size"])
        self.pg_ = pg.loc[pg["size"] >= self.min_count, "median"].to_dict()
        self.genre_ = frame.groupby("Genre")["_y"].median().to_dict()
        self.global_ = float(frame["_y"].median())
        return self

    def predict(self, X):
        keys = X[["Publisher", "Genre"]].astype(str)
        out = []
        for pub, gen in keys.itertuples(index=False):
            out.append(self.pg_.get((pub, gen), self.genre_.get(gen, self.global_)))
        return np.asarray(out, dtype=float)
