"""Chronological evaluation protocol, metrics, permutation importance and conformal intervals.

Protocol used for experiments run in the dashboard
---------------------------------------------------
* Development period: release years <= cutoff (default 2012, the saved split).
* Model selection: year-ahead expanding-window CV inside the development period.
  Fold k trains on all years < v_k and validates on the single release year v_k,
  where v_k are the last ``n_folds`` years up to the cutoff. A release year is never
  split between training and validation.
* Retrospective holdout: release years > cutoff. It is reported, never used for
  selection or tuning. Because the report already used this period during
  development, it is labelled *retrospective*, not independent.
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TARGET


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    clipped = np.clip(y_pred, 0, None)
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(y_true)) ** 2))),
    }
    return out


METRIC_DIRECTION = {"RMSE": "min", "MAE": "min", "RMSLE": "min", "R2": "max"}
METRIC_HELP = {
    "RMSE": "Root mean squared error in millions of copies. Dominated by large errors on blockbusters. Lower is better.",
    "MAE": "Mean absolute error in millions of copies. Lower is better.",
    "R2": "Share of variance explained relative to predicting the evaluation-set mean. Can be negative. Higher is better; not an accuracy percentage.",
    "RMSLE": "RMSE of log1p(sales); measures relative error and is less dominated by blockbusters. Lower is better.",
}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def year_ahead_folds(years: np.ndarray, cutoff: int, n_folds: int, min_train_rows: int = 200):
    """Yield (validation_year, train_idx, valid_idx) for expanding year-ahead CV."""
    years = np.asarray(years, dtype=int)
    dev_years = np.unique(years[years <= cutoff])
    val_years = dev_years[-n_folds:]
    for v in val_years:
        train_idx = np.where(years < v)[0]
        valid_idx = np.where(years == v)[0]
        if len(train_idx) < min_train_rows or len(valid_idx) == 0:
            continue
        yield int(v), train_idx, valid_idx


def describe_protocol(cutoff: int, n_folds: int, years: np.ndarray) -> str:
    years = np.asarray(years, dtype=int)
    dev = np.unique(years[years <= cutoff])
    vals = dev[-n_folds:]
    hold = years[years > cutoff]
    return (
        f"Year-ahead expanding CV: validate on each of {vals.min()}-{vals.max()} ({len(vals)} folds), "
        f"training on all earlier years; final fit on {years.min()}-{cutoff}; "
        f"retrospective holdout {hold.min() if len(hold) else '-'}-{hold.max() if len(hold) else '-'} (n={len(hold)})."
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    model: str
    feature_config: str
    feature_label: str
    params: dict
    tuning_status: str
    protocol: str
    cutoff: int
    seed: int
    status: str = "ok"
    error: str = ""
    fold_scores: list = field(default_factory=list)
    cv_predictions: pd.DataFrame | None = None
    holdout_predictions: pd.DataFrame | None = None
    holdout_metrics: dict = field(default_factory=dict)
    cv_time_s: float = float("nan")
    fit_time_s: float = float("nan")
    inference_time_s: float = float("nan")
    fitted: object = None
    train_rows: int = 0
    holdout_rows: int = 0
    is_baseline: bool = False

    def cv_summary(self) -> dict:
        if not self.fold_scores:
            return {}
        frame = pd.DataFrame(self.fold_scores)
        out = {}
        for m in ["RMSE", "MAE", "R2", "RMSLE"]:
            out[f"CV {m} mean"] = float(frame[m].mean())
            out[f"CV {m} std"] = float(frame[m].std(ddof=1)) if len(frame) > 1 else float("nan")
        return out


def run_single(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cutoff: int,
    n_folds: int,
    *,
    feature_config: str,
    feature_label: str,
    params: dict,
    tuning_status: str,
    seed: int,
    progress=None,
    is_baseline: bool = False,
) -> ExperimentResult:
    """Cross-validate, refit on the development period and score the holdout.

    Any exception is captured in the result instead of being raised.
    """
    years = meta["Year"].to_numpy(dtype=int)
    res = ExperimentResult(
        model=name,
        feature_config=feature_config,
        feature_label=feature_label,
        params=params,
        tuning_status=tuning_status,
        protocol=describe_protocol(cutoff, n_folds, years),
        cutoff=cutoff,
        seed=seed,
        is_baseline=is_baseline,
    )
    try:
        folds = list(year_ahead_folds(years, cutoff, n_folds))
        if not folds:
            raise ValueError("No valid CV folds for this cut-off / fold setting.")
        cv_rows = []
        t0 = time.perf_counter()
        for i, (vy, tr, va) in enumerate(folds, start=1):
            if progress:
                progress(f"{name}: CV fold {i}/{len(folds)} (validate {vy})")
            model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[va])
            scores = regression_metrics(y.iloc[va], pred)
            res.fold_scores.append({"fold": i, "valid_year": vy, "n_train": len(tr), "n_valid": len(va), **scores})
            cv_rows.append(pd.DataFrame({"row": va, "valid_year": vy, "actual": y.iloc[va].to_numpy(), "predicted": pred}))
        res.cv_time_s = time.perf_counter() - t0
        res.cv_predictions = pd.concat(cv_rows, ignore_index=True)

        dev = np.where(years <= cutoff)[0]
        hold = np.where(years > cutoff)[0]
        res.train_rows, res.holdout_rows = len(dev), len(hold)
        if progress:
            progress(f"{name}: final fit on {years.min()}-{cutoff}")
        t0 = time.perf_counter()
        fitted = clone(estimator).fit(X.iloc[dev], y.iloc[dev])
        res.fit_time_s = time.perf_counter() - t0
        if len(hold):
            t0 = time.perf_counter()
            pred = fitted.predict(X.iloc[hold])
            res.inference_time_s = time.perf_counter() - t0
            res.holdout_predictions = pd.DataFrame(
                {"row": hold, "Year": years[hold], "actual": y.iloc[hold].to_numpy(), "predicted": pred}
            )
            res.holdout_metrics = regression_metrics(y.iloc[hold], pred)
        res.fitted = fitted
    except Exception as exc:  # noqa: BLE001 - one failing model must not stop the run
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        res.fitted = None
        res.traceback = traceback.format_exc()
    return res


def result_row(res: ExperimentResult) -> dict:
    row = {
        "Model": res.model,
        "Source": "Session (retrained)",
        "Status": res.status,
        "Feature configuration": res.feature_label,
        "Hyperparameters": res.tuning_status,
        "Evaluation protocol": res.protocol,
        "Holdout period": f"{res.cutoff + 1}+",
        "Seed": res.seed,
    }
    row.update({f"Holdout {k}": v for k, v in res.holdout_metrics.items()})
    row.update(res.cv_summary())
    row.update({"CV time (s)": res.cv_time_s, "Final fit time (s)": res.fit_time_s, "Holdout inference time (s)": res.inference_time_s})
    if res.error:
        row["Error"] = res.error
    return row


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
def permutation_importance_frame(model, X, y, n_repeats: int = 5, seed: int = 42, scoring="neg_root_mean_squared_error",
                                 max_rows: int | None = None) -> pd.DataFrame:
    if max_rows and len(X) > max_rows:
        sample = X.sample(max_rows, random_state=seed).index
        X, y = X.loc[sample], y.loc[sample]
    imp = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (
        pd.DataFrame({"feature": list(X.columns), "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Conformal prediction interval (log scale)
# ---------------------------------------------------------------------------
def conformal_quantile(actual, predicted, alpha: float) -> float:
    """Split-conformal quantile of |log1p(actual) - log1p(predicted)|.

    Uses the finite-sample rank ceil((n+1)(1-alpha)). Coverage is guaranteed only if
    future titles are exchangeable with the calibration titles - an assumption that
    temporal drift can violate.
    """
    a = np.log1p(np.asarray(actual, dtype=float))
    p = np.log1p(np.clip(np.asarray(predicted, dtype=float), 0, None))
    scores = np.sort(np.abs(a - p))
    n = len(scores)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def conformal_interval(pred, q: float):
    lp = np.log1p(np.clip(np.asarray(pred, dtype=float), 0, None))
    return np.clip(np.expm1(lp - q), 0, None), np.expm1(lp + q)


def empirical_coverage(actual, predicted, q: float) -> float:
    lo, hi = conformal_interval(predicted, q)
    a = np.asarray(actual, dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)))


# ---------------------------------------------------------------------------
# Group error analysis
# ---------------------------------------------------------------------------
def group_errors(frame: pd.DataFrame, group: str, min_n: int = 30) -> pd.DataFrame:
    """MAE / RMSE / median absolute error by group with sample counts."""
    f = frame.assign(err=frame["predicted"] - frame["actual"], abs_err=(frame["predicted"] - frame["actual"]).abs())
    f = f.assign(log_err=np.log1p(np.clip(f["predicted"], 0, None)) - np.log1p(f["actual"]))
    g = f.groupby(group, observed=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "MAE": g["abs_err"].mean(),
            "RMSE": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
            "Median abs. error": g["abs_err"].median(),
            "RMSLE": g["log_err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
            "Mean signed error": g["err"].mean(),
            "Median actual": g["actual"].median(),
        }
    ).reset_index()
    overall_mae = f["abs_err"].mean()
    out["MAE / overall MAE"] = out["MAE"] / overall_mae if overall_mae > 0 else np.nan
    out["Small sample (n < %d)" % min_n] = out["n"] < min_n
    return out.sort_values("MAE", ascending=False).reset_index(drop=True)


def year_group_label(year: int) -> str:
    if year <= 1999:
        return "<=1999"
    if year <= 2004:
        return "2000-2004"
    if year <= 2008:
        return "2005-2008"
    if year <= 2012:
        return "2009-2012"
    if year <= 2016:
        return str(year)
    return "2017-2020"
