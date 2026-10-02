"""Reproduce the report's time-based model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from models.stacking import TimeSeriesStackingRegressor


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "ADY201m_enhanced_v2.xlsx"
SEED = 42
N_SPLITS = 5
STACK_SPLITS = 3
TRAIN_END_YEAR = 2012
N_JOBS = 2

CAT_FEATURES = [
    "Genre",
    "Platform",
    "Publisher_Tier",
    "Platform_Manufacturer",
    "Platform_Lifecycle_Stage",
    "ConsoleGen",
]
NUM_FEATURES = [
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
FEATURES = CAT_FEATURES + NUM_FEATURES


def add_platform_cumulative_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative platform history using strictly earlier release years."""
    platform_year = (
        data.groupby(["Platform", "Year"], as_index=False, sort=False)
        .agg(
            current_year_titles=("Platform", "size"),
            current_year_sales=("Global_Sales", "sum"),
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
        platform_year[
            ["Platform", "Year", "Platform_Cumulative_Titles", "Platform_Cumulative_Sales"]
        ],
        on=["Platform", "Year"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(numeric_steps)
    return ColumnTransformer(
        [
            ("categorical", categorical, CAT_FEATURES),
            ("numeric", numeric, NUM_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric)),
            ("model", model),
        ]
    )


def target_model(estimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def make_model_zoo():
    linear = make_pipeline(LinearRegression(), scale_numeric=True)
    ridge = make_pipeline(Ridge(alpha=1.0), scale_numeric=True)
    knn = make_pipeline(
        KNeighborsRegressor(n_neighbors=15, weights="distance"),
        scale_numeric=True,
    )
    random_forest = make_pipeline(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
    )
    xgboost = make_pipeline(
        XGBRegressor(
            n_estimators=414,
            max_depth=4,
            learning_rate=0.0541,
            subsample=0.6064,
            colsample_bytree=0.9769,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=N_JOBS,
            verbosity=0,
        )
    )
    lightgbm = make_pipeline(
        LGBMRegressor(
            n_estimators=494,
            num_leaves=63,
            learning_rate=0.0482,
            min_child_samples=53,
            random_state=SEED,
            n_jobs=N_JOBS,
            verbosity=-1,
        )
    )
    catboost = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        cat_features=tuple(CAT_FEATURES),
        random_seed=SEED,
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
    )
    stack = TimeSeriesStackingRegressor(
        estimators=[
            ("xgboost", xgboost),
            ("lightgbm", lightgbm),
            ("catboost", catboost),
            ("random_forest", random_forest),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_splits=STACK_SPLITS,
    )
    return {
        "Linear Regression": target_model(linear),
        "Ridge": target_model(ridge),
        "kNN": target_model(knn),
        "Random Forest": target_model(random_forest),
        "XGBoost (tuned)": target_model(xgboost),
        "LightGBM (tuned)": target_model(lightgbm),
        "CatBoost": target_model(catboost),
        "Stacking (XGB + LGBM + CatBoost + RF)": target_model(stack),
    }


def evaluate(y_true, predictions):
    return {
        "RMSE (M)": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAE (M)": float(mean_absolute_error(y_true, predictions)),
        "R2": float(r2_score(y_true, predictions)),
    }


def time_series_cv(estimator, X, y, model_name):
    fold_scores = []
    for fold, (train_indices, valid_indices) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS).split(X), start=1
    ):
        print(f"  {model_name}: fold {fold}/{N_SPLITS}", flush=True)
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        fold_model = clone(estimator).fit(X_train, y_train)
        fold_scores.append(evaluate(y_valid, fold_model.predict(X_valid)))

    return {
        "RMSE_CV_Mean": float(np.mean([score["RMSE (M)"] for score in fold_scores])),
        "RMSE_CV_Std": float(np.std([score["RMSE (M)"] for score in fold_scores], ddof=1)),
        "MAE_CV_Mean": float(np.mean([score["MAE (M)"] for score in fold_scores])),
        "MAE_CV_Std": float(np.std([score["MAE (M)"] for score in fold_scores], ddof=1)),
        "R2_CV_Mean": float(np.mean([score["R2"] for score in fold_scores])),
        "R2_CV_Std": float(np.std([score["R2"] for score in fold_scores], ddof=1)),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    data = pd.read_excel(DATA_PATH)
    data = add_platform_cumulative_features(data)
    data[CAT_FEATURES] = data[CAT_FEATURES].fillna("Unknown").astype(str)
    data = data.dropna(subset=FEATURES + ["Global_Sales"])
    data = data.sort_values("Year", kind="mergesort").reset_index(drop=True)

    train = data[data["Year"] <= TRAIN_END_YEAR]
    test = data[data["Year"] > TRAIN_END_YEAR]
    if len(train) != 14237 or len(test) != 2090:
        raise ValueError(f"Unexpected temporal split: train={len(train)}, test={len(test)}")

    X_train, y_train = train[FEATURES], train["Global_Sales"]
    X_test, y_test = test[FEATURES], test["Global_Sales"]
    models = make_model_zoo()

    partial_cv_path = MODEL_DIR / "cv_results.partial.csv"
    if partial_cv_path.exists():
        cv_rows = pd.read_csv(partial_cv_path).to_dict(orient="records")
    else:
        cv_rows = []
    completed_cv = {row["Model"] for row in cv_rows}
    fitted_models = {}
    predictions = {}
    for name, estimator in models.items():
        if name not in completed_cv:
            print(f"Cross-validating {name}...", flush=True)
            cv_rows.append({"Model": name, **time_series_cv(estimator, X_train, y_train, name)})
            pd.DataFrame(cv_rows).to_csv(partial_cv_path, index=False)
        else:
            print(f"Reusing completed CV checkpoint for {name}.", flush=True)

        print(f"Fitting {name} on training split...", flush=True)
        fitted = clone(estimator).fit(X_train, y_train)
        fitted_models[name] = fitted
        predictions[name] = fitted.predict(X_test)

    median_prediction = np.full(len(y_test), float(y_train.median()))
    predictions["Median (baseline)"] = median_prediction
    test_rows = [
        {"Model": name, **evaluate(y_test, pred)}
        for name, pred in predictions.items()
    ]
    test_results = pd.DataFrame(test_rows).sort_values("RMSE (M)").reset_index(drop=True)
    cv_results = pd.DataFrame(cv_rows).sort_values("RMSE_CV_Mean").reset_index(drop=True)

    best_name = cv_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    test_results.to_csv(MODEL_DIR / "test_results.csv", index=False)
    cv_results.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    (MODEL_DIR / "cv_results.partial.csv").unlink(missing_ok=True)

    tuning_log = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Configuration": "n_estimators=414, max_depth=4, learning_rate=0.0541, subsample=0.6064, colsample_bytree=0.9769",
                "Source": "Final tuned configuration reported in Final_Report_LNCS_v5.docx",
            },
            {
                "Model": "LightGBM",
                "Configuration": "n_estimators=494, num_leaves=63, learning_rate=0.0482, min_child_samples=53",
                "Source": "Final tuned configuration reported in Final_Report_LNCS_v5.docx",
            },
        ]
    )
    tuning_log.to_csv(MODEL_DIR / "tuning_log.csv", index=False)

    genre_errors = pd.DataFrame(
        {"Genre": test["Genre"].to_numpy(), "absolute_error": np.abs(y_test - best_predictions)}
    )
    genre_errors.groupby("Genre")["absolute_error"].agg([("mean", "mean"), ("count", "size")]).to_csv(
        MODEL_DIR / "error_by_genre.csv"
    )

    from sklearn.inspection import permutation_importance

    print("Calculating permutation importance on the held-out test set...", flush=True)
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).set_index("feature").sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(MODEL_DIR / "permutation_importance.csv")

    stacking_metrics = evaluate(y_test, best_predictions)
    metadata = {
        "model": best_name,
        "model_version": "2.0.0",
        "task": "regression",
        "target": "Global_Sales",
        "target_transform": "log1p; predictions inverse-transformed with expm1",
        "features_cat": CAT_FEATURES,
        "features_num": NUM_FEATURES,
        "metrics_test": {
            "RMSE": stacking_metrics["RMSE (M)"],
            "MAE": stacking_metrics["MAE (M)"],
            "R2": stacking_metrics["R2"],
        },
        "train_years": [int(data["Year"].min()), TRAIN_END_YEAR],
        "test_years": [TRAIN_END_YEAR + 1, int(data["Year"].max())],
        "train_rows": len(train),
        "test_rows": len(test),
        "cv_strategy": "TimeSeriesSplit(5) for model selection; TimeSeriesSplit(3) for chronological stacking OOF",
        "evaluation_note": "Retrospective rerun on the report's reused holdout; not an independent test set. Publisher_Tier and Num_Platforms retain the limitations discussed in the report.",
        "tuning_note": "Uses tuned configurations stated in the report. RandomizedSearchCV search distributions were not included, so the search itself was not repeated.",
        "top_features": importance_frame.head(10).index.tolist(),
    }
    with (MODEL_DIR / "model_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("Test results:")
    print(test_results.to_string(index=False))
    print("\nCV results:")
    print(cv_results.to_string(index=False))
    print(f"\nSaved best model: {best_name}")


if __name__ == "__main__":
    main()