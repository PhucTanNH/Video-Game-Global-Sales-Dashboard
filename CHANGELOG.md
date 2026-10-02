# Changelog

## v2.0.0 - Current

- Replaced the 25-feature XGBoost artifact with the 27-feature model selected by training-only temporal cross-validation; the candidate set includes a stacking ensemble of XGBoost, LightGBM, CatBoost, and Random Forest with a Ridge meta-learner.
- Added CatBoost and stacking to the eight-model comparison; training uses `log1p` and inverse `expm1` for metrics in million copies.
- Used five outer temporal folds for model comparison and three chronological OOF folds inside the stack.
- Added platform cumulative-title and cumulative-sales features using earlier release years only.
- Recomputed cross-validation, test metrics, permutation importance, and genre error analysis from the reproducible training script.
- Rerun result: CV selected tuned XGBoost (CV RMSE 1.269M; holdout RMSE 0.970M, MAE 0.372M, R² 0.389). Stacking scored holdout RMSE 0.984M and MAE 0.365M, so it was not selected.
- The report's stacking RMSE 0.942M was not reproduced; the rerun uses documented estimator settings, prior-year cumulative features, and the available preprocessing reconstruction.
- Marked the reported test holdout as retrospective because it was reused during report development.
- Added model/version and rerun notes to the Streamlit dashboard.

## v1.1.0 - Previous

- Align displayed model, target, feature count, and test metrics with the deployed model metadata.
- Calculate prediction features from historical records available before the selected release year.
- Prevent historical lookups from falling back to future records.
- Label the MAE-based interval as an approximate prediction range rather than a calibrated 95% confidence interval.
- Fix rendering of the mixed-type feature details table.
- Show the current version and update notes in the Streamlit dashboard.

## v1.0.0 - Legacy baseline

- Previous published dashboard baseline, preserved at commit `eebbf49` for comparison.
- Includes the original analysis tabs and tuned XGBoost sales prediction workflow.