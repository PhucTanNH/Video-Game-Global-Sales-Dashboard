# Changelog

## v3.0.0 - Current (branch `feature/research-dashboard-v3`)

**Structure**
- Reorganised the app into six sections: Overview, EDA, Model Comparison, Error Analysis, Leakage Audit and Prediction.
- Moved the code into `dashboard/` (UI) and `vgsales_core/` (UI-free logic). `app.py` is now a thin entry point.
- Removed the duplicated cumulative-feature code. `train_models.py` now imports the shared helper; its behaviour and split are unchanged (14,237 / 2,090 rows).
- Backed up the previous app as `legacy/app_v2.0.0.py`; the `v2.0.0` tag still exists.

**Model comparison**
- Added an interactive experiment runner for Linear Regression, Ridge, kNN, Random Forest, XGBoost, LightGBM, CatBoost and Stacking, plus median and publisher–genre median baselines.
- Models train only when you press Run. Progress is shown per fold, a failing model is recorded instead of crashing the run, and identical configurations are reused from the session cache.
- Hyperparameters, cut-off year (2010–2012), number of year-ahead folds and the seed are configurable. Each run is labelled as report-tuned, fixed defaults or user-set.
- Added year-ahead expanding CV, so a release year is never split, and a retrospective holdout that is never used for selection.
- Stacking now uses chronological out-of-fold meta-features over blocks of release years.
- Added RMSLE, fit, inference and CV timing to the results.
- Saved artefacts and session runs appear in separate tables. Rankings happen only within comparable blocks, and the best value per metric is highlighted in the correct direction.
- Added CSV downloads for results, per-fold scores and all held-out predictions.
- Added charts: RMSE/MAE, R², per-fold CV variability, change vs the median baseline, and runtime.
- Added out-of-sample diagnostics: actual vs predicted, residual histogram, residual vs predicted, absolute-error distribution and ECDF, and a largest-errors table.
- Permutation importance can now be computed on held-out data for session models, with configurable repeats and seed. Models are compared side by side only when they share a feature configuration.

**Analysis**
- EDA now uses medians, IQRs and n per group, log-scale options, and filters for genre, platform, publisher, generation and year. Empty filter results show an explicit message.
- Error analysis groups by genre, platform, generation, year group and publisher tier. It flags small samples and checks the saved-model genre MAE against `error_by_genre.csv`.
- Added a leakage audit that recomputes every Original-27 column from raw rows. It found that `Prev_Franchise_*` use same-year, sales-ordered information, that `Publisher_Tier` uses a full-period ranking, and that several columns follow undocumented definitions.
- Added the Strict-18 and Rebuilt-27 (gaps 0, 1 and 2) feature configurations and a configuration-comparison experiment.
- Added a guard that blocks target-derived columns from every configuration.

**Prediction**
- The prediction page now works with the saved model or any session model.
- History features use only titles released before the release year. The previous version used whole-dataset publisher tiers.
- Added handling for a new publisher or new IP, a check for release years before the platform launch, and warnings for unseen categories and out-of-range inputs.
- Replaced the MAE-based range with a split-conformal interval in log space and stated its coverage assumptions. For session models, the coverage achieved on the holdout is reported.

**Fixes**
- Removed hard-coded "insight" captions that were not computed from the data. They are replaced by captions calculated from the current filter.
- Removed the external Zhan 2026 reference line from the R² chart, because it was not a like-for-like comparison.

**Tests**
- Added `tests/` with 27 pytest and AppTest checks.

## v2.0.0 - Previous

- Replaced the 25-feature XGBoost artifact with the 27-feature model selected by training-only temporal cross-validation. The candidate set includes a stacking ensemble of XGBoost, LightGBM, CatBoost and Random Forest with a Ridge meta-learner.
- Added CatBoost and stacking to the eight-model comparison. Training uses `log1p` and inverse `expm1`, so metrics are in million copies.
- Used five outer temporal folds for model comparison and three chronological OOF folds inside the stack.
- Added platform cumulative-title and cumulative-sales features using earlier release years only.
- Recomputed cross-validation, test metrics, permutation importance and genre error analysis from the reproducible training script.
- Rerun result: CV selected tuned XGBoost (CV RMSE 1.269M; holdout RMSE 0.970M, MAE 0.372M, R² 0.389). Stacking scored holdout RMSE 0.984M and MAE 0.365M, so it was not selected.
- The report's stacking RMSE of 0.942M was not reproduced. The rerun uses the documented estimator settings, prior-year cumulative features and the available preprocessing reconstruction.
- Marked the reported test holdout as retrospective because it was reused during report development.
- Added model/version and rerun notes to the Streamlit dashboard.

## v1.1.0

- Aligned the displayed model, target, feature count and test metrics with the deployed model metadata.
- Calculated prediction features from historical records available before the selected release year.
- Prevented historical lookups from falling back to future records.
- Labelled the MAE-based interval as an approximate prediction range rather than a calibrated 95% confidence interval.
- Fixed rendering of the mixed-type feature details table.
- Showed the current version and update notes in the Streamlit dashboard.

## v1.0.0 - Legacy baseline

- Previous published dashboard baseline, preserved at commit `eebbf49` for comparison.
- Includes the original analysis tabs and the tuned XGBoost sales prediction workflow.
