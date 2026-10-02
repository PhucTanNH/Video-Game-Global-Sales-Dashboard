# Changelog

## v1.1.0 - Current

- Align displayed model, target, feature count, and test metrics with the deployed model metadata.
- Calculate prediction features from historical records available before the selected release year.
- Prevent historical lookups from falling back to future records.
- Label the MAE-based interval as an approximate prediction range rather than a calibrated 95% confidence interval.
- Fix rendering of the mixed-type feature details table.
- Show the current version and update notes in the Streamlit dashboard.

## v1.0.0 - Legacy baseline

- Previous published dashboard baseline, preserved at commit `eebbf49` for comparison.
- Includes the original analysis tabs and tuned XGBoost sales prediction workflow.