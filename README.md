# Video Game Global Sales Prediction: Research Dashboard (ADY201m)

A Streamlit dashboard for the ADY201m project. It covers dataset exploration, chronological model comparison, error analysis, a data-leakage audit, and pre-release sales prediction.

## Versions
- **Current:** v3.0.0 (branch `feature/research-dashboard-v3`)
- **Previous:** v2.0.0, v1.1.0 and v1.0.0, preserved as Git tags. A copy of the v2.0.0 app is also kept in `legacy/app_v2.0.0.py`.
- See [CHANGELOG.md](CHANGELOG.md).

## Install and run
Python 3.10 or newer is required. Tested with Python 3.13, Streamlit 1.66, pandas 3.0 and scikit-learn 1.9.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

All paths are resolved relative to the project folder, so the app can be launched from any working directory.

### Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```
`tests/test_core.py` holds unit tests for features, leakage guards, folds, baselines, prediction features and conformal coverage. `tests/test_app.py` holds Streamlit `AppTest` smoke tests: every section, empty filters, a quick experiment, prediction, a failed run, and missing model files.

### Reproduce the saved artefacts
```bash
python train_models.py
```
This step is unchanged from v2.0.0, apart from importing the shared `add_platform_cumulative_features` helper. It rewrites `models/*`.

## Dashboard sections
| Section | What it answers |
|---|---|
| 1. Dataset Overview | Size, coverage and data dictionary with the leakage role of every column. Also shows the saved model card (holdout metrics recomputed live and checked against `model_meta.json`) and the skewed sales distribution. |
| 2. Exploratory Data Analysis | Sales by genre, platform, publisher tier, generation and publisher (n, median, IQR, mean). Also covers regional composition, release-year trends, Spearman/Pearson correlations and a genre × generation heatmap. All filters apply here, and all results are descriptive only. |
| 3. Model Comparison | **Run experiment**: 8 models plus 2 baselines, with configurable hyperparameters, cut-off, folds and seed. **Results tables**: best value highlighted per metric within comparable blocks only. **Charts**: RMSE/MAE, R², CV variability, gain over the median baseline, runtime. **Out-of-sample diagnostics**: actual vs predicted, residuals, error distribution, largest errors, CSV export. **Feature importance**: permutation importance on held-out data. |
| 4. Error Analysis | MAE, RMSE, median absolute error and RMSLE by genre, platform, generation, year group and publisher tier. Sample counts are shown, small groups are hatched, and high-error groups are flagged. |
| 5. Leakage Audit | Variable roles, recomputation evidence, the forbidden-column guard, and an experiment that compares Original-27, Strict-18 and Rebuilt-27 at temporal gaps of 0, 1 and 2 years. |
| 6. Sales Prediction | Prediction from the saved model or any session model. History features are built only from earlier years. The page validates inputs, warns about unseen categories and out-of-range inputs, and shows a split-conformal interval. |

## Evaluation protocol
**Saved artefacts (`models/*.csv`, `best_model.joblib`)**
- Produced by `train_models.py`: training on 1980–2012 (14,237 rows) and holdout on 2013–2020 (2,090 rows).
- Models were selected with row-based `TimeSeriesSplit(5)`, whose folds can split a release year.
- The deployed model is tuned XGBoost on the 27 Original inputs.
- The dashboard recomputes its holdout scores from the artefact and reproduces RMSE 0.970, MAE 0.372 and R² 0.389 exactly.

**Session experiments**
- **Year-ahead expanding CV** inside the development period (years ≤ cut-off). Each fold validates on one whole release year and trains on all earlier years.
- The final fit uses all years ≤ cut-off. The cut-off can be 2010, 2011 or 2012; 2012 is the saved split.
- Years after the cut-off are a **retrospective holdout**. They are reported but never used for selection or tuning. The 2013+ period was already used while the report was developed, so it is labelled retrospective rather than independent.
- Preprocessing (imputation, one-hot encoding, scaling) and the `log1p` target transform sit inside each model's pipeline, so they are re-fitted on every training fold.
- Stacking builds its meta-features from chronological out-of-fold predictions over blocks of release years.
- Comparisons are ranked only inside blocks that share the same feature configuration and protocol. The baselines appear in every block with the same protocol.

## Leakage audit findings (verified by recomputation)
| Input | Finding | Status |
|---|---|---|
| `Prev_Franchise_MaxSales`, `Prev_Franchise_TitleCount` | These include **same-year** franchise releases taken in descending-sales order: the rank correlation is −1 in 100% of the 2,904 testable franchise-years. As a result, 2,864 rows "inherit" the sales of a same-year port (e.g. Saints Row 2 PS3 2008 takes 3.03 from the X360 version). These are the saved model's two most important inputs. | leaky |
| `Publisher_Tier` | Top 10 / 11–50 publishers ranked by title count over the whole 1980–2020 period. | leaky |
| `Num_Platforms`, `Is_Multiplatform` | Platforms of the same title released in years ≤ Y (100% match), so same-year ports are counted. This is pre-release only if those platforms were announced in advance. | uncertain |
| `PubGenre_Specialization`, `GenrePlatform_Fit`, `Publisher_MarketShare_Prev3Y` | The underlying averages follow an undocumented rule. Their counts match prior-year windows. | uncertain |
| `Is_Sequel` | Derived from the title; the derivation code is not in the repository. | uncertain |
| `Pub_/Genre_/Platform_Prev3Y_*`, `SameGenre_Releases_PrevYear`, `Genre_vs_Market_Avg`, `Genre_Trend_Momentum`, `Platform_Cumulative_*` | Strictly prior-year (100% / ≥ 99.8% match). They still use the *lifetime* sales of earlier titles, which were not final at the forecast origin. | prior-year (verified) |
| Regional sales, regional shares, `Dominant_Region`, `Log_Global_Sales` | Derived from the target. A guard blocks them from every configuration. | forbidden |

Rank, critic-score and user-score columns are not present in this Excel file.

**Feature configurations**
- **Original-27** is the saved model's inputs as stored in the file.
- **Strict-18** keeps only the verified prior-year columns plus static attributes. It is defined by this dashboard; the report's own "Strict-13" list was not available.
- **Rebuilt-27 (gap g)** recomputes the 27-column schema from raw rows. History for release year Y uses only titles released up to Y−1−g. A unit test perturbs every invisible year's sales and checks that the features stay identical.

### Example session results (cut-off 2012, 3 year-ahead folds, seed 42, retrospective holdout 2013–2020)
These come from this environment, so rerun them in the app to confirm on your machine.

| XGBoost (report-tuned settings) | CV RMSE | Holdout RMSE | Holdout R² | Holdout RMSLE |
|---|---:|---:|---:|---:|
| Original-27 | 1.016 | 0.988 | 0.366 | 0.291 |
| Strict-18 | 1.139 | 1.155 | 0.133 | 0.339 |
| Rebuilt-27, gap 0 | 0.961 | 0.912 | 0.459 | 0.298 |
| Rebuilt-27, gap 1 | 0.985 | 0.951 | 0.413 | 0.312 |
| Rebuilt-27, gap 2 | 1.045 | 1.029 | 0.311 | 0.319 |
| Median baseline | 1.285 | 1.280 | −0.066 | 0.421 |

How to read this table:
- Removing the franchise history altogether (Strict-18) costs much more than the leaky same-year component contributes. Strict franchise history (Rebuilt, gap 0) actually scores *better* than the file's version.
- Scores fall as the information gap grows. This shows how much the models depend on recent sales history that would not be fully known at a real forecast date.

Retraining the saved XGBoost configuration with 10 different seeds gives holdout RMSEs of 0.953–0.989. The saved artefact's 0.970 sits within that range. Most gaps between the top models (≈0.96–0.99) are therefore within seed-to-seed noise.

## Project structure
```
app.py                     Streamlit entry point (navigation, filters)
dashboard/                 One module per section, plus state.py (caching and session results)
vgsales_core/              UI-free core: config, data, features, models, evaluation, predict, plots
models/                    Saved artefacts from train_models.py (unchanged)
train_models.py            Reproduces the saved artefacts
tests/                     pytest unit tests and Streamlit AppTest smoke tests
legacy/app_v2.0.0.py       Backup of the previous dashboard
ADY201m_enhanced_v2.xlsx   Dataset (16,327 rows, 74 columns; unchanged)
```

## Limitations
- The 2013+ holdout was reused during report development, so no fully independent test set exists.
- Global_Sales are lifetime totals from a later snapshot. Every history feature, including the rebuilt ones with a gap, still contains some post-origin information. A clean pre-release evaluation would need sales as they stood at each release date.
- The report PDF and its "Strict-13" list were not available while building v3.0.0. Wording that refers to the report is based on the repository's own notes.
- Saved results lack per-fold CV scores, holdout RMSLE (except for the deployed model) and timings, so those cells show "None".
- Session results are stored in memory per browser session. Use the CSV downloads to keep them.
- The prediction page builds franchise and publisher-tier inputs strictly from earlier years. The saved model was trained on the leaky file versions, so its predictions can be shifted. Session models trained on Rebuilt features do not have this mismatch.
- The conformal interval assumes that future titles are exchangeable with the calibration titles. The app reports the coverage achieved on the holdout for session models.
