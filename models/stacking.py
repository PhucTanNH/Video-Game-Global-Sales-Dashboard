from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesStackingRegressor(RegressorMixin, BaseEstimator):
    """Stack base predictions from expanding, chronological validation folds."""

    def __init__(self, estimators, final_estimator=None, n_splits=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    @staticmethod
    def _take_rows(values, indices):
        if hasattr(values, "iloc"):
            return values.iloc[indices]
        return np.asarray(values)[indices]

    def fit(self, X, y):
        target = np.asarray(y)
        oof = np.full((len(target), len(self.estimators)), np.nan, dtype=float)
        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        for train_indices, valid_indices in splitter.split(X):
            X_train = self._take_rows(X, train_indices)
            y_train = self._take_rows(y, train_indices)
            X_valid = self._take_rows(X, valid_indices)
            for column, (_, estimator) in enumerate(self.estimators):
                fold_model = clone(estimator).fit(X_train, y_train)
                oof[valid_indices, column] = fold_model.predict(X_valid)

        valid_rows = np.isfinite(oof).all(axis=1)
        self.final_estimator_ = clone(self.final_estimator or Ridge()).fit(
            oof[valid_rows], target[valid_rows]
        )
        self.estimators_ = [
            (name, clone(estimator).fit(X, y))
            for name, estimator in self.estimators
        ]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        base_predictions = np.column_stack(
            [estimator.predict(X) for _, estimator in self.estimators_]
        )
        return self.final_estimator_.predict(base_predictions)