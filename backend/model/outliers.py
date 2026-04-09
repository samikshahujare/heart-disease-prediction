from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = float(iqr_factor)

    def fit(self, X: np.ndarray, y: Any = None) -> "OutlierCapper":
        X_arr = np.asarray(X, dtype=float)
        q1 = np.percentile(X_arr, 25, axis=0)
        q3 = np.percentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.iqr_factor * iqr
        self.upper_bounds_ = q3 + self.iqr_factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(list(input_features), dtype=object)

