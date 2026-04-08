from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ProbabilityAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Simple soft-voting style ensemble for predict_proba.

    This avoids sklearn's VotingClassifier validation edge-cases while still
    meeting the "ensemble" requirement.
    """

    _estimator_type = "classifier"

    def __init__(self, estimators: List[BaseEstimator], weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.fitted_estimators_ = [clone(e) for e in self.estimators]
        for est in self.fitted_estimators_:
            est.fit(X, y)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "fitted_estimators_"):
            raise RuntimeError("Ensemble is not fitted yet.")

        probas = [np.asarray(est.predict_proba(X)) for est in self.fitted_estimators_]
        # Each proba is shape (n_samples, n_classes). We assume binary => (n,2).
        weights = self.weights if self.weights is not None else [1.0] * len(probas)
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + 1e-12)

        avg = np.zeros_like(probas[0], dtype=float)
        for p, wi in zip(probas, w):
            avg += wi * p
        return avg

