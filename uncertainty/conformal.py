"""
Split Conformal Prediction wrapper for EcoTrust agents.

Each agent wraps its base model with this class to produce calibrated
prediction intervals. The interval width is used by the governance layer
to trigger human escalation.

Reference:
    Angelopoulos & Bates (2023). "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification." arXiv:2107.07511
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from typing import Optional


class SplitConformalRegressor:
    """
    Split conformal prediction wrapper for any scikit-learn regressor.

    Produces prediction intervals with guaranteed marginal coverage:
        P(Y ∈ [ŷ - q, ŷ + q]) >= 1 - alpha
    where q is the calibrated quantile of nonconformity scores.
    """

    def __init__(self, base_model: BaseEstimator, alpha: float = 0.10):
        self.base_model = base_model
        self.alpha = alpha        # miscoverage rate; 0.10 → 90% coverage
        self.q_hat: Optional[float] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalRegressor":
        """Fit base model on train split, calibrate on held-out cal split."""
        self.model_ = clone(self.base_model)
        self.model_.fit(X_train, y_train)

        cal_preds = self.model_.predict(X_cal)
        scores = np.abs(y_cal - cal_preds)           # nonconformity scores
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict:
        """
        Returns:
            dict with keys:
                'point'  - point prediction
                'lower'  - lower bound of prediction interval
                'upper'  - upper bound of prediction interval
                'width'  - interval width (used for escalation)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        point = self.model_.predict(X)
        return {
            "point": point,
            "lower": point - self.q_hat,
            "upper": point + self.q_hat,
            "width": np.full(len(X), 2 * self.q_hat),
        }

    @property
    def coverage_level(self) -> float:
        return 1.0 - self.alpha


class SplitConformalClassifier:
    """
    Split conformal prediction sets for binary classification.
    Used by the Migration Risk Agent to produce risk probability intervals.
    """

    def __init__(self, base_model: BaseEstimator, alpha: float = 0.10):
        self.base_model = base_model
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        self.model_ = clone(self.base_model)
        self.model_.fit(X_train, y_train)

        cal_proba = self.model_.predict_proba(X_cal)[:, 1]
        # Nonconformity score: 1 - predicted probability for the true class
        scores = np.where(y_cal == 1, 1 - cal_proba, cal_proba)
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        proba = self.model_.predict_proba(X)[:, 1]
        # Prediction set: include class 1 if score <= q_hat
        include_high_risk = (1 - proba) <= self.q_hat
        # Uncertainty score: distance to threshold
        uncertainty = np.abs(proba - 0.5)
        return {
            "risk_score": proba,
            "include_high_risk": include_high_risk,
            "uncertainty": uncertainty,
            "q_hat": self.q_hat,
        }
