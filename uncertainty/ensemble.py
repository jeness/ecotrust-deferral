"""
Ensemble Uncertainty Estimation for EcoTrust agents.

Trains RF + XGBoost + GradientBoosting in parallel and computes:
  - mean prediction (ensemble consensus)
  - variance / std across members (epistemic uncertainty / disagreement)
  - entropy of mean probability (for classification)

This gives two complementary uncertainty signals:
  1. Ensemble disagreement  → captures model uncertainty
  2. CP prediction interval → captures data uncertainty (see conformal.py)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from xgboost import XGBClassifier, XGBRegressor
from typing import List


def _make_pipeline(base_estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", base_estimator),
    ])


# --------------------------------------------------------------------------- #
# Classification ensemble (binary risk prediction)
# --------------------------------------------------------------------------- #

CLASSIFIER_MEMBERS = [
    RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                  random_state=1, eval_metric="logloss", verbosity=0),
    GradientBoostingClassifier(n_estimators=100, max_depth=3,
                               learning_rate=0.05, random_state=2),
]


class EnsembleClassifier:
    """
    Ensemble of RF + XGBoost + GradientBoosting for binary classification.

    Outputs:
        mean_prob   : average P(y=1) across members
        std_prob    : std of P(y=1) across members (disagreement)
        entropy     : H(mean_prob) — higher = more uncertain
    """

    def __init__(self, members=None):
        self.members = [_make_pipeline(clone(m))
                        for m in (members or CLASSIFIER_MEMBERS)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> dict:
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.members], axis=1)
        mean_prob = probs.mean(axis=1)
        std_prob  = probs.std(axis=1)
        eps = 1e-8
        entropy = -(mean_prob * np.log(mean_prob + eps)
                    + (1 - mean_prob) * np.log(1 - mean_prob + eps))
        return {
            "mean_prob":  mean_prob,
            "std_prob":   std_prob,       # main disagreement signal
            "entropy":    entropy,
            "member_probs": probs,        # shape (N, n_members)
        }


# --------------------------------------------------------------------------- #
# Regression ensemble (continuous risk score)
# --------------------------------------------------------------------------- #

REGRESSOR_MEMBERS = [
    RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
    XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                 random_state=1, eval_metric="rmse", verbosity=0),
    GradientBoostingRegressor(n_estimators=100, max_depth=3,
                              learning_rate=0.05, random_state=2),
]


class EnsembleRegressor:
    """
    Ensemble of RF + XGBoost + GradientBoosting for regression.

    Outputs:
        mean_pred : average prediction across members
        std_pred  : std of predictions (disagreement / epistemic uncertainty)
    """

    def __init__(self, members=None):
        self.members = [_make_pipeline(clone(m))
                        for m in (members or REGRESSOR_MEMBERS)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleRegressor":
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> dict:
        preds = np.stack([m.predict(X) for m in self.members], axis=1)
        mean_pred = preds.mean(axis=1)
        std_pred  = preds.std(axis=1)
        return {
            "mean_pred":    mean_pred,
            "std_pred":     std_pred,     # main disagreement signal
            "member_preds": preds,        # shape (N, n_members)
        }
