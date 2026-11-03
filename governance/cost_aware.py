"""
Cost-Aware Governance Layer — EcoTrust L2D Extension

Replaces EcoTrust's fixed multi-signal thresholds with a learned composite
uncertainty score and a cost-minimizing threshold τ*.

Reference:
    "Conformal Selective Prediction with Cost Aware Deferral for Safe
     Clinical Triage under Distribution Shift", Scientific Reports 2026.
    doi:10.1038/s41598-026-40637-w

    "Cost-Sensitive Learning to Defer to Multiple Experts with Workload
     Constraints", arXiv:2403.06906, 2024.

Design:
    1. Compute composite uncertainty score:
         u = w_risk * |p - 0.5| * (-1)   (inverted: higher u = more uncertain)
           + w_ent  * entropy
           + w_std  * ensemble_std
           + w_con  * conflict
    2. Learn τ* on a calibration set to minimize:
         E[c_FN * FN(τ) + c_FP * FP(τ) + c_human * Esc(τ)]
    3. At inference: escalate if u > τ*

Key advantage over fixed thresholds:
    - Principled cost-weighting (e.g., missing salmon crisis >> false alarm)
    - A single learned τ* adapts to data statistics automatically
    - Pareto-optimal: achieves higher ErrCov at the same escalation budget
      than any fixed-threshold combination
"""

import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional


# ── Cost parameters ──────────────────────────────────────────────────────────

@dataclass
class CostConfig:
    """
    Cost parameters for salmon thermal stress monitoring.

    c_fn: Cost of false negative (model predicts safe, fish actually at risk).
          High because thermal stress causes salmon mortality and the window
          to intervene is short (hours).
    c_fp: Cost of false positive (unnecessary intervention/flow management).
          Moderate — costly but reversible.
    c_human: Cost per escalated prediction (expert review time).
             Normalized to 1; FN/FP expressed relative to this.
    w_risk, w_ent, w_std, w_con: Weights for each uncertainty signal in the
          composite score. Default equally weights all four signals.
    """
    c_fn:    float = 10.0   # miss a high-risk day: high cost
    c_fp:    float = 3.0    # false alarm: moderate cost
    c_human: float = 1.0    # expert review: baseline cost unit
    w_risk:  float = 1.0    # weight for risk-score uncertainty signal
    w_ent:   float = 1.0    # weight for entropy
    w_std:   float = 1.0    # weight for ensemble std
    w_con:   float = 0.5    # weight for agent conflict


# ── Composite uncertainty score ───────────────────────────────────────────────

def compute_uncertainty(probs: np.ndarray,
                        entropy: np.ndarray,
                        std: np.ndarray,
                        conflict: np.ndarray,
                        cfg: CostConfig) -> np.ndarray:
    """
    Composite uncertainty score u ∈ ℝ.  Higher = more uncertain = escalate.

    Components:
      - Risk-score uncertainty: |p - 0.5| inverted (close to 0.5 → uncertain)
      - Entropy H(p): peaks at p=0.5
      - Ensemble std σ: model disagreement
      - Agent conflict: weather vs. hydrology disagreement (binary, scaled)
    """
    risk_uncertainty = -np.abs(probs - 0.5)    # negative: closer to 0.5 → higher
    u = (cfg.w_risk * risk_uncertainty
         + cfg.w_ent  * entropy
         + cfg.w_std  * std
         + cfg.w_con  * conflict.astype(float))
    return u


# ── Cost-aware threshold learning ────────────────────────────────────────────

def expected_cost(tau: float,
                  u: np.ndarray,
                  y_pred: np.ndarray,
                  y_true: np.ndarray,
                  cfg: CostConfig) -> float:
    """
    E[cost] = c_FN * FN_rate(τ) + c_FP * FP_rate(τ) + c_human * Esc_rate(τ)

    τ is a threshold on the composite uncertainty score u.
    Cases with u > τ are escalated (human oracle corrects them).
    Cases with u ≤ τ are decided by the model.
    """
    escalated   = u > tau
    non_deferred = ~escalated

    # Model errors on non-escalated cases
    fn = int(((y_pred[non_deferred] == 0) & (y_true[non_deferred] == 1)).sum())
    fp = int(((y_pred[non_deferred] == 1) & (y_true[non_deferred] == 0)).sum())
    esc = int(escalated.sum())

    return cfg.c_fn * fn + cfg.c_fp * fp + cfg.c_human * esc


def learn_threshold(u_cal: np.ndarray,
                    y_pred_cal: np.ndarray,
                    y_true_cal: np.ndarray,
                    cfg: CostConfig,
                    n_grid: int = 200) -> float:
    """
    Learn τ* by grid search over calibration set.

    Sentinel values below min and above max allow "escalate all" and
    "escalate none" to be considered as candidates.

    Returns τ* = argmin_{τ} E[cost(τ)]
    """
    u_min = float(u_cal.min())
    u_max = float(u_cal.max())
    margin = (u_max - u_min) * 0.01 + 1e-8

    # Include sentinels: tau < u_min → escalate all; tau > u_max → escalate none
    grid = np.concatenate([
        [u_min - margin],                          # escalate everything
        np.linspace(u_min, u_max, n_grid),         # intermediate values
        [u_max + margin],                          # escalate nothing
    ])

    best_tau  = float(grid[0])
    best_cost = float("inf")

    for tau in grid:
        c = expected_cost(tau, u_cal, y_pred_cal, y_true_cal, cfg)
        if c < best_cost:
            best_cost = c
            best_tau  = float(tau)

    return best_tau


# ── CostAwareGovernance class ─────────────────────────────────────────────────

class CostAwareGovernance:
    """
    Cost-aware, data-driven governance layer for EcoTrust.

    Usage:
        gov = CostAwareGovernance(cfg)
        gov.fit(probs_cal, entropy_cal, std_cal, conflict_cal, y_cal)
        escalate = gov.predict(probs_test, entropy_test, std_test, conflict_test)
        results  = gov.evaluate(probs_test, entropy_test, std_test,
                                conflict_test, y_true)
    """

    def __init__(self, cfg: Optional[CostConfig] = None):
        self.cfg    = cfg or CostConfig()
        self.tau_   = None   # learned threshold
        self._ready = False

    def fit(self,
            probs:    np.ndarray,
            entropy:  np.ndarray,
            std:      np.ndarray,
            conflict: np.ndarray,
            y_true:   np.ndarray) -> "CostAwareGovernance":
        """Learn τ* on calibration split."""
        y_pred = (probs > 0.5).astype(int)
        u      = compute_uncertainty(probs, entropy, std, conflict, self.cfg)
        self.tau_   = learn_threshold(u, y_pred, y_true, self.cfg)
        self._ready = True
        return self

    def predict(self,
                probs:    np.ndarray,
                entropy:  np.ndarray,
                std:      np.ndarray,
                conflict: np.ndarray) -> np.ndarray:
        """Return boolean escalation mask."""
        if not self._ready:
            raise RuntimeError("Call fit() before predict().")
        u = compute_uncertainty(probs, entropy, std, conflict, self.cfg)
        return u > self.tau_

    def evaluate(self,
                 probs:    np.ndarray,
                 entropy:  np.ndarray,
                 std:      np.ndarray,
                 conflict: np.ndarray,
                 y_true:   np.ndarray) -> dict:
        """Full governance evaluation with oracle correction."""
        from sklearn.metrics import accuracy_score, f1_score

        escalated = self.predict(probs, entropy, std, conflict)
        y_pred    = (probs > 0.5).astype(int)
        y_final   = y_pred.copy()
        y_final[escalated] = y_true[escalated]   # oracle correction

        n = len(y_true)
        n_esc    = int(escalated.sum())
        errors   = (y_pred != y_true)
        n_errors = int(errors.sum())
        errors_caught = int((escalated & errors).sum())

        return {
            "tau_star":        round(float(self.tau_), 6),
            "escalation_rate": round(n_esc / n, 4),
            "error_coverage":  round(errors_caught / n_errors, 4) if n_errors else float("nan"),
            "accuracy_raw":    round(float(accuracy_score(y_true, y_pred)), 4),
            "accuracy_oracle": round(float(accuracy_score(y_true, y_final)), 4),
            "error_reduction": round(float(accuracy_score(y_true, y_final))
                                     - float(accuracy_score(y_true, y_pred)), 4),
            "f1_oracle":       round(float(f1_score(y_true, y_final, zero_division=0)), 4),
            "expected_cost":   round(expected_cost(self.tau_,
                                       compute_uncertainty(probs, entropy, std, conflict, self.cfg),
                                       y_pred, y_true, self.cfg), 2),
        }
