"""
EcoTrust Decision Learning Layer
================================

Replaces the hand-weighted composite uncertainty score
    u = -w_r|p-0.5| + w_e H + w_s sigma + w_c conflict
and its grid-searched threshold tau* with a two-stage *learned* policy:

  Stage 1 — Meta Risk Estimator  g_phi : s(x) -> P(model errs | s(x))
      A small model (logistic regression / gradient boosting) fitted on a
      temporally held-out calibration split. Inputs are the uncertainty
      signals produced by the trustworthy decision layer; the target is the
      binary event "the frozen predictor M is wrong on this case".
      Nothing about M is retrained.

  Stage 2 — Cost-aware Decision Policy  pi
      Given the estimated error probability r_hat, the predicted label y_hat,
      the deployment costs (c_fn, c_fp, c_h) and the reviewer accuracy a,
      the expected cost of each of the two available actions is:

          E[cost | auto]  = r_hat * c_err(y_hat),
                            with c_err(1) = c_fp, c_err(0) = c_fn
          E[cost | defer] = c_h + (1 - a) * (p_hat * c_fn + (1 - p_hat) * c_fp)

      The policy takes the cheaper action, i.e. it defers iff

          r_hat * c_err(y_hat) > c_h + (1 - a) * (p_hat*c_fn + (1-p_hat)*c_fp)

      With a perfect reviewer (a = 1) this collapses to an *asymmetric,
      closed-form* threshold on the estimated error probability:

          defer iff r_hat > c_h / c_fn   when y_hat = 0   (a miss is expensive)
          defer iff r_hat > c_h / c_fp   when y_hat = 1   (a false alarm is cheap)

      No grid search is required: the decision threshold is analytic in the
      costs, and the only learned object is the risk model g_phi.

This is the classical Bayes decision rule applied to a *learned* posterior
error probability. Its optimality (Prop. 1 in the paper) holds over all
measurable policies, not merely over monotone thresholds of a fixed score.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss


# --------------------------------------------------------------------------- #
# Deployment costs
# --------------------------------------------------------------------------- #

@dataclass
class DeploymentCost:
    """
    Deployment cost structure.

    c_fn : cost of a missed thermal-stress day (fish die; short intervention
           window). Dominant term for salmon management.
    c_fp : cost of a false alarm (unnecessary flow/hatchery action).
    c_h  : cost of one expert review (normalisation unit).
    reviewer_acc : probability that the human expert decides a deferred case
           correctly. 1.0 = the perfect-oracle assumption used by most of the
           human-deferral literature; < 1.0 = a realistic reviewer.
    """
    c_fn: float = 100.0
    c_fp: float = 3.0
    c_h: float = 1.0
    reviewer_acc: float = 1.0

    def err_cost(self, y_hat: np.ndarray) -> np.ndarray:
        """Cost incurred if the automatic decision y_hat turns out to be wrong."""
        return np.where(y_hat == 1, self.c_fp, self.c_fn)


# --------------------------------------------------------------------------- #
# Uncertainty signal bundle
# --------------------------------------------------------------------------- #

#: Canonical signal groups. Each maps to one or more columns of the signal
#: matrix. Grouping matters for the ablation: "conformal" is one *source* of
#: evidence even though it contributes two numbers.
SIGNAL_GROUPS = {
    "confidence": ["margin"],                     # |p - 0.5|
    "entropy":    ["entropy"],                    # H(p)
    "variance":   ["ens_std"],                    # std across ensemble members
    "conflict":   ["conflict"],                   # weather vs. hydrology agents
    "conformal":  ["cp_score", "cp_ambiguous"],   # split-conformal nonconformity
    "novelty":    ["novelty"],                    # distance to the training support
}

ALL_SIGNAL_COLUMNS = [c for g in SIGNAL_GROUPS.values() for c in g]


def build_signals(p_hat: np.ndarray,
                  entropy: np.ndarray,
                  ens_std: np.ndarray,
                  conflict: np.ndarray,
                  cp_score: np.ndarray,
                  cp_ambiguous: np.ndarray,
                  novelty: np.ndarray) -> dict:
    """Assemble the raw uncertainty signals into a named dict of 1-D arrays."""
    return {
        "margin":       np.abs(np.asarray(p_hat, dtype=float) - 0.5),
        "entropy":      np.asarray(entropy, dtype=float),
        "ens_std":      np.asarray(ens_std, dtype=float),
        "conflict":     np.asarray(conflict, dtype=float),
        "cp_score":     np.asarray(cp_score, dtype=float),
        "cp_ambiguous": np.asarray(cp_ambiguous, dtype=float),
        "novelty":      np.asarray(novelty, dtype=float),
    }


# --------------------------------------------------------------------------- #
# Support gate: when is the learned risk model allowed to speak?
# --------------------------------------------------------------------------- #

class SupportGate:
    """
    Detects whether a deployment case lies inside the region where the meta risk
    estimator was actually fitted.

    The estimator g_phi is trained on in-distribution calibration data, so its
    output is only meaningful for inputs drawn from (approximately) the same
    distribution. Outside that support g_phi *extrapolates*, and — as the
    cross-watershed experiment shows — it extrapolates towards "confident and
    correct", which is precisely the wrong direction.

    By default the gate is a squared empirical-Mahalanobis distance to the
    training feature distribution, thresholded at the (1 - eps) empirical
    quantile of the training distances.  Ledoit--Wolf Mahalanobis and squared
    standardized Euclidean distances are available for sensitivity analysis.
    It uses no labels and no target-domain data, so it can be evaluated on the
    first day of deployment at a new site.
    """

    def __init__(self, eps: float = 0.05,
                 metric: str = "mahalanobis_empirical"):
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must lie strictly between 0 and 1")
        if metric not in {"mahalanobis_empirical", "mahalanobis_ledoit_wolf",
                          "euclidean"}:
            raise ValueError("unsupported support-gate metric")
        self.eps = eps
        self.metric = metric
        self.prep_: Optional[Pipeline] = None
        self.cov_ = None
        self.d_thresh_: Optional[float] = None

    def fit(self, X_train: np.ndarray) -> "SupportGate":
        from sklearn.covariance import EmpiricalCovariance, LedoitWolf
        self.prep_ = Pipeline([("i", SimpleImputer(strategy="median")),
                               ("s", StandardScaler())]).fit(X_train)
        Z = self.prep_.transform(X_train)
        if self.metric == "mahalanobis_empirical":
            self.cov_ = EmpiricalCovariance().fit(Z)
            d_tr = self.cov_.mahalanobis(Z)
        elif self.metric == "mahalanobis_ledoit_wolf":
            self.cov_ = LedoitWolf().fit(Z)
            d_tr = self.cov_.mahalanobis(Z)
        else:
            self.cov_ = None
            d_tr = np.square(Z).sum(axis=1)
        self.d_thresh_ = float(np.quantile(d_tr, 1.0 - self.eps))
        return self

    def distance(self, X: np.ndarray) -> np.ndarray:
        Z = self.prep_.transform(X)
        if self.metric == "euclidean":
            return np.square(Z).sum(axis=1)
        return self.cov_.mahalanobis(Z)

    def novelty_feature(self, X: np.ndarray) -> np.ndarray:
        """log(1 + d) — a numerically tame version of the raw distance."""
        return np.log1p(self.distance(X))

    def out_of_support(self, X: np.ndarray) -> np.ndarray:
        return self.distance(X) > self.d_thresh_


def signals_to_matrix(signals: dict, groups: Sequence[str]) -> np.ndarray:
    """Stack the columns belonging to the requested signal groups."""
    cols = [c for g in groups for c in SIGNAL_GROUPS[g]]
    if not cols:
        raise ValueError("At least one signal group is required.")
    return np.column_stack([signals[c] for c in cols])


# --------------------------------------------------------------------------- #
# Stage 1: Meta Risk Estimator
# --------------------------------------------------------------------------- #

class MetaRiskEstimator:
    """
    Learns g_phi : uncertainty signals -> P(the frozen predictor is wrong).

    The target is constructed *without any label of the human decision*: it is
    simply 1[y_hat != y] on the calibration split, so the estimator can be
    fitted from ordinary held-out supervised data. This is the key practical
    difference from Learning-to-Defer, which needs recorded expert decisions.
    """

    def __init__(self,
                 model: str = "logistic",
                 groups: Optional[Sequence[str]] = None,
                 random_state: int = 42):
        self.model_name = model
        self.groups = list(groups) if groups is not None else list(SIGNAL_GROUPS)
        self.random_state = random_state
        self.pipeline_: Optional[Pipeline] = None
        self.fit_diagnostics_: dict = {}

    def _make_model(self):
        if self.model_name == "logistic":
            return LogisticRegression(max_iter=1000,
                                      random_state=self.random_state)
        if self.model_name == "gbm":
            # Shallow + heavily regularised: the calibration split is small
            # (n ~ 500), so an unconstrained booster would memorise it.
            return GradientBoostingClassifier(n_estimators=60,
                                              max_depth=2,
                                              learning_rate=0.05,
                                              subsample=0.8,
                                              random_state=self.random_state)
        raise ValueError(f"unknown meta model: {self.model_name}")

    def fit(self, signals: dict, y_hat: np.ndarray, y_true: np.ndarray) -> "MetaRiskEstimator":
        X = signals_to_matrix(signals, self.groups)
        err = (np.asarray(y_hat) != np.asarray(y_true)).astype(int)

        # The Bayes rule of Prop. 1 consumes r_hat as a *probability*, not as a
        # ranking: the deferral threshold c_h / c_err cuts the probability axis
        # directly, so a miscalibrated r_hat defers at the wrong rate even when
        # its ranking is perfect. An in-sample fit systematically over-estimates
        # the error rate (it has seen these errors), which inflates escalation.
        # Cross-fitted Platt scaling makes the estimate honest: each risk value
        # is produced by a model that did not see that case.
        inner = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", self._make_model()),
        ])

        # Degenerate calibration split (no errors at all): fall back to a
        # constant risk estimate rather than crashing.
        if err.sum() == 0 or err.sum() == len(err):
            self.pipeline_ = None
            self._const = float(err.mean())
            self.fit_diagnostics_ = {"degenerate": True, "const_risk": self._const}
            return self

        n_folds = int(min(5, max(2, err.sum())))
        self.pipeline_ = CalibratedClassifierCV(inner, method="sigmoid", cv=n_folds)
        self.pipeline_.fit(X, err)
        r_in = self.pipeline_.predict_proba(X)[:, 1]
        self.fit_diagnostics_ = {
            "degenerate": False,
            "n_cal": int(len(err)),
            "n_folds": n_folds,
            "cal_error_rate": round(float(err.mean()), 4),
            "mean_predicted_risk": round(float(r_in.mean()), 4),
            "cal_auc": round(float(roc_auc_score(err, r_in)), 4),
            "cal_brier": round(float(brier_score_loss(err, r_in)), 4),
        }
        return self

    def predict_risk(self, signals: dict) -> np.ndarray:
        """Estimated probability that the frozen predictor errs on each case."""
        if self.pipeline_ is None:
            return np.full(len(next(iter(signals.values()))), self._const)
        X = signals_to_matrix(signals, self.groups)
        return self.pipeline_.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Stage 2: Cost-aware decision policy
# --------------------------------------------------------------------------- #

def deferral_rule(r_hat: np.ndarray,
                  p_hat: np.ndarray,
                  y_hat: np.ndarray,
                  cost: DeploymentCost) -> np.ndarray:
    """
    Bayes-optimal deferral decision (Prop. 1).

        defer <=> r_hat * c_err(y_hat) > c_h + (1-a) * (p_hat*c_fn + (1-p_hat)*c_fp)

    Returns a boolean escalation mask.
    """
    auto = np.asarray(r_hat, dtype=float) * cost.err_cost(np.asarray(y_hat))
    a = cost.reviewer_acc
    human = cost.c_h + (1.0 - a) * (np.asarray(p_hat, dtype=float) * cost.c_fn
                                    + (1.0 - np.asarray(p_hat, dtype=float)) * cost.c_fp)
    return auto > human


def expected_saving(r_hat: np.ndarray,
                    p_hat: np.ndarray,
                    y_hat: np.ndarray,
                    cost: DeploymentCost) -> np.ndarray:
    """
    Expected cost saved by deferring a case instead of auto-deciding it.
    Ranking by this quantity yields the budget-constrained optimum (Prop. 2).
    """
    auto = np.asarray(r_hat, dtype=float) * cost.err_cost(np.asarray(y_hat))
    a = cost.reviewer_acc
    human = cost.c_h + (1.0 - a) * (np.asarray(p_hat, dtype=float) * cost.c_fn
                                    + (1.0 - np.asarray(p_hat, dtype=float)) * cost.c_fp)
    return auto - human


class CostAwareDecisionPolicy:
    """
    The full EcoTrust decision-learning policy: meta risk estimator + Bayes rule.

    fit()      : learn g_phi on the calibration split (no retraining of M)
    escalate() : apply the analytic cost rule at deployment
    evaluate() : deployment metrics under a perfect or noisy reviewer
    """

    def __init__(self,
                 cost: Optional[DeploymentCost] = None,
                 meta_model: str = "logistic",
                 groups: Optional[Sequence[str]] = None,
                 support_gate: Optional[SupportGate] = None,
                 risk_ceiling: float = 0.5,
                 random_state: int = 42):
        self.cost = cost or DeploymentCost()
        self.risk = MetaRiskEstimator(model=meta_model, groups=groups,
                                      random_state=random_state)
        # `support_gate = None` disables the fallback: this is the ablation that
        # isolates what the gate contributes (see the cross-watershed table).
        self.support_gate = support_gate
        # Maximum-entropy risk: the value a binary error indicator takes when
        # the estimator carries no usable information about it.
        self.risk_ceiling = risk_ceiling

    # -- learning ----------------------------------------------------------- #
    def fit(self, signals_cal: dict, p_cal: np.ndarray, y_cal: np.ndarray) -> "CostAwareDecisionPolicy":
        y_hat_cal = (np.asarray(p_cal) > 0.5).astype(int)
        self.risk.fit(signals_cal, y_hat_cal, np.asarray(y_cal))
        return self

    # -- deployment --------------------------------------------------------- #
    def deployed_risk(self, signals: dict, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Support-aware risk estimate (Prop. 4).

            r_tilde(x) = r_hat(x)                        if x is in-support
                         max(r_hat(x), risk_ceiling)     otherwise

        Inside the support the learned estimate is used as-is. Outside it, the
        estimate is an unvalidated extrapolation, so it is floored at the
        maximum-entropy value 1/2. Under the Bayes rule this floor by itself
        triggers deferral whenever an error costs more than twice a review
        (c_h / c_err < 1/2) — the fallback is *derived* from the cost structure
        rather than hand-set.
        """
        r = self.risk.predict_risk(signals)
        if self.support_gate is not None and X is not None:
            oos = self.support_gate.out_of_support(X)
            r = np.where(oos, np.maximum(r, self.risk_ceiling), r)
        return r

    def escalate(self, signals: dict, p_hat: np.ndarray,
                 X: Optional[np.ndarray] = None) -> np.ndarray:
        r = self.deployed_risk(signals, X)
        y_hat = (np.asarray(p_hat) > 0.5).astype(int)
        return deferral_rule(r, p_hat, y_hat, self.cost)

    def escalate_budget(self, signals: dict, p_hat: np.ndarray, budget: float,
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """Defer the top-`budget` fraction ranked by expected cost saving."""
        r = self.deployed_risk(signals, X)
        y_hat = (np.asarray(p_hat) > 0.5).astype(int)
        s = expected_saving(r, p_hat, y_hat, self.cost)
        k = int(round(budget * len(s)))
        mask = np.zeros(len(s), dtype=bool)
        if k > 0:
            mask[np.argsort(s)[-k:]] = True
        return mask


# --------------------------------------------------------------------------- #
# Shared evaluation: any escalation mask -> deployment metrics
# --------------------------------------------------------------------------- #

def evaluate_policy(escalated: np.ndarray,
                    p_hat: np.ndarray,
                    y_true: np.ndarray,
                    cost: DeploymentCost,
                    reviewer_acc: Optional[float] = None,
                    n_trials: int = 200,
                    seed: int = 42) -> dict:
    """
    Deployment metrics for an arbitrary escalation mask.

    With reviewer_acc = 1.0 the human oracle is perfect and the result is
    deterministic (the usual upper bound reported in the literature). With
    reviewer_acc < 1.0 the reviewer flips each deferred decision with
    probability 1 - a; we average over `n_trials` Monte-Carlo draws and also
    report the standard deviation, so the noisy-reviewer numbers carry an
    uncertainty band rather than a single optimistic point.

    `deployment_cost` is the realised per-day cost
        (c_fn*FN + c_fp*FP + c_h*Esc) / n
    computed on the *final* decisions, so it charges human mistakes on
    escalated cases exactly like model mistakes on automatic cases.
    """
    escalated = np.asarray(escalated, dtype=bool)
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(p_hat) > 0.5).astype(int)
    a = cost.reviewer_acc if reviewer_acc is None else reviewer_acc

    n = len(y_true)
    errors = (y_hat != y_true)
    n_err = int(errors.sum())
    n_esc = int(escalated.sum())

    def _metrics(y_final: np.ndarray) -> tuple:
        fn = int(((y_final == 0) & (y_true == 1)).sum())
        fp = int(((y_final == 1) & (y_true == 0)).sum())
        dep = (cost.c_fn * fn + cost.c_fp * fp + cost.c_h * n_esc) / n
        return float(accuracy_score(y_true, y_final)), dep, fn, fp

    if a >= 1.0:
        y_final = y_hat.copy()
        y_final[escalated] = y_true[escalated]
        acc, dep, fn, fp = _metrics(y_final)
        acc_sd = dep_sd = 0.0
    else:
        rng = np.random.default_rng(seed)
        accs, deps, fns, fps = [], [], [], []
        for _ in range(n_trials):
            y_final = y_hat.copy()
            if n_esc:
                truth = y_true[escalated]
                correct = rng.random(n_esc) < a
                y_final[escalated] = np.where(correct, truth, 1 - truth)
            acc_t, dep_t, fn_t, fp_t = _metrics(y_final)
            accs.append(acc_t); deps.append(dep_t); fns.append(fn_t); fps.append(fp_t)
        acc, dep = float(np.mean(accs)), float(np.mean(deps))
        acc_sd, dep_sd = float(np.std(accs)), float(np.std(deps))
        fn, fp = float(np.mean(fns)), float(np.mean(fps))

    # Cost of the fully automatic system (no governance) — the reference point.
    fn0 = int(((y_hat == 0) & (y_true == 1)).sum())
    fp0 = int(((y_hat == 1) & (y_true == 0)).sum())
    dep_auto = (cost.c_fn * fn0 + cost.c_fp * fp0) / n

    return {
        "escalation_rate": round(n_esc / n, 4),
        "error_coverage": round(float((escalated & errors).sum() / n_err), 4) if n_err else float("nan"),
        "accuracy_raw": round(float(accuracy_score(y_true, y_hat)), 4),
        "accuracy_final": round(acc, 4),
        "accuracy_final_sd": round(acc_sd, 4),
        "deployment_cost": round(dep, 3),
        "deployment_cost_sd": round(dep_sd, 3),
        "deployment_cost_auto": round(dep_auto, 3),
        "cost_reduction_pct": round(100.0 * (dep_auto - dep) / dep_auto, 1) if dep_auto > 0 else float("nan"),
        "reviewer_acc": a,
        "fn": fn, "fp": fp,
    }


def breakeven_reviewer_accuracy(escalated: np.ndarray,
                                p_hat: np.ndarray,
                                y_true: np.ndarray) -> float:
    """
    Prop. 3: deferral raises expected accuracy iff the reviewer is more accurate
    than the model *on the deferred subpopulation*. The breakeven accuracy is
    therefore exactly the model's accuracy on the escalated cases.
    """
    escalated = np.asarray(escalated, dtype=bool)
    if escalated.sum() == 0:
        return float("nan")
    y_hat = (np.asarray(p_hat) > 0.5).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return float((y_hat[escalated] == y_true[escalated]).mean())
