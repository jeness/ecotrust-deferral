"""
EcoTrust Decision Learning — full evaluation suite.

Protocol (identical to cost_aware_eval.py, so numbers are comparable):
    train : Aug-Nov, year < 2011          (fit the frozen predictor M)
    cal   : Aug-Nov, 2011 <= year < 2015  (fit the meta risk estimator g_phi)
    test  : Aug-Nov, year >= 2015         (report everything here)

Rows whose water temperature is missing are dropped everywhere: the label
y = 1[water_temp > 18C] is undefined for them. (This matches the protocol of
cross_watershed_eval.py; the Salmon River site has 639 such days.)

Experiments
    E1  Main comparison vs. deferral baselines (matched cost regime)
    E2  Signal contribution: every subset of the 6 uncertainty signal groups
    E3  Cost sensitivity: sweep c_FN -> optimal escalation / coverage / cost
    E4  Review-budget sweep: 5..50% -> error coverage / cost
    E5  Reviewer-accuracy sweep: 70/80/90/95/100%, policy aware vs. unaware
    E6  Meta risk estimator quality (discrimination + calibration on test)
    E7  Zero-shot cross-watershed, with and without the support gate
    E8  Cost-budget Pareto frontier across all risk scores

Run:  cd code-artifacts/experiments && python3 decision_learning_eval.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, itertools
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier

from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error
from uncertainty.conformal import SplitConformalClassifier
from governance.decision_learning import (
    DeploymentCost, CostAwareDecisionPolicy, SupportGate,
    SIGNAL_GROUPS, build_signals, deferral_rule, expected_saving,
    evaluate_policy, breakeven_reviewer_accuracy, signals_to_matrix,
)
from l2d_modern import PostHocL2D, simulated_expert_predictions

DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = CODE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ENV_FEATURES = [
    "TMAX", "TMIN", "PRCP", "AWND",
    "discharge_cfs", "gage_height_ft",
    "discharge_7d_mean", "discharge_anomaly",
]
CONFLICT_THRESHOLD = 0.35
SEED = 42

# Salmon-critical deployment costs: a missed thermal-stress day is ~33x a false
# alarm and 100x an expert review.
MAIN_COST = dict(c_fn=100.0, c_fp=3.0, c_h=1.0)


# ── data ──────────────────────────────────────────────────────────────────── #

def load(name="merged/columbia_merged.csv", year_min=None):
    df = pd.read_csv(DATA_DIR / name)
    df["date"] = pd.to_datetime(df["date"])
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    # The label is only defined where water temperature was actually recorded.
    if "water_temp_c" in df.columns:
        df = df.dropna(subset=["water_temp_c"])
    for col in ENV_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    df = df[df["month"].isin([8, 9, 10, 11])].copy()
    if year_min is not None:
        df = df[df["date"].dt.year >= year_min].copy()
    return df.sort_values("date").reset_index(drop=True)


def monthly_p20(df):
    return df.groupby(df["date"].dt.month)["discharge_cfs"].quantile(0.2)


def conflict_of(df, p20_ref):
    weather = (df["TMAX"] > 28).astype(float).values
    thr = df["date"].dt.month.map(p20_ref.to_dict())
    hydro = (df["discharge_cfs"] < thr).astype(float).values
    return np.abs(weather - hydro) > CONFLICT_THRESHOLD


# ── signal construction ───────────────────────────────────────────────────── #

def make_signals(ens_out, conflict, cp_out, q_hat, novelty):
    p = ens_out["mean_prob"]
    cp_p = cp_out["risk_score"]
    cp_score = np.where(cp_p > 0.5, 1.0 - cp_p, cp_p)
    class1_in = (1.0 - cp_p) <= q_hat
    class0_in = cp_p <= q_hat
    cp_ambiguous = (class1_in & class0_in).astype(float)
    return build_signals(p, ens_out["entropy"], ens_out["std_prob"],
                         conflict.astype(float), cp_score, cp_ambiguous, novelty)


# ── deferral baselines (all operate on the same frozen predictor) ─────────── #

def chow_risk(p_hat, y_hat):
    """The model's own confidence, read as an error probability."""
    return np.where(y_hat == 1, 1.0 - p_hat, p_hat)


def chow_cost_mask(p_hat, y_hat, cost):
    """
    Cost-sensitive reject option (Chow's rule): the same Bayes decision rule as
    EcoTrust, but the error probability is the model's OWN confidence instead of
    a learned estimate. The gap to EcoTrust isolates what the meta risk
    estimator buys.
    """
    return deferral_rule(chow_risk(p_hat, y_hat), p_hat, y_hat, cost)


def selective_mask_at_cost(p_cal, y_cal, p_te, cost):
    """Selective classification with the confidence threshold tuned on the
    calibration split to minimise deployment cost (so SC gets the same cost
    information EcoTrust gets)."""
    conf_cal = np.maximum(p_cal, 1 - p_cal)
    y_hat_cal = (p_cal > 0.5).astype(int)
    best_theta, best_cost = 0.5, np.inf
    for theta in np.linspace(0.5, 1.0, 201):
        esc = conf_cal < theta
        keep = ~esc
        fn = int(((y_hat_cal[keep] == 0) & (y_cal[keep] == 1)).sum())
        fp = int(((y_hat_cal[keep] == 1) & (y_cal[keep] == 0)).sum())
        c = cost.c_fn * fn + cost.c_fp * fp + cost.c_h * int(esc.sum())
        if c < best_cost:
            best_cost, best_theta = c, float(theta)
    return (np.maximum(p_te, 1 - p_te) < best_theta), best_theta


def l2d_rejector_risk(X_cal, p_cal, y_cal, X_te, seed=SEED):
    """
    Post-hoc Learning-to-Defer / learning-with-rejection baseline
    (predictor-rejector decomposition with a frozen predictor). The rejector is
    trained on the RAW input features to predict the model's error. The gap to
    EcoTrust isolates the value of using *uncertainty signals* rather than raw
    covariates as the rejector's input.
    """
    y_hat_cal = (p_cal > 0.5).astype(int)
    err_cal = (y_hat_cal != y_cal).astype(int)
    if err_cal.sum() in (0, len(err_cal)):
        return np.zeros(len(X_te))
    rej = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("s", StandardScaler()),
                    ("m", GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                                     learning_rate=0.05,
                                                     random_state=seed))])
    rej.fit(X_cal, err_cal)
    return rej.predict_proba(X_te)[:, 1]


def fixed_mask(p, ent, std, conflict):
    """EcoTrust v1: union of four hand-tuned thresholds."""
    return (p > 0.60) | (ent > 0.40) | (std > 0.08) | conflict.astype(bool)


def hand_u(p, e, s, c):
    """EcoTrust v1.5 hand-weighted composite score."""
    return -1.0 * np.abs(p - 0.5) + 1.0 * e + 1.0 * s + 0.5 * np.asarray(c, dtype=float)


def grid_u_mask(p_cal, e_cal, s_cal, c_cal, y_cal, p_te, e_te, s_te, c_te, cost):
    u_cal, u_te = hand_u(p_cal, e_cal, s_cal, c_cal), hand_u(p_te, e_te, s_te, c_te)
    y_hat_cal = (p_cal > 0.5).astype(int)
    lo, hi = float(u_cal.min()), float(u_cal.max())
    grid = np.concatenate([[lo - 1e-3], np.linspace(lo, hi, 200), [hi + 1e-3]])
    best_tau, best_cost = grid[0], np.inf
    for tau in grid:
        esc = u_cal > tau
        keep = ~esc
        fn = int(((y_hat_cal[keep] == 0) & (y_cal[keep] == 1)).sum())
        fp = int(((y_hat_cal[keep] == 1) & (y_cal[keep] == 0)).sum())
        c = cost.c_fn * fn + cost.c_fp * fp + cost.c_h * int(esc.sum())
        if c < best_cost:
            best_cost, best_tau = c, float(tau)
    return (u_te > best_tau), best_tau



def temporal_folds(y, years, n_folds=5):
    """Year-blocked expanding folds with a two-class warm-up period."""
    y = np.asarray(y)
    years = np.asarray(years)
    unique_years = np.unique(years)
    warmup_stop = None
    for yr in unique_years:
        fit_idx = np.flatnonzero(years <= yr)
        # The conformal base model itself is fitted on the first 80% of the
        # available history, so that prefix (not merely the full window) must
        # contain both classes before temporal validation can begin.
        inner_cut = int(len(fit_idx) * 0.8)
        if inner_cut > 0 and np.unique(y[fit_idx[:inner_cut]]).size == 2:
            warmup_stop = yr
            break
    if warmup_stop is None or warmup_stop == unique_years[-1]:
        raise ValueError("temporal cross-fitting needs both classes before the final year")
    validation_years = unique_years[unique_years > warmup_stop]
    for year_block in np.array_split(validation_years, min(n_folds, len(validation_years))):
        fit_idx = np.flatnonzero(years < year_block[0])
        valid_idx = np.flatnonzero(np.isin(years, year_block))
        yield fit_idx, valid_idx


def fit_auxiliary_signals(X_fit, y_fit):
    """Fit conformal and support models using only data available so far."""
    cut = int(len(X_fit) * 0.8)
    cp_base = Pipeline([("i", SimpleImputer(strategy="median")),
                        ("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=500, random_state=SEED))])
    cp = SplitConformalClassifier(cp_base, alpha=0.10)
    cp.fit(X_fit[:cut], y_fit[:cut], X_fit[cut:], y_fit[cut:])
    gate = SupportGate(eps=0.05).fit(X_fit)
    return cp, float(cp.q_hat), gate


def oof_meta_data(train_df, X_tr, y_tr, n_folds=5):
    """
    Honest meta-training data from expanding-window temporal cross-fitting.

    The calibration block alone (488 days, ~44 error events) starves the meta
    risk estimator. For each validation block, every learned signal is fitted
    strictly on earlier dates: the backend, conformal model, and support gate.
    The first block is a warm-up period and is not used as meta-training data.
    """
    keys = ["margin", "entropy", "ens_std", "conflict", "cp_score", "cp_ambiguous", "novelty"]
    chunks = {k: [] for k in keys}
    p_chunks, y_chunks = [], []
    years = train_df["date"].dt.year.to_numpy()
    for tr_i, va_i in temporal_folds(y_tr, years, n_folds=n_folds):
        e = EnsembleClassifier().fit(X_tr[tr_i], y_tr[tr_i])
        o = e.predict(X_tr[va_i])
        cp_f, q_f, gate_f = fit_auxiliary_signals(X_tr[tr_i], y_tr[tr_i])
        p20_f = monthly_p20(train_df.iloc[tr_i])
        conf_f = conflict_of(train_df.iloc[va_i], p20_f)
        s = make_signals(o, conf_f, cp_f.predict(X_tr[va_i]), q_f,
                         gate_f.novelty_feature(X_tr[va_i]))
        for k in keys:
            chunks[k].append(s[k])
        p_chunks.append(o["mean_prob"])
        y_chunks.append(y_tr[va_i])
    oof = {k: np.concatenate(v) for k, v in chunks.items()}
    return oof, np.concatenate(p_chunks), np.concatenate(y_chunks)


def concat_signals(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


# ── main ──────────────────────────────────────────────────────────────────── #

def main():
    np.random.seed(SEED)
    out = {}

    print("=" * 78)
    print("EcoTrust Decision Learning — evaluation suite")
    print("=" * 78)

    df = load()
    train = df[df["date"].dt.year < 2011]
    cal = df[(df["date"].dt.year >= 2011) & (df["date"].dt.year < 2015)]
    test = df[df["date"].dt.year >= 2015]

    X_tr, y_tr = train[ENV_FEATURES].values, train["high_migration_risk"].values.astype(int)
    X_cl, y_cl = cal[ENV_FEATURES].values, cal["high_migration_risk"].values.astype(int)
    X_te, y_te = test[ENV_FEATURES].values, test["high_migration_risk"].values.astype(int)

    p20_tr = monthly_p20(train)
    conf_cl, conf_te = conflict_of(cal, p20_tr), conflict_of(test, p20_tr)

    print(f"\ntrain n={len(y_tr)} ({y_tr.mean():.1%} pos) | "
          f"cal n={len(y_cl)} ({y_cl.mean():.1%} pos) | "
          f"test n={len(y_te)} ({y_te.mean():.1%} pos)")

    # Frozen predictor M — never retrained anywhere below.
    ens = EnsembleClassifier().fit(X_tr, y_tr)
    eo_cl, eo_te = ens.predict(X_cl), ens.predict(X_te)
    p_cl, p_te = eo_cl["mean_prob"], eo_te["mean_prob"]
    y_hat_te = (p_te > 0.5).astype(int)
    err_te = (y_hat_te != y_te).astype(int)

    # Auxiliary signal models, fitted inside the TRAIN block only.
    cp, q_hat, gate = fit_auxiliary_signals(X_tr, y_tr)

    sig_cl = make_signals(eo_cl, conf_cl, cp.predict(X_cl), q_hat, gate.novelty_feature(X_cl))
    sig_te = make_signals(eo_te, conf_te, cp.predict(X_te), q_hat, gate.novelty_feature(X_te))

    # Cross-fitted meta-training set: OOF(train) + calibration block.
    sig_oof, p_oof, y_oof = oof_meta_data(train, X_tr, y_tr)
    sig_meta = concat_signals(sig_oof, sig_cl)
    p_meta = np.concatenate([p_oof, p_cl])
    y_meta = np.concatenate([y_oof, y_cl])

    acc_raw = float(accuracy_score(y_te, y_hat_te))
    oos_te = gate.out_of_support(X_te)
    print(f"frozen predictor M: raw test accuracy = {acc_raw:.4f}, q_hat = {q_hat:.3f}")
    print(f"support gate: {oos_te.mean():.1%} of the in-distribution test set is "
          f"flagged out-of-support (eps=5%)")

    cost = DeploymentCost(**MAIN_COST, reviewer_acc=1.0)

    # ── E1: main comparison ──────────────────────────────────────────────── #
    print("\n" + "-" * 78)
    print("E1  Main comparison @ c_FN:c_FP:c_h = 100:3:1")
    print("-" * 78)
    rows = []

    def add(name, mask, extra=None):
        r = evaluate_policy(mask, p_te, y_te, cost, reviewer_acc=1.0)
        r90 = evaluate_policy(mask, p_te, y_te, cost, reviewer_acc=0.90, seed=SEED)
        r["accuracy_at_90"] = r90["accuracy_final"]
        r["deployment_cost_at_90"] = r90["deployment_cost"]
        r["breakeven_reviewer_acc"] = round(breakeven_reviewer_accuracy(mask, p_te, y_te), 4)
        r["method"] = name
        if extra:
            r.update(extra)
        rows.append(r)
        print(f"{name:40s} esc={r['escalation_rate']:6.1%} cov={r['error_coverage']:6.1%} "
              f"cost={r['deployment_cost']:6.3f} acc={r['accuracy_final']:.4f} "
              f"acc@90={r['accuracy_at_90']:.4f}")
        return r

    def add_multiseed(name, models, X_eval, p_eval, y_eval, extra=None):
        evals, evals90, masks = [], [], []
        for model in models:
            mask = model.escalate(X_eval, p_eval)
            masks.append(mask)
            evals.append(evaluate_policy(mask, p_eval, y_eval, cost, reviewer_acc=1.0))
            evals90.append(evaluate_policy(mask, p_eval, y_eval, cost,
                                           reviewer_acc=0.90, seed=SEED))
        keys = ["escalation_rate", "error_coverage", "accuracy_raw",
                "accuracy_final", "deployment_cost", "deployment_cost_auto"]
        r = {"method": name, "n_seeds": len(models)}
        for key in keys:
            vals = np.asarray([ev[key] for ev in evals], dtype=float)
            digits = 3 if "cost" in key else 4
            r[key] = round(float(vals.mean()), digits)
            r[f"{key}_sd"] = round(float(vals.std(ddof=0)), digits)
        vals90 = np.asarray([ev["accuracy_final"] for ev in evals90], dtype=float)
        costs90 = np.asarray([ev["deployment_cost"] for ev in evals90], dtype=float)
        r["accuracy_at_90"] = round(float(vals90.mean()), 4)
        r["accuracy_at_90_sd"] = round(float(vals90.std(ddof=0)), 4)
        r["deployment_cost_at_90"] = round(float(costs90.mean()), 3)
        r["deployment_cost_at_90_sd"] = round(float(costs90.std(ddof=0)), 3)
        r["breakeven_reviewer_acc"] = round(float(np.mean([
            breakeven_reviewer_accuracy(m, p_eval, y_eval) for m in masks
        ])), 4)
        if extra:
            r.update(extra)
        rows.append(r)
        print(f"{name:40s} esc={r['escalation_rate']:6.1%} cov={r['error_coverage']:6.1%} "
              f"cost={r['deployment_cost']:6.3f}+-{r['deployment_cost_sd']:.3f} "
              f"acc={r['accuracy_final']:.4f} acc@90={r['accuracy_at_90']:.4f}")
        return r

    # Modern L2D baselines.  Both use the exact published surrogate but in a
    # post-hoc restriction: the frozen backend class logits never change and
    # only the defer logit is learned.  Synthetic 90%-accurate expert decisions
    # are available on temporal OOF training rows, as required by L2D.
    l2d_groups = list(SIGNAL_GROUPS)
    Z_oof = signals_to_matrix(sig_oof, l2d_groups)
    Z_cl = signals_to_matrix(sig_cl, l2d_groups)
    Z_te = signals_to_matrix(sig_te, l2d_groups)
    l2d_seeds = [40, 41, 42]
    modern_l2d_models = {"ms": [], "ova": []}
    for method in modern_l2d_models:
        for seed in l2d_seeds:
            expert_oof = simulated_expert_predictions(y_oof, accuracy=0.90, seed=seed)
            # Mozannar--Sontag explicitly introduces alpha as a surrogate
            # hyperparameter.  Select it on the calibration period, together
            # with the operating threshold, instead of giving alpha=1 an
            # arbitrary advantage/disadvantage.  OvA has no alpha analogue.
            alpha_grid = [0.0, 0.5, 1.0] if method == "ms" else [1.0]
            candidates = []
            for alpha in alpha_grid:
                candidate = PostHocL2D(
                    method=method, seed=seed, alpha=alpha
                ).fit(
                    Z_oof, p_oof, y_oof, expert_oof
                ).tune_cost_threshold(Z_cl, p_cl, y_cl, cost)
                candidates.append(candidate)
            model = min(
                candidates,
                key=lambda m: (m.fit_diagnostics_["calibration_cost_total"],
                               m.fit_diagnostics_["calibration_escalations"]),
            )
            model.fit_diagnostics_["selected_alpha"] = model.alpha
            model.fit_diagnostics_["alpha_grid"] = alpha_grid
            modern_l2d_models[method].append(model)

    add("A0 No governance (argmax)", np.zeros(len(y_te), dtype=bool))

    # With c_FN = 100 a rational operator would simply lower the decision
    # threshold and never hire a reviewer. Any deferral method must beat THIS.
    t_cs = MAIN_COST["c_fp"] / (MAIN_COST["c_fp"] + MAIN_COST["c_fn"])
    y_cs = (p_te > t_cs).astype(int)
    fn_cs = int(((y_cs == 0) & (y_te == 1)).sum())
    fp_cs = int(((y_cs == 1) & (y_te == 0)).sum())
    cost_cs = (cost.c_fn * fn_cs + cost.c_fp * fp_cs) / len(y_te)
    rows.append({
        "method": "A1 Cost-sensitive threshold (no human)",
        "escalation_rate": 0.0, "error_coverage": 0.0,
        "accuracy_raw": round(acc_raw, 4),
        "accuracy_final": round(float(accuracy_score(y_te, y_cs)), 4),
        "accuracy_at_90": round(float(accuracy_score(y_te, y_cs)), 4),
        "deployment_cost": round(cost_cs, 3),
        "deployment_cost_at_90": round(cost_cs, 3),
        "threshold": round(t_cs, 4),
        "note": "fully automatic; cost-optimal Bayes threshold on p_hat",
    })
    print(f"{'A1 Cost-sensitive threshold (no human)':40s} esc=  0.0% cov=  0.0% "
          f"cost={cost_cs:6.3f} acc={accuracy_score(y_te, y_cs):.4f}")

    add("B1 EcoTrust v1 (4 fixed thresholds)",
        fixed_mask(p_te, eo_te["entropy"], eo_te["std_prob"], conf_te))

    m_grid, tau_grid = grid_u_mask(p_cl, eo_cl["entropy"], eo_cl["std_prob"], conf_cl, y_cl,
                                   p_te, eo_te["entropy"], eo_te["std_prob"], conf_te, cost)
    add("B2 EcoTrust v1.5 (hand-weighted u, grid tau)", m_grid, {"tau_star": round(tau_grid, 4)})

    m_sc, theta_sc = selective_mask_at_cost(p_cl, y_cl, p_te, cost)
    add("B3 Selective classification (cost-tuned)", m_sc, {"theta": round(theta_sc, 4)})

    m_chow = chow_cost_mask(p_te, y_hat_te, cost)
    add("B4 Cost-sensitive reject option (Chow)", m_chow)

    r_l2d_te = l2d_rejector_risk(X_cl, p_cl, y_cl, X_te)
    m_l2d_raw = deferral_rule(r_l2d_te, p_te, y_hat_te, cost)
    add("B5 Learning-to-defer rejector (raw feats)", m_l2d_raw)
    r_l2d_ms = add_multiseed(
        "B6 L2D-MS (post-hoc, signals)", modern_l2d_models["ms"], Z_te, p_te, y_te,
        {"protocol": "Mozannar-Sontag Eq.6; frozen class logits; 90% expert demos",
         "fit_diagnostics": [m.fit_diagnostics_ for m in modern_l2d_models["ms"]]},
    )
    r_l2d_ova = add_multiseed(
        "B7 L2D-OvA (post-hoc, signals)", modern_l2d_models["ova"], Z_te, p_te, y_te,
        {"protocol": "Verma-Nalisnick Eq.8; frozen class logits; 90% expert demos",
         "fit_diagnostics": [m.fit_diagnostics_ for m in modern_l2d_models["ova"]]},
    )

    # EcoTrust decision learning.
    pol = CostAwareDecisionPolicy(cost=cost, meta_model="logistic",
                                  support_gate=gate).fit(sig_meta, p_meta, y_meta)
    m_dl = pol.escalate(sig_te, p_te, X_te)
    r_dl = add("C1 EcoTrust-DL (logistic meta risk)", m_dl,
               {"meta_diagnostics": pol.risk.fit_diagnostics_})

    pol_gb = CostAwareDecisionPolicy(cost=cost, meta_model="gbm",
                                     support_gate=gate).fit(sig_meta, p_meta, y_meta)
    add("C2 EcoTrust-DL (GBM meta risk)", pol_gb.escalate(sig_te, p_te, X_te),
        {"meta_diagnostics": pol_gb.risk.fit_diagnostics_})

    # Ablation: same policy without the support gate.
    pol_ng = CostAwareDecisionPolicy(cost=cost, meta_model="logistic",
                                     support_gate=None).fit(sig_meta, p_meta, y_meta)
    m_ng = pol_ng.escalate(sig_te, p_te)
    add("C3 EcoTrust-DL, no support gate", m_ng)

    rng = np.random.default_rng(SEED)
    k = int(r_dl["escalation_rate"] * len(y_te))
    m_rand = np.zeros(len(y_te), dtype=bool)
    m_rand[rng.choice(len(y_te), size=k, replace=False)] = True
    add("B0 Random deferral (rate-matched)", m_rand)

    out["E1_main_comparison"] = {
        "protocol": "train<2011 | cal 2011-2014 | test 2015+",
        "costs": MAIN_COST, "n_test": int(len(y_te)),
        "oos_rate_in_distribution": round(float(oos_te.mean()), 4),
        "rows": rows,
    }

    # ── E2: signal contribution (all subsets) ───────────────────────────── #
    print("\n" + "-" * 78)
    print("E2  Signal contribution — every subset of the 6 signal groups")
    print("-" * 78)
    names = list(SIGNAL_GROUPS)
    subsets = []
    for r_ in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r_):
            pol_s = CostAwareDecisionPolicy(cost=cost, meta_model="logistic",
                                            groups=list(combo),
                                            support_gate=gate).fit(sig_meta, p_meta, y_meta)
            mask = pol_s.escalate(sig_te, p_te, X_te)
            ev = evaluate_policy(mask, p_te, y_te, cost, reviewer_acc=1.0)
            r_hat = pol_s.risk.predict_risk(sig_te)
            auc = float(roc_auc_score(err_te, r_hat)) if err_te.sum() else float("nan")
            subsets.append({
                "signals": list(combo), "k": len(combo),
                "risk_auc_test": round(auc, 4),
                "escalation_rate": ev["escalation_rate"],
                "error_coverage": ev["error_coverage"],
                "deployment_cost": ev["deployment_cost"],
                "accuracy_final": ev["accuracy_final"],
            })
    subsets.sort(key=lambda d: d["deployment_cost"])
    for s in subsets[:5]:
        print(f"  {'+'.join(s['signals']):52s} AUC={s['risk_auc_test']:.3f} "
              f"esc={s['escalation_rate']:5.1%} cost={s['deployment_cost']:6.3f}")
    print("  ...")
    for s in subsets[-2:]:
        print(f"  {'+'.join(s['signals']):52s} AUC={s['risk_auc_test']:.3f} "
              f"esc={s['escalation_rate']:5.1%} cost={s['deployment_cost']:6.3f}")
    # Leave-one-out relative to the full bank.
    full = [s for s in subsets if s["k"] == len(names)][0]
    loo = []
    for drop in names:
        keep = [n for n in names if n != drop]
        s = [x for x in subsets if x["signals"] == keep][0]
        loo.append({"dropped": drop,
                    "risk_auc_test": s["risk_auc_test"],
                    "deployment_cost": s["deployment_cost"],
                    "delta_cost_vs_full": round(s["deployment_cost"] - full["deployment_cost"], 3),
                    "delta_auc_vs_full": round(s["risk_auc_test"] - full["risk_auc_test"], 4)})
    print("\n  Leave-one-out (positive delta_cost = the signal was helping):")
    for l in sorted(loo, key=lambda d: -d["delta_cost_vs_full"]):
        print(f"    drop {l['dropped']:12s} cost {l['deployment_cost']:6.3f} "
              f"({l['delta_cost_vs_full']:+.3f})  AUC {l['risk_auc_test']:.3f} "
              f"({l['delta_auc_vs_full']:+.4f})")
    out["E2_signal_contribution"] = {"subsets": subsets, "full": full, "leave_one_out": loo}

    # ── E3: cost sensitivity ─────────────────────────────────────────────── #
    print("\n" + "-" * 78)
    print("E3  Cost sensitivity — sweep c_FN (c_FP=3, c_h=1)")
    print("-" * 78)
    sweep = []
    for c_fn in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        cst = DeploymentCost(c_fn=float(c_fn), c_fp=3.0, c_h=1.0, reviewer_acc=1.0)
        pol_c = CostAwareDecisionPolicy(cost=cst, meta_model="logistic",
                                        support_gate=gate).fit(sig_meta, p_meta, y_meta)
        ev = evaluate_policy(pol_c.escalate(sig_te, p_te, X_te), p_te, y_te, cst, reviewer_acc=1.0)
        ev_chow = evaluate_policy(chow_cost_mask(p_te, y_hat_te, cst), p_te, y_te, cst,
                                  reviewer_acc=1.0)
        sweep.append({
            "c_fn": c_fn,
            "tau_pred_neg": round(1.0 / c_fn, 4),   # analytic: c_h / c_fn
            "tau_pred_pos": round(1.0 / 3.0, 4),    # analytic: c_h / c_fp
            "escalation_rate": ev["escalation_rate"],
            "error_coverage": ev["error_coverage"],
            "deployment_cost": ev["deployment_cost"],
            "deployment_cost_auto": ev["deployment_cost_auto"],
            "cost_reduction_pct": ev["cost_reduction_pct"],
            "chow_escalation_rate": ev_chow["escalation_rate"],
            "chow_deployment_cost": ev_chow["deployment_cost"],
        })
        print(f"  c_FN={c_fn:4d}  esc={ev['escalation_rate']:6.1%} cov={ev['error_coverage']:6.1%} "
              f"cost={ev['deployment_cost']:7.3f} (auto {ev['deployment_cost_auto']:7.3f}, "
              f"-{ev['cost_reduction_pct']:5.1f}%) | Chow: esc={ev_chow['escalation_rate']:6.1%} "
              f"cost={ev_chow['deployment_cost']:7.3f}")
    out["E3_cost_sensitivity"] = sweep

    # ── E4 + E8: fixed review-rate comparison across risk scores ────────── #
    print("\n" + "-" * 78)
    print("E4/E8  Cost at fixed review rates (exactly k reviews)")
    print("-" * 78)
    r_dl_te = pol.deployed_risk(sig_te, X_te)
    r_self_te = chow_risk(p_te, y_hat_te)
    u_te = hand_u(p_te, eo_te["entropy"], eo_te["std_prob"], conf_te)
    r_ms_te = np.mean([m.defer_score(Z_te, p_te) for m in modern_l2d_models["ms"]], axis=0)
    r_ova_te = np.mean([m.defer_score(Z_te, p_te) for m in modern_l2d_models["ova"]], axis=0)

    # Every method ranks by expected cost saving under ITS risk estimate, so the
    # comparison isolates the quality of the risk estimate alone.
    scorers = {
        "EcoTrust-DL (learned risk)": expected_saving(r_dl_te, p_te, y_hat_te, cost),
        "Chow / self-confidence": expected_saving(r_self_te, p_te, y_hat_te, cost),
        "L2D rejector (raw feats)": expected_saving(r_l2d_te, p_te, y_hat_te, cost),
        "L2D-MS (post-hoc)": r_ms_te,
        "L2D-OvA (post-hoc)": r_ova_te,
        "Selective classification": -np.maximum(p_te, 1 - p_te),   # cost-blind by definition
        "Hand-weighted score u": u_te,
    }
    budget_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    frontier = {}
    for nm, sc in scorers.items():
        pts = []
        for b in budget_grid:
            kk = int(round(b * len(y_te)))
            m = np.zeros(len(y_te), dtype=bool)
            if kk:
                m[np.argsort(sc)[-kk:]] = True
            ev = evaluate_policy(m, p_te, y_te, cost, reviewer_acc=1.0)
            ev90 = evaluate_policy(m, p_te, y_te, cost, reviewer_acc=0.90, seed=SEED)
            pts.append({"budget": b,
                        "error_coverage": ev["error_coverage"],
                        "deployment_cost": ev["deployment_cost"],
                        "accuracy_final": ev["accuracy_final"],
                        "accuracy_at_90": ev90["accuracy_final"],
                        "deployment_cost_at_90": ev90["deployment_cost"]})
        frontier[nm] = pts
        row = "  ".join(f"{p['deployment_cost']:6.3f}" for p in pts)
        print(f"  {nm:28s} cost@budget: {row}")
    print(f"  {'budgets':28s}             " +
          "  ".join(f"{b:6.0%}" for b in budget_grid))
    cov_row = "  ".join(f"{p['error_coverage']:6.1%}"
                        for p in frontier["EcoTrust-DL (learned risk)"])
    print(f"  {'EcoTrust-DL ErrCov':28s}          {cov_row}")
    # This is deliberately called a fixed-rate curve, not an optimal frontier:
    # forcing exactly k reviews can increase realised cost after useful cases are
    # exhausted. The convex frontier in Prop. 2 instead uses AT MOST k reviews
    # and truncates at positive saving.
    out["E8_fixed_review_rate"] = {"budgets": budget_grid, "curves": frontier,
                                   "protocol": "exactly k reviews; no positive-saving truncation"}
    out["E8_pareto_frontier"] = {"deprecated_name": True, "budgets": budget_grid,
                                 "frontier": frontier}

    # Oracle diagnostic for Prop. 2: true per-case saving, at most k reviews,
    # positive-saving truncation. This is not a deployable baseline.
    true_saving = np.where(err_te.astype(bool), cost.err_cost(y_hat_te), 0.0) - cost.c_h
    oracle_pts = []
    positive_order = np.flatnonzero(true_saving > 0)
    positive_order = positive_order[np.argsort(true_saving[positive_order])[::-1]]
    for b in budget_grid:
        kk = min(int(round(b * len(y_te))), len(positive_order))
        m = np.zeros(len(y_te), dtype=bool)
        m[positive_order[:kk]] = True
        ev = evaluate_policy(m, p_te, y_te, cost, reviewer_acc=1.0)
        oracle_pts.append({"budget": b, "reviews_used": int(m.sum()),
                           "review_rate_used": round(float(m.mean()), 4),
                           "deployment_cost": ev["deployment_cost"],
                           "error_coverage": ev["error_coverage"]})
    out["E8_oracle_optimal_frontier"] = {
        "protocol": "at most k reviews; true saving; truncate at saving > 0",
        "points": oracle_pts,
    }
    out["E4_budget"] = frontier["EcoTrust-DL (learned risk)"] + [{
        "budget": r_dl["escalation_rate"], "unconstrained": True,
        "error_coverage": r_dl["error_coverage"],
        "deployment_cost": r_dl["deployment_cost"],
        "accuracy_final": r_dl["accuracy_final"],
        "accuracy_at_90": r_dl["accuracy_at_90"],
    }]

    # ── E5: reviewer-accuracy sweep, policy aware vs. unaware ────────────── #
    print("\n" + "-" * 78)
    print("E5  Reviewer accuracy — oracle-unaware vs. reviewer-aware policy")
    print("-" * 78)
    rev = []
    reviewer_baselines = {
        "Chow": [m_chow], "Selective": [m_sc], "L2D_raw": [m_l2d_raw],
        "EcoTrust_DL": [m_dl],
        "L2D_MS": [m.escalate(Z_te, p_te) for m in modern_l2d_models["ms"]],
        "L2D_OvA": [m.escalate(Z_te, p_te) for m in modern_l2d_models["ova"]],
    }
    reviewer_baseline_rows = []
    for a in [0.70, 0.80, 0.90, 0.95, 1.00]:
        ev_un = evaluate_policy(m_dl, p_te, y_te, cost, reviewer_acc=a, seed=SEED)
        cst_a = DeploymentCost(**MAIN_COST, reviewer_acc=a)
        pol_a = CostAwareDecisionPolicy(cost=cst_a, meta_model="logistic",
                                        support_gate=gate).fit(sig_meta, p_meta, y_meta)
        m_a = pol_a.escalate(sig_te, p_te, X_te)
        ev_aw = evaluate_policy(m_a, p_te, y_te, cst_a, reviewer_acc=a, seed=SEED)
        rev.append({
            "reviewer_acc": a,
            "unaware_escalation": ev_un["escalation_rate"],
            "unaware_accuracy": ev_un["accuracy_final"],
            "unaware_accuracy_sd": ev_un["accuracy_final_sd"],
            "unaware_cost": ev_un["deployment_cost"],
            "aware_escalation": ev_aw["escalation_rate"],
            "aware_accuracy": ev_aw["accuracy_final"],
            "aware_accuracy_sd": ev_aw["accuracy_final_sd"],
            "aware_cost": ev_aw["deployment_cost"],
        })
        for baseline_name, masks in reviewer_baselines.items():
            evals = [evaluate_policy(mask, p_te, y_te, cost, reviewer_acc=a,
                                     seed=SEED + i)
                     for i, mask in enumerate(masks)]
            reviewer_baseline_rows.append({
                "reviewer_acc": a, "method": baseline_name,
                "n_policy_seeds": len(masks),
                "escalation_rate": round(float(np.mean(
                    [e["escalation_rate"] for e in evals])), 4),
                "accuracy": round(float(np.mean(
                    [e["accuracy_final"] for e in evals])), 4),
                "accuracy_between_policy_sd": round(float(np.std(
                    [e["accuracy_final"] for e in evals])), 4),
                "mean_monte_carlo_accuracy_sd": round(float(np.mean(
                    [e["accuracy_final_sd"] for e in evals])), 4),
                "deployment_cost": round(float(np.mean(
                    [e["deployment_cost"] for e in evals])), 3),
                "deployment_cost_between_policy_sd": round(float(np.std(
                    [e["deployment_cost"] for e in evals])), 3),
            })
        print(f"  a={a:.0%}  unaware: esc={ev_un['escalation_rate']:6.1%} "
              f"acc={ev_un['accuracy_final']:.4f}+-{ev_un['accuracy_final_sd']:.4f} "
              f"cost={ev_un['deployment_cost']:6.3f}  ||  aware: esc={ev_aw['escalation_rate']:6.1%} "
              f"acc={ev_aw['accuracy_final']:.4f}+-{ev_aw['accuracy_final_sd']:.4f} "
              f"cost={ev_aw['deployment_cost']:6.3f}")
    be = breakeven_reviewer_accuracy(m_dl, p_te, y_te)
    print(f"\n  Prop.3: predicted breakeven reviewer accuracy = model accuracy on the "
          f"escalated set = {be:.4f}")
    out["E5_reviewer_accuracy"] = {
        "rows": rev, "baseline_rows": reviewer_baseline_rows,
        "predicted_breakeven": round(be, 4),
        "raw_accuracy": round(acc_raw, 4),
    }

    # ── E6: meta risk estimator quality ──────────────────────────────────── #
    print("\n" + "-" * 78)
    print("E6  Meta risk estimator quality on the test split")
    print("-" * 78)
    r_meta = pol.risk.predict_risk(sig_te)     # ungated, to score the estimator itself
    quality = {
        "n_test": int(len(y_te)),
        "test_error_rate": round(float(err_te.mean()), 4),
        "meta_auc": round(float(roc_auc_score(err_te, r_meta)), 4),
        "meta_brier": round(float(brier_score_loss(err_te, r_meta)), 4),
        "selfconf_auc": round(float(roc_auc_score(err_te, r_self_te)), 4),
        "selfconf_brier": round(float(brier_score_loss(err_te, r_self_te)), 4),
        "l2d_auc": round(float(roc_auc_score(err_te, r_l2d_te)), 4),
        "l2d_brier": round(float(brier_score_loss(err_te, r_l2d_te)), 4),
        "l2d_ms_defer_auc": round(float(roc_auc_score(err_te, r_ms_te)), 4),
        "l2d_ova_defer_auc": round(float(roc_auc_score(err_te, r_ova_te)), 4),
        "cal_diagnostics": pol.risk.fit_diagnostics_,
    }
    bins = np.quantile(r_meta, np.linspace(0, 1, 11)); bins[-1] += 1e-9
    rel = []
    for i in range(10):
        m = (r_meta >= bins[i]) & (r_meta < bins[i + 1])
        if m.sum() >= 5:
            rel.append({"bin": i + 1, "n": int(m.sum()),
                        "mean_predicted": round(float(r_meta[m].mean()), 4),
                        "empirical_error": round(float(err_te[m].mean()), 4)})
    quality["reliability_deciles"] = rel
    quality["approx_monotone"] = bool(all(
        rel[i]["empirical_error"] <= rel[i + 1]["empirical_error"] + 0.05
        for i in range(len(rel) - 1)))
    print(f"  learned risk (signals) : AUC={quality['meta_auc']:.4f}  Brier={quality['meta_brier']:.4f}")
    print(f"  model self-confidence  : AUC={quality['selfconf_auc']:.4f}  Brier={quality['selfconf_brier']:.4f}")
    print(f"  L2D rejector (raw x)   : AUC={quality['l2d_auc']:.4f}  Brier={quality['l2d_brier']:.4f}")
    print(f"  L2D-MS defer score     : AUC={quality['l2d_ms_defer_auc']:.4f}")
    print(f"  L2D-OvA defer score    : AUC={quality['l2d_ova_defer_auc']:.4f}")
    print(f"  approx. monotone reliability: {quality['approx_monotone']}")
    out["E6_meta_quality"] = quality

    # ── E7: zero-shot cross-watershed, gate ablation ─────────────────────── #
    print("\n" + "-" * 78)
    print("E7  Zero-shot multi-basin transfer — does the system know it does not know?")
    print("-" * 78)
    metadata_path = DATA_DIR / "multibasin" / "station_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is required; run data/fetch_multibasin.py first"
        )
    with open(metadata_path) as f:
        station_metadata = json.load(f)["sites"]
    ood_rows = []
    ood_eval_cache = []
    for site_meta in station_metadata:
        site = site_meta["station_name"]
        fname = f"merged/multibasin/{site_meta['site_no']}.csv"
        dfo = load(fname, year_min=2015)
        Xo = dfo[ENV_FEATURES].values
        yo = dfo["high_migration_risk"].values.astype(int)
        eo_o = ens.predict(Xo)
        p_o = eo_o["mean_prob"]
        y_hat_o = (p_o > 0.5).astype(int)
        # The conflict signal is watershed-specific (it uses Columbia discharge
        # percentiles), so it is disabled off-basin — a conservative choice.
        conf_o = np.zeros(len(yo), dtype=bool)
        sig_o = make_signals(eo_o, conf_o, cp.predict(Xo), q_hat, gate.novelty_feature(Xo))
        Z_o = signals_to_matrix(sig_o, l2d_groups)

        r_gate = pol.deployed_risk(sig_o, Xo)      # support-aware
        r_nogate = pol_ng.risk.predict_risk(sig_o)  # ablation
        r_self_o = chow_risk(p_o, y_hat_o)
        true_err = float((y_hat_o != yo).mean())

        ev_g = evaluate_policy(pol.escalate(sig_o, p_o, Xo), p_o, yo, cost, reviewer_acc=1.0)
        m_n = pol_ng.escalate(sig_o, p_o)
        ev_n = evaluate_policy(m_n, p_o, yo, cost, reviewer_acc=1.0)
        ev_c = evaluate_policy(chow_cost_mask(p_o, y_hat_o, cost), p_o, yo, cost, reviewer_acc=1.0)
        m_b30 = pol.escalate_budget(sig_o, p_o, 0.30, Xo)
        ev_b30 = evaluate_policy(m_b30, p_o, yo, cost, reviewer_acc=1.0)

        def eval_modern_l2d(models):
            evs = [evaluate_policy(m.escalate(Z_o, p_o), p_o, yo, cost,
                                   reviewer_acc=1.0) for m in models]
            result = {}
            for key in ["escalation_rate", "error_coverage", "deployment_cost"]:
                vals = np.asarray([ev[key] for ev in evs], dtype=float)
                digits = 3 if key == "deployment_cost" else 4
                result[key] = round(float(vals.mean()), digits)
                result[f"{key}_sd"] = round(float(vals.std(ddof=0)), digits)
            return result

        ev_ms = eval_modern_l2d(modern_l2d_models["ms"])
        ev_ova = eval_modern_l2d(modern_l2d_models["ova"])

        row = {
            "site": site, "site_no": site_meta["site_no"],
            "usgs_url": site_meta["usgs_url"],
            "n": int(len(yo)), "prevalence": round(float(yo.mean()), 4),
            "accuracy_raw": round(float(accuracy_score(yo, y_hat_o)), 4),
            "true_error_rate": round(true_err, 4),
            "out_of_support_rate": round(float(gate.out_of_support(Xo).mean()), 4),
            "mean_risk_selfconf": round(float(r_self_o.mean()), 4),
            "mean_risk_ungated": round(float(r_nogate.mean()), 4),
            "mean_risk_gated": round(float(r_gate.mean()), 4),
            "esc_gated": ev_g["escalation_rate"], "cov_gated": ev_g["error_coverage"],
            "acc_gated": ev_g["accuracy_final"], "cost_gated": ev_g["deployment_cost"],
            "esc_ungated": ev_n["escalation_rate"], "cov_ungated": ev_n["error_coverage"],
            "cost_ungated": ev_n["deployment_cost"],
            "esc_chow": ev_c["escalation_rate"], "cov_chow": ev_c["error_coverage"],
            "cost_chow": ev_c["deployment_cost"],
            "cost_auto": ev_g["deployment_cost_auto"],
            "budget30_cov": ev_b30["error_coverage"], "budget30_acc": ev_b30["accuracy_final"],
            "budget30_cost": ev_b30["deployment_cost"],
            "l2d_ms": ev_ms,
            "l2d_ova": ev_ova,
        }
        ood_rows.append(row)
        ood_eval_cache.append({"site": site, "site_no": site_meta["site_no"],
                               "X": Xo, "p": p_o, "y": yo, "Z": Z_o,
                               "risk_gated": r_gate, "risk_chow": r_self_o,
                               "ungated_mask": m_n})
        print(f"\n  {site}  (n={row['n']}, prevalence {row['prevalence']:.1%})")
        print(f"    raw accuracy {row['accuracy_raw']:.3f}  |  TRUE error rate "
              f"{row['true_error_rate']:.3f}  |  out-of-support {row['out_of_support_rate']:.1%}")
        print(f"    estimated risk:  self-conf {row['mean_risk_selfconf']:.3f}   "
              f"ungated {row['mean_risk_ungated']:.3f}   gated {row['mean_risk_gated']:.3f}")
        print(f"    escalation:      Chow {row['esc_chow']:6.1%}   ungated {row['esc_ungated']:6.1%}"
              f"   gated {row['esc_gated']:6.1%}")
        print(f"    error coverage:  Chow {row['cov_chow']:6.1%}   ungated {row['cov_ungated']:6.1%}"
              f"   gated {row['cov_gated']:6.1%}")
        print(f"    deployment cost: auto {row['cost_auto']:7.3f}  Chow {row['cost_chow']:7.3f}  "
              f"ungated {row['cost_ungated']:7.3f}  gated {row['cost_gated']:7.3f}")
        print(f"    modern L2D:      MS esc {ev_ms['escalation_rate']:6.1%} / cov "
              f"{ev_ms['error_coverage']:6.1%} / cost {ev_ms['deployment_cost']:.3f}; "
              f"OvA esc {ev_ova['escalation_rate']:6.1%} / cov "
              f"{ev_ova['error_coverage']:6.1%} / cost {ev_ova['deployment_cost']:.3f}")
        print(f"    30% budget:      cov {row['budget30_cov']:.1%}  acc {row['budget30_acc']:.3f}")
    out["E7_cross_watershed"] = {
        "protocol": "Columbia-trained frozen predictor and governance models; zero-shot evaluation on ten USGS stations; official station names read from USGS metadata",
        "in_distribution_reference": {
            "mean_risk_gated": round(float(r_dl_te.mean()), 4),
            "true_error_rate": round(float(err_te.mean()), 4),
            "escalation_rate": r_dl["escalation_rate"],
            "out_of_support_rate": round(float(oos_te.mean()), 4),
        },
        "sites": ood_rows,
    }
    method_cost_keys = {
        "automatic": "cost_auto", "Chow": "cost_chow",
        "EcoTrust_ungated": "cost_ungated", "EcoTrust_gated": "cost_gated",
    }
    win_counts = {name: 0 for name in [*method_cost_keys, "L2D_MS", "L2D_OvA"]}
    for row in ood_rows:
        costs = {name: row[key] for name, key in method_cost_keys.items()}
        costs.update({"L2D_MS": row["l2d_ms"]["deployment_cost"],
                      "L2D_OvA": row["l2d_ova"]["deployment_cost"]})
        best = min(costs.values())
        for name, value in costs.items():
            if abs(value - best) < 1e-12:
                win_counts[name] += 1
    out["E7_cross_watershed"]["summary"] = {
        "n_sites": len(ood_rows),
        "n_gate_oos_ge_99pct": sum(r["out_of_support_rate"] >= 0.99 for r in ood_rows),
        "n_gated_cost_le_reviewer_cost": sum(r["cost_gated"] <= cost.c_h + 1e-12
                                               for r in ood_rows),
        "n_chow_silent_esc_le_10pct_and_error_ge_20pct": sum(
            r["esc_chow"] <= 0.10 and r["true_error_rate"] >= 0.20 for r in ood_rows
        ),
        "lowest_mean_cost_win_counts_including_ties": win_counts,
    }
    out["E10_modern_l2d"] = {
        "protocol": "published surrogate losses; frozen backend class logits; 90% synthetic expert demonstrations; MS alpha and all thresholds cost-tuned on calibration",
        "seeds": l2d_seeds,
        "in_distribution": [r_l2d_ms, r_l2d_ova],
        "cross_watershed": [{"site": r["site"], "l2d_ms": r["l2d_ms"],
                              "l2d_ova": r["l2d_ova"]} for r in ood_rows],
    }

    # ── E11: support-gate sensitivity ───────────────────────────────────── #
    print("\n" + "-" * 78)
    print("E11 Support-gate sensitivity to epsilon and distance estimator")
    print("-" * 78)
    gate_sensitivity = []
    for metric in ["mahalanobis_empirical", "mahalanobis_ledoit_wolf", "euclidean"]:
        for eps_gate in [0.01, 0.025, 0.05, 0.10, 0.20]:
            gate_alt = SupportGate(eps=eps_gate, metric=metric).fit(X_tr)
            oos_id = gate_alt.out_of_support(X_te)
            ev_id = evaluate_policy(m_ng | oos_id, p_te, y_te, cost,
                                    reviewer_acc=1.0)
            site_rows = []
            for cached in ood_eval_cache:
                oos_site = gate_alt.out_of_support(cached["X"])
                ev_site = evaluate_policy(
                    cached["ungated_mask"] | oos_site,
                    cached["p"], cached["y"], cost, reviewer_acc=1.0,
                )
                site_rows.append({
                    "site": cached["site"], "site_no": cached["site_no"],
                    "out_of_support_rate": round(float(oos_site.mean()), 4),
                    "escalation_rate": ev_site["escalation_rate"],
                    "error_coverage": ev_site["error_coverage"],
                    "deployment_cost": ev_site["deployment_cost"],
                })
            config = {
                "metric": metric, "eps": eps_gate,
                "threshold": round(float(gate_alt.d_thresh_), 5),
                "train_flagged": round(float(gate_alt.out_of_support(X_tr).mean()), 4),
                "calibration_flagged": round(float(gate_alt.out_of_support(X_cl).mean()), 4),
                "test_flagged": round(float(oos_id.mean()), 4),
                "test_escalation": ev_id["escalation_rate"],
                "test_error_coverage": ev_id["error_coverage"],
                "test_cost": ev_id["deployment_cost"],
                "n_transfer_sites_flagged_ge_99pct": sum(
                    r["out_of_support_rate"] >= 0.99 for r in site_rows
                ),
                "transfer_sites": site_rows,
            }
            gate_sensitivity.append(config)
            print(f"  {metric:27s} eps={eps_gate:5.3f}: ID flagged "
                  f"{config['test_flagged']:6.1%}, cost {config['test_cost']:.3f}; "
                  f"transfer >=99% OOS {config['n_transfer_sites_flagged_ge_99pct']}/"
                  f"{len(site_rows)}")
    out["E11_gate_sensitivity"] = {
        "protocol": "gate fit on Columbia training covariates only; same frozen ungated policy; OOS cases are added to the escalation set",
        "configurations": gate_sensitivity,
    }

    # ── E12: shifted deployment with finite review and noisy reviewers ───── #
    # The unconstrained gate sends every transfer case to review. This grid asks
    # what remains when deployment instead imposes a hard workload budget and
    # the reviewer is imperfect. Every learned score is fixed before seeing
    # target labels; labels are used only to report realised cost and accuracy.
    print("\n" + "-" * 78)
    print("E12 Shifted deployment: reviewer accuracy x review budget")
    print("-" * 78)

    def top_k_mask(score, budget):
        score = np.asarray(score, dtype=float)
        k = int(round(float(budget) * len(score)))
        mask = np.zeros(len(score), dtype=bool)
        if k > 0:
            mask[np.argsort(score, kind="stable")[-k:]] = True
        return mask

    shifted_rows = []
    for a in [0.80, 0.90, 0.95]:
        cost_a = DeploymentCost(**MAIN_COST, reviewer_acc=a)
        for budget in [0.10, 0.30, 0.50]:
            site_method_rows = {name: [] for name in
                                ["Automatic", "EcoTrust_gate", "Chow",
                                 "L2D_MS", "L2D_OvA"]}
            for site_idx, cached in enumerate(ood_eval_cache):
                p_o, y_o = cached["p"], cached["y"]
                y_hat_o = (p_o > 0.5).astype(int)
                eco_score = expected_saving(
                    cached["risk_gated"], p_o, y_hat_o, cost_a
                )
                chow_score = expected_saving(
                    cached["risk_chow"], p_o, y_hat_o, cost_a
                )
                ms_score = np.mean([
                    model.defer_score(cached["Z"], p_o)
                    for model in modern_l2d_models["ms"]
                ], axis=0)
                ova_score = np.mean([
                    model.defer_score(cached["Z"], p_o)
                    for model in modern_l2d_models["ova"]
                ], axis=0)
                masks = {
                    "Automatic": np.zeros(len(y_o), dtype=bool),
                    "EcoTrust_gate": top_k_mask(eco_score, budget),
                    "Chow": top_k_mask(chow_score, budget),
                    "L2D_MS": top_k_mask(ms_score, budget),
                    "L2D_OvA": top_k_mask(ova_score, budget),
                }
                for method_idx, (method, mask) in enumerate(masks.items()):
                    ev = evaluate_policy(
                        mask, p_o, y_o, cost_a, reviewer_acc=a,
                        seed=SEED + 100 * site_idx + method_idx,
                    )
                    site_method_rows[method].append({
                        "site": cached["site"],
                        "deployment_cost": ev["deployment_cost"],
                        "accuracy": ev["accuracy_final"],
                        "escalation_rate": ev["escalation_rate"],
                    })
            for method, site_rows in site_method_rows.items():
                costs = np.asarray([row["deployment_cost"] for row in site_rows])
                accuracies = np.asarray([row["accuracy"] for row in site_rows])
                shifted_rows.append({
                    "reviewer_acc": a, "budget": budget, "method": method,
                    "mean_site_cost": round(float(costs.mean()), 3),
                    "site_cost_sd": round(float(costs.std(ddof=0)), 3),
                    "mean_site_accuracy": round(float(accuracies.mean()), 4),
                    "sites": site_rows,
                })
            summary = {row["method"]: row["mean_site_cost"]
                       for row in shifted_rows
                       if row["reviewer_acc"] == a and row["budget"] == budget}
            print(f"  a={a:.0%}, budget={budget:.0%}: "
                  + ", ".join(f"{name} {value:.3f}"
                              for name, value in summary.items()))
    out["E12_shift_reviewer_budget"] = {
        "protocol": "ten zero-shot USGS stations; exact review budgets; reviewer errors are class-independent Monte Carlo flips; EcoTrust and Chow rank by reviewer-aware expected saving; L2D ranks by mean defer score over three seeds",
        "aggregation": "unweighted mean and standard deviation across stations",
        "rows": shifted_rows,
    }

    # ── E9: when does a LEARNED risk model beat the model's own confidence? ── #
    # Hypothesis: self-confidence is a sufficient error signal exactly when the
    # backend is well calibrated. As calibration degrades, the learned
    # multi-signal estimator should pull ahead. This is the mechanism that
    # decides whether the meta-learner is worth its complexity.
    print("\n" + "-" * 78)
    print("E9  Backend calibration vs. the value of a learned risk model")
    print("-" * 78)
    # B1 (backend bank): a spread of off-the-shelf, un-tuned classifiers whose
    # native calibration varies widely, so the ECE->value relationship rests on
    # real models rather than a single deliberately-broken backend.
    def band(est, label, seed):
        """Train one backend seed, then compare Chow vs. EcoTrust-DL."""
        pl = Pipeline([("i", SimpleImputer(strategy="median")),
                       ("s", StandardScaler()), ("m", est)])
        pl.fit(X_tr, y_tr)
        pc, pt = pl.predict_proba(X_cl)[:, 1], pl.predict_proba(X_te)[:, 1]
        yh_c, yh_t = (pc > 0.5).astype(int), (pt > 0.5).astype(int)
        ece = float(expected_calibration_error(y_te, pt)["ece"])

        # A single model has no ensemble spread, so the variance signal is 0.
        eps = 1e-8

        def _sig(pr, cf, Xs, cp_obj=cp, q_obj=q_hat, gate_obj=gate):
            H = -(pr * np.log(pr + eps) + (1 - pr) * np.log(1 - pr + eps))
            return make_signals({"mean_prob": pr, "entropy": H, "std_prob": np.zeros(len(pr))},
                                cf, cp_obj.predict(Xs), q_obj, gate_obj.novelty_feature(Xs))

        st_ = _sig(pt, conf_te, X_te)

        # Same cross-fitted meta-training protocol as the main experiment, so
        # this row is comparable with E1 rather than handicapped by a smaller
        # calibration block.
        oof_sig_chunks, oof_pr_chunks, oof_y_chunks = [], [], []
        train_years = train["date"].dt.year.to_numpy()
        for tr_i, va_i in temporal_folds(y_tr, train_years, n_folds=5):
            pf = Pipeline([("i", SimpleImputer(strategy="median")),
                           ("s", StandardScaler()), ("m", clone(est))])
            pf.fit(X_tr[tr_i], y_tr[tr_i])
            oof_pr_f = pf.predict_proba(X_tr[va_i])[:, 1]
            cp_f, q_f, gate_f = fit_auxiliary_signals(X_tr[tr_i], y_tr[tr_i])
            p20_f = monthly_p20(train.iloc[tr_i])
            conf_f = conflict_of(train.iloc[va_i], p20_f)
            oof_sig_chunks.append(_sig(oof_pr_f, conf_f, X_tr[va_i], cp_f, q_f, gate_f))
            oof_pr_chunks.append(oof_pr_f)
            oof_y_chunks.append(y_tr[va_i])
        sig_oof_b = {k: np.concatenate([s[k] for s in oof_sig_chunks])
                     for k in oof_sig_chunks[0]}
        oof_pr = np.concatenate(oof_pr_chunks)
        oof_y = np.concatenate(oof_y_chunks)
        sm_ = concat_signals(sig_oof_b, _sig(pc, conf_cl, X_cl))
        pm_ = np.concatenate([oof_pr, pc])
        ym_ = np.concatenate([oof_y, y_cl])

        p_dl = CostAwareDecisionPolicy(cost=cost, meta_model="logistic",
                                       support_gate=gate, random_state=seed).fit(sm_, pm_, ym_)
        dl_mask = p_dl.escalate(st_, pt, X_te)
        chow_mask = chow_cost_mask(pt, yh_t, cost)
        ev_dl = evaluate_policy(dl_mask, pt, y_te, cost, reviewer_acc=1.0)
        ev_ch = evaluate_policy(chow_mask, pt, y_te, cost, reviewer_acc=1.0)

        e_t = (yh_t != y_te).astype(int)
        dl_risk = p_dl.risk.predict_risk(st_)
        chow_r = chow_risk(pt, yh_t)
        auc_dl = float(roc_auc_score(e_t, dl_risk)) if e_t.sum() else float("nan")
        auc_ch = float(roc_auc_score(e_t, chow_r)) if e_t.sum() else float("nan")
        brier_dl = float(brier_score_loss(e_t, dl_risk))
        brier_ch = float(brier_score_loss(e_t, chow_r))
        row = {
            "backend": label, "seed": seed, "ece": round(ece, 4),
            "accuracy_raw": round(float(accuracy_score(y_te, yh_t)), 4),
            "chow_cost": ev_ch["deployment_cost"], "dl_cost": ev_dl["deployment_cost"],
            "cost_advantage_dl": round(ev_ch["deployment_cost"] - ev_dl["deployment_cost"], 3),
            "chow_auc": round(auc_ch, 4), "dl_auc": round(auc_dl, 4),
            "auc_advantage_dl": round(auc_dl - auc_ch, 4),
            "chow_brier": round(brier_ch, 4), "dl_brier": round(brier_dl, 4),
            "chow_esc": ev_ch["escalation_rate"], "dl_esc": ev_dl["escalation_rate"],
            "chow_cov": ev_ch["error_coverage"], "dl_cov": ev_dl["error_coverage"],
        }

        auto_cost = np.where(
            (yh_t == 0) & (y_te == 1), cost.c_fn,
            np.where((yh_t == 1) & (y_te == 0), cost.c_fp, 0.0),
        )
        raw = {
            "y": y_te,
            "p": pt,
            "chow_daily_cost": np.where(chow_mask, cost.c_h, auto_cost),
            "dl_daily_cost": np.where(dl_mask, cost.c_h, auto_cost),
        }
        return row, raw

    def aggregate_runs(label, run_rows):
        keys = [
            "ece", "accuracy_raw", "chow_cost", "dl_cost", "cost_advantage_dl",
            "chow_auc", "dl_auc", "auc_advantage_dl", "chow_brier", "dl_brier",
            "chow_esc", "dl_esc", "chow_cov", "dl_cov",
        ]
        out_row = {"backend": label, "n_seeds": len(run_rows),
                   "seeds": [int(r["seed"]) for r in run_rows]}
        for key in keys:
            vals = np.asarray([r[key] for r in run_rows], dtype=float)
            digits = 3 if key in {"chow_cost", "dl_cost", "cost_advantage_dl"} else 4
            out_row[key] = round(float(vals.mean()), digits)
            out_row[f"{key}_sd"] = round(float(vals.std(ddof=0)), digits)
        return out_row

    STRAWMAN_LABEL = "XGBoost (over-confident)"
    b1_seeds = [40, 41, 42]
    specs = [
        ("Random Forest (well calib.)",
         lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
        ("Gradient Boosting",
         lambda s: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.05, random_state=s)),
        ("XGBoost",
         lambda s: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 random_state=s, eval_metric="logloss", verbosity=0)),
        # Off-the-shelf models spanning naturally well-calibrated to naturally
        # over-confident settings. Hyperparameters are fixed before evaluation.
        ("Logistic Regression",
         lambda s: LogisticRegression(max_iter=1000, random_state=s)),
        ("Gaussian Naive Bayes", lambda s: GaussianNB()),
        ("k-NN (k=15)", lambda s: KNeighborsClassifier(n_neighbors=15)),
        ("SVM (RBF, Platt)", lambda s: SVC(probability=True, random_state=s)),
        ("Decision Tree (depth 3)",
         lambda s: DecisionTreeClassifier(max_depth=3, random_state=s)),
        ("Decision Tree (deep)", lambda s: DecisionTreeClassifier(random_state=s)),
        ("MLP (64, default)",
         lambda s: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=s)),
        ("AdaBoost", lambda s: AdaBoostClassifier(random_state=s)),
        ("Extra Trees",
         lambda s: ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
        # One deliberately over-confident control point, kept separate in all
        # robustness statistics.
        (STRAWMAN_LABEL,
         lambda s: XGBClassifier(n_estimators=400, max_depth=10, learning_rate=0.3,
                                 reg_lambda=0.0, random_state=s,
                                 eval_metric="logloss", verbosity=0)),
    ]

    bands, per_seed_rows, raw_by_backend = [], [], {}
    for label, factory in specs:
        seed_rows, seed_raw = [], []
        for seed in b1_seeds:
            row, raw = band(factory(seed), label, seed)
            seed_rows.append(row)
            seed_raw.append(raw)
            per_seed_rows.append(row)
        agg = aggregate_runs(label, seed_rows)
        bands.append(agg)
        raw_by_backend[label] = seed_raw
        print(f"  {label:34s} ECE={agg['ece']:.3f}+-{agg['ece_sd']:.3f}  "
              f"cost: Chow {agg['chow_cost']:6.3f} vs DL {agg['dl_cost']:6.3f} "
              f"({agg['cost_advantage_dl']:+.3f})  AUC: {agg['chow_auc']:.3f} vs "
              f"{agg['dl_auc']:.3f}  Brier: {agg['chow_brier']:.3f} vs {agg['dl_brier']:.3f}")

    # The ensemble backend, already computed above.
    ece_ens = float(expected_calibration_error(y_te, p_te)["ece"])
    ev_ch_ens = evaluate_policy(chow_cost_mask(p_te, y_hat_te, cost), p_te, y_te, cost,
                                reviewer_acc=1.0)
    ens_label = "EcoTrust ensemble (RF+XGB+GB)"
    ens_row = {
        "backend": ens_label, "n_seeds": 1, "seeds": [SEED], "ece": round(ece_ens, 4),
        "accuracy_raw": round(acc_raw, 4),
        "chow_cost": ev_ch_ens["deployment_cost"], "dl_cost": r_dl["deployment_cost"],
        "cost_advantage_dl": round(ev_ch_ens["deployment_cost"] - r_dl["deployment_cost"], 3),
        "chow_auc": quality["selfconf_auc"], "dl_auc": quality["meta_auc"],
        "auc_advantage_dl": round(quality["meta_auc"] - quality["selfconf_auc"], 4),
        "chow_brier": quality["selfconf_brier"], "dl_brier": quality["meta_brier"],
        "chow_esc": ev_ch_ens["escalation_rate"], "dl_esc": r_dl["escalation_rate"],
        "chow_cov": ev_ch_ens["error_coverage"], "dl_cov": r_dl["error_coverage"],
    }
    for key in [k for k in list(ens_row) if k not in {"backend", "n_seeds", "seeds"}]:
        ens_row[f"{key}_sd"] = 0.0
    bands.append(ens_row)
    chow_ens_mask = chow_cost_mask(p_te, y_hat_te, cost)
    ens_auto_cost = np.where(
        (y_hat_te == 0) & (y_te == 1), cost.c_fn,
        np.where((y_hat_te == 1) & (y_te == 0), cost.c_fp, 0.0),
    )
    raw_by_backend[ens_label] = [{
        "y": y_te, "p": p_te,
        "chow_daily_cost": np.where(chow_ens_mask, cost.c_h, ens_auto_cost),
        "dl_daily_cost": np.where(m_dl, cost.c_h, ens_auto_cost),
    }]
    print(f"  {ens_label:34s} ECE={ece_ens:.3f}  cost: Chow "
          f"{ev_ch_ens['deployment_cost']:6.3f} vs DL {r_dl['deployment_cost']:6.3f} "
          f"({bands[-1]['cost_advantage_dl']:+.3f})  AUC: {quality['selfconf_auc']:.3f} vs "
          f"{quality['meta_auc']:.3f}  Brier: {quality['selfconf_brier']:.3f} vs "
          f"{quality['meta_brier']:.3f}")

    # ── R004: does the ECE -> value trend survive WITHOUT the strawman? ──
    eces = np.array([b["ece"] for b in bands])
    advs = np.array([b["cost_advantage_dl"] for b in bands])
    corr = float(np.corrcoef(eces, advs)[0, 1]) if len(bands) > 2 else float("nan")
    rho = float(spearmanr(eces, advs).correlation)

    # Anti-strawman check: trend and win-count over REAL models only.
    real = [b for b in bands if b["backend"] != STRAWMAN_LABEL]
    e_r = np.array([b["ece"] for b in real])
    a_r = np.array([b["cost_advantage_dl"] for b in real])
    corr_real = float(np.corrcoef(e_r, a_r)[0, 1]) if len(real) > 2 else float("nan")
    rho_real = float(spearmanr(e_r, a_r).correlation)
    wins_real = [b["backend"] for b in real if b["cost_advantage_dl"] > 0]

    # Year-block bootstrap. Resampling whole migration seasons preserves the
    # strong within-season temporal dependence and re-computes ECE, cost gain,
    # and their cross-backend association on the same resampled days.
    years = test["date"].dt.year.to_numpy()
    unique_years = np.unique(years)
    rng = np.random.default_rng(SEED)
    boot_all_p, boot_all_s, boot_real_p, boot_real_s = [], [], [], []
    for _ in range(2000):
        sampled_years = rng.choice(unique_years, size=len(unique_years), replace=True)
        idx = np.concatenate([np.flatnonzero(years == yr) for yr in sampled_years])
        ece_b, gain_b = [], []
        for b in bands:
            raw_runs = raw_by_backend[b["backend"]]
            ece_seed = [expected_calibration_error(r["y"][idx], r["p"][idx])["ece"]
                        for r in raw_runs]
            gain_seed = [float(r["chow_daily_cost"][idx].mean()
                               - r["dl_daily_cost"][idx].mean()) for r in raw_runs]
            ece_b.append(float(np.mean(ece_seed)))
            gain_b.append(float(np.mean(gain_seed)))
        ece_b, gain_b = np.asarray(ece_b), np.asarray(gain_b)
        if np.std(ece_b) > 1e-12 and np.std(gain_b) > 1e-12:
            boot_all_p.append(float(np.corrcoef(ece_b, gain_b)[0, 1]))
            boot_all_s.append(float(spearmanr(ece_b, gain_b).correlation))
        keep = np.asarray([b["backend"] != STRAWMAN_LABEL for b in bands])
        if np.std(ece_b[keep]) > 1e-12 and np.std(gain_b[keep]) > 1e-12:
            boot_real_p.append(float(np.corrcoef(ece_b[keep], gain_b[keep])[0, 1]))
            boot_real_s.append(float(spearmanr(ece_b[keep], gain_b[keep]).correlation))

    def ci95(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return [round(float(np.percentile(finite, 2.5)), 3),
                round(float(np.percentile(finite, 97.5)), 3)]

    # Leave-one-backend-out sensitivity over real models exposes any result
    # driven by a single naturally extreme backend.
    loo = []
    for dropped in real:
        kept = [b for b in real if b["backend"] != dropped["backend"]]
        x = np.asarray([b["ece"] for b in kept])
        y = np.asarray([b["cost_advantage_dl"] for b in kept])
        loo.append({
            "dropped": dropped["backend"],
            "pearson": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "spearman": round(float(spearmanr(x, y).correlation), 3),
        })

    fit_all = np.polyfit(eces, advs, 1)
    fit_real = np.polyfit(e_r, a_r, 1)

    print(f"\n  corr(ECE, DL cost advantage): Pearson {corr:+.3f} "
          f"(year-block 95% CI {ci95(boot_all_p)})  Spearman {rho:+.3f} "
          f"(CI {ci95(boot_all_s)})")
    print(f"  excluding the strawman: Pearson {corr_real:+.3f} "
          f"(CI {ci95(boot_real_p)}), Spearman {rho_real:+.3f} "
          f"(CI {ci95(boot_real_s)}) over {len(real)} real models")
    print(f"  real backends where the learned risk model WINS (gain>0): "
          f"{len(wins_real)}/{len(real)} -> {', '.join(wins_real) if wins_real else 'none'}")

    out["E9_calibration_vs_value"] = {
        "rows": bands,
        "per_seed_rows": per_seed_rows,
        "seed_protocol": b1_seeds,
        "bootstrap_protocol": "2000 resamples of complete test years; metrics averaged over seeds",
        "corr_ece_vs_advantage": round(corr, 3),
        "spearman_ece_vs_advantage": round(rho, 3),
        "corr_ci95": ci95(boot_all_p),
        "spearman_ci95": ci95(boot_all_s),
        "corr_excluding_strawman": round(corr_real, 3),
        "spearman_excluding_strawman": round(rho_real, 3),
        "corr_excluding_strawman_ci95": ci95(boot_real_p),
        "spearman_excluding_strawman_ci95": ci95(boot_real_s),
        "n_real_backends": len(real),
        "real_wins_gain_gt0": wins_real,
        "leave_one_real_backend_out": loo,
        "linear_fit_all": {"slope": round(float(fit_all[0]), 4),
                           "intercept": round(float(fit_all[1]), 4)},
        "linear_fit_excluding_strawman": {"slope": round(float(fit_real[0]), 4),
                                          "intercept": round(float(fit_real[1]), 4)},
    }
    # Dedicated B1 artifact, separate from the canonical suite output.
    with open(RESULTS_DIR / "backend_bank_results.json", "w") as f:
        json.dump(out["E9_calibration_vs_value"], f, indent=2)
    # Dedicated B2 artifact keeps the modern L2D protocol and its shift
    # results reviewable without depending on the larger canonical JSON.
    with open(RESULTS_DIR / "modern_l2d_results.json", "w") as f:
        json.dump(out["E10_modern_l2d"], f, indent=2)
    with open(RESULTS_DIR / "multibasin_results.json", "w") as f:
        json.dump(out["E7_cross_watershed"], f, indent=2)
    pd.json_normalize(out["E7_cross_watershed"]["sites"], sep="_").to_csv(
        RESULTS_DIR / "multibasin_results.csv", index=False
    )
    with open(RESULTS_DIR / "gate_sensitivity_results.json", "w") as f:
        json.dump(out["E11_gate_sensitivity"], f, indent=2)
    gate_flat = []
    for config in out["E11_gate_sensitivity"]["configurations"]:
        for site_row in config["transfer_sites"]:
            gate_flat.append({"metric": config["metric"], "eps": config["eps"],
                              "id_test_flagged": config["test_flagged"],
                              "id_test_cost": config["test_cost"], **site_row})
    pd.DataFrame(gate_flat).to_csv(RESULTS_DIR / "gate_sensitivity_results.csv",
                                   index=False)
    with open(RESULTS_DIR / "reviewer_sensitivity_results.json", "w") as f:
        json.dump(out["E5_reviewer_accuracy"], f, indent=2)
    pd.DataFrame(out["E5_reviewer_accuracy"]["baseline_rows"]).to_csv(
        RESULTS_DIR / "reviewer_sensitivity_results.csv", index=False
    )
    with open(RESULTS_DIR / "shift_reviewer_budget_results.json", "w") as f:
        json.dump(out["E12_shift_reviewer_budget"], f, indent=2)
    pd.DataFrame([{k: v for k, v in row.items() if k != "sites"}
                  for row in out["E12_shift_reviewer_budget"]["rows"]]).to_csv(
        RESULTS_DIR / "shift_reviewer_budget_results.csv", index=False
    )
    pd.DataFrame([{
        "point_id": i + 1, "ece": b["ece"],
        "cost_advantage_dl": b["cost_advantage_dl"],
        "ece_sd": b["ece_sd"], "cost_advantage_dl_sd": b["cost_advantage_dl_sd"],
        "is_strawman": int(b["backend"] == STRAWMAN_LABEL),
    } for i, b in enumerate(bands)]).to_csv(RESULTS_DIR / "backend_bank_plot_data.csv", index=False)
    print("  Saved -> results/backend_bank_results.json")
    print("  Saved -> results/backend_bank_plot_data.csv")
    print("  Saved -> results/modern_l2d_results.json")
    print("  Saved -> results/multibasin_results.json/.csv")
    print("  Saved -> results/gate_sensitivity_results.json/.csv")
    print("  Saved -> results/reviewer_sensitivity_results.json/.csv")
    print("  Saved -> results/shift_reviewer_budget_results.json/.csv")

    path = RESULTS_DIR / "decision_learning_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → results/decision_learning_results.json")


if __name__ == "__main__":
    main()
