"""
Within-season evaluation — the canonical experiment for EcoTrust.
Evaluates within Aug-Nov migration season using environmental features only.
No temporal features (month, day_of_year) are used by any model.

Conflict signal computed from weather/hydrology risk proxies (TMAX>28 vs low-flow).

Run: python3 within_season_eval.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
sys.path.insert(0, "..")
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error
from uncertainty.conformal import SplitConformalClassifier
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

# Environmental features only — NO temporal features
ENV_FEATURES = [
    "TMAX", "TMIN", "PRCP", "AWND",
    "discharge_cfs", "gage_height_ft",
    "discharge_7d_mean", "discharge_anomaly",
]
CONFLICT_THRESHOLD = 0.35

def load():
    df = pd.read_csv(DATA_DIR / "merged/columbia_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    for col in ENV_FEATURES:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    # Within-season only
    return df[df["month"].isin([8,9,10,11])].copy()

def compute_monthly_p20(df):
    """Compute monthly 20th percentile discharge thresholds from a reference dataframe."""
    month = df["date"].dt.month
    return df.groupby(month)["discharge_cfs"].quantile(0.2)

def compute_conflict(df, monthly_p20_ref):
    """Compute agent conflict flag using pre-computed training-set discharge thresholds."""
    weather_risk = (df["TMAX"] > 28).astype(float).values
    month = df["date"].dt.month
    monthly_p20 = month.map(monthly_p20_ref.to_dict())
    flood_risk = (df["discharge_cfs"] < monthly_p20).astype(float).values
    return np.abs(weather_risk - flood_risk) > CONFLICT_THRESHOLD

def pipe(est):
    return Pipeline([("i",SimpleImputer(strategy="median")),
                     ("s",StandardScaler()),("m",est)])

def eval_model(name, y_true, y_pred, y_prob=None, gov_sum=None):
    r = {"method": name,
         "accuracy": round(accuracy_score(y_true, y_pred), 4),
         "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)}
    if y_prob is not None:
        r["auc"] = round(roc_auc_score(y_true, y_prob), 4)
        r["ece"] = round(expected_calibration_error(y_true, y_prob)["ece"], 4)
    if gov_sum:
        r.update({k: round(v, 4) for k, v in gov_sum.items()
                  if isinstance(v, float)})
    return r

def main():
    np.random.seed(42)
    df = load()
    train = df[df["date"].dt.year < 2015]
    test  = df[df["date"].dt.year >= 2015]
    X_tr = train[ENV_FEATURES].values; y_tr = train["high_migration_risk"].values.astype(int)
    X_te = test[ENV_FEATURES].values;  y_te = test["high_migration_risk"].values.astype(int)
    # Compute conflict signal using TRAINING-SET discharge percentiles (non-transductive)
    train_p20 = compute_monthly_p20(train)
    conflict_te = compute_conflict(test, train_p20)
    print(f"Train:{len(X_tr)} Test:{len(X_te)} Prevalence:{y_te.mean()*100:.1f}%")
    print(f"Conflict cases (test): {conflict_te.sum()} ({conflict_te.mean()*100:.1f}%)")

    results = []

    # B0: always high-risk
    results.append(eval_model("B0 Always-High-Risk", y_te, np.ones(len(y_te),dtype=int)))

    # B_month: month-only logistic regression (upper bound for calendar shortcut)
    month_te = test["date"].dt.month.values.reshape(-1, 1)
    month_tr = train["date"].dt.month.values.reshape(-1, 1)
    lr_month = LogisticRegression(random_state=42, max_iter=200)
    lr_month.fit(month_tr, y_tr)
    p_month = lr_month.predict_proba(month_te)[:, 1]
    results.append(eval_model("B_month Month-only LR", y_te, (p_month>0.5).astype(int), p_month))

    # B1: Single RF
    rf = pipe(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    rf.fit(X_tr, y_tr); p = rf.predict_proba(X_te)[:,1]
    results.append(eval_model("B1 Single RF", y_te, (p>0.5).astype(int), p))

    # B2: XGBoost
    xg = pipe(XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                              random_state=42,eval_metric="logloss",verbosity=0))
    xg.fit(X_tr, y_tr); p = xg.predict_proba(X_te)[:,1]
    results.append(eval_model("B2 XGBoost", y_te, (p>0.5).astype(int), p))

    # B3: EcoTrust Ensemble (no governance)
    ens = EnsembleClassifier(); ens.fit(X_tr, y_tr)
    eo = ens.predict(X_te); p_ens = eo["mean_prob"]
    results.append(eval_model("B3 Ensemble", y_te, (p_ens>0.5).astype(int), p_ens))

    # B4: Full EcoTrust — using real conflict signal from weather/hydrology proxies
    gov = HumanGovernanceLayer(EscalationConfig())
    mdf = pd.DataFrame({"date": test["date"].values, "migration_risk": p_ens,
                         "ensemble_std": eo["std_prob"], "entropy": eo["entropy"],
                         "conflict": conflict_te, "ground_truth": y_te.astype(float)})
    gdf = gov.evaluate(mdf); gs = gov.summary()
    yf = gdf["final_decision"].values.astype(int)
    results.append(eval_model("B4 EcoTrust", y_te, yf, p_ens, gs))

    # CP coverage: SplitConformalClassifier on 20% calibration split
    from sklearn.pipeline import Pipeline as _Pipeline
    cal_split = int(len(X_tr) * 0.80)
    X_fit, y_fit = X_tr[:cal_split], y_tr[:cal_split]
    X_cal, y_cal = X_tr[cal_split:], y_tr[cal_split:]
    from sklearn.linear_model import LogisticRegression as _LR
    cp_base = _Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("m", _LR(max_iter=500, random_state=42))])
    cp = SplitConformalClassifier(cp_base, alpha=0.10)
    cp.fit(X_fit, y_fit, X_cal, y_cal)
    cp_preds = cp.predict(X_te)
    include_1 = cp_preds["include_high_risk"]  # class 1 in set: proba >= 1 - q_hat
    q_hat = float(cp_preds["q_hat"])
    proba_cp = cp_preds["risk_score"]
    # Correct binary conformal coverage:
    #   y=1: covered if class 1 in set = include_high_risk
    #   y=0: covered if class 0 in set = proba <= q_hat
    class_0_in_set = proba_cp <= q_hat
    covered = np.where(y_te == 1, include_1, class_0_in_set)
    cp_coverage = float(covered.mean())
    cp_result = {"alpha": 0.10, "nominal_coverage": 0.90,
                 "empirical_coverage": round(cp_coverage, 4),
                 "q_hat": round(q_hat, 4),
                 "class_1_coverage": round(float(include_1[y_te == 1].mean()), 4),
                 "class_0_coverage": round(float(class_0_in_set[y_te == 0].mean()), 4)}
    print(f"\nCP empirical coverage: {cp_coverage:.4f} (90% nominal)")

    # Ablation: single-signal vs multi-signal
    y_pred_ens = (p_ens > 0.5).astype(int)
    total_errors = (y_pred_ens != y_te).sum()
    ablation = []
    for mname, esc_mask in [
        ("Risk-score only (p>0.60)",  p_ens > 0.60),
        ("Entropy only (H>0.40)",     eo["entropy"] > 0.40),
        ("EnsStd only (sigma>0.08)",  eo["std_prob"] > 0.08),
        ("Conflict only",             conflict_te),
        ("EcoTrust multi-signal",     (p_ens > 0.60) | (eo["entropy"] > 0.40) | (eo["std_prob"] > 0.08) | conflict_te),
    ]:
        ec = float((esc_mask & (y_pred_ens != y_te)).sum() / total_errors) if total_errors > 0 else 0.0
        yf_ab = y_pred_ens.copy(); yf_ab[esc_mask] = y_te[esc_mask]
        ablation.append({"method": mname, "esc_rate": round(float(esc_mask.mean()), 4),
                         "err_cov": round(ec, 4), "acc_after": round(float(accuracy_score(y_te, yf_ab)), 4)})

    print("\n=== Results ===")
    for r in results:
        print(r)
    print("\nEscalation reason breakdown:", gs["reason_breakdown"])
    print("\nCP coverage:", cp_result)

    # Budget-constrained analysis: escalate top-k% by composite uncertainty score
    unc_score = np.abs(p_ens - 0.5) * (-1) + eo["entropy"] + eo["std_prob"] + conflict_te.astype(float) * 0.5
    y_pred_budget = (p_ens > 0.5).astype(int)
    total_err = (y_pred_budget != y_te).sum()
    budget_results = []
    for budget_pct in [10, 20, 30, 40, 50, 70]:
        k = int(len(y_te) * budget_pct / 100)
        top_idx = np.argsort(unc_score)[-k:]
        esc = np.zeros(len(y_te), dtype=bool); esc[top_idx] = True
        ec = float((esc & (y_pred_budget != y_te)).sum() / total_err) if total_err > 0 else 0.0
        yf_b = y_pred_budget.copy(); yf_b[esc] = y_te[esc]
        budget_results.append({"budget_pct": budget_pct, "esc_rate": round(budget_pct/100, 4),
                                "err_cov": round(ec, 4), "acc_after": round(float(accuracy_score(y_te, yf_b)), 4)})
    # Default operating point: actual governance escalation rate (not a hardcoded formula)
    default_pct = int(round(float(gs["escalation_rate"]) * 100))
    budget_results.append({"budget_pct": default_pct, "esc_rate": round(float(gs["escalation_rate"]), 4),
                           "err_cov": round(float(gs["error_coverage"]), 4),
                           "acc_after": round(float(gs["accuracy_after"]), 4)})

    with open(RESULTS_DIR / "within_season_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "within_season_gov_detail.json", "w") as f:
        json.dump(gs, f, indent=2)
    with open(RESULTS_DIR / "cp_coverage.json", "w") as f:
        json.dump(cp_result, f, indent=2)
    with open(RESULTS_DIR / "abstention_comparison.json", "w") as f:
        json.dump(ablation, f, indent=2)
    with open(RESULTS_DIR / "budget_analysis.json", "w") as f:
        json.dump(budget_results, f, indent=2)
    print("\nSaved.")

if __name__ == "__main__":
    main()
