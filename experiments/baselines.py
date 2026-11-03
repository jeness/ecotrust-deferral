"""
Baseline comparison for EcoTrust.

B0: Majority class (always predict 0 / no risk)
B1: Single RF — no ensemble, no uncertainty, no governance
B2: Single XGBoost — no ensemble, no uncertainty, no governance
B3: EcoTrust ensemble — no governance (no human escalation)
B4: EcoTrust ensemble + Human Governance  (full framework = S1 results)

All methods use the same features and train/test split.
Ground truth: water_temp_c > 18°C AND month in Aug-Nov (from Columbia River USGS)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty.ensemble import EnsembleClassifier
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Same feature set as the migration agent (fused from upstream agents)
FEATURE_COLS = [
    "weather_risk", "weather_ensemble_std",
    "flood_risk",   "flood_ensemble_std",
    "month",
]


def load_migration_features(split_year=2015):
    """
    Load pre-computed agent outputs (run run_scenarios.py first).
    Falls back to building from raw merged data if not available.
    """
    merged_path = DATA_DIR / "merged" / "columbia_merged.csv"
    merged = pd.read_csv(merged_path)
    merged["date"] = pd.to_datetime(merged["date"])

    # Re-derive the same features the migration agent uses
    # (weather_risk proxy: TMAX > 28°C probability from rolling stats)
    merged["weather_risk"] = (merged["TMAX"] > 28).astype(float)
    merged["weather_ensemble_std"] = 0.01   # constant proxy for standalone baseline

    # low-flow risk proxy: discharge < 20th pctile for that month
    monthly_p20 = merged.groupby("month")["discharge_cfs"].quantile(0.2)
    merged["monthly_p20"] = merged["month"].map(monthly_p20.to_dict())
    merged["flood_risk"]  = (merged["discharge_cfs"] < merged["monthly_p20"]).astype(float)
    merged["flood_ensemble_std"] = 0.01

    train = merged[merged["date"].dt.year < split_year].copy()
    test  = merged[merged["date"].dt.year >= split_year].copy()
    return train, test


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def evaluate(name, y_true, y_pred, y_prob=None, escalation_rate=None,
             error_coverage=None, accuracy_after=None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    return {
        "method":           name,
        "accuracy":         round(acc, 4),
        "f1_score":         round(f1, 4),
        "roc_auc":          round(auc, 4) if auc else "—",
        "escalation_rate":  round(escalation_rate, 4) if escalation_rate is not None else "—",
        "error_coverage":   round(error_coverage, 4) if error_coverage is not None else "—",
        "accuracy_w_human": round(accuracy_after, 4) if accuracy_after is not None else "—",
    }


def run_with_governance(probs, y_true, threshold=0.60):
    """Simulate governance: escalate if prob > threshold, human oracle corrects."""
    y_pred    = (probs > 0.5).astype(int)
    escalated = probs > threshold
    errors    = (y_pred != y_true)

    errors_escalated = (escalated & errors).sum()
    error_coverage   = errors_escalated / errors.sum() if errors.sum() > 0 else 0.0

    # Human oracle: fix all escalated errors
    y_final = y_pred.copy()
    y_final[escalated] = y_true[escalated]   # perfect human override
    acc_after = accuracy_score(y_true, y_final)

    return escalated.mean(), error_coverage, acc_after


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data ...")
    train, test = load_migration_features(split_year=2015)

    X_train = train[FEATURE_COLS].values
    y_train = train["high_migration_risk"].values.astype(int)
    X_test  = test[FEATURE_COLS].values
    y_test  = test["high_migration_risk"].values.astype(int)

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Prevalence: {y_test.mean()*100:.1f}%\n")

    results = []

    # ------------------------------------------------------------------ #
    # B0: Majority class baseline
    # ------------------------------------------------------------------ #
    y_pred_maj = np.zeros(len(y_test), dtype=int)
    results.append(evaluate("B0 Majority Class", y_test, y_pred_maj, y_prob=None))
    print("B0 done")

    # ------------------------------------------------------------------ #
    # B1: Single Random Forest (no ensemble, no governance)
    # ------------------------------------------------------------------ #
    rf = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf > 0.5).astype(int)
    esc_rate, err_cov, acc_after = run_with_governance(y_prob_rf, y_test)
    results.append(evaluate("B1 Single RF", y_test, y_pred_rf, y_prob_rf,
                             esc_rate, err_cov, acc_after))
    print("B1 done")

    # ------------------------------------------------------------------ #
    # B2: Single XGBoost (no ensemble, no governance)
    # ------------------------------------------------------------------ #
    xgb = make_pipeline(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       random_state=42, eval_metric="logloss", verbosity=0))
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_prob_xgb > 0.5).astype(int)
    esc_rate, err_cov, acc_after = run_with_governance(y_prob_xgb, y_test)
    results.append(evaluate("B2 Single XGBoost", y_test, y_pred_xgb, y_prob_xgb,
                             esc_rate, err_cov, acc_after))
    print("B2 done")

    # ------------------------------------------------------------------ #
    # B3: Ensemble only (no governance)
    # ------------------------------------------------------------------ #
    ens = EnsembleClassifier()
    ens.fit(X_train, y_train)
    ens_out   = ens.predict(X_test)
    y_prob_ens = ens_out["mean_prob"]
    y_pred_ens = (y_prob_ens > 0.5).astype(int)
    esc_rate, err_cov, acc_after = run_with_governance(y_prob_ens, y_test)
    results.append(evaluate("B3 Ensemble (no governance)", y_test, y_pred_ens, y_prob_ens,
                             esc_rate, err_cov, acc_after))
    print("B3 done")

    # ------------------------------------------------------------------ #
    # B4: Full EcoTrust (ensemble + governance) — reproduces S1 from run_scenarios
    # ------------------------------------------------------------------ #
    gov = HumanGovernanceLayer(EscalationConfig())
    migration_out_df = pd.DataFrame({
        "date":           test["date"].values,
        "migration_risk": y_prob_ens,
        "ensemble_std":   ens_out["std_prob"],
        "entropy":        ens_out["entropy"],
        "conflict":       False,
        "ground_truth":   y_test.astype(float),
    })
    gov_df  = gov.evaluate(migration_out_df)
    gov_sum = gov.summary()
    y_final = gov_df["final_decision"].values.astype(int)
    results.append(evaluate(
        "B4 EcoTrust (ensemble + governance)",
        y_test, y_final, y_prob_ens,
        gov_sum["escalation_rate"],
        gov_sum["error_coverage"],
        gov_sum["accuracy_after"],
    ))
    print("B4 done\n")

    # ------------------------------------------------------------------ #
    # Print table
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(results)
    print("=" * 80)
    print("BASELINE COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Migration season subset (Aug-Nov only — where decisions matter)
    season_mask = test["month"].isin([8, 9, 10, 11]).values
    X_season = X_test[season_mask]
    y_season = y_test[season_mask]
    print(f"Migration season (Aug-Nov): {season_mask.sum():,} days, "
          f"{y_season.mean()*100:.1f}% high-risk\n")

    season_results = []
    for model, probs in [
        ("B1 Single RF",     rf.predict_proba(X_season)[:, 1]),
        ("B2 Single XGBoost", xgb.predict_proba(X_season)[:, 1]),
        ("B3 Ensemble",       ens.predict(X_season)["mean_prob"]),
    ]:
        y_pred_s = (probs > 0.5).astype(int)
        esc, ecov, aafter = run_with_governance(probs, y_season)
        season_results.append(evaluate(model, y_season, y_pred_s, probs, esc, ecov, aafter))

    df_season = pd.DataFrame(season_results)
    print("MIGRATION SEASON SUBSET (Aug-Nov)")
    print("=" * 80)
    print(df_season.to_string(index=False))

    # Save
    out_path = RESULTS_DIR / "baseline_comparison.csv"
    df.to_csv(out_path, index=False)
    df_season.to_csv(RESULTS_DIR / "baseline_comparison_season.csv", index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
