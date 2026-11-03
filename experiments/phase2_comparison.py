"""
Phase 2: Model-Agnostic EcoTrust Comparison

Proves that EcoTrust's Decision Governance Layer is model-agnostic by
running the SAME governance policy on THREE prediction backends:

    Backend A: Random Forest (classical ML)
    Backend B: XGBoost      (gradient boosting)
    Backend C: Chronos      (time-series foundation model, zero-shot)

For each backend, evaluates:
    - WITHOUT EcoTrust: raw model predictions, no governance
    - WITH EcoTrust:    same model + escalation policy

Key claim: EcoTrust consistently improves Error Coverage and Error
Reduction regardless of which prediction backend is used.

Usage:
    # Phase 1 backends only (CPU, fast):
    python phase2_comparison.py --backends rf xgb

    # Full Phase 2 (requires Chronos install):
    python phase2_comparison.py --backends rf xgb chronos
"""

import argparse
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error, reliability_diagram_text
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

FEATURE_COLS = [
    "weather_risk", "weather_ensemble_std",
    "flood_risk",   "flood_ensemble_std",
    "month",
]


def load_features(split_year=2015):
    """Build migration agent features from merged dataset."""
    merged = pd.read_csv(DATA_DIR / "merged" / "columbia_merged.csv")
    merged["date"] = pd.to_datetime(merged["date"])

    merged["weather_risk"]          = (merged["TMAX"] > 28).astype(float)
    merged["weather_ensemble_std"]  = 0.01
    p20 = merged.groupby("month")["discharge_cfs"].quantile(0.2)
    merged["monthly_p20"]           = merged["month"].map(p20.to_dict())
    merged["flood_risk"]            = (merged["discharge_cfs"] < merged["monthly_p20"]).astype(float)
    merged["flood_ensemble_std"]    = 0.01

    train = merged[merged["date"].dt.year < split_year]
    test  = merged[merged["date"].dt.year >= split_year]
    return train, test, merged


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def evaluate_without_ecotrust(name, y_true, y_prob) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    cal = expected_calibration_error(y_true, y_prob)
    return {
        "backend":         name,
        "setting":         "Without EcoTrust",
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_score":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":         round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else None,
        "ece":             cal["ece"],
        "brier_score":     cal["brier_score"],
        "escalation_rate": "—",
        "error_coverage":  "—",
        "error_reduction": "—",
    }


def evaluate_with_ecotrust(name, y_true, y_prob, ensemble_std, entropy, config) -> dict:
    gov = HumanGovernanceLayer(config)
    df_gov = pd.DataFrame({
        "date":           range(len(y_true)),
        "migration_risk": y_prob,
        "ensemble_std":   ensemble_std,
        "entropy":        entropy,
        "conflict":       False,
        "ground_truth":   y_true.astype(float),
    })
    gov_df  = gov.evaluate(df_gov)
    gov_sum = gov.summary()

    y_pred  = (y_prob > 0.5).astype(int)
    y_final = gov_df["final_decision"].values.astype(int)
    cal_before = expected_calibration_error(y_true, y_prob)

    return {
        "backend":         name,
        "setting":         "With EcoTrust",
        "accuracy":        round(accuracy_score(y_true, y_final), 4),
        "f1_score":        round(f1_score(y_true, y_final, zero_division=0), 4),
        "roc_auc":         round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else None,
        "ece":             cal_before["ece"],   # calibration of raw model (before governance)
        "brier_score":     cal_before["brier_score"],
        "escalation_rate": gov_sum.get("escalation_rate", 0),
        "error_coverage":  gov_sum.get("error_coverage", np.nan),
        "error_reduction": gov_sum.get("error_reduction", np.nan),
    }


def run_rf(X_train, y_train, X_test, y_test, config):
    print("  Running Random Forest ...")
    model = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    std   = np.full(len(probs), 0.05)   # single model → fixed pseudo-std
    entropy = -(probs * np.log(probs+1e-8) + (1-probs)*np.log(1-probs+1e-8))
    r1 = evaluate_without_ecotrust("Random Forest", y_test, probs)
    r2 = evaluate_with_ecotrust("Random Forest", y_test, probs, std, entropy, config)
    return r1, r2


def run_xgb(X_train, y_train, X_test, y_test, config):
    print("  Running XGBoost ...")
    model = make_pipeline(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         random_state=42, eval_metric="logloss", verbosity=0))
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    std   = np.full(len(probs), 0.05)
    entropy = -(probs * np.log(probs+1e-8) + (1-probs)*np.log(1-probs+1e-8))
    r1 = evaluate_without_ecotrust("XGBoost", y_test, probs)
    r2 = evaluate_with_ecotrust("XGBoost", y_test, probs, std, entropy, config)
    return r1, r2


def run_ensemble(X_train, y_train, X_test, y_test, config):
    print("  Running EcoTrust Ensemble (RF+XGB+GB) ...")
    ens = EnsembleClassifier()
    ens.fit(X_train, y_train)
    out    = ens.predict(X_test)
    probs  = out["mean_prob"]
    std    = out["std_prob"]
    entropy = out["entropy"]
    r1 = evaluate_without_ecotrust("EcoTrust Ensemble", y_test, probs)
    r2 = evaluate_with_ecotrust("EcoTrust Ensemble", y_test, probs, std, entropy, config)
    return r1, r2


def run_chronos(merged_train, merged_test, config):
    print("  Running Chronos (foundation model, zero-shot) ...")
    from agents.chronos_agent import run_chronos_comparison
    metrics, out, gov_df = run_chronos_comparison(merged_train, merged_test, config)

    valid = out.dropna(subset=["ground_truth"])
    y_true = valid["ground_truth"].values.astype(int)
    y_prob = valid["migration_risk"].values

    r1 = evaluate_without_ecotrust("Chronos-small", y_true, y_prob)
    r2 = {
        "backend":         "Chronos-small",
        "setting":         "With EcoTrust",
        "accuracy":        metrics["accuracy_after"],
        "f1_score":        metrics["f1_score"],
        "roc_auc":         metrics["roc_auc"],
        "ece":             metrics["ece"],
        "brier_score":     metrics["brier_score"],
        "escalation_rate": metrics["escalation_rate"],
        "error_coverage":  metrics["error_coverage"],
        "error_reduction": metrics["error_reduction"],
    }
    return r1, r2


def print_table(results: list):
    cols = ["backend", "setting", "accuracy", "f1_score", "roc_auc",
            "ece", "escalation_rate", "error_coverage", "error_reduction"]
    df = pd.DataFrame(results)[cols]
    print("\n" + "=" * 100)
    print("PHASE 2: MODEL-AGNOSTIC EcoTrust COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    print("Key: error_coverage = % of model errors caught by EcoTrust escalation")
    print("     error_reduction = accuracy improvement after human governance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+",
                        default=["rf", "xgb", "ensemble"],
                        choices=["rf", "xgb", "ensemble", "chronos"])
    parser.add_argument("--split_year", type=int, default=2015)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features ...")
    train, test, merged = load_features(args.split_year)
    X_train = train[FEATURE_COLS].values
    y_train = train["high_migration_risk"].values.astype(int)
    X_test  = test[FEATURE_COLS].values
    y_test  = test["high_migration_risk"].values.astype(int)

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Prevalence: {y_test.mean()*100:.1f}%\n")

    config = EscalationConfig()
    all_results = []

    for backend in args.backends:
        if backend == "rf":
            r1, r2 = run_rf(X_train, y_train, X_test, y_test, config)
        elif backend == "xgb":
            r1, r2 = run_xgb(X_train, y_train, X_test, y_test, config)
        elif backend == "ensemble":
            r1, r2 = run_ensemble(X_train, y_train, X_test, y_test, config)
        elif backend == "chronos":
            r1, r2 = run_chronos(train, test, config)
        all_results.extend([r1, r2])

    print_table(all_results)

    out_path = RESULTS_DIR / "phase2_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "phase2_comparison.csv", index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
