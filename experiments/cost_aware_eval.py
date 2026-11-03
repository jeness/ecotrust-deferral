"""
Cost-Aware Governance Evaluation — L2D vs. Fixed Threshold Comparison

Compares three governance strategies:
  1. EcoTrust Fixed:     hand-tuned thresholds (current default)
  2. CostAware-10:3:1   c_FN=10, c_FP=3, c_human=1 (salmon: miss > false alarm)
  3. CostAware-5:1:1    c_FN=5,  c_FP=1, c_human=1 (conservative)

Key claim: CostAware achieves the same or better ErrCov at lower escalation rate
than EcoTrust's fixed multi-signal thresholds.

Results saved to: code-artifacts/results/cost_aware_results.json
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
sys.path.insert(0, "..")
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict

from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error
from governance.escalation import HumanGovernanceLayer, EscalationConfig
from governance.cost_aware import CostAwareGovernance, CostConfig, compute_uncertainty

DATA_DIR    = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

ENV_FEATURES = [
    "TMAX", "TMIN", "PRCP", "AWND",
    "discharge_cfs", "gage_height_ft",
    "discharge_7d_mean", "discharge_anomaly",
]
CONFLICT_THRESHOLD = 0.35
CAL_FRAC = 0.20   # fraction of training data used for threshold calibration


def load():
    df = pd.read_csv(DATA_DIR / "merged/columbia_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    for col in ENV_FEATURES:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    return df[df["month"].isin([8, 9, 10, 11])].copy()


def compute_monthly_p20(df):
    """Compute monthly 20th percentile discharge thresholds from a reference dataframe."""
    month = df["date"].dt.month
    return df.groupby(month)["discharge_cfs"].quantile(0.2)

def compute_conflict(df, monthly_p20_ref):
    """Compute conflict flag using pre-computed reference-set discharge thresholds."""
    weather_risk = (df["TMAX"] > 28).astype(float).values
    month = df["date"].dt.month
    monthly_p20 = month.map(monthly_p20_ref.to_dict())
    flood_risk = (df["discharge_cfs"] < monthly_p20).astype(float).values
    return np.abs(weather_risk - flood_risk) > CONFLICT_THRESHOLD


def run_ecotrust_fixed(X_tr, y_tr, X_te, y_te, conflict_te):
    """Baseline: EcoTrust default fixed thresholds."""
    ens = EnsembleClassifier(); ens.fit(X_tr, y_tr)
    eo  = ens.predict(X_te)
    p_ens = eo["mean_prob"]

    gov = HumanGovernanceLayer(EscalationConfig())
    mdf = pd.DataFrame({
        "date":           range(len(y_te)),
        "migration_risk": p_ens,
        "ensemble_std":   eo["std_prob"],
        "entropy":        eo["entropy"],
        "conflict":       conflict_te,
        "ground_truth":   y_te.astype(float),
    })
    gdf = gov.evaluate(mdf)
    gs  = gov.summary()

    return {
        "method": "EcoTrust Fixed",
        "description": "risk>0.60 OR entropy>0.40 OR std>0.08 OR conflict",
        "escalation_rate":  gs["escalation_rate"],
        "error_coverage":   gs["error_coverage"],
        "accuracy_raw":     round(float(accuracy_score(y_te, (p_ens>0.5).astype(int))), 4),
        "accuracy_oracle":  gs["accuracy_after"],
        "error_reduction":  gs["error_reduction"],
        "f1_oracle":        round(float(f1_score(y_te,
                                gdf["final_decision"].values.astype(int),
                                zero_division=0)), 4),
        "tau_star":         None,
        "expected_cost":    None,
    }, ens, eo, p_ens


def run_cost_aware(ens, eo, p_ens, X_tr, y_tr, X_te, y_te, conflict_te,
                   cfg: CostConfig, label: str):
    """
    L2D cost-aware governance with learned threshold τ*.

    Protocol: use the SAME ensemble (trained on full X_tr) to predict on
    the calibration holdout. This ensures the uncertainty score distribution
    is consistent between calibration and test, avoiding train/cal mismatch.
    """
    n_cal  = max(1, int(len(X_tr) * CAL_FRAC))
    X_cal  = X_tr[-n_cal:]
    y_cal  = y_tr[-n_cal:]

    # Same ensemble for calibration probs (not re-trained)
    eo_cal = ens.predict(X_cal)

    # Conflict on calibration set
    tmax_col  = ENV_FEATURES.index("TMAX")
    disch_col = ENV_FEATURES.index("discharge_cfs")
    p20_tr    = np.percentile(X_tr[:, disch_col], 20)
    weather_cal  = (X_cal[:, tmax_col] > 28).astype(float)
    flood_cal    = (X_cal[:, disch_col] < p20_tr).astype(float)
    conflict_cal = np.abs(weather_cal - flood_cal) > CONFLICT_THRESHOLD

    gov_ca = CostAwareGovernance(cfg)
    gov_ca.fit(eo_cal["mean_prob"], eo_cal["entropy"], eo_cal["std_prob"],
               conflict_cal, y_cal)

    result = gov_ca.evaluate(p_ens, eo["entropy"], eo["std_prob"], conflict_te, y_te)
    result["method"]      = label
    result["description"] = (f"Learned τ*={result['tau_star']:.4f}, "
                             f"c_FN={cfg.c_fn}, c_FP={cfg.c_fp}, c_human={cfg.c_human}")
    return result


def main():
    """
    Temporal calibration protocol:
      train: Aug-Nov rows where year < 2011 (n≈1086)
      cal:   Aug-Nov rows 2011 ≤ year < 2015, OOS from training (n≈488)
      test:  Aug-Nov rows 2015+ (n=1208, standard test set)
    This avoids in-sample bias when calibrating τ*.
    """
    np.random.seed(42)
    print("=== Cost-Aware Governance Evaluation ===\n")
    print("Protocol: train<2011 | cal 2011-2014 (OOS) | test 2015+\n")

    df    = load()
    train = df[df["date"].dt.year < 2011]
    cal   = df[(df["date"].dt.year >= 2011) & (df["date"].dt.year < 2015)]
    test  = df[df["date"].dt.year >= 2015]

    X_tr  = train[ENV_FEATURES].values; y_tr = train["high_migration_risk"].values.astype(int)
    X_cl  = cal[ENV_FEATURES].values;   y_cl = cal["high_migration_risk"].values.astype(int)
    X_te  = test[ENV_FEATURES].values;  y_te = test["high_migration_risk"].values.astype(int)
    # Compute conflict using TRAINING-set (pre-2011) discharge percentiles (non-transductive)
    train_p20 = compute_monthly_p20(train)
    conflict_cl = compute_conflict(cal, train_p20)
    conflict_te = compute_conflict(test, train_p20)

    print(f"Train: n={len(X_tr)}, positive={y_tr.mean()*100:.1f}%")
    print(f"Cal:   n={len(X_cl)}, positive={y_cl.mean()*100:.1f}%")
    print(f"Test:  n={len(X_te)}, positive={y_te.mean()*100:.1f}%\n")

    results = []

    # ── 1. EcoTrust Fixed (trained on full X_tr, tested on X_te) ─────────────
    print("Running EcoTrust Fixed ...")
    r_fixed, ens, eo, p_ens = run_ecotrust_fixed(X_tr, y_tr, X_te, y_te, conflict_te)
    results.append(r_fixed)
    print(f"  esc={r_fixed['escalation_rate']:.1%}  ErrCov={r_fixed['error_coverage']:.1%}  "
          f"acc_oracle={r_fixed['accuracy_oracle']:.4f}")

    # Override run_cost_aware with calibration set for proper OOS protocol
    eo_cl = ens.predict(X_cl)

    def run_ca_oos(cfg, label):
        gov_ca = CostAwareGovernance(cfg)
        gov_ca.fit(eo_cl["mean_prob"], eo_cl["entropy"], eo_cl["std_prob"], conflict_cl, y_cl)
        res = gov_ca.evaluate(p_ens, eo["entropy"], eo["std_prob"], conflict_te, y_te)
        res["method"]      = label
        res["description"] = (f"τ*={res['tau_star']:.4f}, "
                              f"c_FN={cfg.c_fn}, c_FP={cfg.c_fp}, c_human={cfg.c_human}")
        return res

    # ── 2. CostAware moderate (c_FN=10) ──────────────────────────────────────
    print("\nRunning CostAware-10:3:1 (moderate cost ratio) ...")
    r_ca10 = run_ca_oos(CostConfig(c_fn=10, c_fp=3, c_human=1), "CostAware (10:3:1)")
    results.append(r_ca10)
    print(f"  τ*={r_ca10['tau_star']:.4f}  esc={r_ca10['escalation_rate']:.1%}  "
          f"ErrCov={r_ca10['error_coverage']:.1%}  acc_oracle={r_ca10['accuracy_oracle']:.4f}")

    # ── 3. CostAware salmon-critical (c_FN=100) ───────────────────────────────
    print("\nRunning CostAware-100:3:1 (salmon-critical: thermal stress mortality) ...")
    r_ca100 = run_ca_oos(CostConfig(c_fn=100, c_fp=3, c_human=1), "CostAware (100:3:1)")
    results.append(r_ca100)
    print(f"  τ*={r_ca100['tau_star']:.4f}  esc={r_ca100['escalation_rate']:.1%}  "
          f"ErrCov={r_ca100['error_coverage']:.1%}  acc_oracle={r_ca100['accuracy_oracle']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Method':30s}  {'Esc%':>6}  {'ErrCov':>7}  {'Acc(oracle)':>11}  {'τ*':>8}")
    print("-" * 80)
    for r in results:
        tau_str = f"{r['tau_star']:.4f}" if r["tau_star"] is not None else "  fixed"
        print(f"{r['method']:30s}  {r['escalation_rate']:>6.1%}  "
              f"{r['error_coverage']:>7.1%}  {r['accuracy_oracle']:>11.4f}  {tau_str:>8}")

    # ── Efficiency gain: CostAware-100 vs Fixed ───────────────────────────────
    esc_fixed = r_fixed["escalation_rate"]
    cov_fixed = r_fixed["error_coverage"]
    print(f"\nPareto-dominance check (CostAware-100:3:1 vs Fixed):")
    print(f"  Escalation: {esc_fixed:.1%} → {r_ca100['escalation_rate']:.1%}  "
          f"({r_ca100['escalation_rate']-esc_fixed:+.1%})")
    print(f"  ErrCov:     {cov_fixed:.1%} → {r_ca100['error_coverage']:.1%}  "
          f"({r_ca100['error_coverage']-cov_fixed:+.1%})")
    dominated = (r_ca100["escalation_rate"] < esc_fixed and
                 r_ca100["error_coverage"]  > cov_fixed)
    print(f"  Pareto-dominant: {dominated}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "protocol": "temporal split: train<2011, cal 2011-2014 (OOS), test 2015+",
        "n_train": int(len(X_tr)),
        "n_cal":   int(len(X_cl)),
        "n_test":  int(len(X_te)),
        "results": results,
    }
    with open(RESULTS_DIR / "cost_aware_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → code-artifacts/results/cost_aware_results.json")


if __name__ == "__main__":
    main()
