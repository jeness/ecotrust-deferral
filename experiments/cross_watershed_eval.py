"""
Cross-Watershed Generalization Evaluation for EcoTrust.

Protocol:
  TRAIN: Columbia River (1996-2014, within-season Aug-Nov, ~1574 days)
  TEST:  McKenzie River, OR (2015-2024, within-season Aug-Nov, zero-shot)

This is the strongest generalization test: the model is trained on one
watershed (Columbia, arid/gorge climate) and tested on a completely different
watershed (McKenzie, Cascade Range snowmelt river) without any McKenzie data
in the training set.

McKenzie River differs from Columbia in:
  - Climate: wet maritime/Cascade vs arid Columbia Gorge
  - Hydrology: snowmelt-dominated vs regulated/mixed
  - Thermal regime: rarely exceeds 18°C vs frequently
  - Geography: 220km south, different elevation profile

Results saved to:
  code-artifacts/results/cross_watershed_results.json
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
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

ENV_FEATURES = [
    "TMAX", "TMIN", "PRCP", "AWND",
    "discharge_cfs", "gage_height_ft",
    "discharge_7d_mean", "discharge_anomaly",
]
CONFLICT_THRESHOLD = 0.35


def load_columbia():
    """Load Columbia River training data (1996-2014, Aug-Nov)."""
    df = pd.read_csv(DATA_DIR / "merged/columbia_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    for col in ENV_FEATURES:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    train = df[(df["date"].dt.year < 2015) & df["month"].isin([8,9,10,11])].copy()
    return train


def load_mckenzie(split_year=2015):
    """Load McKenzie River test data (split_year+, Aug-Nov)."""
    df = pd.read_csv(DATA_DIR / "merged/mckenzie_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    for col in ENV_FEATURES:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    # Within-season
    df = df[df["month"].isin([8,9,10,11])].copy()
    return df[df["date"].dt.year >= split_year].copy()


def compute_monthly_p20(df):
    """Compute monthly 20th percentile discharge thresholds from a reference dataframe."""
    month = df["date"].dt.month
    return df.groupby(month)["discharge_cfs"].quantile(0.2)

def compute_conflict(df, monthly_p20_ref):
    """Compute agent conflict flag using pre-computed reference discharge thresholds."""
    weather_risk = (df["TMAX"] > 28).astype(float).values
    month = df["date"].dt.month
    monthly_p20 = month.map(monthly_p20_ref.to_dict())
    flood_risk = (df["discharge_cfs"] < monthly_p20).astype(float).values
    return np.abs(weather_risk - flood_risk) > CONFLICT_THRESHOLD


def pipe(est):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()), ("m", est)])


def eval_model(name, y_true, y_pred, y_prob=None, gov_sum=None):
    r = {"method": name,
         "accuracy": round(accuracy_score(y_true, y_pred), 4),
         "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)}
    if y_prob is not None and len(np.unique(y_true)) > 1:
        r["auc"]  = round(roc_auc_score(y_true, y_prob), 4)
        r["ece"]  = round(expected_calibration_error(y_true, y_prob)["ece"], 4)
    if gov_sum:
        r.update({k: round(v, 4) for k, v in gov_sum.items() if isinstance(v, (float, int))})
    return r


def main():
    np.random.seed(42)
    print("=== EcoTrust Cross-Watershed Generalization ===")
    print("Train: Columbia River 1996-2014 (Aug-Nov)")
    print("Test:  McKenzie River 2015-2024 (Aug-Nov, ZERO-SHOT)\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df  = load_columbia()
    test_df   = load_mckenzie(split_year=2015)
    # Discharge-based conflict signal is NOT transferable cross-watershed:
    # Columbia's discharge percentile thresholds don't apply to McKenzie (different river system).
    # For zero-shot evaluation, we rely only on model-based uncertainty signals
    # (ensemble disagreement, entropy, risk score). Conflict is disabled in this setting.
    conflict  = np.zeros(len(test_df), dtype=bool)

    X_tr = train_df[ENV_FEATURES].values
    y_tr = train_df["high_migration_risk"].values.astype(int)
    X_te = test_df[ENV_FEATURES].values
    y_te = test_df["high_migration_risk"].values.astype(int)

    print(f"Train (Columbia):  n={len(X_tr)}, positive={y_tr.mean()*100:.1f}%")
    print(f"Test  (McKenzie):  n={len(X_te)}, positive={y_te.mean()*100:.1f}%")
    print(f"Conflict cases:    {conflict.sum()} ({conflict.mean()*100:.1f}%)\n")

    results = []

    # B0: Always-high-risk
    results.append(eval_model("B0 Always-High-Risk", y_te, np.ones(len(y_te), dtype=int)))

    # B_month: month-only LR (calendar ceiling)
    month_tr = train_df["date"].dt.month.values.reshape(-1, 1)
    month_te = test_df["date"].dt.month.values.reshape(-1, 1)
    lr_m = LogisticRegression(random_state=42, max_iter=200)
    lr_m.fit(month_tr, y_tr)
    p_m = lr_m.predict_proba(month_te)[:, 1]
    results.append(eval_model("B_month Month-only LR", y_te, (p_m>0.5).astype(int), p_m))

    # B1: Single RF
    rf = pipe(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    results.append(eval_model("B1 Single RF", y_te, (p>0.5).astype(int), p))

    # B2: XGBoost
    xg = pipe(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=42, eval_metric="logloss", verbosity=0))
    xg.fit(X_tr, y_tr)
    p = xg.predict_proba(X_te)[:, 1]
    results.append(eval_model("B2 XGBoost", y_te, (p>0.5).astype(int), p))

    # B3: EcoTrust Ensemble (no governance)
    ens = EnsembleClassifier()
    ens.fit(X_tr, y_tr)
    eo = ens.predict(X_te)
    p_ens = eo["mean_prob"]
    results.append(eval_model("B3 Ensemble", y_te, (p_ens>0.5).astype(int), p_ens))

    # B4: Full EcoTrust (trained on Columbia, tested on McKenzie zero-shot)
    gov = HumanGovernanceLayer(EscalationConfig())
    mdf = pd.DataFrame({
        "date":          test_df["date"].values,
        "migration_risk": p_ens,
        "ensemble_std":   eo["std_prob"],
        "entropy":        eo["entropy"],
        "conflict":       conflict,
        "ground_truth":   y_te.astype(float),
    })
    gdf = gov.evaluate(mdf)
    gs  = gov.summary()
    yf  = gdf["final_decision"].values.astype(int)
    results.append(eval_model("B4 EcoTrust (zero-shot)", y_te, yf, p_ens, gs))

    # ── Print ─────────────────────────────────────────────────────────────────
    print("=== Results (cross-watershed: Columbia → McKenzie) ===")
    for r in results:
        esc  = f"esc={r.get('escalation_rate',0):.1%} " if "escalation_rate" in r else ""
        cov  = f"ErrCov={r.get('error_coverage',0):.1%}" if "error_coverage" in r else ""
        print(f"  {r['method']:35s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"auc={r.get('auc','N/A')} {esc}{cov}")
    print()
    print(f"Escalation breakdown: {gs['reason_breakdown']}")

    # ── Compare against within-watershed ─────────────────────────────────────
    try:
        with_res = json.load(open(RESULTS_DIR / "within_season_results.json"))
        within_ens = next(r for r in with_res if r["method"] == "B3 Ensemble")
        within_eco = next(r for r in with_res if r["method"] == "B4 EcoTrust")
        cross_ens  = next(r for r in results   if "Ensemble"   in r["method"])
        cross_eco  = next(r for r in results   if "zero-shot"  in r["method"])
        print("=== Generalization Gap ===")
        print(f"  Ensemble accuracy:   within={within_ens['accuracy']:.4f}  "
              f"cross={cross_ens['accuracy']:.4f}  "
              f"gap={within_ens['accuracy']-cross_ens['accuracy']:.4f}")
        print(f"  EcoTrust ErrCov:     within={within_eco.get('error_coverage',0):.4f}  "
              f"cross={cross_eco.get('error_coverage',0):.4f}  "
              f"gap={within_eco.get('error_coverage',0)-cross_eco.get('error_coverage',0):.4f}")
        print(f"  EcoTrust acc_after:  within={within_eco.get('accuracy_after',0):.4f}  "
              f"cross={cross_eco.get('accuracy_after',0):.4f}")
    except Exception as e:
        print(f"  (Could not load within-season results for comparison: {e})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "train_watershed": "Columbia River at The Dalles OR (USGS 14105700)",
        "test_watershed":  "McKenzie River near Vida OR (USGS 14150000)",
        "protocol":        "zero-shot cross-watershed (no McKenzie data in training)",
        "train_n":         int(len(X_tr)),
        "test_n":          int(len(X_te)),
        "train_prevalence": float(round(y_tr.mean(), 4)),
        "test_prevalence":  float(round(y_te.mean(), 4)),
        "results":          results,
        "gov_detail":       {k: v for k, v in gs.items() if not isinstance(v, dict)},
        "reason_breakdown": gs["reason_breakdown"],
    }
    with open(RESULTS_DIR / "cross_watershed_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → code-artifacts/results/cross_watershed_results.json")


if __name__ == "__main__":
    main()
