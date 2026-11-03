"""
Within-season Phase 2: Model-Agnostic EcoTrust Comparison (CANONICAL)

Matches the evaluation protocol of within_season_eval.py:
  - Aug-Nov data only
  - ENV_FEATURES only (no month, no day_of_year)
  - Temporal split: train <2015, test >=2015
  - Conflict computed from weather/hydrology risk proxies

For each backend (RF, XGBoost, Ensemble), evaluates:
  - WITHOUT governance: raw model accuracy
  - WITH governance: governance using risk score + ensemble std + entropy + conflict

Results saved to: results/within_season_phase2.json
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
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
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    # Within-season only (Aug-Nov)
    return df[df["month"].isin([8, 9, 10, 11])].copy()


def pipe(est):
    return Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler()),
        ("m", est),
    ])


def compute_monthly_p20(df):
    """Compute monthly 20th percentile discharge thresholds from a reference dataframe."""
    return df.groupby(df["date"].dt.month)["discharge_cfs"].quantile(0.2)

def compute_conflict(df, monthly_p20_ref):
    """Compute conflict flag using pre-computed training-set discharge thresholds."""
    weather_risk = (df["TMAX"] > 28).astype(float)
    month = df["date"].dt.month
    monthly_p20 = month.map(monthly_p20_ref.to_dict())
    flood_risk = (df["discharge_cfs"] < monthly_p20).astype(float)
    conflict = (np.abs(weather_risk.values - flood_risk.values) > CONFLICT_THRESHOLD)
    return conflict


def evaluate_backend(name, X_tr, y_tr, X_te, y_te, conflict_te):
    """Train, evaluate without and with governance. Returns two dicts."""
    if name == "Random Forest":
        model = pipe(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        std = np.full(len(probs), 0.05)
        entropy = -(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))
    elif name == "XGBoost":
        model = pipe(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    random_state=42, eval_metric="logloss", verbosity=0))
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        std = np.full(len(probs), 0.05)
        entropy = -(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))
    elif name == "EcoTrust Ensemble":
        ens = EnsembleClassifier()
        ens.fit(X_tr, y_tr)
        out = ens.predict(X_te)
        probs = out["mean_prob"]
        std = out["std_prob"]
        entropy = out["entropy"]
    else:
        raise ValueError(f"Unknown backend: {name}")

    y_pred = (probs > 0.5).astype(int)
    cal = expected_calibration_error(y_te, probs)

    r_without = {
        "backend":        name,
        "setting":        "Without EcoTrust",
        "accuracy":       round(accuracy_score(y_te, y_pred), 4),
        "f1":             round(f1_score(y_te, y_pred, zero_division=0), 4),
        "auc":            round(roc_auc_score(y_te, probs), 4),
        "ece":            round(cal["ece"], 4),
        "escalation_rate": None,
        "error_coverage": None,
        "error_reduction": None,
    }

    # With governance — using REAL conflict signal
    gov = HumanGovernanceLayer(EscalationConfig())
    mdf = pd.DataFrame({
        "date":           range(len(y_te)),
        "migration_risk": probs,
        "ensemble_std":   std,
        "entropy":        entropy,
        "conflict":       conflict_te,
        "ground_truth":   y_te.astype(float),
    })
    gdf = gov.evaluate(mdf)
    gs = gov.summary()
    y_final = gdf["final_decision"].values.astype(int)

    r_with = {
        "backend":         name,
        "setting":         "With EcoTrust",
        "accuracy":        round(accuracy_score(y_te, y_final), 4),
        "f1":              round(f1_score(y_te, y_final, zero_division=0), 4),
        "auc":             round(roc_auc_score(y_te, probs), 4),
        "ece":             round(cal["ece"], 4),
        "escalation_rate": round(gs["escalation_rate"], 4),
        "error_coverage":  round(gs["error_coverage"], 4),
        "error_reduction": round(gs["error_reduction"], 4),
        "reason_breakdown": gs["reason_breakdown"],
    }
    return r_without, r_with


def main():
    np.random.seed(42)
    df = load()
    train = df[df["date"].dt.year < 2015]
    test = df[df["date"].dt.year >= 2015]

    X_tr = train[ENV_FEATURES].values
    y_tr = train["high_migration_risk"].values.astype(int)
    X_te = test[ENV_FEATURES].values
    y_te = test["high_migration_risk"].values.astype(int)
    train_p20 = compute_monthly_p20(train)
    conflict_te = compute_conflict(test, train_p20)

    print(f"Train: {len(X_tr)} | Test: {len(X_te)} | Prevalence: {y_te.mean()*100:.1f}%")
    print(f"Conflict cases (test): {conflict_te.sum()} ({conflict_te.mean()*100:.1f}%)")

    results = []
    for name in ["Random Forest", "XGBoost", "EcoTrust Ensemble"]:
        print(f"Running {name}...")
        r1, r2 = evaluate_backend(name, X_tr, y_tr, X_te, y_te, conflict_te)
        results.extend([r1, r2])
        print(f"  w/o governance: acc={r1['accuracy']:.4f}, ece={r1['ece']:.4f}")
        print(f"  w/ governance:  acc={r2['accuracy']:.4f}, ErrCov={r2['error_coverage']:.4f}, esc={r2['escalation_rate']:.4f}")
        print(f"  reason breakdown: {r2['reason_breakdown']}")

    out_path = RESULTS_DIR / "within_season_phase2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table for paper
    print("\n=== Paper Table Summary ===")
    for r in results:
        esc = f"{r['escalation_rate']:.1%}" if r['escalation_rate'] is not None else "—"
        cov = f"{r['error_coverage']:.1%}" if r['error_coverage'] is not None else "—"
        red = f"+{r['error_reduction']:.1%}" if r['error_reduction'] is not None else "—"
        print(f"{r['backend']:20s} {r['setting']:20s} acc={r['accuracy']:.3f} ece={r['ece']:.3f} ErrCov={cov} ΔAcc={red}")


if __name__ == "__main__":
    main()
