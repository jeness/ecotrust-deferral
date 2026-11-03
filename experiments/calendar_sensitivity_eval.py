"""
Calendar sensitivity diagnostic (BigData P0).

Question: the month-only baseline (94.1%) beats the environmental ensemble
(92.7%). Is EcoTrust's central negative finding -- "richer risk signals do not
reliably beat calibrated confidence / Chow" -- merely an artifact of EXCLUDING a
strong calendar predictor?

We freeze three predictors under the identical within-season temporal split and,
for each, compare Chow (confidence-only) deployment cost vs a multi-signal
cost-aware (EcoTrust-style) deployment cost at the paper's primary 100:3:1 ratio
with a perfect reviewer. If Chow stays <= EcoTrust across ALL three predictors --
including Environmental+Calendar, the strongest -- the negative finding is robust.

Run: python3 calendar_sensitivity_eval.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
sys.path.insert(0, "..")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error

DATA_DIR = Path("../data"); RESULTS_DIR = Path("../results"); RESULTS_DIR.mkdir(exist_ok=True)

ENV = ["TMAX","TMIN","PRCP","AWND","discharge_cfs","gage_height_ft",
       "discharge_7d_mean","discharge_anomaly"]
CAL = ["month","day_of_year"]
FEATURE_SETS = {"Calendar": CAL, "Environmental": ENV, "Env+Calendar": ENV + CAL}

# Primary deployment cost ratio (paper): FN:FP:review = 100:3:1
C_FN, C_FP, C_H = 100.0, 3.0, 1.0

def load():
    df = pd.read_csv(DATA_DIR / "merged/columbia_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    for c in ENV:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(df[c].median()).fillna(0.0)
    return df[df["month"].isin([8,9,10,11])].copy()

def deployment_cost(p, y, defer_mask):
    """Per-day cost with a perfect reviewer (a=1): deferred cases pay only c_h."""
    yhat = (p > 0.5).astype(int)
    nd = ~defer_mask
    fn = int(((yhat[nd]==0)&(y[nd]==1)).sum())
    fp = int(((yhat[nd]==1)&(y[nd]==0)).sum())
    return (C_FN*fn + C_FP*fp + C_H*int(defer_mask.sum())) / len(y)

def chow_cost(p_cal, y_cal, p_te, y_te):
    """Learn a confidence band on cal that minimizes cost; apply on test.
    Chow defers the least-confident cases: max(p,1-p) < t  <=>  |p-0.5| < t-0.5."""
    conf_cal = np.maximum(p_cal, 1-p_cal); conf_te = np.maximum(p_te, 1-p_te)
    grid = np.unique(np.concatenate([[0.0], np.linspace(0.5,1.0,201),[1.01]]))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        c = deployment_cost(p_cal, y_cal, conf_cal < t)
        if c < best_c: best_c, best_t = c, t
    return deployment_cost(p_te, y_te, conf_te < best_t), float(best_t)

def ecotrust_cost(sig_cal, p_cal, y_cal, sig_te, p_te, y_te):
    """Multi-signal cost-aware rule: learn a threshold on a composite risk score
    (confidence + entropy + ensemble std) minimizing cost on cal; apply on test."""
    def score(sig): return -np.abs(sig["p"]-0.5) + sig["entropy"] + sig["std"]
    u_cal, u_te = score(sig_cal), score(sig_te)
    grid = np.concatenate([[u_cal.min()-1], np.linspace(u_cal.min(),u_cal.max(),201),[u_cal.max()+1]])
    best_t, best_c = grid[0], np.inf
    for t in grid:
        c = deployment_cost(p_cal, y_cal, u_cal > t)
        if c < best_c: best_c, best_t = c, t
    return deployment_cost(p_te, y_te, u_te > best_t), float(best_t)

def main():
    np.random.seed(42)
    df = load()
    tr_all = df[df["date"].dt.year < 2015].sort_values("date")
    te     = df[df["date"].dt.year >= 2015]
    # carve a calibration tail (last 20% of train by time) for threshold learning
    k = int(len(tr_all)*0.8); fit, cal = tr_all.iloc[:k], tr_all.iloc[k:]
    y_fit = fit["high_migration_risk"].values.astype(int)
    y_cal = cal["high_migration_risk"].values.astype(int)
    y_te  = te["high_migration_risk"].values.astype(int)
    print(f"fit={len(fit)} cal={len(cal)} test={len(te)} test_prev={y_te.mean()*100:.1f}%\n")

    rows = []
    for name, feats in FEATURE_SETS.items():
        ens = EnsembleClassifier(); ens.fit(fit[feats].values, y_fit)
        oc, ot = ens.predict(cal[feats].values), ens.predict(te[feats].values)
        p_cal, p_te = oc["mean_prob"], ot["mean_prob"]
        sig_cal = {"p":p_cal, "entropy":oc["entropy"], "std":oc["std_prob"]}
        sig_te  = {"p":p_te,  "entropy":ot["entropy"], "std":ot["std_prob"]}
        yhat = (p_te>0.5).astype(int)
        chow_c, chow_t = chow_cost(p_cal, y_cal, p_te, y_te)
        eco_c, eco_t   = ecotrust_cost(sig_cal, p_cal, y_cal, sig_te, p_te, y_te)
        auto_c = deployment_cost(p_te, y_te, np.zeros(len(y_te),bool))  # no deferral
        row = {"predictor": name,
               "accuracy": round(accuracy_score(y_te, yhat),4),
               "f1": round(f1_score(y_te, yhat, zero_division=0),4),
               "auc": round(roc_auc_score(y_te, p_te),4),
               "ece": round(expected_calibration_error(y_te, p_te)["ece"],4),
               "brier": round(brier_score_loss(y_te, p_te),4),
               "auto_cost": round(auto_c,4),
               "chow_cost": round(chow_c,4),
               "ecotrust_cost": round(eco_c,4),
               "winner": "Chow" if chow_c <= eco_c else "EcoTrust",
               "chow_beats_ecotrust": bool(chow_c <= eco_c)}
        rows.append(row); print(row)

    all_chow_wins = all(r["chow_beats_ecotrust"] for r in rows)
    verdict = ("ROBUST: Chow <= EcoTrust for ALL predictors incl. Env+Calendar; "
               "the negative finding is NOT an artifact of excluding calendar."
               if all_chow_wins else
               "SENSITIVE: the winning policy changes across predictors; the "
               "negative finding may depend on which predictor is frozen.")
    print("\nVERDICT:", verdict)
    out = {"cost_ratio": f"{int(C_FN)}:{int(C_FP)}:{int(C_H)}", "reviewer": "perfect(a=1)",
           "rows": rows, "all_chow_win": all_chow_wins, "verdict": verdict}
    (RESULTS_DIR / "calendar_sensitivity_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> results/calendar_sensitivity_results.json")

if __name__ == "__main__":
    main()
