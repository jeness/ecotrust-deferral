"""
Calibration-matched confidence-only baselines (reviewer CRITICAL/MAJOR, R2-R3).

Isolates the value of the SIX-signal risk bank from ordinary supervised
recalibration. For every backend we fit deferral policies that share the *same*
meta-training rows, the *same* year-blocked outer cross-fitting, the *same*
logistic model, and the *same* cross-fitted Platt scaling. Only the FEATURES the
risk estimator sees differ:

  1. raw Chow            -- backend confidence, no supervised recalibration
  2. confonly (margin)   -- MetaRiskEstimator(groups=["confidence"]): |p-0.5|
  3. confonly-classaware -- logistic on [margin, y_hat, margin*y_hat]: a
                            class-aware recalibrator (addresses the reviewer note
                            that costs are class-asymmetric and unsigned margin
                            pools predicted-positive and predicted-negative)
  4. full six-signal     -- the EcoTrust risk estimator (all groups)

Backends: the 12 off-the-shelf models AND the primary RF+XGB+GB ensemble (13th),
so the ensemble's variance signal is active for at least one backend. For single
models the variance signal is identically zero (no ensemble spread), so the bank
has five active groups on those backends; the ensemble activates all six.

If the full bank does not beat the class-aware confidence recalibrator, the
contribution is supervised recalibration, not multi-signal learned risk. Paired
year-block bootstrap intervals quantify the residual.

Run from code-artifacts/ or code-artifacts/experiments/ (import path is added).
"""
import sys, json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
for p in (str(CODE_DIR), str(CODE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from xgboost import XGBClassifier

from experiments.decision_learning_eval import (
    load, monthly_p20, conflict_of, make_signals, concat_signals,
    fit_auxiliary_signals, temporal_folds, chow_cost_mask, ENV_FEATURES,
)
from governance.decision_learning import (
    DeploymentCost, CostAwareDecisionPolicy, evaluate_policy, deferral_rule,
)
from uncertainty.ensemble import EnsembleClassifier
from uncertainty.calibration import expected_calibration_error

RESULTS_DIR = CODE_DIR / "results"
SEED = 42
eps = 1e-8


# --- backend adapters: fit(X,y)->self ; predict_ms(X)->(mean_prob, std_prob) ---
class SingleModel:
    def __init__(self, est):
        self.pl = Pipeline([("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler()), ("m", est)])
    def fit(self, X, y): self.pl.fit(X, y); return self
    def predict_ms(self, X):
        p = self.pl.predict_proba(X)[:, 1]
        return p, np.zeros(len(p))  # single model -> no ensemble spread


class EnsembleModel:
    def fit(self, X, y): self.m = EnsembleClassifier().fit(X, y); return self
    def predict_ms(self, X):
        o = self.m.predict(X); return o["mean_prob"], o["std_prob"]


def setup():
    df = load()
    train = df[df["date"].dt.year < 2011]
    cal = df[(df["date"].dt.year >= 2011) & (df["date"].dt.year < 2015)]
    test = df[df["date"].dt.year >= 2015]
    X_tr, y_tr = train[ENV_FEATURES].values, train["high_migration_risk"].values.astype(int)
    X_cl, y_cl = cal[ENV_FEATURES].values, cal["high_migration_risk"].values.astype(int)
    X_te, y_te = test[ENV_FEATURES].values, test["high_migration_risk"].values.astype(int)
    p20_tr = monthly_p20(train)
    conf_cl, conf_te = conflict_of(cal, p20_tr), conflict_of(test, p20_tr)
    cp, q_hat, gate = fit_auxiliary_signals(X_tr, y_tr)
    return dict(train=train, X_tr=X_tr, y_tr=y_tr, X_cl=X_cl, y_cl=y_cl,
                X_te=X_te, y_te=y_te, conf_cl=conf_cl, conf_te=conf_te,
                cp=cp, q_hat=q_hat, gate=gate)


def classaware_confonly_risk(pm_, ym_, pt, gate, X_te):
    """Class-aware confidence recalibrator: identical CalibratedClassifierCV +
    logistic machinery as MetaRiskEstimator, but features are [margin, y_hat,
    margin*y_hat] so the recalibration is per predicted class."""
    def feats(p):
        m = np.abs(p - 0.5); yh = (p > 0.5).astype(float)
        return np.column_stack([m, yh, m * yh])
    yh_tr = (pm_ > 0.5).astype(int)
    err_tr = (yh_tr != ym_).astype(int)
    inner = Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler()),
                      ("m", LogisticRegression(max_iter=1000, random_state=SEED))])
    clf = CalibratedClassifierCV(inner, method="sigmoid", cv=5)
    clf.fit(feats(pm_), err_tr)
    r = clf.predict_proba(feats(pt))[:, 1]
    # apply the identical support-gate floor as the full policy
    oos = gate.out_of_support(X_te)
    r = np.where(oos, np.maximum(r, 0.5), r)
    return r


def band(make, label, seed, ctx, cost):
    train = ctx["train"]; X_tr, y_tr = ctx["X_tr"], ctx["y_tr"]
    X_cl, y_cl = ctx["X_cl"], ctx["y_cl"]; X_te, y_te = ctx["X_te"], ctx["y_te"]
    conf_cl, conf_te = ctx["conf_cl"], ctx["conf_te"]
    cp, q_hat, gate = ctx["cp"], ctx["q_hat"], ctx["gate"]

    bk = make().fit(X_tr, y_tr)
    pc, sc = bk.predict_ms(X_cl)
    pt, stt = bk.predict_ms(X_te)
    yh_t = (pt > 0.5).astype(int)
    ece = float(expected_calibration_error(y_te, pt)["ece"])

    def _sig(pr, sd, cf, Xs, cp_obj=cp, q_obj=q_hat, gate_obj=gate):
        H = -(pr * np.log(pr + eps) + (1 - pr) * np.log(1 - pr + eps))
        return make_signals({"mean_prob": pr, "entropy": H, "std_prob": sd},
                            cf, cp_obj.predict(Xs), q_obj, gate_obj.novelty_feature(Xs))

    st_ = _sig(pt, stt, conf_te, X_te)

    # Outer year-blocked cross-fitting for the meta-training rows (backend refit
    # from scratch each fold via make()).
    oof_sig, oof_pr, oof_y = [], [], []
    train_years = train["date"].dt.year.to_numpy()
    for tr_i, va_i in temporal_folds(y_tr, train_years, n_folds=5):
        bf = make().fit(X_tr[tr_i], y_tr[tr_i])
        pr_f, sd_f = bf.predict_ms(X_tr[va_i])
        cp_f, q_f, gate_f = fit_auxiliary_signals(X_tr[tr_i], y_tr[tr_i])
        p20_f = monthly_p20(train.iloc[tr_i])
        conf_f = conflict_of(train.iloc[va_i], p20_f)
        oof_sig.append(_sig(pr_f, sd_f, conf_f, X_tr[va_i], cp_f, q_f, gate_f))
        oof_pr.append(pr_f); oof_y.append(y_tr[va_i])
    sig_oof_b = {k: np.concatenate([s[k] for s in oof_sig]) for k in oof_sig[0]}
    oof_pr = np.concatenate(oof_pr); oof_y = np.concatenate(oof_y)
    sm_ = concat_signals(sig_oof_b, _sig(pc, sc, conf_cl, X_cl))
    pm_ = np.concatenate([oof_pr, pc]); ym_ = np.concatenate([oof_y, y_cl])

    ev_ch = evaluate_policy(chow_cost_mask(pt, yh_t, cost), pt, y_te, cost, reviewer_acc=1.0)

    p_full = CostAwareDecisionPolicy(cost=cost, meta_model="logistic",
                                     support_gate=gate, random_state=seed).fit(sm_, pm_, ym_)
    ev_full = evaluate_policy(p_full.escalate(st_, pt, X_te), pt, y_te, cost, reviewer_acc=1.0)

    p_conf = CostAwareDecisionPolicy(cost=cost, meta_model="logistic", groups=["confidence"],
                                     support_gate=gate, random_state=seed).fit(sm_, pm_, ym_)
    ev_conf = evaluate_policy(p_conf.escalate(st_, pt, X_te), pt, y_te, cost, reviewer_acc=1.0)

    r_ca = classaware_confonly_risk(pm_, ym_, pt, gate, X_te)
    mask_ca = deferral_rule(r_ca, pt, yh_t, cost)
    ev_ca = evaluate_policy(mask_ca, pt, y_te, cost, reviewer_acc=1.0)

    e_t = (yh_t != y_te).astype(int)
    auc_full = float(roc_auc_score(e_t, p_full.risk.predict_risk(st_))) if e_t.sum() else float("nan")
    auc_ca = float(roc_auc_score(e_t, r_ca)) if e_t.sum() else float("nan")

    # per-day cost vectors for paired bootstrap
    auto = np.where((yh_t == 0) & (y_te == 1), cost.c_fn,
                    np.where((yh_t == 1) & (y_te == 0), cost.c_fp, 0.0))
    full_day = np.where(p_full.escalate(st_, pt, X_te), cost.c_h, auto)
    ca_day = np.where(mask_ca, cost.c_h, auto)

    return {
        "backend": label, "seed": seed, "ece": round(ece, 4),
        "chow_cost": ev_ch["deployment_cost"],
        "confonly_cost": ev_conf["deployment_cost"],
        "confonly_ca_cost": ev_ca["deployment_cost"],
        "full_cost": ev_full["deployment_cost"],
        "full_vs_chow": round(ev_ch["deployment_cost"] - ev_full["deployment_cost"], 3),
        "full_vs_confonly": round(ev_conf["deployment_cost"] - ev_full["deployment_cost"], 3),
        "full_minus_ca": round(ev_full["deployment_cost"] - ev_ca["deployment_cost"], 3),  # +=> recalibrator cheaper
        "ca_vs_chow": round(ev_ch["deployment_cost"] - ev_ca["deployment_cost"], 3),
        "auc_full": round(auc_full, 4), "auc_confonly_ca": round(auc_ca, 4),
        "_full_day": full_day, "_ca_day": ca_day,
    }


def main():
    np.random.seed(SEED)
    ctx = setup()
    cost = DeploymentCost()
    # Seed-parametrized factories: matches the declared three-seed protocol of
    # the main backend-bank experiment (seeds 40/41/42).
    specs = [
        ("Random Forest (well calib.)", lambda s: SingleModel(RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1))),
        ("Gradient Boosting", lambda s: SingleModel(GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=s))),
        ("XGBoost", lambda s: SingleModel(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=s, eval_metric="logloss", verbosity=0))),
        ("Logistic Regression", lambda s: SingleModel(LogisticRegression(max_iter=1000, random_state=s))),
        ("Gaussian Naive Bayes", lambda s: SingleModel(GaussianNB())),
        ("k-NN (k=15)", lambda s: SingleModel(KNeighborsClassifier(n_neighbors=15))),
        ("SVM (RBF, Platt)", lambda s: SingleModel(SVC(probability=True, random_state=s))),
        ("Decision Tree (depth 3)", lambda s: SingleModel(DecisionTreeClassifier(max_depth=3, random_state=s))),
        ("Decision Tree (deep)", lambda s: SingleModel(DecisionTreeClassifier(random_state=s))),
        ("MLP (64, default)", lambda s: SingleModel(MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=s))),
        ("AdaBoost", lambda s: SingleModel(AdaBoostClassifier(random_state=s))),
        ("Extra Trees", lambda s: SingleModel(ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=-1))),
        ("EcoTrust ensemble (RF+XGB+GB)", lambda s: EnsembleModel()),
    ]
    seeds = [40, 41, 42]
    bands = []
    for label, make_s in specs:
        seed_rows = [band(lambda ms=make_s, ss=s: ms(ss), label, s, ctx, cost) for s in seeds]
        agg = {"backend": label}
        for k in ["ece", "chow_cost", "confonly_cost", "confonly_ca_cost", "full_cost",
                  "full_vs_chow", "full_vs_confonly", "full_minus_ca",
                  "ca_vs_chow", "auc_full", "auc_confonly_ca"]:
            agg[k] = round(float(np.mean([r[k] for r in seed_rows])), 4)
        # Seed-averaged per-day cost gap: sign = full - recalibrator (positive =>
        # recalibrator cheaper), averaged over the three seeds before bootstrap.
        agg["_gap_day"] = np.mean([r["_full_day"] - r["_ca_day"] for r in seed_rows], axis=0)
        agg["_ca_range_seedmax"] = round(float(np.max([r["full_minus_ca"] for r in seed_rows])), 4)
        agg["_ca_range_seedmin"] = round(float(np.min([r["full_minus_ca"] for r in seed_rows])), 4)
        bands.append(agg)
        print(f"  {label:30s} ECE={agg['ece']:.3f}  Chow={agg['chow_cost']:.3f}  "
              f"ConfCA={agg['confonly_ca_cost']:.3f}  Full={agg['full_cost']:.3f}  "
              f"(full-minus-CA {agg['full_minus_ca']:+.3f})")

    real = [x for x in bands if "ensemble" not in x["backend"]]  # off-the-shelf
    all_b = bands                                                # incl. ensemble

    # Single sign convention throughout: full_minus_ca = full_cost - CA_cost.
    # POSITIVE => the class-aware recalibrator is cheaper (full worse).
    def summarize(group, tag):
        ece = np.array([x["ece"] for x in group])
        fmca = np.array([x["full_cost"] - x["confonly_ca_cost"] for x in group])
        return {
            f"n_{tag}": len(group),
            f"n_full_beats_ca_{tag}": int(sum(v < 0 for v in fmca)),   # full cheaper
            f"n_ca_beats_chow_{tag}": int(sum(x["ca_vs_chow"] > 0 for x in group)),
            f"n_full_beats_chow_{tag}": int(sum(x["full_vs_chow"] > 0 for x in group)),
            f"full_minus_ca_range_{tag}": [round(float(fmca.min()), 3), round(float(fmca.max()), 3)],
            f"pearson_ece_vs_full_minus_ca_{tag}": round(float(pearsonr(ece, fmca)[0]), 4),
            f"spearman_ece_vs_full_minus_ca_{tag}": round(float(spearmanr(ece, fmca).correlation), 4),
        }

    # Paired year-block bootstrap over test years of the mean full-minus-CA
    # per-day cost gap, using seed-averaged per-backend daily gaps (12 off-shelf).
    df = load(); test = df[df["date"].dt.year >= 2015]
    years = test["date"].dt.year.to_numpy()
    uniq = np.unique(years)
    rng = np.random.default_rng(SEED)
    stacked = np.vstack([x["_gap_day"] for x in real]).mean(axis=0)  # avg backend gap/day
    boot = []
    for _ in range(2000):
        yrs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(years == y)[0] for y in yrs])
        boot.append(stacked[idx].mean())
    boot = np.sort(boot)
    ci = [round(float(boot[49]), 4), round(float(boot[1949]), 4)]
    mean_gap = round(float(stacked.mean()), 4)

    summary = {
        "description": "Calibration-matched confidence-only baselines (R2-R4). Policies "
                       "share rows/folds/logistic/Platt and support gate; only risk "
                       "FEATURES differ. confonly=margin; confonly_ca=[margin,yhat,"
                       "margin*yhat] class-aware; full=six-signal bank. Perfect reviewer, "
                       "100:3:1. Ensemble is the 13th backend (variance signal active). "
                       "All rows averaged over seeds 40/41/42. Sign convention: "
                       "full_minus_ca = full_cost - CA_cost; POSITIVE => recalibrator cheaper.",
        "seeds": seeds,
        **summarize(real, "offshelf"),
        **summarize(all_b, "all13"),
        "paired_bootstrap_full_minus_ca_meancost": {
            "mean_gap_per_day": mean_gap,
            "ci95_yearblock": ci,
            "note": "seed-averaged, then avg over 12 off-the-shelf backends of "
                    "(full - class-aware-conf) per-day cost; POSITIVE => recalibrator "
                    "cheaper. 2000 year-block resamples.",
        },
        "per_backend": [{k: v for k, v in x.items() if not k.startswith("_")} for x in bands],
    }
    out = RESULTS_DIR / "confonly_baseline_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY (seeds 40/41/42 averaged) ===")
    print(f"  [off-the-shelf 12] full beats class-aware conf on "
          f"{summary['n_full_beats_ca_offshelf']}/12; "
          f"CA beats raw Chow on {summary['n_ca_beats_chow_offshelf']}/12")
    print(f"  [all 13 incl. ens] full beats class-aware conf on "
          f"{summary['n_full_beats_ca_all13']}/13")
    print(f"  full-minus-CA range (off-shelf, +=>CA cheaper): {summary['full_minus_ca_range_offshelf']}")
    print(f"  ECE vs (full-CA) off-shelf: Pearson "
          f"{summary['pearson_ece_vs_full_minus_ca_offshelf']}, "
          f"Spearman {summary['spearman_ece_vs_full_minus_ca_offshelf']}")
    print(f"  paired year-block mean (full-CA)/day (+=>CA cheaper): {mean_gap}, 95% CI {ci}")
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
