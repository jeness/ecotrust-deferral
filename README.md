# EcoTrust: Cost-Aware Post-Hoc Deferral Under Calibration and Shift

Code and data for *"Cost-Aware Post-Hoc Deferral Under Calibration and Shift:
An Environmental AI Case Study"* (IEEE ICTAI 2026).

Authors: Haoran Yu, Lifei Liu, Danping Zhang.

EcoTrust is a **post-hoc decision layer** around a *frozen* classifier: for
each prediction it decides, by cost, whether the AI should act automatically
or the case should be **deferred to a human reviewer** — under
class-asymmetric error costs, imperfect reviewers, review quotas, and detected
distribution shift. The study is a controlled characterization, not a claim of
algorithmic dominance.

## Repository structure

```text
data/          Data acquisition + merge → the frozen predictor's inputs
agents/        Per-view predictors (weather, hydrology, migration, chronos)
uncertainty/   Six-signal machinery: calibration, conformal, ensemble
governance/    The EcoTrust decision layer (risk estimator + cost rule + gate)
experiments/   Evaluation suite (main comparison, baselines, transfer, sweeps)
figures/       Regenerate the paper figures from result artifacts
results/        Pre-computed result files (JSON/CSV) + figure .tex
tests/         Unit tests (support gate, modern L2D surrogates)
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# The merged datasets and all result files ship with the repo,
# so you can regenerate every figure without fetching raw data:
python figures/make_results_figures.py       # writes the paper figures

# Re-run the main evaluation from the merged dataset:
python experiments/decision_learning_eval.py  # main comparison + reviewer/gate sweeps
```

No credentials are needed to reproduce the results: the merged datasets
(`data/merged/*.csv`) and every result file (`results/*.json`, `*.csv`) are
checked in. Only the raw re-download scripts (`data/fetch_noaa.py`) need a free
NOAA CDO token (`export NOAA_TOKEN=...`); USGS and Open-Meteo need none.

## Core idea

A frozen classifier can only emit a probability and a binary label. The
operational question is **who should act on that prediction** — the model, or
a human. That gate matters because errors are not equally costly: in our
illustrative setting a missed thermal-stress day costs 100, a false alarm 3,
and an expert review 1.

EcoTrust compares two actions per case and picks the cheaper one:

```text
cost(auto)   = estimated error risk × cost of being wrong (by predicted class)
cost(defer)  = review fee + expected cost of the reviewer also being wrong
defer  ⇔  cost(auto) > cost(defer)
```

Central question:

```text
Do richer, multi-signal uncertainty estimates improve deployment cost over a
calibration-matched, class-aware confidence estimator — in a human-review
setting with asymmetric errors, imperfect review, quotas, and detected shift?
```

Both the confidence baseline (Chow) and EcoTrust use the **same frozen
predictor** and the **same cost-sensitive rule**; the only difference is where
the error-risk estimate comes from. That isolates exactly what we study.

## Main components

- **Frozen predictor** — a tree-based ensemble (random forest + XGBoost +
  gradient boosting) over NOAA weather and USGS hydrology features, with
  per-view agents in `agents/`. Water temperature is the target only and is
  never an input feature.
- **Meta risk estimator** (`governance/decision_learning.py`) — a calibrated
  logistic model over six uncertainty-signal groups (confidence, entropy,
  ensemble disagreement, agent conflict, two conformal quantities, and a
  Mahalanobis novelty distance) estimating P(prediction is wrong); fitted with
  cross-fitted Platt scaling, no target-site labels.
- **Cost-aware decision rule** (`governance/cost_aware.py`) — maps a risk
  estimate to auto-vs-review under class-asymmetric costs and reviewer
  accuracy; yields class-dependent thresholds (e.g. 0.01 when predicting safe,
  0.333 when predicting danger, at 100:3:1).
- **Support gate + escalation** (`governance/escalation.py`) — a Mahalanobis
  out-of-support detector that raises risk to a conservative floor and routes
  flagged cases to review (a fail-safe, not a robustness guarantee).
- **Calibration-matched control** (`experiments/confonly_baseline_eval.py`) —
  a class-aware recalibrated confidence estimator with the same data, folds,
  and gate, isolating "value of extra signals" from "value of recalibration".

## Reproducing results

```bash
pip install -r requirements.txt
```

### Main comparison — better ranking ≠ lower cost (§VI, Fig. 2)

```bash
python experiments/decision_learning_eval.py   # also writes reviewer/gate/backend sweeps
python experiments/baselines.py                 # baseline bank A0–B5
python experiments/l2d_modern.py                # post-hoc L2D surrogates B6–B7
```

### When does learned risk pay? — calibration-matched (§VII, Table II)

```bash
python experiments/confonly_baseline_eval.py
```

### Cost sensitivity and reviewer quality (§VIII, Fig. 4)

```bash
python experiments/cost_aware_eval.py           # miss-cost sweep (cFN 1→500)
# reviewer-accuracy sweep is produced by decision_learning_eval.py
```

### Detected extrapolation — station transfer (§IX, Fig. 5)

```bash
python experiments/cross_watershed_eval.py
```

### Discussion — calendar baseline diagnostic

```bash
python experiments/calendar_sensitivity_eval.py
```

### Regenerate all paper figures

```bash
python figures/make_results_figures.py          # Fig. 2–5 from results/*.json
```

| Script | Description | Paper |
|--------|-------------|-------|
| `data/build_dataset.py` | Merge NOAA + USGS into the labeled dataset (water temp as target only) | §V |
| `data/fetch_noaa.py` / `fetch_usgs.py` / `fetch_yakima.py` / ... | Raw re-download (optional; NOAA needs a token) | §V |
| `agents/weather_agent.py`, `hydrology_agent.py`, `migration_agent.py`, `chronos_agent.py` | Per-view predictors feeding the frozen ensemble | §IV |
| `uncertainty/calibration.py`, `conformal.py`, `ensemble.py` | The six-signal uncertainty machinery | §IV |
| `governance/decision_learning.py` | Meta risk estimator (EcoTrust C1) | §IV |
| `governance/cost_aware.py` | Cost-aware auto-vs-review rule + thresholds | §IV |
| `governance/escalation.py` | Support gate + human-in-the-loop escalation | §IV |
| `experiments/decision_learning_eval.py` | Main comparison; reviewer, gate, backend-bank sweeps | §VI, §VIII, Fig. 2 |
| `experiments/confonly_baseline_eval.py` | Calibration-matched class-aware confidence control | §VII, Table II |
| `experiments/cost_aware_eval.py` | Miss-cost sweep and review-budget frontier | §VIII, Fig. 4 |
| `experiments/baselines.py` | Baseline bank (no governance, Chow, selective classification, rejector) | §VI |
| `experiments/l2d_modern.py` | Frozen-backend post-hoc L2D surrogates (Mozannar–Sontag, one-vs-all) | §VI |
| `experiments/cross_watershed_eval.py` | Transfer to ten river stations | §IX, Fig. 5 |
| `experiments/calendar_sensitivity_eval.py` | Month-only calendar baseline diagnostic | Discussion |
| `figures/make_results_figures.py` | Regenerate Fig. 2–5 from result artifacts | §VI–§IX |

## Key results

On a Columbia River thermal-stress testbed (The Dalles, 1996–2024; strictly
temporal train/calibration/test split), with the frozen ensemble at 92.7%
test accuracy:

| Finding | Result |
| --- | --- |
| Error-ranking AUC (learned risk vs. confidence) | 0.889 vs. 0.869 (Brier 0.056 vs. 0.058) |
| Deployment cost / day (Chow vs. EcoTrust) | **0.416 vs. 0.567** — the simple rule is cheaper |
| Review rate (EcoTrust vs. Chow) | 44.0% vs. 31.0% of days |
| Learned risk wins on | 6 of 13 real backends |
| Calibration-matched (full bank vs. class-aware) | full bank cheaper on 7/12; mean Δ +0.0082 [−0.0002, 0.0172], CI crosses zero |
| Reviewer break-even accuracy | 93.05% (measured crossing between a = 0.90 and 0.95) |
| Deferral beats automation once misses are costly | around cFN = 5–10; 78.4% cheaper at cFN = 100 |
| Station transfer (10 sites) | gate flags every case; cheapest on 8 sites only with a perfect, unconstrained reviewer |
| Calendar baseline | a month-only logistic hits 94.1% in Aug–Nov (> ensemble's 92.7%) |

The headline: **better error ranking does not guarantee lower cost**, the
extra signals' advantage over calibrated confidence is largely recalibration,
and detecting extrapolation is not the same as transferring.

## Scope and limitations

This is a controlled case study on one task; it bounds how the results read:

- it does **not** claim algorithmic dominance — learned risk wins on only 6 of
  13 backends, and the calibration-matched test does not resolve a mean cost
  difference
- the costs (100:3:1) are **illustrative**, not measured management costs; we
  sweep the miss cost from 1 to 500 but do not estimate real-world costs
- reviewer errors and the L2D expert decisions are **simulated** and
  class-independent; actual difficulty-dependent behavior may differ
- the support gate detects **covariate**, not concept, shift, and in transfer
  flags every case — a conservative fallback, not transfer competence
- the meta risk model uses ~1,200 rows, too few to resolve the extreme 1%
  operating tail; the ten transfer stations broaden geography but not task
- `B6`/`B7` freeze class logits for comparability and are not full joint
  classifier–rejector reproductions

## Artifact contents

This repository includes:

- the full pipeline: data merge, agents, uncertainty signals, the EcoTrust
  governance layer, and the evaluation suite
- the **merged datasets** (`data/merged/*.csv`) and the multi-basin transfer
  data, so results reproduce without re-downloading
- **pre-computed result files** (`results/*.json`, `*.csv`) and the figure
  `.tex` sources, plus the script that regenerates all paper figures

It does not include: LaTeX build artifacts (git-ignored). Raw NOAA re-download
requires a free token; USGS and Open-Meteo do not.

## Key parameters

- Thermal-stress label: water temperature > 18 °C (Richter & Kolmes, 2005)
- Primary cost ratio: cFN : cFP : c_h = 100 : 3 : 1 (one review = unit cost)
- Temporal split: train < 2011 (n = 1,086), calibration 2011–2014 (n = 488), test 2015+ (n = 1,208)
- Meta risk estimator: ℓ2 logistic over six signal groups, cross-fitted Platt scaling (5 folds)
- Support gate: Mahalanobis distance, (1 − ε) training quantile, ε = 0.05
- Evaluation season: August–November migration window

## Citation

```bibtex
@inproceedings{yu2026ecotrust,
  title={Cost-Aware Post-Hoc Deferral Under Calibration and Shift: An Environmental AI Case Study},
  author={Haoran Yu and Lifei Liu and Danping Zhang},
  booktitle={IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
