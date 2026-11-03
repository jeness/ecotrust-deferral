"""
Run all 4 EcoTrust experimental scenarios.

Ground truth: water_temp_c > 18°C during migration season (Aug-Nov)
              — from USGS, independent of all input features.

Scenarios:
    S1: Normal observations
    S2: Sensor failure (30% feature drop)
    S3: Noisy observations (Gaussian noise)
    S4: Human escalation analysis

Usage:
    python run_scenarios.py --site issaquah_creek
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.weather_agent import WeatherAgent
from agents.hydrology_agent import HydrologyAgent
from agents.migration_agent import MigrationRiskAgent
from governance.escalation import HumanGovernanceLayer, EscalationConfig

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

NOAA_SITE_MAP = {
    "issaquah_creek": "issaquah",
    "columbia_river": "columbia",
}


def load_data(site: str):
    # Try merged dataset first
    merged_path = DATA_DIR / "merged" / f"{site.replace('_creek','').replace('_river','')}_merged.csv"
    if not merged_path.exists():
        merged_path = DATA_DIR / "merged" / "columbia_merged.csv"

    noaa_name = NOAA_SITE_MAP.get(site, "issaquah")
    noaa_path = DATA_DIR / "noaa" / f"{noaa_name}_daily.csv"
    usgs_path = DATA_DIR / "usgs" / f"{site}_daily.csv"

    if not noaa_path.exists() or not usgs_path.exists():
        print(f"Raw data not found. Run fetch_*.py first.")
        return None, None, None

    if not merged_path.exists():
        print("Merged dataset not found. Run data/build_dataset.py first.")
        return None, None, None

    noaa   = pd.read_csv(noaa_path)
    usgs   = pd.read_csv(usgs_path)
    merged = pd.read_csv(merged_path)
    merged["date"] = pd.to_datetime(merged["date"])
    return noaa, usgs, merged


def temporal_split(df, split_year=2015):
    """Split by year so test set is truly held-out future data."""
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"].dt.year < split_year].copy()
    test  = df[df["date"].dt.year >= split_year].copy()
    return train, test


def train_agents(noaa_train, usgs_train, merged_train, alpha=0.10):
    print("  Weather Agent  (ensemble RF+XGB+GB + CP) ...")
    weather = WeatherAgent(alpha=alpha)
    weather.fit(noaa_train)

    print("  Hydrology Agent (ensemble RF+XGB+GB + CP) ...")
    hydro = HydrologyAgent(alpha=alpha)
    hydro.fit(usgs_train)

    weather_out = weather.predict(noaa_train)
    hydro_out   = hydro.predict(usgs_train)

    print("  Migration Agent (ensemble RF+XGB+GB + CP) ...")
    migration = MigrationRiskAgent(alpha=alpha)
    migration.fit(weather_out, hydro_out, merged_train)
    return weather, hydro, migration


def eval_metrics(migration_out: pd.DataFrame) -> dict:
    """Compute prediction metrics on rows with valid ground truth."""
    df = migration_out.dropna(subset=["ground_truth"])
    if len(df) == 0:
        return {}
    y_true = df["ground_truth"].values.astype(int)
    y_prob  = df["migration_risk"].values
    y_pred  = (y_prob > 0.5).astype(int)

    metrics = {
        "n_samples":       len(df),
        "n_positive":      int(y_true.sum()),
        "prevalence":      round(y_true.mean(), 4),
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_score":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mean_ensemble_std": round(df["ensemble_std"].mean(), 4),
        "mean_entropy":      round(df["entropy"].mean(), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def run_scenario(name, noaa_test, usgs_test, merged_test,
                 weather_agent, hydro_agent, migration_agent,
                 governance) -> dict:
    print(f"\n--- {name} ---")

    noaa_in = noaa_test.copy()
    usgs_in = usgs_test.copy()

    if name == "S2_sensor_failure":
        noaa_in = weather_agent.simulate_sensor_failure(noaa_in, drop_frac=0.30)
        usgs_in = hydro_agent.simulate_sensor_failure(usgs_in, drop_frac=0.30)
    elif name == "S3_noisy":
        noaa_in = weather_agent.add_noise(noaa_in, noise_frac=0.10)

    weather_out   = weather_agent.predict(noaa_in)
    hydro_out     = hydro_agent.predict(usgs_in)
    migration_out = migration_agent.predict(weather_out, hydro_out, merged_test)

    gov_df  = governance.evaluate(migration_out)
    gov_sum = governance.summary()

    pred_metrics = eval_metrics(migration_out)

    # CP empirical coverage on migration season rows
    season_rows = migration_out[migration_out["ground_truth"].notna()].copy()
    if len(season_rows) > 0:
        cp_coverage = float(
            (season_rows["cp_in_high_risk"] == season_rows["ground_truth"].astype(bool))
            .mean()
        )
    else:
        cp_coverage = float("nan")

    result = {
        "scenario":           name,
        **pred_metrics,
        "cp_coverage":        round(cp_coverage, 4),
        "escalation_rate":    gov_sum.get("escalation_rate", 0),
        "error_coverage":     gov_sum.get("error_coverage", float("nan")),
        "error_reduction":    gov_sum.get("error_reduction", float("nan")),
        "accuracy_before":    gov_sum.get("accuracy_before", float("nan")),
        "accuracy_after":     gov_sum.get("accuracy_after", float("nan")),
        "conflict_rate":      round(gov_sum.get("conflict_cases", 0) / max(len(gov_df), 1), 4),
        "reason_breakdown":   gov_sum.get("reason_breakdown", {}),
    }

    summary = {k: v for k, v in result.items() if k != "reason_breakdown"}
    print(json.dumps(summary, indent=2, default=str))
    print("  Escalation reasons:", result["reason_breakdown"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",       default="issaquah_creek")
    parser.add_argument("--alpha",      type=float, default=0.10)
    parser.add_argument("--split_year", type=int,   default=2015)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Always load Columbia River USGS (has water_temp_c, matches merged dataset)
    noaa_path   = DATA_DIR / "noaa"  / "columbia_daily.csv"
    usgs_path   = DATA_DIR / "usgs"  / "columbia_river_daily.csv"
    merged_path = DATA_DIR / "merged" / "columbia_merged.csv"
    if not all(p.exists() for p in [noaa_path, usgs_path, merged_path]):
        print("Data not found. Run fetch_*.py and build_dataset.py first.")
        return

    noaa   = pd.read_csv(noaa_path)
    usgs   = pd.read_csv(usgs_path)
    merged = pd.read_csv(merged_path)
    merged["date"] = pd.to_datetime(merged["date"])

    print(f"Dataset: {len(merged):,} rows | split year: {args.split_year}")
    print(f"High migration risk prevalence: {merged['high_migration_risk'].mean()*100:.1f}%")

    noaa_train, noaa_test     = temporal_split(noaa,   args.split_year)
    usgs_train, usgs_test     = temporal_split(usgs,   args.split_year)
    merged_train, merged_test = temporal_split(merged, args.split_year)

    print(f"Train: {len(noaa_train):,} days | Test: {len(noaa_test):,} days")

    print("\nTraining agents ...")
    weather, hydro, migration = train_agents(
        noaa_train, usgs_train, merged_train, args.alpha
    )
    governance = HumanGovernanceLayer(EscalationConfig())

    scenarios = ["S1_normal", "S2_sensor_failure", "S3_noisy", "S4_governance"]
    all_results = []
    for s in scenarios:
        r = run_scenario(s, noaa_test, usgs_test, merged_test,
                         weather, hydro, migration, governance)
        all_results.append(r)

    out_path = RESULTS_DIR / f"scenario_results_{args.site}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")

    # Summary table
    cols = ["scenario", "accuracy", "f1_score", "roc_auc",
            "mean_ensemble_std", "mean_entropy", "cp_coverage",
            "escalation_rate", "error_coverage", "error_reduction"]
    df = pd.DataFrame(all_results)
    available = [c for c in cols if c in df.columns]
    print("\n=== Summary Table ===")
    print(df[available].to_string(index=False))
    df[available].to_csv(RESULTS_DIR / f"summary_table_{args.site}.csv", index=False)


if __name__ == "__main__":
    main()
