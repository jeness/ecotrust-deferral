"""
Build cross-watershed dataset for EcoTrust validation.

Watershed: McKenzie River near Vida, OR (USGS 14150000)
Weather: Open-Meteo at McKenzie watershed coordinates (44.127°N, -122.457°W)

Same pipeline as Columbia River (build_dataset.py), same feature engineering.
Output: data/merged/mckenzie_merged.csv

Key differences from Columbia:
  - McKenzie is a Cascade Range river (western slopes), Columbia is high-desert/gorge
  - McKenzie has cold snowmelt inputs → different thermal regime
  - Chinook salmon habitat: spring Chinook use McKenzie heavily
  - Thermal stress threshold same (18°C)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = DATA_DIR / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_mckenzie():
    print("Building McKenzie River cross-watershed dataset...")

    # Load USGS hydrology (water_temp_c is ground truth, not a feature)
    usgs = pd.read_csv(DATA_DIR / "usgs" / "mckenzie_river_daily.csv")
    usgs["date"] = pd.to_datetime(usgs["date"])

    # Load Open-Meteo weather
    noaa = pd.read_csv(DATA_DIR / "noaa" / "mckenzie_daily.csv")
    noaa["date"] = pd.to_datetime(noaa["date"])

    # Inner join
    df = noaa.merge(usgs, on="date", how="inner")
    print(f"  After join: {len(df)} rows, date {df['date'].min().date()} to {df['date'].max().date()}")

    # Temporal features (used for filtering, NOT as model inputs)
    df["month"]      = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year

    # Fill missing hydrology
    for col in ["gage_height_ft", "AWND", "SNOW"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Streamflow features (no water_temp_c)
    df["discharge_cfs"] = df["discharge_cfs"].fillna(df["discharge_cfs"].rolling(7, min_periods=1).median())
    df["discharge_7d_mean"] = df["discharge_cfs"].rolling(7, min_periods=1).mean()
    monthly_p90 = df.groupby("month")["discharge_cfs"].quantile(0.9)
    df["monthly_p90"]       = df["month"].map(monthly_p90.to_dict())
    df["discharge_anomaly"] = df["discharge_cfs"] - df["monthly_p90"]

    df["year"] = df["date"].dt.year

    # ─── Ground truth ─────────────────────────────────────────────────────────
    # water_temp_c > 18°C during Aug-Nov migration season
    df["thermal_stress"]   = (df["water_temp_c"] > 18.0).astype(int)
    df["migration_season"] = df["month"].isin([8, 9, 10, 11]).astype(int)
    df["high_migration_risk"] = (
        (df["thermal_stress"] == 1) & (df["migration_season"] == 1)
    ).astype(int)

    # Drop rows without ground truth
    n_before = len(df)
    df = df.dropna(subset=["water_temp_c"])
    print(f"  Dropped {n_before - len(df)} rows missing water_temp_c")

    # Season summary
    season = df[df["migration_season"] == 1]
    print(f"  Total rows: {len(df)}")
    print(f"  Migration season (Aug-Nov) rows: {len(season)}")
    print(f"  High-risk days: {season['high_migration_risk'].sum()} "
          f"({season['high_migration_risk'].mean()*100:.1f}% of season)")
    print(f"  Max water temp: {df['water_temp_c'].max():.1f}°C")

    # Save
    out_path = OUT_DIR / "mckenzie_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df


if __name__ == "__main__":
    build_mckenzie()
