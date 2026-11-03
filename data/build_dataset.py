"""
Build the merged EcoTrust dataset with a proper independent ground truth.

Ground truth definition:
    thermal_stress = water_temp_c > 18°C  (salmon thermal stress threshold)

This threshold is well-documented in salmon biology:
    - Brett (1952): optimal swimming performance drops sharply above 18°C
    - NOAA Pacific Salmon: 18°C is the critical threshold for Chinook
    - Materna (2001): mortality risk begins above 20-21°C

Key design principle: water_temp_c is the TARGET, never an input feature.
All agents must predict thermal stress using only upstream variables:
    - Weather Agent  : NOAA air temperature / precipitation
    - Hydrology Agent: USGS streamflow / gage height (NO water_temp_c)
    - Migration Agent: fused weather + hydrology risk signals

Outputs:
    data/merged/columbia_merged.csv   — full merged daily dataset
    data/merged/dataset_schema.md     — dataset card
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = DATA_DIR / "merged"


def load_and_align(noaa_name: str, usgs_name: str) -> pd.DataFrame:
    noaa = pd.read_csv(DATA_DIR / "noaa" / f"{noaa_name}_daily.csv")
    usgs = pd.read_csv(DATA_DIR / "usgs" / f"{usgs_name}_daily.csv")
    noaa["date"] = pd.to_datetime(noaa["date"])
    usgs["date"] = pd.to_datetime(usgs["date"])

    # Inner join on date
    df = noaa.merge(usgs, on="date", how="inner")

    df["year"] = df["date"].dt.year

    # Temporal features
    df["month"]      = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["season"] = pd.cut(
        df["month"],
        bins=[0, 3, 6, 9, 12],
        labels=["winter", "spring", "summer", "fall"],
    )

    # ------------------------------------------------------------------ #
    # Ground truth: salmon thermal stress (water_temp_c > 18°C)
    # This is the ONLY column that uses water_temp_c — all agents must
    # predict this label WITHOUT seeing water_temp_c as a feature.
    # ------------------------------------------------------------------ #
    df["thermal_stress"] = (df["water_temp_c"] > 18.0).astype(int)

    # Migration season mask (Aug–Nov = active salmon migration)
    df["migration_season"] = df["month"].isin([8, 9, 10, 11]).astype(int)

    # Combined label: thermal stress DURING migration season
    df["high_migration_risk"] = (
        (df["thermal_stress"] == 1) & (df["migration_season"] == 1)
    ).astype(int)

    # Fill sparse columns with 0
    for col in ["AWND", "SNOW", "gage_height_ft"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # Streamflow features (no water_temp_c)
    df["discharge_7d_mean"] = df["discharge_cfs"].rolling(7, min_periods=1).mean()
    monthly_p90 = df.groupby("month")["discharge_cfs"].quantile(0.9)
    df["monthly_p90"]       = df["month"].map(monthly_p90.to_dict())
    df["discharge_anomaly"] = df["discharge_cfs"] - df["monthly_p90"]

    return df


def write_schema(df: pd.DataFrame, path: Path):
    schema = f"""# EcoTrust Dataset Schema

## Ground Truth
- `thermal_stress`: 1 if water_temp_c > 18°C (salmon thermal stress threshold)
- `high_migration_risk`: 1 if thermal_stress=1 AND month in Aug–Nov
- `water_temp_c`: **EXCLUDED from all agent input features** — target only

## Weather Agent Features (NOAA)
{', '.join(['TMAX', 'TMIN', 'PRCP', 'AWND', 'SNOW', 'month', 'day_of_year'])}

## Hydrology Agent Features (USGS — NO water_temp_c)
{', '.join(['discharge_cfs', 'gage_height_ft', 'discharge_7d_mean', 'discharge_anomaly', 'month', 'day_of_year'])}

## Environmental Predictor Features
TMAX, TMIN, PRCP, AWND, discharge_cfs, gage_height_ft,
discharge_7d_mean, discharge_anomaly

## Dataset Statistics
- Date range:  {df['date'].min().date()} to {df['date'].max().date()}
- Total rows:  {len(df):,}
- Thermal stress days: {df['thermal_stress'].sum():,} ({df['thermal_stress'].mean()*100:.1f}%)
- High migration risk days: {df['high_migration_risk'].sum():,} ({df['high_migration_risk'].mean()*100:.1f}%)
- Migration season rows: {df['migration_season'].sum():,}
- Missing water_temp_c: {df['water_temp_c'].isna().sum():,} days
"""
    path.write_text(schema)
    print(schema)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Columbia River: has water_temp_c (1996-2024, 66% coverage)
    # NOAA: Portland Int'l Airport (columbia proxy)
    print("Building merged dataset (Columbia River) ...")
    df = load_and_align("columbia", "columbia_river")

    # Drop rows where ground truth is missing (no temp measurement)
    n_before = len(df)
    df = df.dropna(subset=["water_temp_c"])
    print(f"Dropped {n_before - len(df)} rows with missing water_temp_c")

    out_path = OUT_DIR / "columbia_merged.csv"   # Columbia River at The Dalles (USGS 14105700)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")

    write_schema(df, OUT_DIR / "dataset_schema.md")


if __name__ == "__main__":
    main()
