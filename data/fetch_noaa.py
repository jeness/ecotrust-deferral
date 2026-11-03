"""
Fetch daily weather data from NOAA Climate Data Online (CDO) API.
Requires a free API token: https://www.ncdc.noaa.gov/cdo-web/token

Usage:
    export NOAA_TOKEN=your_token_here
    python fetch_noaa.py
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
TOKEN = os.environ.get("NOAA_TOKEN", "")

# Washington State stations near salmon watersheds
STATIONS = {
    "issaquah": "GHCND:USW00094290",   # Seattle-Tacoma Intl Airport (proxy)
    "columbia":  "GHCND:USW00024157",  # Portland Int'l Airport (Columbia basin proxy)
}

DATATYPES = ["TMAX", "TMIN", "PRCP", "AWND", "SNOW"]
START_DATE = "1990-01-01"
END_DATE   = "2024-12-31"
OUT_DIR    = Path(__file__).parent.parent / "data" / "noaa"


def fetch_year(station_id: str, year: int, headers: dict) -> list:
    """Fetch one calendar year of data (CDO API max range is 1 year)."""
    all_records = []
    offset = 1
    limit = 1000
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": ",".join(DATATYPES),
            "startdate": start,
            "enddate": end,
            "limit": limit,
            "offset": offset,
            "units": "metric",
        }
        r = requests.get(f"{NOAA_BASE}/data", headers=headers, params=params)
        r.raise_for_status()
        body = r.json()
        results = body.get("results", [])
        if not results:
            break
        all_records.extend(results)
        total = body.get("metadata", {}).get("resultset", {}).get("count", 0)
        offset += limit
        if offset > total:
            break
        time.sleep(0.25)

    return all_records


def fetch_station(station_id: str, start: str, end: str) -> pd.DataFrame:
    headers = {"token": TOKEN}
    start_year = int(start[:4])
    end_year   = int(end[:4])
    all_records = []

    for year in range(start_year, end_year + 1):
        print(f"    year {year} ...", end=" ", flush=True)
        try:
            records = fetch_year(station_id, year, headers)
            all_records.extend(records)
            print(f"{len(records)} records")
        except Exception as e:
            print(f"SKIP ({e})")
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
    df.columns.name = None
    df = df.reset_index()
    return df


def fetch_station_incremental(station_id: str, out_path: Path) -> None:
    """Fetch year by year, appending to CSV after each year (crash-safe)."""
    headers = {"token": TOKEN}

    # Determine which years we already have
    existing_years = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["date"])
        existing_years = set(pd.to_datetime(existing["date"]).dt.year.unique())
        print(f"  Already have years: {sorted(existing_years)}")

    start_year = int(START_DATE[:4])
    end_year   = int(END_DATE[:4])

    for year in range(start_year, end_year + 1):
        if year in existing_years:
            print(f"    year {year} ... skipped (already saved)")
            continue

        print(f"    year {year} ...", end=" ", flush=True)
        try:
            records = fetch_year(station_id, year, headers)
            if not records:
                print("0 records")
                continue

            df_year = pd.DataFrame(records)
            df_year["date"] = pd.to_datetime(df_year["date"]).dt.date
            df_year = df_year.pivot_table(
                index="date", columns="datatype", values="value", aggfunc="mean"
            )
            df_year.columns.name = None
            df_year = df_year.reset_index()

            # Append to CSV (write header only if file doesn't exist yet)
            write_header = not out_path.exists()
            df_year.to_csv(out_path, mode="a", header=write_header, index=False)
            print(f"{len(df_year)} rows saved")

        except Exception as e:
            print(f"SKIP ({e})")

        time.sleep(0.5)


def main():
    if not TOKEN:
        raise EnvironmentError("Set NOAA_TOKEN environment variable first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, station_id in STATIONS.items():
        out_path = OUT_DIR / f"{name}_daily.csv"
        print(f"Fetching {name} ({station_id}) -> {out_path}")
        fetch_station_incremental(station_id, out_path)
        total = sum(1 for _ in open(out_path)) - 1  # subtract header
        print(f"  Done. Total rows on disk: {total}")


if __name__ == "__main__":
    main()
