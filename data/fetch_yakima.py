"""
Fetch Yakima River cross-watershed validation data.

Watershed: Yakima River at Parker, WA
- USGS 12500450: discharge + water temperature (ground truth)
- Open-Meteo API: weather features (TMAX, TMIN, PRCP, AWND) — no token needed

Output files:
  data/usgs/yakima_river_daily.csv   (discharge_cfs, water_temp_c, gage_height_ft)
  data/noaa/yakima_daily.csv         (TMAX, TMIN, PRCP, AWND)
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_USGS = DATA_DIR / "usgs"
OUT_NOAA = DATA_DIR / "noaa"
OUT_USGS.mkdir(parents=True, exist_ok=True)
OUT_NOAA.mkdir(parents=True, exist_ok=True)

# McKenzie River near Vida, OR — cross-watershed validation site
# Site 14150000: excellent temperature coverage (>99%), important Chinook habitat
# Genuinely different watershed from Columbia River (different climate, elevation, drainage)
YAKIMA_USGS_SITE = "14150000"
YAKIMA_LAT = 44.127    # near Vida, OR (McKenzie watershed)
YAKIMA_LON = -122.457
START_DATE = "1996-01-01"
END_DATE   = "2024-12-31"


# ── USGS NWIS fetch (no API key) ─────────────────────────────────────────────

def fetch_usgs_chunk(site_no: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one date-range chunk from USGS NWIS."""
    url = "https://waterservices.usgs.gov/nwis/dv"
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": "00060,00010,00065",
        "startDT": start,
        "endDT": end,
        "statCd": "00003",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()

    param_map = {
        "00060": "discharge_cfs",
        "00010": "water_temp_c",
        "00065": "gage_height_ft",
    }
    time_series = body.get("value", {}).get("timeSeries", [])
    if not time_series:
        return pd.DataFrame()

    frames = []
    for ts in time_series:
        var_code = ts["variable"]["variableCode"][0]["value"]
        col = param_map.get(var_code, var_code)
        vals = ts["values"][0]["value"]
        df = pd.DataFrame(vals)[["dateTime", "value"]]
        df["date"] = pd.to_datetime(df["dateTime"]).dt.date.astype(str)
        df[col] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df[["date", col]])

    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


def fetch_usgs(site_no: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily streamflow + water temperature from USGS NWIS, in 5-year chunks."""
    print(f"  Fetching USGS site {site_no} ({start} to {end}) in 5-year chunks...")
    start_yr = int(start[:4])
    end_yr   = int(end[:4])
    chunks   = []

    for yr_from in range(start_yr, end_yr + 1, 5):
        yr_to = min(yr_from + 4, end_yr)
        s = f"{yr_from}-01-01"
        e = f"{yr_to}-12-31"
        print(f"    {s} → {e} ...", end=" ", flush=True)
        try:
            chunk = fetch_usgs_chunk(site_no, s, e)
            chunks.append(chunk)
            print(f"{len(chunk)} rows")
        except Exception as ex:
            print(f"FAILED: {ex}")
        time.sleep(0.3)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    result = result.groupby("date").first().reset_index()
    return result.sort_values("date").reset_index(drop=True)


# ── Open-Meteo weather fetch (no API key) ────────────────────────────────────

def fetch_openmeteo_chunk(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo Historical API (free, no token)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily": ",".join([
            "temperature_2m_max",     # TMAX
            "temperature_2m_min",     # TMIN
            "precipitation_sum",      # PRCP
            "wind_speed_10m_max",     # AWND proxy
        ]),
        "timezone": "America/Los_Angeles",
        "wind_speed_unit": "ms",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    daily = body["daily"]
    df = pd.DataFrame({
        "date":  daily["time"],
        "TMAX":  daily["temperature_2m_max"],
        "TMIN":  daily["temperature_2m_min"],
        "PRCP":  daily["precipitation_sum"],
        "AWND":  daily["wind_speed_10m_max"],
        "SNOW":  0.0,    # placeholder (Open-Meteo doesn't provide snowfall easily)
    })
    return df


def fetch_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Open-Meteo limits to ~20 years per request; split by decade if needed.
    """
    start_yr = int(start[:4])
    end_yr   = int(end[:4])
    chunks = []

    print(f"  Fetching Open-Meteo weather ({lat}°N, {lon}°E) ...")
    for yr_from in range(start_yr, end_yr + 1, 10):
        yr_to = min(yr_from + 9, end_yr)
        s = f"{yr_from}-01-01"
        e = f"{yr_to}-12-31"
        print(f"    {s} → {e} ...", end=" ", flush=True)
        try:
            chunk = fetch_openmeteo_chunk(lat, lon, s, e)
            chunks.append(chunk)
            print(f"{len(chunk)} rows")
        except Exception as ex:
            print(f"FAILED: {ex}")
        time.sleep(0.5)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("date").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Yakima River Cross-Watershed Data Fetch ===\n")

    # 1. USGS hydrology
    usgs_path = OUT_USGS / "mckenzie_river_daily.csv"
    if usgs_path.exists():
        print(f"USGS file exists: {usgs_path} — skipping fetch")
        usgs_df = pd.read_csv(usgs_path)
    else:
        usgs_df = fetch_usgs(YAKIMA_USGS_SITE, START_DATE, END_DATE)
        if usgs_df.empty:
            print("ERROR: Failed to fetch USGS data for Yakima River!")
            return
        usgs_df.to_csv(usgs_path, index=False)
        print(f"  Saved {len(usgs_df)} rows → {usgs_path}")

    print(f"\nUSGS data summary:")
    print(f"  Rows: {len(usgs_df)}")
    print(f"  Date range: {usgs_df['date'].min()} to {usgs_df['date'].max()}")
    if "water_temp_c" in usgs_df.columns:
        n_temp = usgs_df["water_temp_c"].notna().sum()
        print(f"  water_temp_c coverage: {n_temp}/{len(usgs_df)} ({n_temp/len(usgs_df)*100:.1f}%)")
        above18 = (usgs_df["water_temp_c"] > 18).sum()
        print(f"  Days >18°C: {above18} ({above18/n_temp*100:.1f}% of temp-available days)")
    else:
        print("  WARNING: No water_temp_c column found!")

    # 2. Open-Meteo weather
    noaa_path = OUT_NOAA / "mckenzie_daily.csv"
    if noaa_path.exists():
        print(f"\nWeather file exists: {noaa_path} — skipping fetch")
        noaa_df = pd.read_csv(noaa_path)
    else:
        print()
        noaa_df = fetch_openmeteo(YAKIMA_LAT, YAKIMA_LON, START_DATE, END_DATE)
        if noaa_df.empty:
            print("ERROR: Failed to fetch Open-Meteo weather data!")
            return
        noaa_df.to_csv(noaa_path, index=False)
        print(f"  Saved {len(noaa_df)} rows → {noaa_path}")

    print(f"\nWeather data summary:")
    print(f"  Rows: {len(noaa_df)}")
    print(f"  TMAX range: {noaa_df['TMAX'].min():.1f} to {noaa_df['TMAX'].max():.1f} °C")
    print(f"  PRCP nulls: {noaa_df['PRCP'].isna().sum()}")

    print("\n✓ Yakima River data ready for cross-watershed validation.")
    print("  Next: run python code-artifacts/data/build_cross_watershed.py")


if __name__ == "__main__":
    main()
