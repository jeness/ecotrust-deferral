"""
Fetch North Umpqua River data for second cross-watershed validation.

Site: North Umpqua River near Toketee Falls, OR (USGS 14137000)
  - discharge, water temperature (ground truth)
  - Open-Meteo weather features

North Umpqua is distinctly different from both Columbia and McKenzie:
  - High-elevation Cascade Range headwaters (snowmelt + groundwater)
  - Much lower thermal stress prevalence (<18°C almost always)
  - Chinook salmon habitat in lower reaches
  - Smaller catchment area, faster flow response
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

# North Umpqua River near Toketee Falls — good temperature coverage
SITE_NO = "14137000"
LAT = 43.262   # near Toketee Falls, OR
LON = -122.428
START_DATE = "1996-01-01"
END_DATE   = "2024-12-31"

def fetch_usgs_chunk(site_no, start, end):
    url = "https://waterservices.usgs.gov/nwis/dv"
    params = {
        "format": "json", "sites": site_no,
        "parameterCd": "00060,00010,00065",
        "startDT": start, "endDT": end, "statCd": "00003",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    ts_list = body.get("value", {}).get("timeSeries", [])
    frames = []
    for ts in ts_list:
        var = ts["variable"]["variableCode"][0]["value"]
        desc = ts["variable"]["variableName"]
        rows = [{"date": v["dateTime"][:10], "value": float(v["value"])}
                for v in ts["values"][0]["value"]
                if v["value"] not in ("-999999", "")]
        if rows:
            df = pd.DataFrame(rows)
            df.columns = ["date", var]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")
    return result

def fetch_usgs():
    print("Fetching USGS data for North Umpqua...")
    chunks = []
    years = list(range(1996, 2025, 5))
    for i, y_start in enumerate(years):
        y_end = min(y_start + 4, 2024)
        start = f"{y_start}-01-01"
        end = f"{y_end}-12-31"
        print(f"  {start} - {end}")
        try:
            chunk = fetch_usgs_chunk(SITE_NO, start, end)
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as e:
            print(f"  Warning: {e}")
        time.sleep(0.5)
    
    if not chunks:
        raise RuntimeError("No USGS data fetched")
    
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    
    # Rename columns
    rename = {}
    for col in df.columns:
        if col == "00060": rename[col] = "discharge_cfs"
        elif col == "00010": rename[col] = "water_temp_c"
        elif col == "00065": rename[col] = "gage_height_ft"
    df = df.rename(columns=rename)
    
    print(f"  USGS: {len(df)} rows")
    if "water_temp_c" in df.columns:
        cov = df["water_temp_c"].notna().mean()
        print(f"  Water temp coverage: {cov:.1%}")
    
    out = OUT_USGS / "north_umpqua_daily.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return df

def fetch_weather():
    print("Fetching weather from Open-Meteo for North Umpqua...")
    dfs = []
    years = list(range(1996, 2025))
    for y in years:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": f"{y}-01-01", "end_date": f"{y}-12-31",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "temperature_unit": "celsius", "wind_speed_unit": "mph",
            "precipitation_unit": "inch", "timezone": "America/Los_Angeles",
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            d = r.json().get("daily", {})
            df = pd.DataFrame({
                "date": d.get("time", []),
                "TMAX": d.get("temperature_2m_max", []),
                "TMIN": d.get("temperature_2m_min", []),
                "PRCP": d.get("precipitation_sum", []),
                "AWND": d.get("wind_speed_10m_max", []),
            })
            # Convert Celsius to Fahrenheit
            if not df.empty:
                df["TMAX"] = df["TMAX"] * 9/5 + 32
                df["TMIN"] = df["TMIN"] * 9/5 + 32
            dfs.append(df)
        except Exception as e:
            print(f"  Warning {y}: {e}")
        time.sleep(0.3)
    
    weather = pd.concat(dfs, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").drop_duplicates("date")
    print(f"  Weather: {len(weather)} rows")
    out = OUT_NOAA / "north_umpqua_daily.csv"
    weather.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return weather

if __name__ == "__main__":
    fetch_usgs()
    fetch_weather()
    print("Done. Run build_north_umpqua.py next.")
