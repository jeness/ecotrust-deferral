"""
Fetch stream discharge and water temperature from USGS NWIS (no API token needed).

Usage:
    python fetch_usgs.py
"""

import requests
import pandas as pd
from pathlib import Path

NWIS_BASE = "https://waterservices.usgs.gov/nwis/dv"

# Site numbers for target watersheds
SITES = {
    "issaquah_creek": "12121600",   # Issaquah Creek near Issaquah, WA
    "columbia_river":  "14105700",  # Columbia River at The Dalles, OR
    "puget_sound_snoqualmie": "12144500",  # Snoqualmie River near Monroe, WA
}

# USGS parameter codes
PARAMS = {
    "00060": "discharge_cfs",       # Streamflow (cubic feet per second)
    "00010": "water_temp_c",        # Water temperature (Celsius)
    "00065": "gage_height_ft",      # Gage height (feet)
}

START_DATE = "1990-01-01"
END_DATE   = "2024-12-31"
OUT_DIR    = Path(__file__).parent.parent / "data" / "usgs"


def fetch_site(site_no: str, param_codes: list, start: str, end: str) -> pd.DataFrame:
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": ",".join(param_codes),
        "startDT": start,
        "endDT": end,
        "statCd": "00003",   # daily mean
    }
    r = requests.get(NWIS_BASE, params=params)
    r.raise_for_status()
    body = r.json()

    time_series = body.get("value", {}).get("timeSeries", [])
    if not time_series:
        return pd.DataFrame()

    frames = []
    for ts in time_series:
        var_code = ts["variable"]["variableCode"][0]["value"]
        col_name = PARAMS.get(var_code, var_code)
        values = ts["values"][0]["value"]
        df = pd.DataFrame(values)[["dateTime", "value"]]
        df["date"] = pd.to_datetime(df["dateTime"]).dt.date
        df[col_name] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df[["date", col_name]])

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    param_codes = list(PARAMS.keys())

    for name, site_no in SITES.items():
        print(f"Fetching {name} (site {site_no}) ...")
        df = fetch_site(site_no, param_codes, START_DATE, END_DATE)
        if df.empty:
            print(f"  No data for {name}")
            continue
        out_path = OUT_DIR / f"{name}_daily.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
