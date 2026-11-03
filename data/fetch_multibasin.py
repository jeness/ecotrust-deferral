"""Fetch and build provenance-safe USGS multi-basin transfer datasets.

Station names and coordinates are read from the USGS response, never typed into
the result table.  This prevents a site-number/name mismatch from silently
changing the scientific interpretation of a transfer experiment.

Outputs
-------
data/multibasin/usgs/<site_no>.csv
data/multibasin/weather/<site_no>.csv
data/merged/multibasin/<site_no>.csv
data/multibasin/station_metadata.json
"""

from __future__ import annotations

import json
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "multibasin"
USGS_DIR = RAW_DIR / "usgs"
WEATHER_DIR = RAW_DIR / "weather"
MERGED_DIR = DATA_DIR / "merged" / "multibasin"
META_PATH = RAW_DIR / "station_metadata.json"

START_DATE = "1990-01-01"
END_DATE = "2024-12-31"

# The training basin (14105700) is intentionally absent.  These ten sites span
# Oregon, Washington, and Idaho and have daily water-temperature and discharge
# records.  Slugs are filenames only; scientific labels come from USGS.
SITES = {
    "14150000": "middle_fork_willamette",
    "14316500": "north_umpqua_river",
    "13317000": "salmon_river",
    "12500450": "yakima_river",
    "14162500": "mckenzie_river",
    "14339000": "rogue_river",
    "12181000": "skagit_river",
    "12210700": "nooksack_river",
    "13340000": "clearwater_river",
    "13269000": "snake_river",
}

PARAM_NAMES = {
    "00010": "water_temp_c",
    "00060": "discharge_cfs",
    "00065": "gage_height_ft",
}


def _get(url, params):
    for attempt in range(6):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=120,
                headers={"User-Agent": "EcoTrust-research/1.0",
                         "Accept-Encoding": "identity"},
            )
        except requests.RequestException as exc:
            if attempt == 5:
                raise
            wait = 10.0 * (attempt + 1)
            print(f"  {type(exc).__name__}; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 5:
            response.raise_for_status()
        wait = max(float(response.headers.get("Retry-After", 0)), 10.0 * (attempt + 1))
        print(f"  HTTP {response.status_code}; retrying in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_usgs(site_no):
    cached_path = USGS_DIR / f"{site_no}.csv"
    if cached_path.exists():
        df = pd.read_csv(cached_path, parse_dates=["date"])
        if {"water_temp_c", "discharge_cfs"}.issubset(df.columns):
            merged_path = MERGED_DIR / f"{site_no}.csv"
            if merged_path.exists():
                station = pd.read_csv(
                    merged_path, usecols=["usgs_station_name"], nrows=1
                ).iloc[0, 0]
                meta = {"site_no": site_no, "slug": SITES[site_no],
                        "station_name": station, "latitude": None,
                        "longitude": None,
                        "usgs_url": f"https://waterdata.usgs.gov/nwis/inventory/?site_no={site_no}"}
            else:
                meta = fetch_station_metadata(site_no)
            return _add_usgs_counts(df, meta)

    frames = []
    meta = None
    for first_year in range(1990, 2025, 5):
        last_year = min(first_year + 4, 2024)
        response = _get(
            "https://waterservices.usgs.gov/nwis/dv/",
            {
                "format": "json", "sites": site_no,
                "parameterCd": ",".join(PARAM_NAMES),
                "startDT": f"{first_year}-01-01",
                "endDT": f"{last_year}-12-31",
                "statCd": "00003", "siteStatus": "all",
            },
        )
        series = response.json().get("value", {}).get("timeSeries", [])
        if meta is None and series:
            meta = _metadata_from_source(site_no, series[0]["sourceInfo"])
        chunk = _parse_usgs_series(series)
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        raise RuntimeError(f"USGS returned no daily series for {site_no}")
    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df = df.groupby("date", as_index=False).first()
    if not {"water_temp_c", "discharge_cfs"}.issubset(df.columns):
        raise RuntimeError(f"{site_no} lacks required temperature/discharge series")
    return _add_usgs_counts(df, meta)


def _metadata_from_source(site_no, source):
    site_codes = source.get("siteCode", [])
    returned_site = next(
        (x["value"] for x in site_codes if x.get("agencyCode") == "USGS"),
        site_codes[0]["value"],
    )
    if returned_site != site_no:
        raise RuntimeError(f"requested {site_no}, USGS returned {returned_site}")
    geo = source["geoLocation"]["geogLocation"]
    return {"site_no": site_no, "slug": SITES[site_no],
            "station_name": source["siteName"],
            "latitude": float(geo["latitude"]),
            "longitude": float(geo["longitude"]),
            "usgs_url": f"https://waterdata.usgs.gov/nwis/inventory/?site_no={site_no}"}


def fetch_station_metadata(site_no):
    response = _get(
        "https://waterservices.usgs.gov/nwis/site/",
        {"format": "rdb", "sites": site_no, "siteOutput": "expanded",
         "siteStatus": "all"},
    )
    frame = pd.read_csv(io.StringIO(response.text), sep="\t", comment="#", dtype=str)
    frame = frame[frame["agency_cd"] == "USGS"]
    if len(frame) != 1 or frame.iloc[0]["site_no"] != site_no:
        raise RuntimeError(f"could not resolve unique USGS metadata for {site_no}")
    row = frame.iloc[0]
    meta = {
        "site_no": site_no,
        "slug": SITES[site_no],
        "station_name": row["station_nm"],
        "latitude": float(row["dec_lat_va"]),
        "longitude": float(row["dec_long_va"]),
        "usgs_url": f"https://waterdata.usgs.gov/nwis/inventory/?site_no={site_no}",
    }
    return meta


def _parse_usgs_series(series):
    columns = {}
    for ts in series:
        code = ts["variable"]["variableCode"][0]["value"]
        if code not in PARAM_NAMES:
            continue
        values = ts.get("values", [{}])[0].get("value", [])
        s = pd.Series(
            {v["dateTime"][:10]: pd.to_numeric(v["value"], errors="coerce")
             for v in values},
            name=PARAM_NAMES[code],
            dtype=float,
        )
        s[s <= -999998] = np.nan
        # Defensive combine: a service can expose more than one matching time
        # series under different internal identifiers.
        columns[s.name] = s if s.name not in columns else columns[s.name].combine_first(s)
    if not columns:
        return pd.DataFrame()
    df = pd.concat(columns.values(), axis=1).rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _add_usgs_counts(df, meta):
    df = df.sort_values("date").drop_duplicates("date")
    meta.update({
        "usgs_start": df["date"].min().date().isoformat(),
        "usgs_end": df["date"].max().date().isoformat(),
        "n_usgs_days": int(len(df)),
        "n_temperature_days": int(df["water_temp_c"].notna().sum()),
        "n_discharge_days": int(df["discharge_cfs"].notna().sum()),
    })
    return df, meta


def fetch_weather(latitude, longitude):
    # Match the Columbia training data units exactly: Celsius, millimetres,
    # and metres/second.  One request per site also stays within API limits.
    response = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max"
            ),
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "wind_speed_unit": "ms",
            "timezone": "America/Los_Angeles",
        },
    )
    daily = response.json()["daily"]
    weather = pd.DataFrame({
        "date": daily["time"],
        "TMAX": daily["temperature_2m_max"],
        "TMIN": daily["temperature_2m_min"],
        "PRCP": daily["precipitation_sum"],
        "AWND": daily["wind_speed_10m_max"],
    })
    weather["date"] = pd.to_datetime(weather["date"])
    return weather.sort_values("date").drop_duplicates("date")


def build_merged(usgs, weather, meta):
    df = weather.merge(usgs, on="date", how="inner").sort_values("date")
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["year"] = df["date"].dt.year
    if "gage_height_ft" not in df:
        df["gage_height_ft"] = 0.0
    else:
        df["gage_height_ft"] = df["gage_height_ft"].fillna(0.0)
    for col in ["AWND", "PRCP", "TMAX", "TMIN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
    df["discharge_cfs"] = df["discharge_cfs"].fillna(
        df["discharge_cfs"].rolling(7, min_periods=1).median()
    )
    df["discharge_7d_mean"] = df["discharge_cfs"].rolling(7, min_periods=1).mean()
    monthly_p90 = df.groupby("month")["discharge_cfs"].quantile(0.9)
    df["monthly_p90"] = df["month"].map(monthly_p90)
    df["discharge_anomaly"] = df["discharge_cfs"] - df["monthly_p90"]

    df["thermal_stress"] = (df["water_temp_c"] > 18.0).astype(int)
    df["migration_season"] = df["month"].isin([8, 9, 10, 11]).astype(int)
    df["high_migration_risk"] = (
        (df["thermal_stress"] == 1) & (df["migration_season"] == 1)
    ).astype(int)
    df["usgs_site_no"] = meta["site_no"]
    df["usgs_station_name"] = meta["station_name"]
    df = df.dropna(subset=["water_temp_c"])

    season = df[df["migration_season"] == 1]
    test = season[season["year"] >= 2015]
    meta.update({
        "n_merged_temperature_days": int(len(df)),
        "n_migration_days": int(len(season)),
        "n_test_migration_days": int(len(test)),
        "test_prevalence": round(float(test["high_migration_risk"].mean()), 4)
        if len(test) else None,
    })
    return df


def main():
    for directory in [USGS_DIR, WEATHER_DIR, MERGED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    metadata = []
    for site_no in SITES:
        print(f"[{site_no}] fetching USGS", flush=True)
        usgs, meta = fetch_usgs(site_no)
        print(f"  {meta['station_name']}", flush=True)
        weather_path = WEATHER_DIR / f"{site_no}.csv"
        if not weather_path.exists() and meta["latitude"] is None:
            # Cached hydrology/merged data can reconstruct the official name,
            # but a missing weather cache still needs official coordinates.
            counts = {k: v for k, v in meta.items() if k.startswith("n_") or
                      k.startswith("usgs_")}
            meta = fetch_station_metadata(site_no)
            meta.update(counts)
        if weather_path.exists():
            weather = pd.read_csv(weather_path, parse_dates=["date"])
            # A previous draft accidentally requested Fahrenheit.  Never trust
            # that cache: PNW daily mean maximums in Celsius stay far below 50.
            if weather["TMAX"].mean() >= 50 or len(weather) < 12000:
                weather = fetch_weather(meta["latitude"], meta["longitude"])
        else:
            weather = fetch_weather(meta["latitude"], meta["longitude"])
        merged = build_merged(usgs, weather, meta)
        usgs.to_csv(USGS_DIR / f"{site_no}.csv", index=False)
        weather.to_csv(weather_path, index=False)
        merged.to_csv(MERGED_DIR / f"{site_no}.csv", index=False)
        metadata.append(meta)
        print(
            f"  temperature={meta['n_temperature_days']}, "
            f"test-season={meta['n_test_migration_days']}, "
            f"prevalence={meta['test_prevalence']}",
            flush=True,
        )
    with open(META_PATH, "w") as f:
        json.dump({
            "source": "USGS NWIS daily values and Open-Meteo archive",
            "fetched_through": END_DATE,
            "weather_units": {"TMAX": "degC", "TMIN": "degC",
                              "PRCP": "mm", "AWND": "m/s"},
            "sites": metadata,
        }, f, indent=2)
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
