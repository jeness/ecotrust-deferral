"""
Download salmon escapement / fish count data from Washington Dept of Fish & Wildlife (WDFW).

WDFW publishes adult salmon escapement data via their SalmonScan portal and direct CSV exports.
This script downloads and preprocesses the publicly available data.

Usage:
    python fetch_wdfw.py
"""

import io
import requests
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "fish_counts"

# WDFW Salmon & Steelhead Stock Inventory (SASSI) — public CSV export
# Alternate: WDFW StreamNet data portal (available via StreamNet API below)
STREAMNET_BASE = "https://www.streamnet.org/api"

# Direct CSV links for Puget Sound Chinook escapement (WDFW annual reports)
WDFW_SOURCES = {
    # WDFW Salmonid Stock Inventory — Puget Sound escapement data
    # Resource ID from data.wa.gov (Salmonid Stock Inventory SASSI)
    "issaquah_chinook": (
        "https://data.wa.gov/resource/9s8d-ykbv.csv"
        "?$where=waterbodyname=%27Issaquah%20Creek%27&$limit=50000"
    ),
    # Fallback: StreamNet adult escapement (Puget Sound Chinook)
    "streamnet_fallback": (
        "https://www.streamnet.org/api/v1/escapement"
        "?species=Chinook&huc=17110019&format=csv"
    ),
}

# Fallback: StreamNet Adult Returns API (requires registration for bulk, but small queries work)
STREAMNET_QUERY = (
    "https://www.cbfish.org/api/Escapement"
    "?species=Chinook&run=fall&location=Issaquah+Creek&format=csv"
)


def fetch_wdfw_data() -> pd.DataFrame:
    """Try to pull from WDFW open data portal."""
    url = WDFW_SOURCES["issaquah_chinook"]
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as e:
        print(f"  WDFW portal fetch failed: {e}")
        return pd.DataFrame()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Attempting to fetch WDFW fish count data ...")
    df = fetch_wdfw_data()

    if df.empty:
        raise RuntimeError(
            "WDFW download failed; no synthetic fallback is permitted. "
            "Provide a verified source file before continuing."
        )

    out_path = OUT_DIR / "issaquah_salmon_counts.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    print()
    print("NOTE: For actual WDFW data, download manually from:")
    print("  https://wdfw.wa.gov/fishing/management/salmon-steelhead/escapement")
    print("  or request via WDFW data portal: https://data.wa.gov/")


if __name__ == "__main__":
    main()
