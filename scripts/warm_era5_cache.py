"""Warm the ERA5 timeseries cache (CDS -> data/era5) WITHOUT touching the DB, so it
runs in parallel with verify/compare. Uses the SAME (start, end) the `wx nwp` step
uses, so the cache keys match and the store step becomes near-instant."""

import sys
from datetime import datetime, timezone

from wx.config import AIRPORTS
from wx.ingestion.nwp_era5 import download_station_timeseries

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 12, 31, tzinfo=timezone.utc)   # matches wx nwp: t1 - 1 day


def main() -> int:
    n = len(AIRPORTS)
    print(f"warming ERA5 timeseries for {n} stations {START:%Y-%m-%d}..{END:%Y-%m-%d}", flush=True)
    for i, a in enumerate(AIRPORTS, 1):
        try:
            path = download_station_timeseries(a.icao, a.lat, a.lon, START, END)
            size = path.stat().st_size / 1e6
            print(f"[{i}/{n}] {a.icao}: ok ({size:.1f} MB)", flush=True)
        except Exception as e:
            print(f"[{i}/{n}] {a.icao}: FAILED {type(e).__name__}: {e}", flush=True)
    print("ERA5 cache warming complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
