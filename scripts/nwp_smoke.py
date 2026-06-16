"""One-day ERA5 smoke test: download 2023-01-15 over Iberia, extract LEMD points,
store to nwp_point, and print a comparison against that day's known fog."""

import cdsapi

from wx.config import ERA5_DIR, settings
from wx.db import repositories as repo
from wx.db.connection import get_connection
from wx.ingestion.nwp_era5 import ERA5_VARIABLES, extract_points, load_dataset

ERA5_DIR.mkdir(parents=True, exist_ok=True)
target = ERA5_DIR / "era5_iberia_test_20230115.nc"

if not target.exists():
    n, w, s, e = settings.era5_area
    print("Requesting ERA5 2023-01-15 (CDS queue)…")
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": list(ERA5_VARIABLES.keys()),
            "year": "2023", "month": "01", "day": "15",
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [n, w, s, e],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
print("Downloaded:", target, f"({target.stat().st_size/1e6:.1f} MB)")

ds = load_dataset(target)
if True:
    print("vars:", list(ds.data_vars))
    with get_connection() as con:
        stations = [dict(zip(("icao", "lat", "lon"), r))
                    for r in con.execute("SELECT icao, lat, lon FROM stations").fetchall()]
        recs = extract_points(ds, stations)
        n_ins = repo.store_nwp_points(con, recs)
        print(f"extracted {len(recs)} rows, stored {n_ins}")
        print("\nLEMD ERA5 vs observed (fog morning):")
        for r in con.execute("""
            SELECT valid_time, round(wind10m_spd,1) spd, round(wind10m_dir,0) dir,
                   round(t2m_c,1) t2m, round(d2m_c,1) d2m, round(tcc,2) tcc,
                   cbh_m, round(tp_mm,2) tp
            FROM nwp_point WHERE icao='LEMD' ORDER BY valid_time
            LIMIT 12""").fetchall():
            print("  ", r)
