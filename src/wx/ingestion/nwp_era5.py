"""ERA5 single-levels ingester (Copernicus CDS, free).

Downloads per-year NetCDF granules over the Iberia box, then extracts the
nearest grid point per airport into ``nwp_point``. The gridded NetCDF stays on
disk under ``data/era5`` (kept out of DuckDB).

Requires ``cdsapi`` credentials in ``~/.cdsapirc`` for downloads; the extraction
step works on any xarray Dataset, so it is testable without CDS access.

Documented caveats (see docs/PLAN.md): ERA5 cloud-base height is biased and
there is no reliable surface-visibility field — visibility is verified on the
METAR/TAF side only.
"""

from __future__ import annotations

import tempfile
import zipfile
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wx.config import ERA5_DIR, settings

DATASET = "reanalysis-era5-single-levels"

# CDS request variable name -> NetCDF short name we read back.
ERA5_VARIABLES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "10m_wind_gust_since_previous_post_processing": "fg10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_cloud_cover": "tcc",
    "low_cloud_cover": "lcc",
    "medium_cloud_cover": "mcc",
    "high_cloud_cover": "hcc",
    "cloud_base_height": "cbh",
    "total_precipitation": "tp",
    "mean_sea_level_pressure": "msl",
}

MS_TO_KT = 1.94384
M_TO_FT = 3.28084


def download_year(year: int, out_dir: Path | None = None) -> Path:
    """Download one calendar year of ERA5 single-levels over Iberia. Returns path."""
    import cdsapi

    out_dir = out_dir or ERA5_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"era5_iberia_{year}.nc"
    if target.exists():
        return target

    n, w, s, e = settings.era5_area
    client = cdsapi.Client()
    client.retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "variable": list(ERA5_VARIABLES.keys()),
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [n, w, s, e],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    return target


def load_dataset(path: Path) -> xr.Dataset:
    """Open an ERA5 download as a single Dataset.

    The new CDS infrastructure returns a ZIP of separate NetCDFs (instantaneous
    vs accumulated streams) even when ``unarchived`` is requested. Detect that,
    extract, and merge the members; otherwise open the NetCDF directly.
    """
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path).load()

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            zf.extractall(tmp)
        datasets = [xr.open_dataset(Path(tmp) / m).load() for m in members]
    # Streams share lat/lon/valid_time; override resolves any attr/coord conflicts.
    return xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")


def _wind_speed_dir(u, v):
    """Speed (m/s) and meteorological 'from' direction (deg) from u/v components."""
    spd = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v))) % 360.0
    return spd, direction


def extract_points(ds: xr.Dataset, stations: list[dict]) -> list[dict]:
    """Nearest-gridpoint time series per station, with unit conversions applied.

    ``stations`` items need ``icao``, ``lat``, ``lon``. The Dataset must expose
    ``latitude``/``longitude`` coords and a time coord ('time' or 'valid_time').
    """
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    times = pd.to_datetime(ds[time_name].values)

    def col(short):
        return ds[short] if short in ds else None

    records: list[dict] = []
    for st in stations:
        pt = ds.sel(latitude=st["lat"], longitude=st["lon"], method="nearest")

        def series(short):
            da = col(short)
            return None if da is None else pt[short].values

        u, v = series("u10"), series("v10")
        spd = direction = None
        if u is not None and v is not None:
            spd, direction = _wind_speed_dir(u, v)

        def conv(short, fn):
            vals = series(short)
            return None if vals is None else fn(vals)

        n = len(times)

        def get(arr, i):
            return None if arr is None else _clean(arr[i])

        gust_short = "fg10" if "fg10" in ds else "i10fg"  # new vs legacy CDS name
        gust = conv(gust_short, lambda a: a * MS_TO_KT)
        t2m = conv("t2m", lambda a: a - 273.15)
        d2m = conv("d2m", lambda a: a - 273.15)
        tcc, lcc, mcc, hcc = (series(s) for s in ("tcc", "lcc", "mcc", "hcc"))
        cbh = series("cbh")  # stored as metres (cbh_m)
        tp = conv("tp", lambda a: a * 1000.0)
        msl = conv("msl", lambda a: a / 100.0)
        spd_kt = None if spd is None else spd * MS_TO_KT

        for i in range(n):
            records.append(
                {
                    "icao": st["icao"],
                    "valid_time": times[i].to_pydatetime().replace(tzinfo=timezone.utc),
                    "source": "era5",
                    "wind10m_spd": get(spd_kt, i),
                    "wind10m_dir": get(direction, i),
                    "gust": get(gust, i),
                    "t2m_c": get(t2m, i),
                    "d2m_c": get(d2m, i),
                    "tcc": get(tcc, i),
                    "lcc": get(lcc, i),
                    "mcc": get(mcc, i),
                    "hcc": get(hcc, i),
                    "cbh_m": get(cbh, i),
                    "tp_mm": get(tp, i),
                    "mslp_hpa": get(msl, i),
                }
            )
    return records


def _clean(v):
    """NaN/masked -> None; numpy scalar -> python float."""
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f
