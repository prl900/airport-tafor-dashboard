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
# ARCO/Zarr timeseries dataset: one point, long period, selected variables.
# It serves a subset of parameters (no low/medium/high cloud split — only tcc);
# we request the full list and unavailable variables are simply absent.
TIMESERIES_DATASET = "reanalysis-era5-single-levels-timeseries"

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
    # Candidate predictors (assessed via `wx feature-importance` before operational use).
    "convective_available_potential_energy": "cape",
    "boundary_layer_height": "blh",
    "total_column_water_vapour": "tcwv",
    "skin_temperature": "skt",
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


def download_month(year: int, month: int, out_dir: Path | None = None) -> Path:
    """Download one calendar month of ERA5 single-levels over Iberia. Returns path.

    Monthly granules keep each CDS request under the per-request cost cap (a full year ×
    all variables × hourly is rejected with 'cost limits exceeded'). Cached per (year, month).
    """
    import cdsapi

    out_dir = out_dir or ERA5_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"era5_iberia_{year}{month:02d}.nc"
    if target.exists():
        return target

    n, w, s, e = settings.era5_area
    cdsapi.Client().retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "variable": list(ERA5_VARIABLES.keys()),
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [n, w, s, e],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    return target


def download_station_timeseries(
    icao: str, lat: float, lon: float, start, end, out_dir: Path | None = None
) -> Path:
    """Download the ERA5 point timeseries for one station over [start, end].

    One request returns the whole hourly series for the point — the efficient
    path for the 2020-2025 per-airport backfill (~one request per station vs
    downloading gridded full years). Cached per (icao, start, end).
    """
    import cdsapi

    out_dir = out_dir or ERA5_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"era5ts_{icao}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    if target.exists():
        return target

    cdsapi.Client().retrieve(
        TIMESERIES_DATASET,
        {
            "variable": list(ERA5_VARIABLES.keys()),
            "location": {"latitude": lat, "longitude": lon},
            "date": [f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"],
            "data_format": "netcdf",
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


def _point_records(pt: xr.Dataset, icao: str) -> list[dict]:
    """Build nwp_point records from a Dataset already reduced to a single point.

    Shared by the gridded path (after nearest-gridpoint .sel) and the timeseries
    path (already a point). Applies unit conversions and derives wind speed/dir.
    """
    time_name = "valid_time" if "valid_time" in pt.coords else "time"
    times = pd.to_datetime(pt[time_name].values)

    def series(short):
        return pt[short].values if short in pt else None

    def conv(short, fn):
        vals = series(short)
        return None if vals is None else fn(vals)

    def get(arr, i):
        return None if arr is None else _clean(arr[i])

    u, v = series("u10"), series("v10")
    spd = direction = None
    if u is not None and v is not None:
        spd, direction = _wind_speed_dir(u, v)
    spd_kt = None if spd is None else spd * MS_TO_KT

    gust_short = "fg10" if "fg10" in pt else "i10fg"  # new vs legacy CDS name
    gust = conv(gust_short, lambda a: a * MS_TO_KT)
    t2m = conv("t2m", lambda a: a - 273.15)
    d2m = conv("d2m", lambda a: a - 273.15)
    tcc, lcc, mcc, hcc = (series(s) for s in ("tcc", "lcc", "mcc", "hcc"))
    cbh = series("cbh")  # metres
    tp = conv("tp", lambda a: a * 1000.0)
    msl = conv("msl", lambda a: a / 100.0)
    cape, blh, tcwv = (series(s) for s in ("cape", "blh", "tcwv"))  # J/kg, m, kg/m^2
    skt = conv("skt", lambda a: a - 273.15)  # K -> C

    records = []
    for i in range(len(times)):
        vt = times[i].to_pydatetime().replace(tzinfo=timezone.utc)
        records.append(
            {
                "icao": icao,
                "valid_time": vt,
                "source": "era5",
                # ERA5 analysis is a degenerate zero-lead forecast (see nwp_point schema).
                "ref_time": vt,
                "step_h": 0,
                "wind10m_spd": get(spd_kt, i), "wind10m_dir": get(direction, i),
                "gust": get(gust, i), "t2m_c": get(t2m, i), "d2m_c": get(d2m, i),
                "tcc": get(tcc, i), "lcc": get(lcc, i), "mcc": get(mcc, i), "hcc": get(hcc, i),
                "cbh_m": get(cbh, i), "tp_mm": get(tp, i), "mslp_hpa": get(msl, i),
                "cape_jkg": get(cape, i), "blh_m": get(blh, i), "tcwv_kgm2": get(tcwv, i),
                "skt_c": get(skt, i),
            }
        )
    return records


def extract_points(ds: xr.Dataset, stations: list[dict]) -> list[dict]:
    """Gridded path: nearest-gridpoint time series per station (needs lat/lon dims)."""
    records: list[dict] = []
    for st in stations:
        pt = ds.sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
        records.extend(_point_records(pt, st["icao"]))
    return records


def extract_timeseries(ds: xr.Dataset, icao: str) -> list[dict]:
    """Timeseries path: the Dataset is already a single point (scalar lat/lon)."""
    return _point_records(ds, icao)


def _clean(v):
    """NaN/masked -> None; numpy scalar -> python float."""
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f
