"""Historical IFS forecast ingester.

Unlike ERA5 (analysis: one value per valid hour), an IFS *forecast* is issued at a run
init time (``ref_time``) and carries many lead steps. Each step becomes one ``nwp_point``
row keyed by ``(ref_time, step_h)`` with ``valid_time = ref_time + step_h`` and
``source='ifs'`` — the run-anchored join in ``wx.ai.dataset`` then reconstructs, for each
(airport, T0, lead), the run initialized at/just-before T0.

The download path uses ``cdsapi`` (mirroring ``nwp_era5``); the **exact CDS dataset id is a
deployment parameter** (``--dataset``) because the free CDS does not host the operational
HRES archive — see docs/ML_PLAN.md / the transition plan for the source decision (CDS
reforecast vs TIGGE via the ECMWF API vs MARS). The extraction + de-accumulation below work
on any xarray Dataset with a step coordinate, so they are fully testable without downloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wx.config import ERA5_DIR, settings
from wx.ingestion.nwp_era5 import MS_TO_KT, _clean, _wind_speed_dir, load_dataset

# CDS dataset id for the historical IFS forecast archive. Intentionally unset: the caller
# must pass it (``--dataset``) once the source is confirmed, so we never silently hit the
# wrong catalogue. See the transition plan's "Confirm before starting" item #1.
IFS_DATASET: str | None = None

# Short names we read back from the forecast granule (same vocabulary as ERA5 so the
# extraction and the nwp_point columns line up 1:1). The CDS *request* parameter names
# depend on the chosen dataset and are supplied alongside --dataset.
IFS_SHORTNAMES = ("u10", "v10", "fg10", "t2m", "d2m", "tcc", "lcc", "mcc", "hcc",
                  "cbh", "tp", "msl", "cape", "blh", "tcwv", "skt")

# TIGGE (ECMWF API) surface parameters that EXIST in the archive, as GRIB param ids.
# Confirmed against confluence.ecmwf.int/display/TIGGE/Parameters. Notably ABSENT from
# TIGGE: 10m wind gust, low/medium/high cloud split (only tcc), cloud base height, and
# boundary-layer height — those features stay NULL for source='ifs' when TIGGE is used.
# 'tcw' (total column water, 136) stands in for tcwv (vapour-only); close moisture proxy.
TIGGE_PARAMS = {
    "165": "u10", "166": "v10", "167": "t2m", "168": "d2m", "151": "msl",
    "228228": "tp", "228164": "tcc", "59": "cape", "136": "tcwv", "235": "skt",
}

# Fields accumulated from forecast start (step 0). De-accumulated to per-interval values to
# match ERA5's hourly semantics. (gust/10fg is a max-since-previous field, handled as-is.)
ACCUMULATED = ("tp",)

IFS_DIR = ERA5_DIR.parent / "ifs"


# --- download -------------------------------------------------------------


def download_ifs(ref_time: datetime, steps: list[int], dataset: str | None = None,
                 area: tuple[float, float, float, float] | None = None,
                 out_dir: Path | None = None) -> Path:
    """Download one IFS run (init=``ref_time``) for the given lead ``steps`` (hours).

    Cached under ``data/ifs/ifs_{YYYYMMDDHH}.nc``; skipped if present. ``dataset`` defaults
    to ``IFS_DATASET`` and must be set. ``area`` defaults to the configured ERA5 box."""
    dataset = dataset or IFS_DATASET
    if not dataset:
        raise ValueError(
            "No IFS CDS dataset id set. Pass dataset=... (or --dataset) once the historical "
            "IFS source is confirmed — the free CDS does not host the operational archive.")
    out_dir = out_dir or IFS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"ifs_{ref_time:%Y%m%d%H}.nc"
    if target.exists():
        return target

    import cdsapi

    area = area or settings.era5_area
    request = {
        "variable": list(IFS_SHORTNAMES),       # caller-confirmed names map via the dataset
        "year": f"{ref_time:%Y}", "month": f"{ref_time:%m}", "day": f"{ref_time:%d}",
        "time": f"{ref_time:%H}:00",
        "leadtime_hour": [str(s) for s in steps],
        "area": list(area),
        "data_format": "netcdf",
    }
    cdsapi.Client().retrieve(dataset, request, str(target))
    return target


def download_tigge(ref_time: datetime, steps: list[int],
                   area: tuple[float, float, float, float] | None = None,
                   origin: str = "ecmf", fc_type: str = "cf",
                   out_dir: Path | None = None) -> Path:
    """Download one ECMWF run from the TIGGE archive via the ECMWF API (free historical
    IFS forecasts, 2006+). Requires ~/.ecmwfapirc credentials and one-time acceptance of the
    TIGGE licence. ``fc_type='cf'`` is the control (deterministic) run; ``origin='ecmf'`` is
    ECMWF. GRIB2 is cached under ``data/ifs/tigge_{YYYYMMDDHH}.grib``.

    Only TIGGE-available params are requested (see TIGGE_PARAMS); cloud layers / cbh / gust /
    blh are not in TIGGE and will be absent (NULL) in the resulting nwp_point rows."""
    out_dir = out_dir or IFS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"tigge_{ref_time:%Y%m%d%H}.grib"
    if target.exists():
        return target

    from ecmwfapi import ECMWFDataServer

    area = area or settings.era5_area
    ECMWFDataServer().retrieve({
        "class": "ti",
        "dataset": "tigge",
        "expver": "prod",
        "origin": origin,
        "levtype": "sfc",
        "type": fc_type,
        "grid": "0.5/0.5",
        "param": "/".join(TIGGE_PARAMS.keys()),
        "date": f"{ref_time:%Y-%m-%d}",
        "time": f"{ref_time:%H}:00:00",
        "step": "/".join(str(s) for s in steps),
        "area": "/".join(str(x) for x in area),   # N/W/S/E
        "target": str(target),
    })
    return target


# --- loaders --------------------------------------------------------------

# ecCodes/cfgrib short names -> our internal vocabulary (so extraction is source-agnostic).
_GRIB_RENAME = {"10u": "u10", "10v": "v10", "2t": "t2m", "2d": "d2m",
                "tcw": "tcwv", "10fg": "fg10", "10fg6": "fg10"}


def load_grib(path: Path) -> xr.Dataset:
    """Open a (multi-param, multi-step) GRIB2 granule and merge into one Dataset, renaming
    GRIB short names to our internal vocabulary. Used for TIGGE and ECMWF Open Data.

    cfgrib splits incompatible hypercubes (e.g. instantaneous vs accumulated tp) into
    separate datasets, so we open them all and merge."""
    import cfgrib

    dss = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    merged = xr.merge([d.load() for d in dss], compat="override", combine_attrs="drop_conflicts")
    rename = {k: v for k, v in _GRIB_RENAME.items() if k in merged.variables}
    return merged.rename(rename) if rename else merged


# --- extraction (testable without CDS) ------------------------------------


_STEP_COORDS = ("step", "forecast_period", "leadtime", "prediction_timedelta")


def _step_hours(pt: xr.Dataset) -> np.ndarray:
    """Lead steps in integer hours from whichever step-like coordinate is present."""
    for name in _STEP_COORDS:
        if name in pt.coords or name in pt.dims:
            vals = pt[name].values
            if np.issubdtype(np.asarray(vals).dtype, np.timedelta64):
                return (vals / np.timedelta64(1, "h")).astype(int)
            return np.asarray(vals).astype(int)
    raise ValueError(f"no step coordinate among {_STEP_COORDS} in {list(pt.coords)}")


def _point_records_fc(pt: xr.Dataset, icao: str, ref_time: datetime) -> list[dict]:
    """Build nwp_point forecast records from a Dataset reduced to one point, indexed by
    lead step. Mirrors ERA5 ``_point_records`` (same units) but stamps ref_time/step_h and
    de-accumulates accumulated fields (tp) across the step dimension."""
    ref_time = ref_time.astimezone(timezone.utc) if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
    steps = _step_hours(pt)
    n = len(steps)

    def series(short):
        return np.asarray(pt[short].values).reshape(n) if short in pt else None

    u, v = series("u10"), series("v10")
    spd = direction = None
    if u is not None and v is not None:
        spd, direction = _wind_speed_dir(u, v)
    spd_kt = None if spd is None else spd * MS_TO_KT

    gust_short = "fg10" if "fg10" in pt else ("i10fg" if "i10fg" in pt else None)
    gust = None if gust_short is None else series(gust_short) * MS_TO_KT
    t2m = None if series("t2m") is None else series("t2m") - 273.15
    d2m = None if series("d2m") is None else series("d2m") - 273.15
    tcc, lcc, mcc, hcc = (series(s) for s in ("tcc", "lcc", "mcc", "hcc"))
    cbh = series("cbh")
    tp = None if series("tp") is None else _deaccumulate(series("tp")) * 1000.0  # m -> mm
    msl = None if series("msl") is None else series("msl") / 100.0               # Pa -> hPa
    cape, blh, tcwv = (series(s) for s in ("cape", "blh", "tcwv"))

    def get(arr, i):
        return None if arr is None else _clean(arr[i])

    records = []
    for i in range(n):
        vt = ref_time + timedelta(hours=int(steps[i]))
        records.append({
            "icao": icao, "valid_time": vt, "source": "ifs",
            "ref_time": ref_time, "step_h": int(steps[i]),
            "wind10m_spd": get(spd_kt, i), "wind10m_dir": get(direction, i),
            "gust": get(gust, i), "t2m_c": get(t2m, i), "d2m_c": get(d2m, i),
            "tcc": get(tcc, i), "lcc": get(lcc, i), "mcc": get(mcc, i), "hcc": get(hcc, i),
            "cbh_m": get(cbh, i), "tp_mm": get(tp, i), "mslp_hpa": get(msl, i),
            "cape_jkg": get(cape, i), "blh_m": get(blh, i), "tcwv_kgm2": get(tcwv, i),
        })
    return records


def _deaccumulate(cum: np.ndarray) -> np.ndarray:
    """Forecast-start accumulation -> per-step increment. Clamped at 0 (guards tiny negative
    diffs from packing). Assumes steps are sorted ascending (as requested)."""
    out = np.diff(cum, prepend=0.0)
    return np.clip(out, 0.0, None)


def extract_points_fc(ds: xr.Dataset, stations: list[dict], ref_time: datetime) -> list[dict]:
    """Gridded forecast: nearest-gridpoint step series per station (needs lat/lon dims)."""
    records: list[dict] = []
    for st in stations:
        pt = ds.sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
        records.extend(_point_records_fc(pt, st["icao"], ref_time))
    return records


def extract_timeseries_fc(ds: xr.Dataset, icao: str, ref_time: datetime) -> list[dict]:
    """Single-point forecast Dataset (scalar lat/lon, step dim)."""
    return _point_records_fc(ds, icao, ref_time)
