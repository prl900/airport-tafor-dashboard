import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from wx.ingestion.nwp_era5 import extract_points


def _synthetic_dataset():
    """Tiny ERA5-like grid over Iberia with known values at one time step."""
    lats = np.array([41.0, 40.5, 40.0])
    lons = np.array([-4.0, -3.5, -3.0])
    times = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]")
    shape = (1, len(lats), len(lons))

    # u=0, v=-5 m/s everywhere => wind FROM the north (360/0 deg), speed 5 m/s.
    u = np.zeros(shape)
    v = np.full(shape, -5.0)
    t2m = np.full(shape, 283.15)   # 10 C
    d2m = np.full(shape, 278.15)   # 5 C
    msl = np.full(shape, 101300.0) # 1013 hPa
    tp = np.full(shape, 0.002)     # 2 mm
    cbh = np.full(shape, 250.0)    # metres

    return xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u),
            "v10": (("time", "latitude", "longitude"), v),
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), d2m),
            "msl": (("time", "latitude", "longitude"), msl),
            "tp": (("time", "latitude", "longitude"), tp),
            "cbh": (("time", "latitude", "longitude"), cbh),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_extract_points_units_and_wind():
    ds = _synthetic_dataset()
    # Station near (40.49, -3.57) -> nearest grid (40.5, -3.5)
    stations = [{"icao": "LEMD", "lat": 40.493, "lon": -3.566}]
    recs = extract_points(ds, stations)
    assert len(recs) == 1
    r = recs[0]
    assert r["icao"] == "LEMD"
    assert r["t2m_c"] == pytest.approx(10.0, abs=1e-6)
    assert r["d2m_c"] == pytest.approx(5.0, abs=1e-6)
    assert r["mslp_hpa"] == pytest.approx(1013.0, abs=1e-6)
    assert r["tp_mm"] == pytest.approx(2.0, abs=1e-6)
    assert r["cbh_m"] == pytest.approx(250.0, abs=1e-6)
    # u=0, v=-5 => speed 5 m/s ~= 9.72 kt, from due north (0/360)
    assert r["wind10m_spd"] == pytest.approx(5 * 1.94384, abs=1e-3)
    assert r["wind10m_dir"] == pytest.approx(0.0, abs=1e-6) or r["wind10m_dir"] == pytest.approx(360.0, abs=1e-6)


def test_extract_points_handles_valid_time_coord():
    ds = _synthetic_dataset().rename({"time": "valid_time"})
    recs = extract_points(ds, [{"icao": "LEMD", "lat": 40.5, "lon": -3.5}])
    assert recs[0]["valid_time"].year == 2023


def test_extract_timeseries_single_point_subset():
    """Timeseries dataset: a single point (scalar lat/lon, valid_time dim) and a
    variable subset (no lcc/mcc/hcc) — extract_timeseries handles both."""
    from wx.ingestion.nwp_era5 import extract_timeseries

    times = np.array(["2023-01-01T00:00:00", "2023-01-01T01:00:00"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "u10": (("valid_time",), np.array([0.0, 0.0])),
            "v10": (("valid_time",), np.array([-5.0, -5.0])),
            "fg10": (("valid_time",), np.array([8.0, 9.0])),
            "t2m": (("valid_time",), np.array([283.15, 284.15])),
            "tcc": (("valid_time",), np.array([1.0, 0.5])),
            "cbh": (("valid_time",), np.array([100.0, 200.0])),
        },
        coords={"valid_time": times, "latitude": 40.5, "longitude": -3.5},
    )
    recs = extract_timeseries(ds, "LEMD")
    assert len(recs) == 2
    assert recs[0]["t2m_c"] == pytest.approx(10.0)
    assert recs[0]["gust"] == pytest.approx(8.0 * 1.94384)
    assert recs[0]["wind10m_dir"] in (pytest.approx(0.0), pytest.approx(360.0))
    assert recs[0]["lcc"] is None  # not in the timeseries subset
    assert recs[0]["cbh_m"] == 100.0


def test_era5_records_are_zero_lead_forecasts():
    """ERA5 analysis is stored as a degenerate forecast: ref_time = valid_time,
    step_h = 0. This lets nwp_point carry both analysis and IFS forecasts, and keeps
    the era5 feature join bit-identical after the forecast dimension was added."""
    ds = _synthetic_dataset()
    recs = extract_points(ds, [{"icao": "LEMD", "lat": 40.5, "lon": -3.5}])
    r = recs[0]
    assert r["step_h"] == 0
    assert r["ref_time"] == r["valid_time"]
