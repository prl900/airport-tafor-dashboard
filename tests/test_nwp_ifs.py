"""IFS forecast extraction: step-indexed records carry ref_time/step_h, units match ERA5,
and accumulated precip is de-accumulated to per-step increments."""

from datetime import datetime, timezone

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from wx.ingestion.nwp_ifs import (
    _deaccumulate,
    extract_points_fc,
    extract_timeseries_fc,
)

UTC = timezone.utc


def test_deaccumulate_is_per_step_increment_clamped():
    cum = np.array([0.0, 0.001, 0.001, 0.004])  # metres, accumulated from step 0
    inc = _deaccumulate(cum)
    assert inc.tolist() == pytest.approx([0.0, 0.001, 0.0, 0.003])
    # tiny negative packing artefacts clamp to 0
    assert _deaccumulate(np.array([0.0, 0.002, 0.0019])).tolist() == pytest.approx([0.0, 0.002, 0.0])


def _synthetic_run():
    """One IFS run init 2023-01-02 00Z, hourly steps 0..3, single point."""
    steps = np.array([0, 1, 2, 3], dtype="timedelta64[h]")
    ds = xr.Dataset(
        {
            "u10": (("step",), np.zeros(4)),
            "v10": (("step",), np.full(4, -5.0)),       # from due north, 5 m/s
            "fg10": (("step",), np.array([8.0, 9.0, 10.0, 11.0])),
            "t2m": (("step",), np.full(4, 283.15)),     # 10 C
            "d2m": (("step",), np.full(4, 278.15)),     # 5 C
            "tcc": (("step",), np.array([0.1, 0.5, 0.9, 1.0])),
            "lcc": (("step",), np.array([0.0, 0.3, 0.8, 1.0])),
            "msl": (("step",), np.full(4, 101300.0)),   # 1013 hPa
            "tp": (("step",), np.array([0.0, 0.001, 0.001, 0.004])),  # accumulated metres
            "cape": (("step",), np.array([10.0, 50.0, 200.0, 400.0])),
        },
        coords={"step": steps, "latitude": 40.5, "longitude": -3.5},
    )
    return ds


def test_extract_timeseries_fc_units_steps_and_reftime():
    ref = datetime(2023, 1, 2, 0, tzinfo=UTC)
    recs = extract_timeseries_fc(_synthetic_run(), "LEMD", ref)
    assert len(recs) == 4
    r0, r2 = recs[0], recs[2]

    # ref_time/step bookkeeping
    assert r0["source"] == "ifs"
    assert r0["ref_time"] == ref
    assert [r["step_h"] for r in recs] == [0, 1, 2, 3]
    assert r2["valid_time"] == datetime(2023, 1, 2, 2, tzinfo=UTC)

    # units mirror ERA5
    assert r0["t2m_c"] == pytest.approx(10.0)
    assert r0["mslp_hpa"] == pytest.approx(1013.0)
    assert r0["wind10m_spd"] == pytest.approx(5 * 1.94384, abs=1e-3)
    assert r0["wind10m_dir"] in (pytest.approx(0.0), pytest.approx(360.0))
    assert r0["gust"] == pytest.approx(8.0 * 1.94384, abs=1e-3)

    # tp de-accumulated to per-step mm: [0, 1, 0, 3]
    assert [r["tp_mm"] for r in recs] == pytest.approx([0.0, 1.0, 0.0, 3.0])
    # candidate var carried through
    assert r2["cape_jkg"] == pytest.approx(200.0)
    assert r2["lcc"] == pytest.approx(0.8)


def test_extract_points_fc_nearest_gridpoint():
    """Gridded run (step,lat,lon): nearest-gridpoint per station."""
    steps = np.array([0, 1], dtype="timedelta64[h]")
    lats, lons = np.array([41.0, 40.5, 40.0]), np.array([-4.0, -3.5, -3.0])
    shape = (2, 3, 3)
    ds = xr.Dataset(
        {
            "u10": (("step", "latitude", "longitude"), np.zeros(shape)),
            "v10": (("step", "latitude", "longitude"), np.full(shape, -5.0)),
            "t2m": (("step", "latitude", "longitude"), np.full(shape, 283.15)),
            "tp": (("step", "latitude", "longitude"), np.zeros(shape)),
        },
        coords={"step": steps, "latitude": lats, "longitude": lons},
    )
    ref = datetime(2023, 1, 2, 6, tzinfo=UTC)
    recs = extract_points_fc(ds, [{"icao": "LEMD", "lat": 40.49, "lon": -3.57}], ref)
    assert len(recs) == 2
    assert {r["step_h"] for r in recs} == {0, 1}
    assert recs[0]["t2m_c"] == pytest.approx(10.0)
    assert recs[0]["ref_time"] == ref
