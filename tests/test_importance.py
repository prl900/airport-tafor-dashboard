"""Assessment harness mechanics: permutation importance runs in the model's own metrics
and the all_nan flag correctly distinguishes ingested variables from not-yet-fetched
candidates (so an un-ingested variable can't masquerade as 'useless')."""

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sklearn")

from wx.ai.dataset import build_samples, temporal_split
from wx.ai.importance import permutation_importance
from wx.ai.models import MultiTaskModel
from wx.db.connection import SCHEMA_PATH, connect

UTC = timezone.utc


@pytest.fixture
def con():
    c = connect(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    # ~2.5 years of 3-hourly obs spanning train(<2024)/val(2024)/test(>=2025). A morning
    # low-cloud/fog pattern drives LIFR; ERA5 lcc is high exactly then, so the cloud_layers
    # group carries real signal while cape/blh/tcwv stay un-ingested (NULL).
    start = datetime(2023, 1, 1, tzinfo=UTC)
    rid = 0
    t = start
    while t < datetime(2025, 7, 1, tzinfo=UTC):
        fog = t.hour in (3, 6)
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?, 'LEMD', ?, ?, ?, 5, 220, 8.0, ?, ?)""",
            [rid, rid, t, 300 if fog else 9999, 100 if fog else None,
             7.5 if fog else 2.0, "LIFR" if fog else "VFR"],
        )
        c.execute(
            """INSERT INTO nwp_point (icao, valid_time, source, ref_time, step_h,
               wind10m_spd, wind10m_dir, gust, t2m_c, d2m_c, tcc, lcc, mcc, hcc,
               cbh_m, tp_mm, mslp_hpa)
               VALUES ('LEMD', ?, 'era5', ?, 0, 6, 220, 9, 8.0, ?, ?, ?, 0.2, 0.1,
                       ?, 0.0, 1013)""",
            [t, t, 7.5 if fog else 2.0, 1.0 if fog else 0.1,
             0.95 if fog else 0.05, 80 if fog else 3000],
        )
        rid += 1
        t += timedelta(hours=3)
    yield c
    c.close()


def test_permutation_importance_flags_data_availability(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3, 6))
    tr, va, te = temporal_split(df)
    model = MultiTaskModel("gbm").fit(tr, val_df=va)
    imp = permutation_importance(model, te, n_repeats=2)

    by_group = imp.set_index("group")
    # Ingested groups have data; un-ingested candidates are flagged all_nan.
    assert not by_group.loc["cloud_layers", "all_nan"]
    assert not by_group.loc["wind", "all_nan"]
    assert by_group.loc["cape", "all_nan"]
    assert by_group.loc["blh", "all_nan"]
    # The result carries the model's own metric deltas.
    assert {"d_hss", "d_mae_vis", "d_mae_ceiling"}.issubset(imp.columns)
