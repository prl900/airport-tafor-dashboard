"""IFS (forecast) NWP join: unlike ERA5 analysis (one value per valid hour), an IFS
forecast has many values per valid_time keyed by run init time + lead. The dataset join
must, for each (icao, T0, lead), use the run initialized at/just-before T0 and snap to the
nearest step of THAT run for both the valid-hour feature (et) and the T0 anchor (e0)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from wx.ai.dataset import build_samples
from wx.db.connection import SCHEMA_PATH, connect

UTC = timezone.utc


def _seed_run(con, ref: datetime, t2m_base: float, steps=range(0, 13)):
    """One IFS run: hourly steps. t2m encodes (base + step) so the test can read back
    exactly which run/step the join selected."""
    for step in steps:
        vt = ref + timedelta(hours=step)
        con.execute(
            """INSERT INTO nwp_point (icao, valid_time, source, ref_time, step_h,
               wind10m_spd, wind10m_dir, gust, t2m_c, d2m_c, tcc, lcc, mcc, hcc,
               cbh_m, tp_mm, mslp_hpa)
               VALUES ('LEMD', ?, 'ifs', ?, ?, 6, 220, 9, ?, 1.0,
                       0.5, 0.4, 0.3, 0.2, 300, 0.0, 1013)""",
            [vt, ref, step, t2m_base + step],
        )


@pytest.fixture
def con():
    c = connect(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    base = datetime(2023, 1, 2, tzinfo=UTC)
    for h in range(24):
        ts = base + timedelta(hours=h)
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?, 'LEMD', ?, 9999, NULL, 5, 220, 8.0, 2.0, 'VFR')""",
            [h, h, ts],
        )
    # Two runs on 2023-01-02: 00Z (t2m = 100+step) and 06Z (t2m = 200+step).
    _seed_run(c, datetime(2023, 1, 2, 0, tzinfo=UTC), 100.0)
    _seed_run(c, datetime(2023, 1, 2, 6, tzinfo=UTC), 200.0)
    yield c
    c.close()


def _row(df, valid_hour, lead):
    vt = pd.Timestamp(datetime(2023, 1, 2, valid_hour, tzinfo=UTC))
    m = (pd.to_datetime(df["valid_time"], utc=True) == vt) & (df["f_lead_h"] == lead)
    return df[m].iloc[0]


def test_ifs_join_uses_latest_run_at_or_before_t0(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3, 6), nwp_source="ifs")
    assert df["f_et_t2m"].notna().all()

    # valid=07Z, lead=1 -> T0=06Z -> run 06Z, step 1 -> 201
    assert _row(df, 7, 1)["f_et_t2m"] == pytest.approx(201.0)
    # valid=07Z, lead=3 -> T0=04Z -> run 00Z (latest <= 04Z), step 7 -> 107
    assert _row(df, 7, 3)["f_et_t2m"] == pytest.approx(107.0)
    # valid=09Z, lead=6 -> T0=03Z -> run 00Z, step 9 -> 109
    assert _row(df, 9, 6)["f_et_t2m"] == pytest.approx(109.0)


def test_ifs_anchor_and_tendency_share_one_run(con):
    """e0 (anchor) and et (valid hour) must come from the SAME run, so the tendency is
    a true within-forecast evolution. valid=09Z, lead=3 -> T0=06Z -> run 06Z:
    et=step3 (203), e0=step0 (200) -> f_tend_t2m = 3."""
    df = build_samples(con, icaos=["LEMD"], leads=(3,), nwp_source="ifs")
    r = _row(df, 9, 3)
    assert r["f_et_t2m"] == pytest.approx(203.0)
    assert r["f_tend_t2m"] == pytest.approx(3.0)


def test_low_med_high_cloud_features_present(con):
    """Low/medium/high cloud layers must surface as features (TAFOR skill driver)."""
    df = build_samples(con, icaos=["LEMD"], leads=(1,), nwp_source="ifs")
    for col in ("f_et_lcc", "f_et_mcc", "f_et_hcc", "f_tend_lcc"):
        assert col in df.columns, f"missing cloud feature {col}"
    assert df["f_et_lcc"].iloc[0] == pytest.approx(0.4)
