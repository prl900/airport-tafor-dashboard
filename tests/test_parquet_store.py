"""Prototype: a Parquet-backed nwp_point (external view) must behave identically to the
DuckDB table for the causal/forecast pipeline — the run-anchored ASOF join still resolves,
and re-ingesting a key is an idempotent upsert (latest write wins)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from wx.ai.dataset import build_samples
from wx.db.connection import SCHEMA_PATH, connect
from wx.db.parquet_store import (
    nwp_point_view_ddl,
    register_nwp_view,
    store_nwp_points_parquet,
)

UTC = timezone.utc


def _run(ref, t2m_base, steps=range(0, 13)):
    return [dict(icao="LEMD", valid_time=ref + timedelta(hours=s), source="ifs",
                 ref_time=ref, step_h=s, t2m_c=t2m_base + s, lcc=0.4) for s in steps]


@pytest.fixture
def con_and_dir(tmp_path):
    pq = tmp_path / "nwp"
    pq.mkdir()
    c = connect(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    base = datetime(2023, 1, 2, tzinfo=UTC)
    for h in range(24):
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?, 'LEMD', ?, 9999, NULL, 5, 220, 8.0, 2.0, 'VFR')""",
            [h, h, base + timedelta(hours=h)],
        )
    yield c, pq
    c.close()


def test_build_samples_over_parquet_view_matches_run_anchored_join(con_and_dir):
    con, pq = con_and_dir
    # Two IFS runs written as separate Parquet batches (no DB writes).
    store_nwp_points_parquet(_run(datetime(2023, 1, 2, 0, tzinfo=UTC), 100.0), base_dir=pq)
    store_nwp_points_parquet(_run(datetime(2023, 1, 2, 6, tzinfo=UTC), 200.0), base_dir=pq)
    register_nwp_view(con, base_dir=pq)
    assert con.execute("SELECT count(*) FROM nwp_point").fetchone()[0] == 26

    df = build_samples(con, icaos=["LEMD"], leads=(1, 3, 6), nwp_source="ifs")

    def at(vh, lead):
        vt = pd.Timestamp(datetime(2023, 1, 2, vh, tzinfo=UTC))
        m = (pd.to_datetime(df["valid_time"], utc=True) == vt) & (df["f_lead_h"] == lead)
        return df[m].iloc[0]

    # Same expectations as the in-table run-anchored join (test_dataset_ifs):
    assert at(7, 1)["f_et_t2m"] == pytest.approx(201.0)   # T0=06Z -> run 06Z, step 1
    assert at(7, 3)["f_et_t2m"] == pytest.approx(107.0)   # T0=04Z -> run 00Z, step 7
    assert at(9, 6)["f_et_t2m"] == pytest.approx(109.0)   # T0=03Z -> run 00Z, step 9
    assert at(7, 1)["f_et_lcc"] == pytest.approx(0.4)     # cloud layer flows through


def test_reingest_is_idempotent_upsert(con_and_dir):
    con, pq = con_and_dir
    ref = datetime(2023, 1, 2, 0, tzinfo=UTC)
    store_nwp_points_parquet(_run(ref, 100.0), base_dir=pq)
    # Re-ingest the same run with corrected values; latest write must win, no duplicates.
    store_nwp_points_parquet(_run(ref, 500.0), base_dir=pq)
    register_nwp_view(con, base_dir=pq)

    rows = con.execute(
        "SELECT count(*) FROM nwp_point WHERE ref_time = ? AND step_h = 5", [ref]
    ).fetchone()[0]
    assert rows == 1
    val = con.execute(
        "SELECT t2m_c FROM nwp_point WHERE ref_time = ? AND step_h = 5", [ref]
    ).fetchone()[0]
    assert val == pytest.approx(505.0)
