"""Phase 4 forecaster tests against an in-memory DuckDB with a known history."""

from datetime import datetime, timezone

import pytest

from wx.ai.generate import ClimatologyForecaster, PersistenceForecaster
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "t.duckdb")
    from wx.db.connection import SCHEMA_PATH

    c.execute(SCHEMA_PATH.read_text())
    # Seed a handful of obs: VFR by day, LIFR fog at 06Z.
    rows = []
    for day in (1, 2, 3):
        for hour in range(24):
            cat = "LIFR" if hour == 6 else "VFR"
            vis = 400 if hour == 6 else 9999
            ceil = 200 if hour == 6 else None
            rows.append((f"2023-01-{day:02d} {hour:02d}:00:00+00", cat, vis, ceil))
    for i, (ts, cat, vis, ceil) in enumerate(rows):
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, flight_category, wind_spd_kt) VALUES (?, ?, 'LEMD', ?, ?, ?, ?, 5)""",
            [i, i, ts, vis, ceil, cat],
        )
    yield c
    c.close()


def test_persistence_carries_last_obs(con):
    issued = datetime(2023, 1, 2, 6, tzinfo=UTC)  # latest obs is the 06Z LIFR fog
    vf = datetime(2023, 1, 2, 7, tzinfo=UTC)
    vt = datetime(2023, 1, 2, 10, tzinfo=UTC)
    hours = PersistenceForecaster().generate(con, "LEMD", issued, vf, vt)
    assert len(hours) == 3
    assert all(h.prevailing["flight_category"] == "LIFR" for h in hours)  # persisted fog


def test_climatology_uses_hour_of_day(con):
    issued = datetime(2023, 1, 5, 0, tzinfo=UTC)
    vf = datetime(2023, 1, 5, 5, tzinfo=UTC)
    vt = datetime(2023, 1, 5, 8, tzinfo=UTC)
    hours = ClimatologyForecaster().generate(con, "LEMD", issued, vf, vt)
    by_hour = {h.valid_hour.hour: h.prevailing["flight_category"] for h in hours}
    assert by_hour[6] == "LIFR"   # 06Z is climatologically foggy
    assert by_hour[5] == "VFR"
    assert by_hour[7] == "VFR"
