"""METAR observation time-series endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
from fastapi import APIRouter, Depends, Query

from wx.api.deps import get_db, parse_dt, rows_to_dicts

router = APIRouter(prefix="/stations", tags=["metar"])


@router.get("/{icao}/metar")
def get_metar(
    icao: str,
    start: str | None = Query(None, description="ISO start (UTC), default now-7d"),
    end: str | None = Query(None, description="ISO end (UTC), default now"),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    now = datetime.now(timezone.utc)
    t0 = parse_dt(start, now - timedelta(days=7))
    t1 = parse_dt(end, now)
    db.execute(
        """
        SELECT observed_at, wind_dir_deg, wind_spd_kt, wind_gust_kt, vis_m,
               temp_c, dewpoint_c, qnh_hpa, ceiling_ft, flight_category, clouds, weather
        FROM metar_obs
        WHERE icao = ? AND observed_at >= ? AND observed_at < ?
        ORDER BY observed_at
        """,
        [icao.upper(), t0, t1],
    )
    return rows_to_dicts(db)
