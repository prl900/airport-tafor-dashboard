"""TAF forecast endpoints: issued TAFs with their decomposed change groups."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
from fastapi import APIRouter, Depends, Query

from wx.api.deps import get_db, parse_dt, rows_to_dicts

router = APIRouter(prefix="/stations", tags=["taf"])


@router.get("/{icao}/taf")
def get_taf(
    icao: str,
    start: str | None = Query(None, description="ISO start (UTC) on issued_at, default now-7d"),
    end: str | None = Query(None, description="ISO end (UTC) on issued_at, default now"),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """TAFs issued in the window, each with its ordered list of change groups."""
    now = datetime.now(timezone.utc)
    t0 = parse_dt(start, now - timedelta(days=7))
    t1 = parse_dt(end, now)

    db.execute(
        """
        SELECT id, icao, issued_at, valid_from, valid_to
        FROM taf_forecast
        WHERE icao = ? AND issued_at >= ? AND issued_at < ?
        ORDER BY issued_at
        """,
        [icao.upper(), t0, t1],
    )
    forecasts = rows_to_dicts(db)
    if not forecasts:
        return []

    ids = [f["id"] for f in forecasts]
    db.execute(
        f"""
        SELECT id, taf_forecast_id, group_type, probability, valid_from, valid_to,
               wind_dir_deg, wind_spd_kt, wind_gust_kt, vis_m, ceiling_ft,
               flight_category, clouds, weather
        FROM taf_group
        WHERE taf_forecast_id IN ({",".join(["?"] * len(ids))})
        ORDER BY taf_forecast_id, valid_from
        """,
        ids,
    )
    groups_by_taf: dict[int, list[dict]] = {}
    for g in rows_to_dicts(db):
        groups_by_taf.setdefault(g["taf_forecast_id"], []).append(g)

    for f in forecasts:
        f["groups"] = groups_by_taf.get(f["id"], [])
    return forecasts
