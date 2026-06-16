"""Station endpoints: list + per-station metadata, with latest observed category."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, HTTPException

from wx.api.deps import get_db, rows_to_dicts

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("")
def list_stations(db: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[dict]:
    """All stations with their most-recent observed flight category (for the map)."""
    db.execute(
        """
        SELECT s.icao, s.name, s.lat, s.lon, s.elevation_m, s.region,
               latest.flight_category AS latest_category,
               latest.observed_at     AS latest_observed_at
        FROM stations s
        LEFT JOIN (
            SELECT DISTINCT ON (icao) icao, flight_category, observed_at
            FROM metar_obs ORDER BY icao, observed_at DESC
        ) latest ON latest.icao = s.icao
        ORDER BY s.region, s.icao
        """
    )
    return rows_to_dicts(db)


@router.get("/{icao}")
def get_station(icao: str, db: duckdb.DuckDBPyConnection = Depends(get_db)) -> dict:
    db.execute("SELECT * FROM stations WHERE icao = ?", [icao.upper()])
    rows = rows_to_dicts(db)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Unknown station {icao}")
    return rows[0]
