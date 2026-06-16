"""ERA5 NWP point-series endpoint (Phase 3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
from fastapi import APIRouter, Depends, Query

from wx.api.deps import get_db, parse_dt, rows_to_dicts

router = APIRouter(prefix="/stations", tags=["nwp"])


@router.get("/{icao}/nwp")
def get_nwp(
    icao: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[dict]:
    """ERA5 nearest-gridpoint series for an airport (empty until `wx nwp` runs)."""
    now = datetime.now(timezone.utc)
    t0 = parse_dt(start, now - timedelta(days=7))
    t1 = parse_dt(end, now)
    db.execute(
        """
        SELECT valid_time, wind10m_spd, wind10m_dir, gust, t2m_c, d2m_c,
               tcc, lcc, mcc, hcc, cbh_m, tp_mm, mslp_hpa
        FROM nwp_point
        WHERE icao = ? AND valid_time >= ? AND valid_time < ?
        ORDER BY valid_time
        """,
        [icao.upper(), t0, t1],
    )
    return rows_to_dicts(db)
