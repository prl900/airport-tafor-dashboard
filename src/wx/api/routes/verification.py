"""Verification endpoints. Scoring is implemented in Phase 2; these endpoints
serve whatever is in verification_hourly and degrade gracefully when empty."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends

from wx.api.deps import get_db, rows_to_dicts

router = APIRouter(prefix="/verification", tags=["verification"])


@router.get("/summary")
def summary(db: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[dict]:
    """Per-airport categorical skill summary (POD/FAR) — empty until Phase 2 runs."""
    db.execute(
        """
        SELECT icao,
               count(*) AS n,
               sum(CASE WHEN category_outcome = 'hit' THEN 1 ELSE 0 END) AS hits,
               sum(CASE WHEN category_outcome = 'miss' THEN 1 ELSE 0 END) AS misses,
               sum(CASE WHEN category_outcome = 'false_alarm' THEN 1 ELSE 0 END) AS false_alarms
        FROM verification_hourly
        WHERE scoring_profile = 'categorical'
        GROUP BY icao ORDER BY icao
        """
    )
    return rows_to_dicts(db)


@router.get("/{icao}")
def station_verification(icao: str, db: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[dict]:
    db.execute(
        """
        SELECT valid_hour, lead_time_h, scoring_profile, fcst_category, obs_category,
               category_outcome, wind_err_kt, vis_err_m, ceiling_err_ft, weighted_score
        FROM verification_hourly
        WHERE icao = ? ORDER BY valid_hour
        """,
        [icao.upper()],
    )
    return rows_to_dicts(db)
