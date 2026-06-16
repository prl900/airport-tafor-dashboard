"""Verification endpoints: categorical skill (POD/FAR/CSI/HSS), per-station
scorecards, and lead-time skill curves. Serves whatever `wx verify` produced."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends

from wx.api.deps import get_db, rows_to_dicts
from wx.verification.scores import skill_scores

router = APIRouter(prefix="/verification", tags=["verification"])


def _outcomes(db, where: str, params: list) -> list[str]:
    db.execute(
        f"SELECT category_outcome FROM verification_hourly WHERE {where}", params
    )
    return [r[0] for r in db.fetchall()]


@router.get("/summary")
def summary(db: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[dict]:
    """Per-airport IFR-event skill summary (POD/FAR/CSI/HSS) over all stored TAFs."""
    icaos = [r[0] for r in db.execute(
        "SELECT DISTINCT icao FROM verification_hourly ORDER BY icao"
    ).fetchall()]
    out = []
    for icao in icaos:
        ss = skill_scores(_outcomes(db, "icao = ?", [icao]))
        out.append({"icao": icao, **ss})
    return out


@router.get("/{icao}/scorecard")
def scorecard(icao: str, db: duckdb.DuckDBPyConnection = Depends(get_db)) -> dict:
    """Headline skill + mean element errors + lead-time skill curve for one station."""
    icao = icao.upper()
    ss = skill_scores(_outcomes(db, "icao = ?", [icao]))

    db.execute(
        """
        SELECT round(avg(abs(vis_err_m)), 0)     AS vis_mae_m,
               round(avg(abs(ceiling_err_ft)), 0) AS ceiling_mae_ft,
               round(avg(abs(wind_err_kt)), 1)   AS wind_mae_kt,
               round(avg(abs(dir_err_deg)), 1)   AS dir_mae_deg,
               round(avg(weighted_score), 3)     AS mean_weighted_score
        FROM verification_hourly WHERE icao = ?
        """,
        [icao],
    )
    errors = rows_to_dicts(db)[0]

    db.execute(
        """
        SELECT (lead_time_h // 6) * 6 AS lead_bucket,
               round(avg(weighted_score), 3) AS mean_score,
               count(*) AS n
        FROM verification_hourly WHERE icao = ? AND lead_time_h >= 0
        GROUP BY 1 ORDER BY 1
        """,
        [icao],
    )
    lead_curve = rows_to_dicts(db)

    return {"icao": icao, "skill": ss, "errors": errors, "lead_curve": lead_curve}


@router.get("/{icao}")
def station_verification(icao: str, db: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[dict]:
    db.execute(
        """
        SELECT valid_hour, lead_time_h, fcst_category, obs_category, category_outcome,
               wind_err_kt, vis_err_m, ceiling_err_ft, weighted_score
        FROM verification_hourly WHERE icao = ? ORDER BY valid_hour
        """,
        [icao.upper()],
    )
    return rows_to_dicts(db)
