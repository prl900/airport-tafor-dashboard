"""Orchestrate verification: expand each TAF -> align METARs -> score -> store.

Idempotent: a TAF already present in verification_hourly (for the 'categorical'
profile) is skipped, so re-runs only process new TAFs."""

from __future__ import annotations

from datetime import datetime

import duckdb

from wx.verification.align import align
from wx.verification.scores import score_hour
from wx.verification.timeline import expand

_GROUP_COLS = (
    "group_type", "probability", "valid_from", "valid_to",
    "wind_dir_deg", "wind_spd_kt", "wind_gust_kt", "vis_m", "ceiling_ft", "flight_category",
)


def _load_groups(con, taf_id: int) -> list[dict]:
    cur = con.execute(
        f"SELECT {', '.join(_GROUP_COLS)} FROM taf_group WHERE taf_forecast_id = ?",
        [taf_id],
    )
    return [dict(zip(_GROUP_COLS, row)) for row in cur.fetchall()]


def _load_obs(con, icao: str, start: datetime, end: datetime) -> list[dict]:
    cols = ("observed_at", "wind_dir_deg", "wind_spd_kt", "vis_m", "ceiling_ft", "flight_category")
    cur = con.execute(
        f"""
        SELECT {', '.join(cols)} FROM metar_obs
        WHERE icao = ? AND observed_at >= ? AND observed_at < ?
        ORDER BY observed_at
        """,
        [icao, start, end],
    )
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def verify_taf(con: duckdb.DuckDBPyConnection, taf: dict) -> int:
    """Verify one TAF (dict: id, icao, issued_at, valid_from, valid_to). Returns
    the number of hourly rows written."""
    groups = _load_groups(con, taf["id"])
    expected = expand(groups, taf["valid_from"], taf["valid_to"])
    if not expected:
        return 0
    obs = _load_obs(con, taf["icao"], expected[0].valid_hour, taf["valid_to"])
    aligned = align(expected, obs)

    rows = []
    for eh, o in aligned:
        scored = score_hour(eh, o)
        if scored is None:
            continue
        lead_h = int((eh.valid_hour - taf["issued_at"]).total_seconds() // 3600)
        rows.append((taf["id"], taf["icao"], eh.valid_hour, lead_h, scored))

    for taf_id, icao, valid_hour, lead_h, s in rows:
        con.execute(
            """
            INSERT INTO verification_hourly
              (taf_forecast_id, icao, valid_hour, lead_time_h, scoring_profile,
               fcst_category, obs_category, category_outcome, wind_err_kt, dir_err_deg,
               temp_err_c, vis_err_m, ceiling_err_ft, weighted_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [taf_id, icao, valid_hour, lead_h, s["scoring_profile"], s["fcst_category"],
             s["obs_category"], s["category_outcome"], s["wind_err_kt"], s["dir_err_deg"],
             s["temp_err_c"], s["vis_err_m"], s["ceiling_err_ft"], s["weighted_score"]],
        )
    return len(rows)


def verify_pending(con: duckdb.DuckDBPyConnection, icaos: list[str] | None = None) -> int:
    """Verify all TAFs that have no verification rows yet. Returns rows written."""
    where = ""
    params: list = []
    if icaos:
        where = f"AND f.icao IN ({','.join(['?'] * len(icaos))})"
        params = list(icaos)
    tafs = con.execute(
        f"""
        SELECT f.id, f.icao, f.issued_at, f.valid_from, f.valid_to
        FROM taf_forecast f
        WHERE f.valid_from IS NOT NULL AND f.valid_to IS NOT NULL {where}
          AND NOT EXISTS (SELECT 1 FROM verification_hourly v WHERE v.taf_forecast_id = f.id)
        ORDER BY f.issued_at
        """,
        params,
    ).fetchall()

    total = 0
    for row in tafs:
        taf = dict(zip(("id", "icao", "issued_at", "valid_from", "valid_to"), row))
        total += verify_taf(con, taf)
    return total
