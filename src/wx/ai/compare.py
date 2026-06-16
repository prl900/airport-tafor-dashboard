"""Score candidate forecasts against the same observations as the official TAFs,
storing them in verification_hourly under the forecaster's name as the
``scoring_profile`` (the official rows use 'categorical'). This makes
"did the candidate beat the official TAF?" a direct query."""

from __future__ import annotations

import duckdb

from wx.ai.generate import FORECASTERS, Forecaster
from wx.verification.align import align
from wx.verification.runner import _load_obs
from wx.verification.scores import score_hour, skill_scores


def run_candidate(con: duckdb.DuckDBPyConnection, forecaster: Forecaster,
                  icaos: list[str] | None = None) -> int:
    """Generate + score one forecaster against every official TAF. Returns rows."""
    where, params = "", []
    if icaos:
        where = f"AND icao IN ({','.join(['?'] * len(icaos))})"
        params = list(icaos)
    tafs = con.execute(
        f"""
        SELECT id, icao, issued_at, valid_from, valid_to FROM taf_forecast
        WHERE valid_from IS NOT NULL AND valid_to IS NOT NULL {where}
        ORDER BY issued_at
        """,
        params,
    ).fetchall()

    from wx.db import repositories as repo

    rows = []
    for taf_id, icao, issued_at, valid_from, valid_to in tafs:
        expected = forecaster.generate(con, icao, issued_at, valid_from, valid_to)
        if not expected:
            continue
        obs = _load_obs(con, icao, expected[0].valid_hour, valid_to)
        for eh, o in align(expected, obs):
            s = score_hour(eh, o)
            if s is None:
                continue
            lead_h = int((eh.valid_hour - issued_at).total_seconds() // 3600)
            rows.append((taf_id, icao, eh.valid_hour, lead_h, forecaster.name,
                         s["fcst_category"], s["obs_category"], s["category_outcome"],
                         s["wind_err_kt"], s["dir_err_deg"], s["temp_err_c"],
                         s["vis_err_m"], s["ceiling_err_ft"], s["weighted_score"]))
    return repo.store_verification(con, rows)


def run_all_candidates(con: duckdb.DuckDBPyConnection, icaos: list[str] | None = None) -> dict:
    return {name: run_candidate(con, f, icaos) for name, f in FORECASTERS.items()}


def comparison(con: duckdb.DuckDBPyConnection, icao: str) -> list[dict]:
    """Per-profile skill + mean weighted score for one station (official vs candidates)."""
    profiles = [r[0] for r in con.execute(
        "SELECT DISTINCT scoring_profile FROM verification_hourly WHERE icao = ?", [icao]
    ).fetchall()]
    out = []
    for p in profiles:
        outcomes = [r[0] for r in con.execute(
            "SELECT category_outcome FROM verification_hourly WHERE icao = ? AND scoring_profile = ?",
            [icao, p],
        ).fetchall()]
        mean_score = con.execute(
            "SELECT avg(weighted_score) FROM verification_hourly WHERE icao = ? AND scoring_profile = ?",
            [icao, p],
        ).fetchone()[0]
        label = "official" if p == "categorical" else p
        out.append({"profile": label, "mean_weighted_score": mean_score, **skill_scores(outcomes)})
    out.sort(key=lambda d: (d["mean_weighted_score"] or 0), reverse=True)
    return out
