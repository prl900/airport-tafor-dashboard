"""Feature builders for candidate-TAF generation.

For the baselines these are simple history lookups (latest observation, hour-of-day
climatology). A trained model (Phase 4+) would extend this to assemble NWP+obs
feature vectors per (airport, valid_hour) — the ERA5 `nwp_point` series joined to
the METAR history — but the Forecaster interface stays the same."""

from __future__ import annotations

from datetime import datetime

import duckdb

OBS_COLS = ("observed_at", "wind_dir_deg", "wind_spd_kt", "vis_m", "ceiling_ft", "flight_category")


def latest_obs_before(con: duckdb.DuckDBPyConnection, icao: str, t: datetime) -> dict | None:
    """Most recent METAR observation at or before ``t`` (the persistence anchor)."""
    row = con.execute(
        f"""
        SELECT {', '.join(OBS_COLS)} FROM metar_obs
        WHERE icao = ? AND observed_at <= ? ORDER BY observed_at DESC LIMIT 1
        """,
        [icao, t],
    ).fetchone()
    return dict(zip(OBS_COLS, row)) if row else None


def hourly_climatology(con: duckdb.DuckDBPyConnection, icao: str) -> dict[int, dict]:
    """Per hour-of-day: median visibility/ceiling and modal flight category.

    Computed from all stored observations for the station. Returns {hour: {...}}.
    """
    clim: dict[int, dict] = {}
    rows = con.execute(
        """
        SELECT extract('hour' FROM observed_at)::INT AS hod,
               median(vis_m)      AS vis_m,
               median(ceiling_ft) AS ceiling_ft,
               mode(flight_category) AS flight_category
        FROM metar_obs WHERE icao = ?
        GROUP BY 1
        """,
        [icao],
    ).fetchall()
    for hod, vis_m, ceiling_ft, cat in rows:
        clim[int(hod)] = {
            "vis_m": vis_m,
            "ceiling_ft": ceiling_ft,
            "flight_category": cat,
            "wind_dir_deg": None,
            "wind_spd_kt": None,
        }
    return clim
