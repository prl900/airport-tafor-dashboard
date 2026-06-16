"""Data-access helpers. Storing raw messages and parsed components are separate,
idempotent stages: the parse stage reads raw rows that have no parsed child yet,
so re-running never duplicates and improving the parser re-parses cleanly.

Bulk writes use DuckDB's vectorized ``INSERT ... SELECT`` from a registered
DataFrame — NOT ``executemany``, which in DuckDB runs the statement once per row
and made the full backfill O(hours) as the conflict index grew.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd

from wx.parsing.metar import parse_metar
from wx.parsing.taf import ParsedTaf, parse_taf


def _insert_select(con, table: str, df: pd.DataFrame, select_sql: str,
                   target_cols: list[str], key_cols: list[str] | None = None) -> int:
    """Vectorized bulk insert: register df, INSERT INTO table(cols) SELECT ... FROM df.

    Idempotency via a vectorized anti-join (NOT EXISTS on ``key_cols``) rather than
    ON CONFLICT — the hot tables carry no unique index (see schema.sql), so this is
    a hash anti-join that stays fast as the table grows. Returns rows inserted.
    """
    if df.empty:
        return 0
    before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    where = ""
    if key_cols:
        conds = " AND ".join(f"t.{k} IS NOT DISTINCT FROM d.{k}" for k in key_cols)
        where = f"WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {conds})"
    con.register("_ins_df", df)
    try:
        con.execute(
            f"INSERT INTO {table} ({', '.join(target_cols)}) "
            f"SELECT {select_sql} FROM _ins_df d {where}"
        )
    finally:
        con.unregister("_ins_df")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] - before


_VERIF_COLS = ["taf_forecast_id", "icao", "valid_hour", "lead_time_h", "scoring_profile",
               "fcst_category", "obs_category", "category_outcome", "wind_err_kt",
               "dir_err_deg", "temp_err_c", "vis_err_m", "ceiling_err_ft", "weighted_score"]


def store_verification(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    """Bulk-insert verification rows (vectorized), deduped on
    (taf_forecast_id, valid_hour, scoring_profile). Rows match _VERIF_COLS order."""
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=_VERIF_COLS)
    return _insert_select(con, "verification_hourly", df, ", ".join(_VERIF_COLS), _VERIF_COLS,
                          key_cols=["taf_forecast_id", "valid_hour", "scoring_profile"])


# --- raw storage -----------------------------------------------------------


def store_raw_metar(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """records: {icao, observed_at, raw_text, source}. Returns rows inserted."""
    if not records:
        return 0
    df = pd.DataFrame(records, columns=["icao", "observed_at", "raw_text", "source"])
    return _insert_select(
        con, "raw_metar", df,
        "icao, observed_at, raw_text, source, now()",
        ["icao", "observed_at", "raw_text", "source", "ingested_at"],
        key_cols=["icao", "observed_at", "source"],
    )


def store_raw_taf(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """records: {icao, issued_at, valid_from, valid_to, raw_text, source}."""
    if not records:
        return 0
    df = pd.DataFrame(
        records, columns=["icao", "issued_at", "valid_from", "valid_to", "raw_text", "source"]
    )
    return _insert_select(
        con, "raw_taf", df,
        "icao, issued_at, valid_from, valid_to, raw_text, source, now()",
        ["icao", "issued_at", "valid_from", "valid_to", "raw_text", "source", "ingested_at"],
        key_cols=["icao", "issued_at", "source"],
    )


def store_nwp_points(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """Upsert ERA5 per-station point series into nwp_point. Returns rows written."""
    if not records:
        return 0
    cols = ["icao", "valid_time", "source", "wind10m_spd", "wind10m_dir", "gust",
            "t2m_c", "d2m_c", "tcc", "lcc", "mcc", "hcc", "cbh_m", "tp_mm", "mslp_hpa"]
    df = pd.DataFrame(records, columns=cols)
    return _insert_select(con, "nwp_point", df, ", ".join(cols), cols,
                          key_cols=["icao", "valid_time", "source"])


# --- parse stage -----------------------------------------------------------

_METAR_COLS = ["raw_metar_id", "icao", "observed_at", "wind_dir_deg", "wind_spd_kt",
               "wind_gust_kt", "vis_m", "temp_c", "dewpoint_c", "qnh_hpa", "ceiling_ft",
               "flight_category", "clouds", "weather"]


def parse_pending_metar(con: duckdb.DuckDBPyConnection) -> int:
    """Parse raw_metar rows lacking a metar_obs child. Returns rows parsed."""
    pending = con.execute(
        """
        SELECT r.id, r.observed_at, r.raw_text
        FROM raw_metar r
        LEFT JOIN metar_obs o ON o.raw_metar_id = r.id
        WHERE o.id IS NULL
        """
    ).fetchall()

    rows = []
    for raw_id, observed_at, raw_text in pending:
        try:
            p = parse_metar(raw_text, observed_at)
        except Exception:
            continue  # leave unparsed; a parser improvement can retry later
        c = p.conditions
        rows.append((raw_id, p.icao, p.observed_at, c.wind_dir_deg, c.wind_spd_kt,
                     c.wind_gust_kt, c.vis_m, p.temp_c, p.dewpoint_c, p.qnh_hpa,
                     c.ceiling_ft, c.flight_category,
                     json.dumps(c.clouds), json.dumps(c.weather)))
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=_METAR_COLS)
    # clouds/weather are JSON text -> cast in the SELECT.
    select = ", ".join("clouds::JSON" if c == "clouds" else "weather::JSON" if c == "weather"
                       else c for c in _METAR_COLS)
    return _insert_select(con, "metar_obs", df, select, _METAR_COLS,
                          key_cols=["raw_metar_id"])


_GROUP_COLS = ["taf_forecast_id", "group_type", "probability", "valid_from", "valid_to",
               "wind_dir_deg", "wind_spd_kt", "wind_gust_kt", "vis_m", "ceiling_ft",
               "flight_category", "clouds", "weather"]


def parse_pending_taf(con: duckdb.DuckDBPyConnection) -> int:
    """Parse raw_taf rows lacking a taf_forecast child. Returns TAFs parsed.

    Forecasts inserted in one vectorized statement; ids mapped back by raw_taf_id;
    then all change groups inserted in one vectorized statement.
    """
    pending = con.execute(
        """
        SELECT r.id, r.issued_at, r.raw_text
        FROM raw_taf r
        LEFT JOIN taf_forecast f ON f.raw_taf_id = r.id
        WHERE f.id IS NULL
        """
    ).fetchall()

    parsed: list[tuple[int, ParsedTaf]] = []
    for raw_id, issued_at, raw_text in pending:
        try:
            parsed.append((raw_id, parse_taf(raw_text, issued_at)))
        except Exception:
            continue
    if not parsed:
        return 0

    fdf = pd.DataFrame(
        [(rid, p.icao, p.issued_at, p.valid_from, p.valid_to) for rid, p in parsed],
        columns=["raw_taf_id", "icao", "issued_at", "valid_from", "valid_to"],
    )
    _insert_select(con, "taf_forecast", fdf,
                   "raw_taf_id, icao, issued_at, valid_from, valid_to",
                   ["raw_taf_id", "icao", "issued_at", "valid_from", "valid_to"],
                   key_cols=["raw_taf_id"])

    idmap = dict(con.execute("SELECT raw_taf_id, id FROM taf_forecast").fetchall())

    group_rows = []
    for rid, p in parsed:
        fid = idmap.get(rid)
        if fid is None:
            continue
        for g in p.groups:
            c = g.conditions
            group_rows.append((fid, g.group_type, g.probability, g.valid_from, g.valid_to,
                               c.wind_dir_deg, c.wind_spd_kt, c.wind_gust_kt, c.vis_m,
                               c.ceiling_ft, c.flight_category,
                               json.dumps(c.clouds), json.dumps(c.weather)))
    if group_rows:
        gdf = pd.DataFrame(group_rows, columns=_GROUP_COLS)
        select = ", ".join("clouds::JSON" if c == "clouds" else "weather::JSON" if c == "weather"
                           else c for c in _GROUP_COLS)
        _insert_select(con, "taf_group", gdf, select, _GROUP_COLS)  # no key: 1:1 with new forecasts
    return len(parsed)
