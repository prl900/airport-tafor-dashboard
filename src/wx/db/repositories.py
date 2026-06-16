"""Data-access helpers. Storing raw messages and parsed components are separate,
idempotent stages: the parse stage reads raw rows that have no parsed child yet,
so re-running never duplicates and improving the parser re-parses cleanly."""

from __future__ import annotations

import json

import duckdb

from wx.parsing.metar import parse_metar
from wx.parsing.taf import ParsedTaf, parse_taf

# --- raw storage -----------------------------------------------------------


def store_raw_metar(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """records: {icao, observed_at, raw_text, source}. Returns rows inserted."""
    if not records:
        return 0
    before = con.execute("SELECT count(*) FROM raw_metar").fetchone()[0]
    con.executemany(
        """
        INSERT INTO raw_metar (icao, observed_at, raw_text, source, ingested_at)
        VALUES (?, ?, ?, ?, now())
        ON CONFLICT DO NOTHING
        """,
        [(r["icao"], r["observed_at"], r["raw_text"], r["source"]) for r in records],
    )
    return con.execute("SELECT count(*) FROM raw_metar").fetchone()[0] - before


def store_raw_taf(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """records: {icao, issued_at, valid_from, valid_to, raw_text, source}."""
    if not records:
        return 0
    before = con.execute("SELECT count(*) FROM raw_taf").fetchone()[0]
    con.executemany(
        """
        INSERT INTO raw_taf (icao, issued_at, valid_from, valid_to, raw_text, source, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, now())
        ON CONFLICT DO NOTHING
        """,
        [
            (r["icao"], r["issued_at"], r.get("valid_from"), r.get("valid_to"),
             r["raw_text"], r["source"])
            for r in records
        ],
    )
    return con.execute("SELECT count(*) FROM raw_taf").fetchone()[0] - before


# --- parse stage -----------------------------------------------------------


_METAR_INSERT = """
    INSERT INTO metar_obs
      (raw_metar_id, icao, observed_at, wind_dir_deg, wind_spd_kt, wind_gust_kt,
       vis_m, temp_c, dewpoint_c, qnh_hpa, ceiling_ft, flight_category, clouds, weather)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT DO NOTHING
"""
_BATCH = 100_000


def parse_pending_metar(con: duckdb.DuckDBPyConnection) -> int:
    """Parse raw_metar rows lacking a metar_obs child (batched). Returns rows parsed.

    Inserts are batched via executemany — millions of single-row autocommit inserts
    would otherwise dominate the full backfill.
    """
    pending = con.execute(
        """
        SELECT r.id, r.observed_at, r.raw_text
        FROM raw_metar r
        LEFT JOIN metar_obs o ON o.raw_metar_id = r.id
        WHERE o.id IS NULL
        """
    ).fetchall()

    rows: list[tuple] = []
    n = 0
    for raw_id, observed_at, raw_text in pending:
        try:
            p = parse_metar(raw_text, observed_at)
        except Exception:
            continue  # leave unparsed; a parser improvement can retry later
        c = p.conditions
        rows.append((
            raw_id, p.icao, p.observed_at, c.wind_dir_deg, c.wind_spd_kt, c.wind_gust_kt,
            c.vis_m, p.temp_c, p.dewpoint_c, p.qnh_hpa, c.ceiling_ft, c.flight_category,
            json.dumps(c.clouds), json.dumps(c.weather),
        ))
        if len(rows) >= _BATCH:
            con.executemany(_METAR_INSERT, rows)
            n += len(rows)
            rows = []
    if rows:
        con.executemany(_METAR_INSERT, rows)
        n += len(rows)
    return n


def store_nwp_points(con: duckdb.DuckDBPyConnection, records: list[dict]) -> int:
    """Upsert ERA5 per-station point series into nwp_point. Returns rows written."""
    if not records:
        return 0
    before = con.execute("SELECT count(*) FROM nwp_point").fetchone()[0]
    con.executemany(
        """
        INSERT INTO nwp_point
          (icao, valid_time, source, wind10m_spd, wind10m_dir, gust, t2m_c, d2m_c,
           tcc, lcc, mcc, hcc, cbh_m, tp_mm, mslp_hpa)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            (r["icao"], r["valid_time"], r["source"], r["wind10m_spd"], r["wind10m_dir"],
             r["gust"], r["t2m_c"], r["d2m_c"], r["tcc"], r["lcc"], r["mcc"], r["hcc"],
             r["cbh_m"], r["tp_mm"], r["mslp_hpa"])
            for r in records
        ],
    )
    return con.execute("SELECT count(*) FROM nwp_point").fetchone()[0] - before


def parse_pending_taf(con: duckdb.DuckDBPyConnection) -> int:
    """Parse raw_taf rows lacking a taf_forecast child (batched). Returns TAFs parsed.

    Forecasts are inserted in one executemany, then their ids are mapped back by
    raw_taf_id to insert all change groups in a second executemany — avoiding a
    per-TAF insert+select+N-inserts round trip over hundreds of thousands of TAFs.
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

    con.executemany(
        """
        INSERT INTO taf_forecast (raw_taf_id, icao, issued_at, valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING
        """,
        [(rid, p.icao, p.issued_at, p.valid_from, p.valid_to) for rid, p in parsed],
    )

    # Map raw_taf_id -> taf_forecast.id for the rows we just inserted.
    idmap = dict(con.execute("SELECT raw_taf_id, id FROM taf_forecast").fetchall())

    group_rows = []
    for rid, p in parsed:
        fid = idmap.get(rid)
        if fid is None:
            continue
        for g in p.groups:
            c = g.conditions
            group_rows.append((
                fid, g.group_type, g.probability, g.valid_from, g.valid_to,
                c.wind_dir_deg, c.wind_spd_kt, c.wind_gust_kt, c.vis_m, c.ceiling_ft,
                c.flight_category, json.dumps(c.clouds), json.dumps(c.weather),
            ))
    if group_rows:
        con.executemany(
            """
            INSERT INTO taf_group
              (taf_forecast_id, group_type, probability, valid_from, valid_to,
               wind_dir_deg, wind_spd_kt, wind_gust_kt, vis_m, ceiling_ft,
               flight_category, clouds, weather)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            group_rows,
        )
    return len(parsed)
