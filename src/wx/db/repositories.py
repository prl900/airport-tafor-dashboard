"""Data-access helpers. Storing raw messages and parsed components are separate,
idempotent stages: the parse stage reads raw rows that have no parsed child yet,
so re-running never duplicates and improving the parser re-parses cleanly."""

from __future__ import annotations

import json
from datetime import datetime

import duckdb

from wx.parsing.metar import ParsedMetar, parse_metar
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

    n = 0
    for raw_id, observed_at, raw_text in pending:
        try:
            p = parse_metar(raw_text, observed_at)
        except Exception:
            continue  # leave unparsed; a parser improvement can retry later
        _insert_metar_obs(con, raw_id, p)
        n += 1
    return n


def _insert_metar_obs(con: duckdb.DuckDBPyConnection, raw_id: int, p: ParsedMetar) -> None:
    c = p.conditions
    con.execute(
        """
        INSERT INTO metar_obs
          (raw_metar_id, icao, observed_at, wind_dir_deg, wind_spd_kt, wind_gust_kt,
           vis_m, temp_c, dewpoint_c, qnh_hpa, ceiling_ft, flight_category, clouds, weather)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            raw_id, p.icao, p.observed_at, c.wind_dir_deg, c.wind_spd_kt, c.wind_gust_kt,
            c.vis_m, p.temp_c, p.dewpoint_c, p.qnh_hpa, c.ceiling_ft, c.flight_category,
            json.dumps(c.clouds), json.dumps(c.weather),
        ],
    )


def parse_pending_taf(con: duckdb.DuckDBPyConnection) -> int:
    """Parse raw_taf rows lacking a taf_forecast child. Returns rows parsed."""
    pending = con.execute(
        """
        SELECT r.id, r.issued_at, r.raw_text
        FROM raw_taf r
        LEFT JOIN taf_forecast f ON f.raw_taf_id = r.id
        WHERE f.id IS NULL
        """
    ).fetchall()

    n = 0
    for raw_id, issued_at, raw_text in pending:
        try:
            p = parse_taf(raw_text, issued_at)
        except Exception:
            continue
        _insert_taf(con, raw_id, p)
        n += 1
    return n


def _insert_taf(con: duckdb.DuckDBPyConnection, raw_id: int, p: ParsedTaf) -> None:
    con.execute(
        """
        INSERT INTO taf_forecast (raw_taf_id, icao, issued_at, valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [raw_id, p.icao, p.issued_at, p.valid_from, p.valid_to],
    )
    row = con.execute("SELECT id FROM taf_forecast WHERE raw_taf_id = ?", [raw_id]).fetchone()
    if row is None:
        return
    fid = row[0]
    for g in p.groups:
        c = g.conditions
        con.execute(
            """
            INSERT INTO taf_group
              (taf_forecast_id, group_type, probability, valid_from, valid_to,
               wind_dir_deg, wind_spd_kt, wind_gust_kt, vis_m, ceiling_ft,
               flight_category, clouds, weather)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fid, g.group_type, g.probability, g.valid_from, g.valid_to,
                c.wind_dir_deg, c.wind_spd_kt, c.wind_gust_kt, c.vis_m, c.ceiling_ft,
                c.flight_category, json.dumps(c.clouds), json.dumps(c.weather),
            ],
        )
