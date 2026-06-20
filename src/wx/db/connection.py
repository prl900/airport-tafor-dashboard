"""DuckDB connection management and schema initialisation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from wx.config import AIRPORTS, settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the spatial extension loaded."""
    path = str(db_path or settings.db_path)
    con = duckdb.connect(path, read_only=read_only)
    # spatial is needed for the stations.geom helpers / map queries.
    con.execute("INSTALL spatial; LOAD spatial;")
    # Render TIMESTAMPTZ in UTC so readback matches what we stored.
    con.execute("SET TimeZone='UTC';")
    return con


@contextmanager
def get_connection(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_db(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Apply the schema (idempotent) and seed the stations table."""
    own = con is None
    con = con or connect()
    try:
        con.execute(SCHEMA_PATH.read_text())
        _migrate(con)
        seed_stations(con)
    finally:
        if own:
            con.close()


def _migrate(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotent migrations for already-populated databases.

    schema.sql uses CREATE TABLE IF NOT EXISTS, so column additions don't reach an
    existing wx.duckdb — apply them here. Safe to re-run; no-ops on a fresh DB."""
    # nwp_point gained a forecast reference dimension (ref_time, step_h). Existing
    # ERA5 rows are backfilled as a zero-lead forecast: ref_time = valid_time, step_h = 0.
    con.execute("ALTER TABLE nwp_point ADD COLUMN IF NOT EXISTS ref_time TIMESTAMPTZ;")
    con.execute("ALTER TABLE nwp_point ADD COLUMN IF NOT EXISTS step_h INTEGER;")
    con.execute(
        "UPDATE nwp_point SET ref_time = valid_time, step_h = 0 WHERE ref_time IS NULL;"
    )
    # Candidate predictors (nullable; backfilled to NULL on existing rows).
    for col in ("cape_jkg", "blh_m", "tcwv_kgm2", "skt_c"):
        con.execute(f"ALTER TABLE nwp_point ADD COLUMN IF NOT EXISTS {col} DOUBLE;")


def seed_stations(con: duckdb.DuckDBPyConnection) -> None:
    """Upsert the configured seed airports into the stations table."""
    rows = [
        (a.icao, a.name, a.lat, a.lon, a.elevation_m, a.region) for a in AIRPORTS
    ]
    con.executemany(
        """
        INSERT INTO stations (icao, name, lat, lon, elevation_m, region)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (icao) DO UPDATE SET
            name = excluded.name,
            lat = excluded.lat,
            lon = excluded.lon,
            elevation_m = excluded.elevation_m,
            region = excluded.region
        """,
        rows,
    )
