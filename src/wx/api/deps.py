"""Shared API dependencies and query helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterator

import duckdb

from wx.db.connection import connect

# Columns stored as JSON text in DuckDB that should be decoded for API responses.
_JSON_COLUMNS = {"clouds", "weather", "prevailing", "tempo", "prob"}


def get_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """Read-only DuckDB connection per request."""
    con = connect(read_only=True)
    try:
        yield con
    finally:
        con.close()


def rows_to_dicts(cur: duckdb.DuckDBPyConnection) -> list[dict]:
    """Turn the last-executed cursor's result into a list of dicts."""
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        for col in _JSON_COLUMNS & d.keys():
            if isinstance(d[col], str):
                try:
                    d[col] = json.loads(d[col])
                except (ValueError, TypeError):
                    pass
        out.append(d)
    return out


def parse_dt(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
