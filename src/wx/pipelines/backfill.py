"""Typer CLI entrypoint (`wx`).

Phase 0 ships `initdb` and `stations`. The staged, idempotent `backfill` command
is implemented in Phase 1 (fetch -> store_raw -> parse -> store_parsed).
"""

from __future__ import annotations

from datetime import datetime, timezone

import typer
from dateutil import parser as dateparser
from rich.console import Console
from rich.table import Table

from wx.config import AIRPORTS, settings
from wx.db import repositories as repo
from wx.db.connection import get_connection, init_db
from wx.ingestion.metar_iem import IemMetarIngester
from wx.ingestion.ogimet import OgimetMetarIngester, OgimetTafIngester

app = typer.Typer(add_completion=False, help="Airport TAFOR dashboard pipelines.")
console = Console()


def _utc(s: str) -> datetime:
    dt = dateparser.parse(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


@app.command()
def initdb() -> None:
    """Create the DuckDB database, apply the schema, and seed stations."""
    init_db()
    with get_connection(read_only=True) as con:
        n = con.execute("SELECT count(*) FROM stations").fetchone()[0]
    console.print(f"[green]Initialised[/] {settings.db_path} — {n} stations seeded.")


@app.command()
def stations() -> None:
    """List the seeded airports."""
    with get_connection(read_only=True) as con:
        rows = con.execute(
            "SELECT icao, name, region, lat, lon FROM stations ORDER BY region, icao"
        ).fetchall()
    table = Table(title="Seeded Spanish airports")
    for col in ("ICAO", "Name", "Region", "Lat", "Lon"):
        table.add_column(col)
    for icao, name, region, lat, lon in rows:
        table.add_row(icao, name, region, f"{lat:.3f}", f"{lon:.3f}")
    console.print(table)


@app.command()
def backfill(
    start: str = typer.Option(..., help="ISO date, inclusive (e.g. 2020-01-01)"),
    end: str = typer.Option(..., help="ISO date, exclusive (e.g. 2026-01-01)"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all seeded"),
    metar: bool = typer.Option(True, help="Ingest METAR observations"),
    taf: bool = typer.Option(True, help="Ingest TAF forecasts (Ogimet gettafor)"),
    metar_source: str = typer.Option("iem", help="METAR source: 'iem' (fast) or 'ogimet'"),
) -> None:
    """Backfill METAR + TAF over [start, end): fetch -> store_raw -> parse -> store_parsed.

    Idempotent: re-running skips already-stored raw rows and already-parsed children.
    Ogimet requests are cached per (region-prefix, year), so the 1-request/minute
    bulk limit is paid at most once per granule even across many stations.
    """
    t0, t1 = _utc(start), _utc(end)
    icaos = station or [a.icao for a in AIRPORTS]

    with get_connection() as con:
        if metar:
            ingester = OgimetMetarIngester() if metar_source == "ogimet" else IemMetarIngester()
            with ingester as ing:
                total = 0
                for icao in icaos:
                    recs = ing.fetch_raw(icao, t0, t1)
                    ins = repo.store_raw_metar(con, recs)
                    total += ins
                    console.print(f"  METAR {icao}: fetched {len(recs)}, new {ins}")
                console.print(f"[cyan]METAR raw stored: {total} new[/]")

        if taf:
            with OgimetTafIngester() as ing:
                total = 0
                for icao in icaos:
                    recs = ing.fetch_raw(icao, t0, t1)
                    ins = repo.store_raw_taf(con, recs)
                    total += ins
                    console.print(f"  TAF {icao}: fetched {len(recs)}, new {ins}")
                console.print(f"[cyan]TAF raw stored: {total} new[/]")

        # parse stage (idempotent over whatever raw exists)
        n_metar = repo.parse_pending_metar(con) if metar else 0
        n_taf = repo.parse_pending_taf(con) if taf else 0
    console.print(f"[green]Parsed[/] {n_metar} METAR obs, {n_taf} TAFs.")


@app.command()
def verify(
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
) -> None:
    """Score stored TAFs against METAR observations (writes verification_hourly)."""
    from wx.verification.runner import verify_pending

    with get_connection() as con:
        n = verify_pending(con, station or None)
    console.print(f"[green]Verified[/] {n} TAF-hours.")


@app.command()
def status() -> None:
    """Show row counts across the pipeline tables."""
    with get_connection(read_only=True) as con:
        table = Table(title="Pipeline status")
        table.add_column("Table")
        table.add_column("Rows", justify="right")
        for t in ("stations", "raw_metar", "metar_obs", "raw_taf", "taf_forecast",
                  "taf_group", "verification_hourly", "nwp_point"):
            n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            table.add_row(t, f"{n:,}")
        console.print(table)


if __name__ == "__main__":
    app()
