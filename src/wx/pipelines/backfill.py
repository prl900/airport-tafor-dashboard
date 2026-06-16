"""Typer CLI entrypoint (`wx`).

Phase 0 ships `initdb` and `stations`. The staged, idempotent `backfill` command
is implemented in Phase 1 (fetch -> store_raw -> parse -> store_parsed).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from wx.config import settings
from wx.db.connection import get_connection, init_db

app = typer.Typer(add_completion=False, help="Airport TAFOR dashboard pipelines.")
console = Console()


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
    station: list[str] = typer.Option(None, help="ICAO(s); default = all seeded"),
) -> None:
    """Backfill METAR + TAF for the given window (implemented in Phase 1)."""
    raise typer.Exit(
        console.print("[yellow]backfill is implemented in Phase 1 — coming next.[/]") or 0
    )


if __name__ == "__main__":
    app()
