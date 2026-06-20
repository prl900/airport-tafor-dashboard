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
    from wx.verification.bulk import run_profiles

    with get_connection() as con:
        n = run_profiles(con, ["categorical"], station or None)["categorical"]
    console.print(f"[green]Verified[/] {n} TAF-hours.")


@app.command()
def nwp(
    start: str = typer.Option(..., help="ISO date, inclusive"),
    end: str = typer.Option(..., help="ISO date, exclusive"),
    mode: str = typer.Option("timeseries", help="'timeseries' (per-station, efficient) or 'gridded'"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
) -> None:
    """Download ERA5 (Copernicus CDS) and extract per-station series into nwp_point.

    Requires cdsapi credentials in ~/.cdsapirc. NetCDF granules are cached in
    data/era5. 'timeseries' uses the ARCO/Zarr point dataset (one request per
    station over the whole range — best for backfill); 'gridded' downloads full
    Iberia years and extracts nearest gridpoints.
    """
    from datetime import timedelta

    from wx.ingestion.nwp_era5 import (
        download_month, download_station_timeseries, extract_points,
        extract_timeseries, load_dataset,
    )

    t0, t1 = _utc(start), _utc(end)
    with get_connection() as con:
        all_st = [
            dict(zip(("icao", "lat", "lon"), row))
            for row in con.execute("SELECT icao, lat, lon FROM stations").fetchall()
        ]
        stations = [s for s in all_st if not station or s["icao"] in station]
        total = 0

        if mode == "timeseries":
            for st in stations:
                console.print(f"  ERA5 ts {st['icao']}: downloading (CDS queue)…")
                path = download_station_timeseries(
                    st["icao"], st["lat"], st["lon"], t0, t1 - timedelta(days=1)
                )
                recs = extract_timeseries(load_dataset(path), st["icao"])
                recs = [r for r in recs if t0 <= r["valid_time"] < t1]
                ins = repo.store_nwp_points(con, recs)
                total += ins
                console.print(f"  ERA5 ts {st['icao']}: {len(recs)} rows, {ins} new")
        else:
            # Monthly granules: a full-year all-variable request exceeds the CDS cost cap.
            month = datetime(t0.year, t0.month, 1, tzinfo=timezone.utc)
            while month < t1:
                console.print(f"  ERA5 {month:%Y-%m}: downloading (CDS queue may take minutes)…")
                ds = load_dataset(download_month(month.year, month.month))
                recs = [r for r in extract_points(ds, stations) if t0 <= r["valid_time"] < t1]
                total += repo.store_nwp_points(con, recs)
                console.print(f"  ERA5 {month:%Y-%m}: {len(recs)} point-rows")
                month = datetime(month.year + (month.month // 12),
                                 (month.month % 12) + 1, 1, tzinfo=timezone.utc)
    console.print(f"[green]Stored[/] {total} nwp_point rows.")


@app.command(name="nwp-ifs")
def nwp_ifs(
    start: str = typer.Option(..., help="ISO date, inclusive (first run init day)"),
    end: str = typer.Option(..., help="ISO date, exclusive (last run init day)"),
    source: str = typer.Option("tigge", "--source",
                               help="historical IFS source: 'tigge' (ECMWF API) | 'cds'"),
    dataset: str = typer.Option(None, "--dataset",
                                help="CDS dataset id (required when --source cds)"),
    cycles: str = typer.Option("0,12", help="comma-separated run init hours, e.g. 0,12"),
    max_step: int = typer.Option(30, help="max lead hour to request (covers LEADS_DEFAULT)"),
    step_every: int = typer.Option(1, help="step spacing in hours (1 hourly, 3 three-hourly)"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
) -> None:
    """Download historical IFS forecasts into nwp_point as source='ifs'.

    One granule per run init (date x cycle); each lead step becomes a forecast row keyed by
    (ref_time, step_h). 'tigge' uses the ECMWF API (needs ~/.ecmwfapirc + TIGGE licence;
    cloud layers / cbh / gust / blh are absent from TIGGE). 'cds' needs a confirmed
    --dataset id (the free CDS does not host the operational HRES archive)."""
    from datetime import timedelta

    from wx.ingestion.nwp_ifs import (
        download_ifs, download_tigge, extract_points_fc, load_dataset, load_grib,
    )

    t0, t1 = _utc(start), _utc(end)
    cyc = [int(c) for c in cycles.split(",") if c.strip() != ""]
    steps = list(range(0, max_step + 1, step_every))
    if source == "cds" and not dataset:
        console.print("[red]--dataset is required when --source cds[/]")
        raise typer.Exit(code=1)
    with get_connection() as con:
        all_st = [
            dict(zip(("icao", "lat", "lon"), row))
            for row in con.execute("SELECT icao, lat, lon FROM stations").fetchall()
        ]
        stations = [s for s in all_st if not station or s["icao"] in station]
        total = runs = 0
        day = t0
        while day < t1:
            for hh in cyc:
                ref = day.replace(hour=hh, minute=0, second=0, microsecond=0)
                console.print(f"  IFS[{source}] {ref:%Y-%m-%d %HZ}: downloading…")
                if source == "tigge":
                    ds = load_grib(download_tigge(ref, steps))
                else:
                    ds = load_dataset(download_ifs(ref, steps, dataset=dataset))
                recs = extract_points_fc(ds, stations, ref)
                ins = repo.store_nwp_points(con, recs)
                total += ins
                runs += 1
                console.print(f"  IFS[{source}] {ref:%Y-%m-%d %HZ}: {len(recs)} rows, {ins} new")
            day += timedelta(days=1)
    console.print(f"[green]Stored[/] {total} IFS nwp_point rows from {runs} runs.")


@app.command()
def compare(
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
) -> None:
    """Generate baseline candidate TAFs and score them against the official TAFs."""
    from wx.ai.compare import comparison
    from wx.verification.bulk import run_profiles

    icaos = station or None
    with get_connection() as con:
        counts = run_profiles(con, ["persistence", "climatology"], icaos)
        console.print(f"[cyan]Candidate rows scored:[/] {counts}")
        targets = icaos or [r[0] for r in con.execute(
            "SELECT DISTINCT icao FROM verification_hourly").fetchall()]
        for icao in targets:
            table = Table(title=f"{icao}: forecaster comparison (higher score = better)")
            for c in ("Forecaster", "Mean score", "POD", "FAR", "HSS", "n"):
                table.add_column(c, justify="right")
            for row in comparison(con, icao):
                table.add_row(
                    row["profile"],
                    f"{row['mean_weighted_score']:.3f}" if row["mean_weighted_score"] else "—",
                    f"{row['POD']:.2f}" if row["POD"] is not None else "—",
                    f"{row['FAR']:.2f}" if row["FAR"] is not None else "—",
                    f"{row['HSS']:.2f}" if row["HSS"] is not None else "—",
                    str(row["n"]),
                )
            console.print(table)


@app.command()
def train(
    rung: str = typer.Option("gbm", help="linreg | rf | gbm | mlp"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
    sample_pct: int = typer.Option(5, help="%% of target obs to sample (dense grid is ~22M)"),
    train_end: str = typer.Option("2024-01-01", help="train < this <= val"),
    val_end: str = typer.Option("2025-01-01", help="val < this <= frozen test"),
    nwp_source: str = typer.Option("era5", "--nwp-source",
                                   help="NWP feature source: era5 (perfect-prognosis) | ifs (forecast)"),
) -> None:
    """Train a model-ladder rung; evaluate on the frozen test split vs the references."""
    from wx.ai.train import train_and_evaluate

    with get_connection(read_only=True) as con:
        try:
            rec = train_and_evaluate(con, rung, icaos=station or None, sample_pct=sample_pct,
                                     train_end=train_end, val_end=val_end,
                                     nwp_source=nwp_source)
        except MemoryError as exc:
            # The dataset memory guard refused this sample_pct (would OOM this box).
            console.print(f"[red]aborted:[/] {exc}")
            raise typer.Exit(code=1) from exc
    t, ref = rec["test"], rec["reference"]
    s = t["skill"]
    console.print(f"[bold]{rung}[/] trained on {rec['n_train']:,} rows, tested on {t['n']:,} "
                  f"(sample {sample_pct}%)")
    fnum = lambda v, d=3: f"{v:.{d}f}" if v is not None else "—"
    console.print(f"  HSS={fnum(s['HSS'])}  POD={fnum(s['POD'],2)}  FAR={fnum(s['FAR'],2)}  "
                  f"Brier={fnum(t['brier'])}  [bold]BSS={fnum(t['bss'])}[/]")
    console.print(f"  vis MAE={fnum(t['mae']['vis'],0)}m  ceil MAE={fnum(t['mae']['ceiling'],0)}ft  "
                  f"wind MAE={fnum(t['mae']['wind'],1)}kt")
    off = ref["official_bss"]
    console.print(f"  references — climatology BSS=0.000  official TAF BSS={fnum(off)}")
    if t["bss"] is not None and t["bss"] > 0:
        console.print("  [green]→ positive BSS: beats climatology[/]"
                      + (" and the official TAF" if off is not None and t["bss"] > off else ""))
    else:
        console.print("  → does not yet beat climatology (BSS<=0)")


@app.command(name="feature-importance")
def feature_importance(
    rung: str = typer.Option("gbm", help="rung to fit for the assessment"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
    sample_pct: int = typer.Option(5, help="%% of target obs to sample"),
    nwp_source: str = typer.Option("era5", "--nwp-source", help="era5 | ifs"),
    ablate: bool = typer.Option(False, "--ablate",
                                help="also run leave-one-group-out ablation (retrains per group)"),
    repeats: int = typer.Option(3, help="permutation repeats"),
) -> None:
    """Rank NWP predictors by the skill they contribute, to decide operational inclusion.

    Permutation importance (cheap) shuffles each variable group on the frozen test split;
    --ablate additionally retrains without each group. A group flagged all_nan has no
    ingested data yet — fetch it (gridded mode for cloud layers / candidates) to assess."""
    from wx.ai.dataset import LEADS_DEFAULT, build_samples, temporal_split
    from wx.ai.importance import ablation, permutation_importance
    from wx.ai.models import MultiTaskModel

    with get_connection(read_only=True) as con:
        df = build_samples(con, icaos=station or None, leads=LEADS_DEFAULT,
                           sample_pct=sample_pct, nwp_source=nwp_source)
        tr, va, te = temporal_split(df)
        if tr.empty or te.empty:
            console.print(f"[red]empty split[/] train={len(tr)} test={len(te)}")
            raise typer.Exit(code=1)
        model = MultiTaskModel(rung).fit(tr, val_df=va)
        imp = permutation_importance(model, te, n_repeats=repeats)

        table = Table(title=f"Permutation importance ({nwp_source}, {rung}, test split)")
        for c in ("group", "ΔHSS", "Δmae_vis", "Δmae_ceil", "data?"):
            table.add_column(c)
        fnum = lambda v, d=4: f"{v:.{d}f}" if v is not None else "—"
        for _, r in imp.iterrows():
            table.add_row(r["group"], fnum(r["d_hss"]), fnum(r["d_mae_vis"], 0),
                          fnum(r["d_mae_ceiling"], 0),
                          "[red]none[/]" if r["all_nan"] else "yes")
        console.print(table)
        console.print("ΔHSS > 0 = variable helps (shuffling it lost skill). "
                      "Δmae > 0 = error rose when removed.")

        if ablate:
            abl = ablation(con, rung=rung, icaos=station or None, sample_pct=sample_pct,
                           nwp_source=nwp_source)
            t2 = Table(title="Leave-one-group-out ablation (retrained)")
            for c in ("group", "HSS", "ΔHSS_vs_full"):
                t2.add_column(c)
            for _, r in abl.iterrows():
                t2.add_row(r["group"], fnum(r["hss"]), fnum(r["d_hss"]))
            console.print(t2)


@app.command()
def promote(
    rung: str = typer.Option("gbm", help="rung to gate, loaded from data/models/<rung>.joblib"),
    station: list[str] = typer.Option(None, "--station", help="ICAO(s); default = all"),
    register: bool = typer.Option(True, help="record as champion if it wins the gate"),
) -> None:
    """Gate a trained model against the champion on the frozen 2025 test; register if it wins.

    Promotion = paired bootstrap CI on the HSS difference excludes zero (ML_PLAN gate).
    """
    from wx.ai.promote import promote_if_better

    with get_connection(read_only=True) as con:
        d = promote_if_better(con, rung, icaos=station or None, register=register)
    fnum = lambda v, dp=3: f"{v:.{dp}f}" if v is not None else "—"
    hd = d["hss_diff"]
    console.print(f"[bold]{d['challenger']}[/] vs champion [bold]{d['champion']}[/] "
                  f"on {d['n_paired']:,} paired test hours")
    console.print(f"  HSS  challenger={fnum(d['challenger_hss'])}  champion={fnum(d['champion_hss'])}  "
                  f"Δ={fnum(hd['diff'])} CI[{fnum(hd['ci_low'])},{fnum(hd['ci_high'])}]")
    console.print(f"  BSS  challenger={fnum(d['challenger_bss'])}  champion={fnum(d['champion_bss'])}")
    if d["promote"]:
        console.print("  [green]→ PROMOTED: HSS gain is significant (CI excludes 0)[/]"
                      + ("  [registered]" if d.get("registered") else ""))
    else:
        console.print("  [yellow]→ not promoted: HSS gain not significant[/]")


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
