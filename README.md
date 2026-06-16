# Airport TAFOR Verification Dashboard

Ingest, parse and verify **METAR** (observations) and **TAFOR/TAF** (forecasts) for
Spanish airports over a fixed historical window (2020-2025), with a FastAPI backend,
a DuckDB store, and a React + MapLibre + Plotly dashboard.

See the full plan in `docs/PLAN.md` (mirrored from the approved planning doc).

## Stack
- **Backend:** Python 3.12, FastAPI, DuckDB (single file `wx.duckdb`)
- **Parsing:** `metar-taf-parser-mivek`
- **NWP:** ERA5 reanalysis via Copernicus CDS (`cdsapi`) — optional extra
- **Frontend:** Vite + React + MapLibre GL JS + Plotly

## Setup
```bash
uv sync                      # core deps
uv sync --extra nwp          # + ERA5 ingestion deps (Phase 3)
uv sync --extra dev          # + pytest / ruff
```

## Initialise the database
```bash
uv run wx initdb             # creates wx.duckdb, applies schema, seeds stations
uv run wx stations           # list seeded Spanish airports
```

## Pipeline commands
```bash
# Backfill METAR (IEM) + TAF (Ogimet) -> store raw -> parse to components
uv run wx backfill --station LEMD --start 2023-01-01 --end 2023-02-01   # smoke test
uv run wx backfill --start 2020-01-01 --end 2026-01-01                  # full seed/window
uv run wx backfill --metar-source ogimet ...                           # METAR from Ogimet too

# Score TAFs against observations (POD/FAR/CSI/HSS, element errors)
uv run wx verify

# Download ERA5 NWP (needs ~/.cdsapirc) and extract per-station series
uv run wx nwp --start 2023-01-01 --end 2023-02-01

# Generate baseline candidate TAFs and compare skill vs the official TAFs
uv run wx compare

uv run wx status     # row counts across the pipeline
```

Ogimet uses the bulk `getmetar`/`gettafor` tools (1 request/minute/IP, fetched per
region-prefix per year and cached — so a full 24-station backfill makes only a
handful of live requests). A full-year granule lives on slow storage (~2-3 min).

## Run the API
```bash
uv run uvicorn wx.api.app:app --reload --port 8000
# docs at http://localhost:8000/docs
```

## Run the dashboard
```bash
cd frontend && npm install && npm run dev
```

## Tests
```bash
uv run pytest
```

## Data sources
- **METAR:** Iowa Environmental Mesonet (IEM) ASOS archive (free, 1 req/s).
- **TAF:** Ogimet (free, scrape — paced + cached). AEMET OpenData as authoritative spot-check.
- **NWP:** ERA5 single-levels via Copernicus CDS.
