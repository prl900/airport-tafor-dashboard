# Airport TAFOR Verification Dashboard — Implementation Plan

## Context

We are building a greenfield system to ingest, store, parse, and analyse airport
weather data for **Spain**, iterating over a **fixed historical window (2020-2025)**
rather than a live feed. The goal is a dashboard that lets users interactively
explore **METAR** (observations) and **TAFOR/TAF** (forecasts) per airport and as
group summaries, and a backend that **decomposes the encoded messages** into
components (wind, visibility, clouds/ceiling, temperature, dewpoint, pressure,
weather phenomena) so it can **score TAF forecast quality against the observed
METARs** (current and historical). Later phases download **ECMWF ERA5** NWP fields
and use **AI models to auto-generate alternative TAFs**, judged by the same
verification engine, with the long-term aim of beating the official forecasts.

The directory `/home/prl900/Downloads/airport-tafor-dashboard` is empty except for
`.claude/`. It currently sits inside the parent `~/Downloads` git repo, so the first
step is `git init` here to make a dedicated repository.

### Confirmed decisions
- **Database:** DuckDB (single-file, zero-ops, excellent for batch 2020-2025 OLAP scans). Accepts a harder pivot to live concurrent writes later — acceptable since the focus is historical iteration.
- **Frontend:** React (Vite) + MapLibre GL JS + Plotly, decoupled SPA against the FastAPI JSON API.
- **NWP source:** ERA5 reanalysis via Copernicus CDS (`cdsapi`), free, hourly, 0.25°.
- **Scope:** Detail all phases 0-4; build in sequence.

## Data sources (all free, historical Spain)
- **METAR (primary):** Iowa Environmental Mesonet (IEM) ASOS archive — `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`. No key; **1 req/s** limit; query per station-year with `data=metar&format=onlycomma&tz=Etc/UTC`. **Store the raw METAR text and re-parse ourselves** (IEM precip is null for non-US; we read phenomena from raw text).
- **TAF (primary):** Ogimet — `https://www.ogimet.com/utafs.phtml.en` (and `display_metars2.php`, `tipo=ALL`). Plain-text/HTML scrape, **fragile**: gentle pacing (1 req / several s), `tenacity` retry/backoff, cache every raw response so re-parsing never re-downloads.
- **Authoritative supplement / spot-check:** AEMET OpenData (`opendata.aemet.es`, free key) — official Spanish source; restrictive quotas, used for validation and as "official TAF" identity in Phase 4.
- **NWP:** ERA5 single-levels via `cdsapi`. Iberia bbox `area=[44,-10,35,4]`. Variables: `10m_u/v_wind`, `10m_wind_gust`, `2m_temperature`, `2m_dewpoint_temperature`, `total/low/medium/high_cloud_cover`, `cloud_base_height`, `total_precipitation`, `mean_sea_level_pressure`. Request **per-year, per-variable NetCDF** granules (resumable). **Known limits to document:** ERA5 CBH is biased; no reliable surface-visibility field (derive a proxy or omit on the NWP side).

### Seed airports (~24, kept in a config/seed table, extensible)
LEMD, LEBL, LEPA, LEMG, LEZL, LEAL, LEVC, LEBB, LEST, LEGE, LEVT, LEXJ, LEAS,
LEMH, LEIB, LEMI, LERS, GCLP, GCXO, GCTS, GCFV, GCRR, GCLA, GEML.

## Parsing
- **Primary lib:** `metar-taf-parser-mivek` (PyPI; GitHub `mivek/python-metar-taf-parser`). The only well-maintained Python lib with **full TAF decomposition** — base + trend groups (`FM/BECMG/TEMPO/PROB30/PROB40/PROB_TEMPO`) each with its own validity window + probability, plus wind/vis/clouds/weather/temp. Use its `MetarParser` and `TAFParser` so METAR and TAF share one object model.
- **Fallback / cross-check:** classic `python-metar` for METAR edge cases.
- Add a thin **`normalize.py`** layer that converts parsed objects into canonical SI components plus a **derived ceiling** (lowest BKN/OVC/VV layer) and **flight category**. Do not store raw library objects. Handle `CAVOK/NSC/SKC`, `VV///`, `VRB` wind, `9999` visibility, metric (ICAO) units — Spain is metric, not statute miles.

## Verification methodology (Phase 2)
Compare each TAF hour-by-hour over its validity against the METAR(s) in that hour:
1. **Flight-category (ceiling+visibility)** — derive VFR/MVFR/IFR/LIFR for forecast and obs; **configurable band table (FAA vs ICAO/metric)**. Categorical contingency: hit/miss/false-alarm/correct-negative → POD, FAR, CSI, bias, Heidke Skill Score, aggregated per airport / category / lead-time.
2. **Element-wise continuous** — wind speed/direction (±5 kt / ±30°), temp, dewpoint, visibility, ceiling: MAE/RMSE/bias + within-tolerance hit rates.
3. **Time-window / change-group matching** — expand TAF groups into an **hourly expected-state timeline**; apply a **pluggable scoring profile**, including the NWS weighted scheme (prevailing ±7, TEMPO ±4, PROB40 ±2, PROB30 ±1).

Store results at **hourly granularity** so they roll up to any aggregation. Design the scorer as a **pure function over (forecast timeline, observations)** so an AI-generated TAF (Phase 4) plugs into the same judge unchanged.

## Architecture & module layout

```
airport-tafor-dashboard/
  pyproject.toml            # uv; fastapi, uvicorn, duckdb, pydantic, pydantic-settings,
                            #   pandas, xarray, netCDF4, cdsapi, httpx, tenacity,
                            #   metar-taf-parser-mivek, typer
  src/wx/
    config.py               # settings + airport seed list
    db/
      connection.py         # DuckDB connection (load 'spatial' extension), migrations
      schema.sql            # table DDL (below)
      repositories.py       # data-access helpers
    ingestion/
      base.py               # Ingester ABC: fetch_raw -> store_raw (retry/backoff/cache)
      metar_iem.py          # IEM ASOS, 1 req/s, per station-year
      taf_ogimet.py         # Ogimet TAF scrape, gentle, cached raw
      aemet.py              # AEMET OpenData supplement/validation
      nwp_era5.py           # cdsapi retrieval, Iberia subset, per-station extraction
    parsing/
      metar.py  taf.py  normalize.py
    verification/
      timeline.py  align.py  scores.py  runner.py
    ai/                     # Phase 4
      features.py  generate.py  compare.py
    pipelines/
      backfill.py           # Typer CLI: idempotent staged backfill(start,end,stations)
      schedule.py           # later: same stages on a rolling window (live pivot)
    api/
      app.py  deps.py
      routes/  stations.py  metar.py  taf.py  verification.py  nwp.py
  frontend/                 # Vite + React + MapLibre GL JS + Plotly
  data/
    raw_cache/              # cached HTTP responses (re-parse without re-download)
    era5/                   # per-year/var NetCDF granules (gridded, kept out of DuckDB)
  tests/
```

Core abstraction: `Ingester` does `fetch_raw()` → `store_raw()`; parsing is a
**separate idempotent stage** (`parse()` → `store_parsed()`), so improving the parser
re-parses all history without re-downloading. Each pipeline stage (fetch → parse →
expand-timeline → verify) is independently re-runnable; idempotency via primary
keys + upsert.

### DuckDB schema sketch (`db/schema.sql`)
- `stations(icao PK, name, lat, lon, elevation_m, region, geom)` — load DuckDB `spatial` ext for `geom`/map queries.
- `raw_metar(id, icao, observed_at, raw_text, source, ingested_at, UNIQUE(icao,observed_at,raw_text))`
- `raw_taf(id, icao, issued_at, valid_from, valid_to, raw_text, source, ingested_at, UNIQUE(icao,issued_at,raw_text))`
- `metar_obs(id, raw_metar_id, icao, observed_at, wind_dir_deg, wind_spd_kt, wind_gust_kt, vis_m, temp_c, dewpoint_c, qnh_hpa, ceiling_ft, flight_category, clouds JSON, weather JSON)`
- `taf_forecast(id, raw_taf_id, icao, issued_at, valid_from, valid_to)`
- `taf_group(id, taf_forecast_id, group_type, probability, valid_from, valid_to, wind_*, vis_m, ceiling_ft, flight_category, clouds JSON, weather JSON)`
- `taf_expected_hourly(taf_forecast_id, icao, valid_hour, prevailing JSON, tempo JSON, prob JSON, PK(taf_forecast_id,valid_hour))`
- `verification_hourly(id, taf_forecast_id, icao, valid_hour, lead_time_h, scoring_profile, fcst_category, obs_category, category_outcome, wind_err_kt, dir_err_deg, temp_err_c, vis_err_m, ceiling_err_ft, weighted_score)`
- `nwp_point(icao, valid_time, source, wind10m_spd, wind10m_dir, gust, t2m_c, d2m_c, tcc, lcc, mcc, hcc, cbh_m, tp_mm, mslp_hpa, PK(icao,valid_time,source))`

DuckDB has no continuous aggregates → precompute dashboard roll-ups as **views or
materialised summary tables** (monthly POD/FAR per airport per category) refreshed
after each verification run.

### Frontend
Vite + React + **MapLibre GL JS** (free vector tiles, no token; airports as a GeoJSON
layer colour-coded by flight category / verification score) + **Plotly** (per-airport
time series, TAF-vs-METAR overlays, verification scorecards). Fully decoupled SPA
consuming FastAPI JSON. CORS enabled in `api/app.py`.

## Phased roadmap

- **Phase 0 — Scaffold (~0.5 wk):** `git init`; `uv` project + deps; DuckDB connection + `schema.sql`; `config.py` + airport seed; `Ingester` base + raw tables; `data/raw_cache` & `data/era5`.
- **Phase 1 — MVP ingest+parse+store+dashboard (~2 wk):** IEM METAR + Ogimet TAF backfill (raw) for ~24 stations, 2020-2025 → parse with `metar-taf-parser-mivek` + normalise → `metar_obs`, `taf_forecast`/`taf_group`. FastAPI read endpoints (`/stations`, `/stations/{icao}/metar`, `/taf`). React map of Spain + per-airport Plotly charts with TAF-vs-METAR overlay.
- **Phase 2 — Verification engine (~1.5 wk):** `timeline.py` expands TAF → hourly expected state; `align.py` aligns METARs; `scores.py` categorical (POD/FAR/CSI/HSS) + element RMSE + NWS-weighted → `verification_hourly`; summary views; `/verification` endpoints; dashboard group summaries by month / lead-time and per-airport scorecards.
- **Phase 3 — NWP ingestion (~1.5 wk):** `cdsapi` ERA5 Iberia subset, per-year/var NetCDF in `data/era5/`; extract per-station point series → `nwp_point`; dashboard NWP overlays; document CBH bias + visibility handling.
- **Phase 4 — AI-generated TAFs (open-ended):** `ai/features.py` builds NWP+obs feature vectors per airport-hour; train a model (natural fit for the `mltrain` cluster + MLflow) to emit candidate TAF components; `ai/compare.py` scores candidate vs official TAF through the **same Phase-2 verifier** for apples-to-apples comparison.

## Verification (how to test end-to-end)
- **Parsing:** unit tests with fixture METAR/TAF strings (incl. CAVOK, VV///, PROB40 TEMPO, VRB wind, 9999) asserting normalised components + derived ceiling/flight category.
- **Ingestion:** run `backfill.py` for one station (e.g. LEMD) over one month against IEM/Ogimet; assert `raw_metar`/`raw_taf` row counts and that re-running is idempotent (no duplicates). Confirm `data/raw_cache` hits avoid re-download.
- **Verification:** seed a known TAF + matching METARs, run `verification/runner.py`, assert contingency counts and weighted scores against hand-computed values.
- **API:** start `uvicorn`, hit `/stations`, `/stations/LEMD/metar?from&to`, `/verification/LEMD`; assert JSON shape.
- **Frontend:** `npm run dev`, confirm Spain map renders airports colour-coded by category and per-airport charts show TAF-vs-METAR overlays.
- **NWP (Phase 3):** small single-day ERA5 request for the Iberia box; assert `nwp_point` extraction for LEMD matches nearest-gridpoint values.

## Biggest risks / flags
- **Ogimet fragility** is the main data-supply risk (TAF source). Mitigate with gentle pacing, caching, retry/backoff; AEMET as authoritative fallback/validation.
- **DuckDB single-writer** makes a future live pivot harder; staged idempotent pipelines keep that migration to a storage swap, not a rewrite.
- **ERA5 CBH bias / no visibility** — calibrate CBH or treat as a feature; verify visibility only on the METAR/TAF side.
