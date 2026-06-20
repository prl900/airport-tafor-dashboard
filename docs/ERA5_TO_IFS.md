# ERA5 → IFS: operational NWP transition

## Context
The model trained in a **perfect-prognosis (PP)** setup: ERA5 *reanalysis* was joined into the
feature frame at both the issue time `T₀` and the valid hour `t`. Reanalysis has no forecast
error, so reported skill was optimistic (docs/ISSUES.md #4). This branch moves the NWP feature
pipeline onto genuine **IFS forecasts** so the model can run operationally and report honest
skill, and adds the data model, ingestion, and assessment tooling needed for it.

Full design + decisions: `~/.claude/plans/make-a-plan-for-zany-cloud.md` (planning notes).

## What changed

### 1. Forecast data model (`nwp_point`)
ERA5 analysis is one value per `(icao, valid_time)`; an IFS forecast has many values per
`valid_time` keyed by run init time and lead. `nwp_point` gained:
- `ref_time` (model init/cycle time) and `step_h` (lead hours); `ref_time + step_h = valid_time`.
- ERA5 is stored as a degenerate zero-lead forecast (`ref_time = valid_time`, `step_h = 0`).
- New dedup key `(icao, valid_time, source, ref_time, step_h)`.
- Idempotent migration for existing DBs in `connection._migrate()` (ALTER + backfill).

Files: `db/schema.sql`, `db/connection.py`, `db/repositories.py` (`store_nwp_points`),
`ingestion/nwp_era5.py` (`_point_records` emits ref_time/step_h).

### 2. Run-anchored join (`ai/dataset.py`)
`_joins(source)` selects NWP features per source:
- **era5**: keyed on `valid_time` (unchanged PP behaviour — bit-identical, since era5 rows are
  step 0).
- **ifs**: ASOF-picks the latest run with `ref_time ≤ T₀`, then ASOF-snaps to the nearest step
  of *that* run for both `et` (valid hour) and `e0` (T₀ anchor) — tolerating coarse steps.

`nwp_source` is threaded through `build_samples`, `build_inference_features` (so train and
serve are identical), and `train_and_evaluate` (IFS models saved with a `_ifs` suffix so the
ERA5-PP champion is preserved for comparison).

### 3. Features: cloud layers + candidate predictors
- **Low/medium/high cloud** (`lcc/mcc/hcc`) are now features (`f_et_lcc/mcc/hcc`, `f_tend_lcc`).
  Previously ingested but unused. (Needs ERA5 *gridded* mode — the timeseries/ARCO dataset only
  has total cloud.)
- **Candidate predictors** added (nullable, NaN-tolerant): `cape`, `blh` (boundary-layer
  height), `tcwv`, `skt` (skin temperature) → `f_et_cape/blh/tcwv/skt`, `f_tend_skt`.

### 4. IFS ingestion (`ingestion/nwp_ifs.py`, new)
- `download_tigge()` — historical ECMWF forecasts via the ECMWF API (TIGGE); `download_ifs()` —
  CDS path (dataset id parameterised).
- `_point_records_fc()` — step-indexed extraction mirroring ERA5 units, stamps ref_time/step_h,
  de-accumulates `tp`. `load_grib()` reads GRIB2 (cfgrib) and renames short names.
- CLI `wx nwp-ifs --source tigge|cds …` (`pipelines/backfill.py`).
- **ERA5 gridded chunking fix:** a full-year CDS request exceeds the cost cap; the gridded path
  now downloads **monthly** granules (`download_month`).

### 5. Feature-importance / assessment harness (`ai/importance.py`, new)
- `permutation_importance()` — shuffle each variable group on the eval split, measure ΔHSS /
  Δvis-MAE / Δceiling-MAE (cheap, no retrain). Flags `all_nan` groups (variable not yet ingested).
- `ablation()` — leave-one-group-out retrain.
- CLI `wx feature-importance [--ablate]`.

### 6. Parquet-backed `nwp_point` (prototype, `db/parquet_store.py`, new)
Optional storage layer: partitioned Parquet (`source=/year=/month=`) read by DuckDB as a view,
with latest-write-wins dedup. Removes DuckDB's single-writer lock (each ingest writes its own
file → parallel-safe) while keeping the ASOF query path unchanged. **Prototype only** — not yet
wired into the default ingest path.

### Tests
`test_dataset_ifs.py` (run-anchored join + cloud features), `test_nwp_ifs.py` (forecast
extraction + de-accumulation), `test_importance.py` (harness + data-availability flag),
`test_parquet_store.py` (build_samples over Parquet view + idempotent upsert), and a
`step_h=0` assertion in `test_nwp.py`. **Full suite: 44 passed.**

### Dependencies
`nwp` extra gains `ecmwf-api-client`, `cfgrib`, `eccodeslib` (GRIB2 reading for TIGGE / Feed A1).

## Empirical findings (2023-train / 2024-val / 2025-test, ERA5 PP, gbm, 10% sample)
Test 2025: **HSS 0.400 · BSS 0.211 · vis MAE 494 m · ceiling MAE 5038 ft · wind MAE 1.9 kt.**

**Permutation importance on 2025** (NWP groups, by ΔHSS): `blh` 0.035 (#1 NWP) · t2m 0.033 ·
wind 0.032 · `tcwv` 0.031 · `cloud_layers` 0.029 · `skt` 0.023 · cbh 0.011 · tcc 0.006 ·
cape 0.003 · tp ~0 · msl ~0. (T−Td spread dominates at 0.231 but it's the METAR anchor, not
NWP.) **`cloud_layers` is by far the largest *ceiling* driver** (Δceiling-MAE 1386 ft; next is
tp at 533). The candidate predictors added in this branch all validate on the full year —
**`blh` is the top NWP feature, `tcwv` #4, `skt` solid** — and `blh`/`tcwv` are not in Open
Data/TIGGE.

**Cloud head-to-head (2025):** layers-only HSS 0.403 / BSS 0.222 / ceil 5043 ≥ full
0.400/0.211/5038 > tcc-only 0.391/0.219/5150 > no-cloud 0.382/0.211/5237.
- layers ≥ full → **`tcc` adds ~nothing on top of the layers (redundant)**.
- layers beat tcc-only and no-cloud; the clearest advantage is **ceiling** (~110 ft better than
  tcc-only, ~195 ft than no-cloud).
- The HSS margin over tcc-only is modest on the full year, but was *much* larger in a
  winter-only test (layers 0.406/0.224 vs tcc-only 0.359/0.140 — tcc-only fell *below*
  no-cloud). Fog season is when the low-cloud split matters most, and it's the operationally
  salient case.

**Parquet vs DuckDB (full scale, 473k `nwp_point` → 852k samples):** identical results,
`build_samples` 6.6 s vs 6.6 s — no regression (was ~8% faster at 210k; dedup-window cost grows
with scale, offset by partition pruning).

*Caveats: ERA5 perfect-prognosis (upper bound — magnitudes shrink on real IFS forecasts, ranking
should hold), 10% sample, no official-TAF baseline yet (no `verify` run).*

## Operational data sourcing (reference)
Cloud layers `lcc/mcc/hcc` availability: **ERA5** (gridded) ✓ · **TIGGE** ✗ (tcc only) ·
**ECMWF Open Data** ✗ (tcc only) · **MARS / licensed real-time feed** ✓. So the operational
model can only use the layers via a licensed HRES feed (e.g. Feed A1) for serving and
MARS/Feed-A1 archive for training.

## Not in this branch (follow-ups)
- Wire Parquet store into the default ingest + `init_db` view (behind a config flag).
- Feed A1 (GCS) operational ingester (`nwp_ifs_feed.py`) + scheduled inference loop.
- MARS historical-forecast training path (`download_mars`, params 186/187/188 + surface set).
- Full-scale 2023–24-train / 2025-test assessment + optimism-gap measurement vs official TAF.
