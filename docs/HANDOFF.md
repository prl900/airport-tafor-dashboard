# Handoff — TAFOR ML, CPU dev box → GPU box

This repo was developed on a 15 GB / 12-core **CPU** box. The tabular model ladder
(linreg → gbm) is done and calibrated; the next work — **mlp** and **Phase D
(sequence/probabilistic: TFT / seq2seq)** — needs a **GPU box**. This doc is the
pickup point for a fresh Claude session there.

## TL;DR state (2026-06-17)
- **Champion: calibrated `gbm`.** On the frozen 2025 test it beats the official TAF
  on **both** primary metrics (HSS) and the probabilistic metric (BSS).
  - gbm: **BSS +0.294** (CI [+0.281, +0.305], excludes 0), **HSS 0.476** (thr 0.32)
  - linreg: BSS +0.264, HSS 0.444
  - official TAF skyline: BSS −0.108, HSS ~0.37 ; climatology floor: BSS 0
- **The fix that mattered was calibration, not data volume.** Tree/NN models had
  good discrimination but garbage probabilities; isotonic calibration + a val-tuned
  decision threshold (fit on the 2024 split) fixed it. See `data/research_log.jsonl`.
- **Data saturates at ~5%** sample for the current tabular feature set (9% ≈ 5% within
  CI). More rows won't help these rungs — features and model class will.

## What's built (phases A–C + part of D)
- **Causal dataset** `src/wx/ai/dataset.py` — ASOF-joined features, strict `obs ≤ T0`
  contract, temporal splits (train <2024 / val 2024 / **frozen test 2025**).
  Now includes **METAR-lag features** (t0−1/−3/−6 h + current−lag tendencies; 33→54
  features) — the deterioration signal for fog onset. **Not yet retrained into the
  champion** (see open task 1).
- **Model ladder** `src/wx/ai/models.py` — `MultiTaskModel` (per-target regressors +
  classifiers, shared preprocessor) + isotonic calibration + HSS-optimal threshold,
  wrapped as `ModelForecaster` so the verifier scores it like the official TAF.
- **Train driver** `src/wx/ai/train.py` — `train_and_evaluate` fits on train, calibrates
  on val, reports val+test, bootstraps a BSS CI; appends to the research log.
- **Promotion gate** `src/wx/ai/promote.py` (+ `wx promote`) — paired bootstrap on the
  HSS difference vs the champion on frozen test; registers `data/models/champion.json`.
- **Memory guard** `src/wx/ai/dataset.py` — estimates peak RAM from row count and refuses
  / clamps over-budget builds. **This is CPU-box-specific** (see "On the GPU box").

## On the GPU box — setup
1. Clone, then `uv sync --extra dev --extra nwp`.
2. **Re-ingest data** (the 913 MB `wx.duckdb` is gitignored, rebuilt from sources):
   ```bash
   uv run wx initdb
   uv run wx backfill --start 2020-01-01 --end 2026-01-01     # METAR + TAF
   uv run wx nwp --start 2020-01-01 --end 2026-01-01          # ERA5 (needs ~/.cdsapirc)
   uv run wx verify                                           # score official TAFs
   ```
3. **Memory guard:** the constant `PEAK_BYTES_PER_ROW` and 60% budget were tuned for a
   15 GB box. On a bigger box, either raise the budget fraction
   (`build_samples(..., mem_fraction=0.8)`) or pass `mem_guard=False`. With more RAM you
   can finally train at high `sample_pct` — but note data saturates at ~5% for the
   *current* features, so spend the headroom on Phase D, not on rows.
4. **GPU:** Phase D models (PyTorch TFT/seq2seq) are not in deps yet — add them in the
   `nwp`/a new `dl` extra. The plan targets the `mltrain` cluster (Ray + MLflow); the
   `mltrain` skill covers job submission and the registry.

## Open tasks (also in docs/ISSUES.md)
1. **Retrain the ladder with lag features** (`wx train --rung gbm` / `linreg`) and re-gate.
   The lag-feature code is committed + tested but the saved champions predate it.
2. **mlp on GPU** — sklearn MLP took ~1 h/run on CPU; move to a GPU MLP (PyTorch) with
   class-aware loss + the same isotonic calibration, then `wx promote --rung mlp`.
3. **Phase D — TFT / seq2seq** — multi-horizon, quantile outputs for PROB/TEMPO groups,
   scored by Brier/CRPS. The biggest expected gain.
4. **PP-optimism validation** — ERA5 is joined at the valid hour (perfect prognosis), so
   absolute skill is optimistic. Validate on a real IFS/ECMWF reforecast archive.

## Verify the handoff landed
```bash
uv run pytest -q                  # 34 tests
uv run wx train --rung gbm        # trains + evaluates a rung (after data re-ingest)
uv run wx promote --rung gbm      # gate vs official TAF on frozen 2025
```
