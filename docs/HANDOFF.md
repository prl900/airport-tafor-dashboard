# Handoff — TAFOR ML (now on the GPU box)

The data move + GPU work is done. The tabular ladder is consolidated, the GPU **MLP
is the champion**, and **Phase D (seq2seq + TFT-with-quantiles)** is built and
benchmarked. This doc is the pickup point for the next session.

## TL;DR state (2026-06-17, GPU box: TITAN Xp 12 GB, 8 core, 31 GB)
- **Champion: calibrated GPU `mlp`** (`data/models/mlp.joblib`, PyTorch). Frozen-2025
  tabular eval, all stations, 9% sample:
  - **mlp: BSS +0.311** (CI [+0.300, +0.321]), **HSS 0.472** — beats gbm
  - gbm: BSS +0.284 (CI [+0.273, +0.295]), HSS 0.463  (prev champion, lag-retrained)
  - linreg: BSS +0.247, HSS 0.425
  - official TAF: BSS −0.108, HSS ~0.37 ; climatology: BSS 0
- **Phase D models beat the official TAF/climatology but NOT the tabular champion on
  BSS/HSS:** seq2seq BSS +0.211, TFT BSS +0.219 (matched 9 leads). The adverse-event
  signal is dominated by current-state + PP-NWP features the tabular models already
  capture. **But the TFT wins where it counts for products:** vis MAE 287 m (vs ~440 m
  tabular) and a **calibrated predictive distribution** (q10/q90 vis interval = 82.8%
  empirical coverage vs 80% nominal).
- **Lag features were marginal** for the tabular models (gbm BSS +0.282→+0.284) —
  confirms the tabular feature set saturates (~5% sample). Calibration remains the lever.

## What's built
- **Tabular ladder** `src/wx/ai/{dataset,models,train,promote}.py` — `MultiTaskModel`
  (sklearn) + isotonic calibration + HSS threshold, `ModelForecaster`, `wx train`/`wx promote`.
- **GPU MLP rung** `src/wx/ai/torch_models.py` — `TorchMultiTaskModel`: shared trunk +
  per-target heads, class-aware loss, same calibration contract, device-safe pickling.
  Wired via `build_model()` in `train.py` (rung `mlp`).
- **Phase D** — `src/wx/ai/seq_dataset.py` (causal windowed sequences: past METAR
  encoder + known-future ERA5 decoder + masked multi-horizon targets),
  `src/wx/ai/seq_models.py` (`SeqForecastModel`, GRU seq2seq),
  `src/wx/ai/tft_models.py` (`TFTModel`: GRN + static enrichment + masked attention +
  **quantile heads**, `predict_quantiles()`). Driver: `scripts/train_seq.py [pct] [seq2seq|tft]`.
- **Tests** — 49 total; Phase-D modules covered in `tests/test_{seq_dataset,torch_models,
  seq_models,tft_models}.py` (causality/alignment, sampling ratio, masking, calibration,
  device-safe roundtrip, quantile contract).

## Environment / setup
- `uv sync --extra dev --extra nwp --extra ml --extra dl`.
- **torch is the cu121 build** (the GPU driver is CUDA 12.2; the default PyPI wheel
  targets CUDA 13 and won't init). Pinned via `[tool.uv.sources]`/`[[tool.uv.index]]`
  (`pytorch-cu121`) in `pyproject.toml`. TITAN Xp is Pascal sm_61 — supported.
- Data lives at repo root (`wx.duckdb`, `wx_serve.duckdb`) + `data/` (era5, raw_cache,
  models). All gitignored except the small `data/models/*.joblib` champions.
- **Memory guard** (`dataset.py`, tabular builds only) budgets 60% of RAM → ~13% max
  sample here; raise `mem_fraction` or pass `mem_guard=False` for more. Sequence builds
  (`seq_dataset.py`) are not guarded but are lighter (one sample per issue hour).

## Open tasks (see docs/ISSUES.md)
1. **PROB/TEMPO generation** from the TFT quantiles — the genuinely new capability
   (turn q10/q90 + P(adverse) into PROB30/TEMPO groups). *(in progress)*
2. **Batch the promotion gate** — `wx promote` is CPU/SQL-bound (~1.5 s/TAF × 37k,
   hours). Batch `build_inference_features` over all 2025 TAF-hours → minutes. *(in progress)*
3. **Tune Phase D for BSS / HPO** — focal loss, capacity, longer training; or accept
   that BSS is tabular-bound and pursue the distribution/products angle instead.
4. **PP-optimism validation** — ERA5 joined at valid hour ⇒ optimistic. Validate on a
   real IFS/ECMWF reforecast archive.

## Verify
```bash
uv run pytest -q                              # 49 tests
uv run python scripts/train_ladder.py 9 mlp   # GPU MLP rung
uv run python scripts/train_seq.py 25 tft     # TFT w/ quantiles
```
