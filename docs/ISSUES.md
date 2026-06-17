# Open issues / roadmap

Ready-to-file GitHub issues for the next phase (GPU box). Once a remote exists:
`gh issue create --title "..." --body "..." --label "..."`. Ordered by priority.

---

## 1. Retrain the ladder with the new METAR-lag features
**labels:** `ml`, `features`, `good-first-task`

The lag/tendency features (t0−1/−3/−6 h + current−lag deltas, 33→54 features) are
implemented in `ai/dataset.py` and unit-tested, but the saved champions
(`gbm`, `linreg`) were trained before them — `ModelForecaster` silently ignores the
extra columns. Retrain and re-gate to measure their effect.

- [ ] `wx train --rung linreg` and `--rung gbm` (5% is enough; data saturates)
- [ ] `wx promote --rung gbm` — does HSS/BSS improve significantly?
- [ ] Update `data/research_log.jsonl` + `champion.json`; note the delta in the log.

## 2. Move `mlp` to the GPU (PyTorch)
**labels:** `ml`, `gpu`, `mlp`

sklearn `MLPRegressor/Classifier` took ~1 h/run on the CPU box (7 nets over 776k rows)
and was abandoned. Reimplement as a multi-task PyTorch MLP on GPU.

- [ ] Multi-task heads (vis/ceiling/wind regression + flight-category classification),
      station embedding + static features (per `ML_PLAN.md`).
- [ ] Class-aware loss for rare adverse events, **then** isotonic calibration on the
      2024 val split (keep the calibration/threshold decoupling that fixed gbm).
- [ ] Slot in as a `ModelForecaster`; `wx promote --rung mlp` vs the gbm champion.

## 3. Phase D — sequence / probabilistic models (TFT, seq2seq)
**labels:** `ml`, `gpu`, `phase-d`, `epic`

The biggest expected gain. Multi-horizon, known-future NWP + observed-past METAR,
**quantile outputs** to drive PROB/TEMPO group generation (scored by Brier/CRPS).

- [ ] Add a `dl` extra (torch + a TFT impl, e.g. pytorch-forecasting) to `pyproject.toml`.
- [ ] Sequence dataset adapter over the existing causal frame (windowed per station).
- [ ] TFT baseline → N-HiTS / PatchTST / DeepAR comparisons via the promotion gate.
- [ ] Run on the `mltrain` cluster (Ray + MLflow); register the winner.

## 4. Validate Perfect-Prognosis optimism
**labels:** `ml`, `validation`, `data`

ERA5 is joined at the valid hour, so current skill is optimistic vs real forecasts.

- [ ] Ingest an IFS/ECMWF reforecast archive (forecast issued at T0, valid at t).
- [ ] Re-evaluate the champion with genuine-forecast NWP; quantify the optimism gap.

## 5. Generalize the memory guard beyond the 15 GB dev box
**labels:** `infra`, `nice-to-have`

`PEAK_BYTES_PER_ROW` / `mem_fraction` are tuned for the CPU box. Auto-calibrate the
bytes/row constant from a tiny probe build instead of a hardcoded estimate, so the
guard is correct on any machine.
