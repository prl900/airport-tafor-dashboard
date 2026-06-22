# ML TAFOR Generation — Research Plan (approved 2026-06-16)

Iterative, benchmark-driven model development (Karpathy "auto-research" style):
the simplest model is the benchmark; a new model replaces the champion **only when
it beats it** on a frozen test set with statistical significance.

## Approved decisions
- **NWP signal:** Perfect Prognosis now — train ERA5-analysis(t) → METAR(t), deploy
  with IFS forecast(T₀→t). PP optimism flagged; ingest IFS/ECMWF reforecast archive
  in a later phase to validate on genuine forecasts.
- **Targets:** multi-task — vis, ceiling, wind (regression) + flight category
  (classification), yielding a full `ExpectedHour` for the verifier.
- **Scope:** one pooled model over all airports with a learned station embedding +
  static features (shares rare adverse-event data; generalizes to new airports).
- **Build order:** Phases A–C first → review vs official TAF → then D–E.

## Causality contract (enforced in `ai/dataset.py`)
Forecast at issue **T₀** for valid **t = T₀+h** may use ONLY:
- METAR(τ) for **τ ≤ T₀** (current + lags −1/−3/−6 h; the Markov state)
- ERA5 at t and T₀ and the T₀→t tendency (PP proxy for IFS forecast)
- lead h; hour-of-day & day-of-year (cyclical); static airport attrs
NEVER METAR(τ>T₀). Target = METAR-derived(t) is a label only. A leakage audit asserts
every observation-feature timestamp ≤ T₀ < t (DuckDB ASOF join on `observed_at ≤ t0`).

## Targets & integration
Per (airport, T₀, h): predict vis_m, ceiling_ft, wind_spd/dir, flight_category. Each
model is wrapped as a `Forecaster.generate() → [ExpectedHour]` (existing interface),
so `ai/compare.py` + `scores.py` score it vs the official TAF and baselines unchanged.

## Data splits
Temporal, blocked, purged: train 2020–2023 / val 2024 / **frozen test 2025**. All hours
of one TAF in one fold; gap between train/test to kill autocorrelation leakage. No
random splits. Training uses dense (t, h) samples for lead balance; evaluation generates
timelines over official-TAF windows and scores with the verifier.

## Benchmark contract
- Single source of truth = the verifier (`scores.py`).
- **Primary metric: HSS** (IFR-or-worse event), stratified by lead time & region.
  Secondary: POD/FAR/CSI, Brier/CRPS, element MAE, weighted_score.
- Reference lines: floor = persistence/climatology; **skyline = official TAF** (LEMD HSS 0.37).
- **Promotion:** challenger replaces champion only if it beats it on the frozen test set
  with a bootstrap CI on the metric difference excluding 0; else archived with notes.

## Model ladder
0. persistence / climatology (built) + official-TAF skyline — initial benchmark
1. Linear/Logistic Regression — first ML champion, interpretable
2. RF → Gradient Boosting (LightGBM/XGBoost) — tabular workhorse
3. MLP — multi-task heads, class-weighted for rare adverse events
4. Time-series/probabilistic: GBM+lags → seq2seq LSTM/CNN → **TFT** (static+known-future
   NWP + observed-past METAR, multi-horizon, quantiles) → N-HiTS/PatchTST/DeepAR.
   Probabilistic outputs enable PROB/TEMPO generation (scored by Brier/CRPS).

## Auto-research loop (on mltrain cluster + MLflow registry)
Each config = an `mltrain run`; metrics → MLflow; registry holds the champion. A
controller iterates a config queue (ladder → Optuna HPO → feature ablations): train →
eval-vs-champion on val → if better, eval on frozen test w/ significance → `mltrain
promote` if it wins, else archive. Every experiment appended to a research log.
Human-gated promotion initially; progressively automate HPO/ablations.

## Phases
- **A** ✅ Causal dataset (`ai/dataset.py`) + leakage audit + frozen splits.
  Extended with METAR-lag features (t0−1/−3/−6 h + tendencies).
- **B** ✅ Benchmark harness: champion/challenger eval (`scores.py`), paired bootstrap
  significance (`ai/promote.py`), official-TAF skyline, research-log scaffold.
- **C** ✅ Ladder rungs linreg/gbm calibrated (isotonic + val-tuned threshold), then a
  **GPU PyTorch MLP** (`torch_models.py`, shared trunk + class-aware loss + same
  calibration) which **beats gbm and is the champion** (BSS +0.311 / HSS 0.472 on frozen
  2025). Key results: calibration was the lever; the tabular feature set saturates at ~5%
  sample (lag features only marginal). → REVIEWED.
- **D** ✅ (GPU, started) Sequence/probabilistic: `seq_dataset.py` (causal windowed
  encoder/decoder) + `seq_models.py` (GRU seq2seq) + `tft_models.py` (**TFT with quantile
  heads**, pinball loss, masked attention). Both beat official/climatology; neither beats
  the tabular champion on adverse BSS (seq2seq +0.211, TFT +0.219 at matched leads), but
  the TFT gives **calibrated quantile distributions** (82.8% coverage) + better vis MAE —
  the foundation for PROB/TEMPO products. Next: PROB/TEMPO generation, HPO, CRPS.
- **E** ⏳ Auto-research controller (config queue, Optuna, ablations) + human gates.

> Handoff for the GPU box: **`docs/HANDOFF.md`**; open issues: **`docs/ISSUES.md`**.
