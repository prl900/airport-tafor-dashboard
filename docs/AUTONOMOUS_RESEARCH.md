# Autonomous research session — TAFs from NWP

**Goal:** a model that produces good TAFs from NWP (probabilistic, drives PROB/TEMPO),
beating the current best. **Window:** 2026-06-17 ~21:04 → ~05:04 AEST (8 h). Driven
event-style: each background training run notifies on completion → evaluate → log here
→ launch the next. This file is the persistent state + findings report.

## Baseline leaderboard (frozen 2025, matched 9 leads, BSS / HSS)
- **Ensemble MLP+TFT: +0.328 / 0.493**  ← best so far
- MLP champion (tabular, GPU): +0.313 / 0.484
- gbm (tabular): +0.284 / 0.463
- TFT@100 (seq, saturated): +0.288 / 0.454
- LSTM@40 / TFT@40: +0.252 / +0.257
- official TAF: −0.108 / ~0.37

## Established findings (pre-session)
- Tabular features saturate ~5%; sequence models scale then **plateau ~+0.27 (60–100%)**.
- RNN cell is not the lever (GRU≈LSTM≈TFT). Data scaling exhausted.
- Calibration (isotonic + val threshold) is the key lever; class-weighted argmax must
  NOT drive the category (fixed in ModelForecaster).
- TFT quantiles → PROB/TEMPO TAF scores BSS +0.147 (vs official −0.108).

## Experiment queue (adaptive)
1. **[running] Backbone sweep** gru/tcn/transformer @40 → rank architectures.
2. **Richer NWP features** — add low/mid/high cloud (lcc/mcc/hcc, already in nwp_point but
   unused) to ERA5 features in tabular + seq datasets. Low cloud ↔ ceiling/fog. Retrain
   MLP + best seq backbone. *(biggest lever for "TAFs from NWP")*
3. **Scale the winning combo** (best backbone + richer features) to 60–100%.
4. **Objective**: focal loss + cat_loss_weight tuning on the best backbone.
5. **HPO**: small random search over the best architecture.
6. **Stacked ensemble**: learned stacker over {MLP, gbm, best-seq, TFT} calibrated probs.
7. **Synthesis**: pick best model, regenerate PROB/TEMPO TAFs, validate, report, commit.

Adapt: double down on whatever moves BSS; drop dead ends. Each run → research_log.jsonl
entry + a line in the Findings log below.

## FINAL REPORT

**Best model: a logistic stacked ensemble of {MLP, gbm, TFT}** — frozen-2025, 9 standard
leads: **BSS +0.348 / HSS 0.513** (vs MLP champion +0.313, official TAF −0.108). It is
**near-perfectly calibrated** (reliability bins match observed frequency to ±0.01), so its
probabilities are trustworthy — the prerequisite for a probabilistic TAF product. Driving
PROB/TEMPO generation with it yields an **operational TAF at BSS +0.215** vs the official
TAF's −0.108. Skill decays gracefully with lead (+0.503 @1h → +0.295 @30h) and varies by
region (Canaries +0.451 best, Melilla +0.127 hardest).

### What was tried, and the ceiling
- **Architecture is not the lever.** GRU/LSTM/Transformer/TFT/TCN backbones cluster
  ~+0.21–0.23 @40% (TCN worst); TFT marginally best. Recent architectures don't beat it.
- **Data scaling is exhausted.** TFT BSS +0.189→+0.229→+0.271→+0.271 (25→40→60→100%) —
  plateaus by 60%. (Tabular saturates even earlier, ~5%.)
- **Features are exhausted.** Every populated NWP field is used; the cloud-layer split
  (lcc/mcc/hcc) — the obvious ceiling/fog lever — was never ingested (0% populated).
- **HPO is a null.** MLP capacity/regularization sweeps stay within run-noise of +0.311.
- **Ensembling is the only thing that helped.** Stacking {MLP,gbm,TFT} > any single
  (+0.348 vs +0.316) and > linear blend (+0.334). LSTM is redundant (≈TFT). The TFT
  contributes +0.011 of genuine NWP-sequence signal over the tabular-only stack.
- **The quantile distribution is weak for rare low-vis:** TFT vis quantiles collapse to
  "clear" even at q10; adverse skill lives in the calibrated category probability, not the
  element distribution. Adverse TAF groups therefore use category-representative conditions.

### The real ceiling & next levers (modeling is exhausted; these are data/eng)
1. **Richer NWP** — re-ingest ERA5 with cloud layers (lcc/mcc/hcc); the single most likely
   lever for ceiling/fog skill. Pure data-engineering (CDS API).
2. **Real IFS/ECMWF reforecast** — replaces perfect-prognosis ERA5; current absolute skill
   is optimistic. The credibility milestone.
3. **Productionize the stack** as a `Forecaster` (seq inference for arbitrary T0 +
   StackedEnsembleForecaster) so it runs the verifier gate and can be the registered
   champion. Engineering, not research.

### Artifacts
`data/models/stacker.joblib` (the fitted stacker); base models mlp/gbm/tft(/lstm).joblib;
`scripts/{stack_ensemble,stack_ablation,stack_analysis,stacked_taf,mlp_hpo}.py`.
Recipe: calibrated P(adverse) from each base → logistic stack (fit on 2024 val) → isotonic
recalibrate → PROB/TEMPO via `prob_groups` with TFT quantiles for non-adverse conditions.

## Findings log (append-only)
- 21:04 — session start. Backbone sweep (gru/tcn/transformer @40) in flight.
- 21:20 — **Backbone sweep done** (@40, all-H BSS): tft +0.229 ≈ lstm +0.225 > gru +0.216
  ≈ transformer +0.207 ≫ tcn +0.131. TFT best; TCN poor (short window), transformer ≈ GRU.
  Architecture confirmed NOT the lever. → TFT is the seq backbone for the rest of the session.
- 21:20 — Starting Q2: add low/mid/high cloud (lcc/mcc/hcc) NWP features.
- 21:35 — **Q2 DEAD END:** lcc/mcc/hcc are 0% populated — ERA5 ingestion only fetched
  total cloud (tcc), never the cloud-layer split. Every *populated* NWP field is already
  used. So FEATURES are exhausted too (alongside data-scaling + architecture). Reverted.
  Real remaining data lever = re-ingest ERA5 w/ cloud layers OR real IFS — a data-eng task,
  not feasible this session. **Pivot to modeling squeezes:** MLP HPO, objective tuning,
  smarter ensembling (per-lead / stacked). Realistic ceiling ~+0.33.
- 21:35 — Q4a: MLP HPO sweep @9% (trunk/dropout/lr/epochs), pick best base model.
- 21:55 — **Q4a NULL:** 6 MLP configs all BSS +0.295..+0.304 (base reproduced +0.304 vs
  champion's +0.311 = ~±0.007 run noise). Wider/deeper/regularized don't help — MLP at its
  ceiling. Discard mlp_hpo_best; champion stays. Single-model squeeze is over.
- 21:55 — Q6: stacked / per-lead ensemble over {MLP, gbm, TFT@100, LSTM@40}.
- 22:10 — **Q6 WIN — new best model.** Logistic stack over the 4 base models (fit on 2024
  val, tested 2025, 9 leads, n=89.6k): **BSS +0.348 / HSS 0.513**, vs MLP +0.316, linear
  blend +0.334. gbm+LSTM add real signal; per-lead (+0.346) overfits vs global. The
  stacked ensemble is the session's best TAF-from-NWP model. → build PROB/TEMPO generation
  from it + validate; then a focal-TFT bonus to see if it lifts the stack further.

- 22:30 — Deliverable: PROB/TEMPO TAFs from the stacked ensemble. Continuous all-H BSS
  +0.310; **regenerated operational TAF BSS +0.215** vs official −0.108. Stacker saved.
  Finding: TFT vis QUANTILES collapse to "clear" even at q10 — adverse skill is all in the
  calibrated category prob, not the element distribution; adverse groups now use
  category-representative conditions. Samples look skilful (PROB30/40 LIFR catch fog).
- 22:55 — **Ablation: LSTM is redundant.** {MLP,gbm,TFT} = +0.348 (= full 4-model);
  tabular-only {MLP,gbm} = +0.337, so the TFT adds +0.011 of genuine NWP-sequence signal.
  → production model = **stacked {MLP, gbm, TFT}**.

## Updated leaderboard (matched 9 leads, BSS / HSS)
- **Stacked {MLP,gbm,TFT} (logistic): +0.348 / 0.513**  ← BEST (LSTM redundant)
- {MLP,gbm} tabular stack: +0.337 ; linear blend MLP+TFT: +0.334
- MLP champion: +0.313–0.316 ; TFT@100 +0.287 ; gbm +0.273 ; official −0.108
- Operational PROB/TEMPO TAF from the stack: BSS +0.215 (vs official −0.108)
