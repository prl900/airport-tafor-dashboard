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

## Updated leaderboard (matched 9 leads)
- **Stacked ensemble {MLP,gbm,TFT,LSTM} (logistic): +0.348 / 0.513**  ← BEST
- linear blend MLP+TFT: +0.334
- MLP champion: +0.313–0.316
- gbm +0.284 ; TFT@100 +0.288 ; official −0.108
