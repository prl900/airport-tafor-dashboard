# Phase D+ Roadmap — toward more capable probabilistic models

Continuation plan after the first Phase-D pass (seq2seq + TFT-with-quantiles). Written
2026-06-17 on the GPU box. Companion to `ML_PLAN.md` (overall plan) and `ISSUES.md`
(actionable tickets).

## What the session established (the planning anchors)
- **Tabular models saturate (~5% sample); sequence models do NOT.** The TFT improved
  with data: @25% → @40% sample gave BSS +0.189 → +0.229, HSS 0.361 → 0.409. So for
  sequence models, **scaling data + compute is a real lever** (it is dead for the trees).
- **The gap to the champion is closing.** MLP champion BSS +0.311; TFT@40% ≈ +0.26 at
  matched leads. A scaled/tuned sequence model could plausibly overtake it.
- **The unique Phase-D value is the distribution.** TFT quantiles → an operational
  PROB/TEMPO TAF scoring BSS +0.147 vs the official TAF's −0.108 — a new *product*,
  independent of the point-BSS race.
- **Infra is ready:** causal sequence dataset, isotonic-calibration contract, a now-fast
  **batched promotion gate**, the research log, and 55 tests. An auto-research loop is
  finally practical.
- **Caveat capping absolute skill:** ERA5 is perfect-prognosis (joined at the valid
  hour) — real-forecast skill will be lower. Validate before trusting the magnitudes.

## Stages (cheapest-first)

### D.1 — Scale & tune what exists  *(highest ROI, low risk)*
- **Data-scaling curve** for the TFT (40% → 60% → max-that-fits); find the plateau.
- **Optuna HPO**: hidden, #heads, depth, dropout, lr, `cat_loss_weight`, `past_len`,
  `horizon`.
- **Loss upgrades**: focal loss for the rare adverse class; **CRPS** as a first-class
  probabilistic objective alongside pinball.

### D.2 — More capable sequence architectures
- **LSTM seq2seq + attention** *(started)* — bidirectional LSTM encoder over past METAR,
  cross-attention decoder over known-future ERA5. Drop-in (`lstm` rung), A/B vs the GRU.
- **DeepAR** — autoregressive, likelihood-based probabilistic RNN; native distributions.
- **N-HiTS / PatchTST** — strong multi-horizon backbones, especially long leads.
- **Full TFT with Variable Selection Networks** — current TFT is *lite* (joint GRN input
  projection); add per-variable VSN for interpretability + possible lift.

### D.3 — Ensembles & hybrids  *(often the real winner)*
- **Stack** the tabular MLP/gbm (strong short-lead) with the TFT (strong long-lead +
  distributions): blend calibrated P(adverse), recalibrate on val.
- **Hybrid product**: tabular head for P(adverse)/HSS; TFT for the quantile distribution
  driving PROB/TEMPO.

### D.4 — Data & objective  *(highest scientific value)*
- Ingest a real **IFS/ECMWF reforecast** archive (issued at T0) to replace PP ERA5;
  quantify the real-forecast skill gap. Add richer NWP (vertical levels, lead-aware).

### E — Auto-research loop  *(unlocked by the fast gate)*
- Optuna config queue → train → val-gate → batched frozen-test gate → promote/archive,
  every run logged. Human-gated promotions first, then automate.

## Experimental protocol (keep it honest)
Same frozen-2025 test, temporal splits, isotonic calibration on val, **BSS + HSS + CRPS**
with bootstrap CIs, promotion via the batched gate. Each config = one
`data/research_log.jsonl` entry.

**Prerequisite for promotion:** wire the seq/TFT models as `Forecaster`s (a
`SeqForecaster` with a batched sequence-inference path), so Phase-D models run the real
gate — today only the tabular models do; seq models are scored via the fast tabular-style
eval in `scripts/train_seq.py`.

## Results so far (2026-06-17, executing D.1–D.3)
- **TFT data-scaling — improves then PLATEAUS at ~+0.271:** all-horizon BSS +0.189 (25%)
  → +0.229 (40%) → +0.271 (60%) → **+0.271 (100%)**. The +0.04/step trend stopped after
  60%; 60→100% gave ~0. Matched-9-lead saturates ~+0.30 — **just short of the MLP
  (+0.311)**. Conclusion: **data scaling alone does NOT beat the tabular champion**; the
  remaining TFT levers are objective/architecture (CRPS, focal loss, HPO, full-VSN) or new
  data/features (real IFS), not more rows.
- **LSTM ≈ TFT** (BSS +0.225 vs +0.229 @40%): the RNN cell is not the lever — **data is**.
  Deprioritize new single-model architectures (DeepAR/N-HiTS/PatchTST) accordingly.
- **Ensemble (MLP + TFT) is the best model** (9 leads): BSS **+0.328** / HSS 0.493 vs MLP
  alone +0.313 — the project best. Blend ~0.65 MLP / 0.35 TFT; the gain (+0.015) is real
  but modest, and capped by the TFT's plateau. Productionizing it needs the SeqForecaster
  gate-wiring (#11) so it can run the verifier gate and be promoted.
- **Verifier gate fix + speed:** found/fixed a `ModelForecaster` bug (raw class-weighted
  argmax over-called adverse, bypassing calibration); the gate now confirms the MLP > gbm
  (0.482 vs 0.466). Batched the gate (features + obs) from ~24 min → **11 min**.

## Recommended sequencing
1. **D.2 LSTM A/B** + **D.1 (scale + HPO + CRPS)** — cheap; directly tests "can a
   sequence model beat the MLP champion?"
2. **`SeqForecaster` + gate wiring** — needed to promote any Phase-D winner.
3. **D.3 ensemble** — likely the pragmatic best operational model.
4. **D.4 real-forecast NWP** — the credibility milestone.
5. **E auto-research** — once 2–3 configs validate the loop.
