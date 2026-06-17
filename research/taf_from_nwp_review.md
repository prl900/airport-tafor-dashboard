# Generating Probabilistic Aerodrome Forecasts (TAFs) from Numerical Weather Prediction: A Benchmark-Driven Model Review

*Working review — Spanish airport network, 2020–2025. Companion artifacts in this repository.*

## Abstract

We study whether machine-learning models can produce skilful, well-calibrated aerodrome
forecasts (TAFs) from numerical weather prediction (NWP) and recent observations, and
whether they can beat the official human-issued TAF. Using a strictly causal feature
frame over 24 Spanish airports (2.5 M METARs, 0.25 M TAFs, ERA5 reanalysis as a
perfect-prognosis NWP proxy), we benchmark a ladder of models — linear, gradient-boosted
trees, a GPU multi-task MLP — and a family of sequence/probabilistic models (GRU, LSTM,
Transformer, TCN, and a Temporal Fusion Transformer with quantile outputs). All models are
scored on a frozen 2025 test set with the Brier Skill Score (BSS) for the IFR-or-worse
event and the Heidke Skill Score (HSS), against the official TAF skyline. The best model is
a **logistic stacked ensemble of {MLP, gradient boosting, TFT}** reaching **BSS +0.348 /
HSS 0.513**, versus the official TAF's **−0.108**, and it is near-perfectly calibrated.
Driving PROB/TEMPO group generation with it yields an operational TAF at **BSS +0.215**.
We find that model **architecture is not the limiting factor**; data scaling, model class,
and the available feature set all plateau around BSS ≈ +0.33, indicating a **data ceiling**
set by the NWP fields ingested and the perfect-prognosis setup.

## 1. Objective

The official TAF is a categorical, human-issued forecast. Our goal is a model that, given
information available at issue time T₀, predicts the hourly aerodrome state over the TAF
validity window (vis, ceiling, wind, and the derived flight category) **probabilistically**,
so it can (a) be scored fairly against the official TAF and (b) drive automatic generation
of PROB/TEMPO change groups. The headline event is **IFR-or-worse** (ceiling/visibility
below instrument-flight thresholds) — the operationally critical, and rare (~3.6%), case.

## 2. Data and problem setup

- **Targets:** METAR-derived vis, ceiling, wind, and flight category at valid time t.
- **Features (strictly causal, enforced structurally):** current + lagged METAR state
  (t₀, −1/−3/−6 h, with tendencies), ERA5 fields at t and T₀ and the T₀→t tendency, lead
  time, cyclical hour/day, and static airport attributes. ERA5 is **perfect-prognosis**:
  reanalysis joined at the valid hour, a stand-in for an IFS forecast issued at T₀.
- **Splits (temporal, blocked):** train < 2024, validation = 2024, **frozen test = 2025**.
  Calibration is fit on validation; the test set is touched only for final scoring.

Full implementation: `src/wx/ai/dataset.py` (causal frame + leakage audit),
`src/wx/ai/seq_dataset.py` (windowed sequences for the deep models).

## 3. Metrics

- **BSS** — Brier Skill Score of P(IFR-or-worse) vs the climatological base rate. The fair,
  hedging-aware probabilistic metric; >0 beats climatology. Bootstrap CIs throughout.
- **HSS** — Heidke Skill Score of the binary adverse decision at a validation-tuned
  threshold (the primary categorical metric in the plan).
- **Reliability** — agreement between forecast probability and observed frequency
  (calibration), essential for a probabilistic product.
- **Reference lines:** climatology (BSS = 0) and the **official TAF** (the skyline).
  Scoring is unified through the verifier (`src/wx/verification/scores.py`).

A key methodological lesson: **calibration is decoupled from the decision threshold**.
Class-weighted models discriminate well but produce poorly-scaled probabilities; isotonic
calibration on validation plus a separately-tuned HSS threshold fixes both. (A subtle bug
where a class-weighted classifier's raw argmax — not the calibrated decision — drove the
forecast category cost the MLP ~0.16 HSS in the verifier path until corrected.)

## 4. Models tested

| Family | Model | Notes |
|--------|-------|-------|
| Baselines | persistence, climatology, **official TAF** | reference / skyline |
| Tabular ladder | linear/logistic, **gradient boosting (gbm)**, **GPU multi-task MLP** | sklearn + a PyTorch shared-trunk MLP |
| Sequence | GRU seq2seq, **LSTM + cross-attention**, Transformer enc-dec, TCN | encoder over past METAR, decoder over known-future ERA5 |
| Probabilistic | **Temporal Fusion Transformer (TFT)** | GRN + static enrichment + masked attention + **quantile heads** |
| Ensemble | linear blend, **logistic stack** | over base-model calibrated probabilities |

All deep models share one framework (`src/wx/ai/{torch_models,seq_models,tft_models}.py`),
the same calibration contract, and the same evaluation, so comparisons isolate the model.

## 5. Results

### 5.1 Model comparison

Every learned model beats the official TAF decisively on probabilistic skill; the official
TAF's over-committed categorical style scores **BSS −0.108**. Among single models the MLP
leads (+0.316); ensembling lifts skill further.

![Model comparison](figures/fig1_model_comparison.png)

### 5.2 Ensembling is the only lever that helped

A logistic stack over base-model probabilities (fit on 2024, tested on 2025) reaches
**BSS +0.348 / HSS 0.513**. Ablation shows **LSTM is redundant** (≈ TFT) — `{MLP, gbm, TFT}`
matches the full four-model stack — and the TFT (sequence model) contributes **+0.011** of
genuine NWP-sequence signal over the tabular-only `{MLP, gbm}` stack.

![Ensemble ablation](figures/fig6_ablation.png)

### 5.3 Calibration

The stacked ensemble is **near-perfectly calibrated**: forecast probability matches observed
frequency to within ±0.01 across the full range — the property that makes PROB30/PROB40
groups meaningful.

![Reliability](figures/fig2_reliability.png)

### 5.4 Skill vs lead time and region

Skill decays gracefully from **+0.503 at +1 h to +0.295 at +30 h**, staying far above the
official TAF at all leads. By region it ranges from the Canaries (+0.451, a predictable
regime) to Melilla (+0.127, hardest, small sample).

![Skill vs lead](figures/fig3_skill_vs_lead.png)
![Per-region skill](figures/fig5_region.png)

### 5.5 The ceiling: scaling and architecture

The TFT improves with data then **plateaus by ~60% sample** (BSS +0.189→+0.229→+0.271→
+0.271 at 25/40/60/100%). The tabular models saturate even earlier (~5%). And across GRU,
LSTM, Transformer, TCN and TFT backbones the skill clusters within ~0.02 — **architecture
is not the lever**. Together these point to a **data ceiling**, not a modelling one.

![Data scaling](figures/fig4_scaling.png)

## 6. Generating PROB/TEMPO TAFs

The stacked P(adverse) is quantized to the TAF's expressible buckets (none / PROB30 /
PROB40 / firm) and combined with the TFT's quantile element forecast to emit an hourly
ExpectedHour timeline with PROB/TEMPO groups (`src/wx/ai/prob_groups.py`). The regenerated
operational TAF scores **BSS +0.215** — versus the official TAF's −0.108. Example
(LEVT, an adverse morning):

```
LEVT issued 2025-06-16 11Z:
  +12h  VFR                       | obs MVFR
  +18h  VFR   PROB40 LIFR         | obs IFR      <- fog risk correctly flagged
  +24h  VFR                       | obs VFR
```

A notable finding: the TFT's **visibility quantiles collapse to "clear" even at q10** — the
rare low-visibility tail is not captured, so the adverse skill lives entirely in the
calibrated category probability, not the element distribution. Adverse groups therefore use
category-representative conditions for realism.

## 7. Discussion

Three independent levers — **data volume, model architecture, and feature set** — all
plateau around BSS ≈ +0.33. The feature set is exhausted because every *populated* NWP
field is already used; notably, ERA5's low/mid/high cloud-cover split (the obvious
ceiling/fog predictor) was never ingested (0% populated). Hyperparameter search on the MLP
returned only run-to-run noise. The single thing that helped beyond a well-calibrated MLP
was **ensembling diverse model classes**, and even that adds only ~+0.03 BSS.

## 8. Limitations and future work

- **Perfect-prognosis optimism.** ERA5 is joined at the valid hour, so absolute skill is
  optimistic relative to a real forecast. Validation against an IFS/ECMWF *reforecast*
  archive (issued at T₀) is the credibility milestone.
- **Richer NWP.** Re-ingesting ERA5 with cloud layers (lcc/mcc/hcc) is the most likely
  lever to raise the ceiling, and is pure data engineering.
- **Productionization.** Wiring the stacked ensemble as a live `Forecaster` (sequence
  inference at arbitrary T₀) so it runs the verifier promotion gate and can be registered
  as champion.

## 9. Conclusion

A calibrated, stacked ensemble of complementary model classes produces aerodrome forecasts
that are **substantially more skilful and better calibrated than the official TAF** on a
frozen 2025 test set (**BSS +0.348 vs −0.108**), and it can drive an operational PROB/TEMPO
TAF that retains most of that skill. The current ceiling is set by the data, not the models:
the highest-value next step is richer / genuine-forecast NWP, not a larger network.

---

*Reproducibility: figures via `research/figures/make_figures.py`; experiment log in
`data/research_log.jsonl`; full session journal in `docs/AUTONOMOUS_RESEARCH.md`; model
ladder and ensemble recipe in `docs/PHASE_D_ROADMAP.md`.*
