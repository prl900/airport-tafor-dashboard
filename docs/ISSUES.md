# Open issues / roadmap

Ordered by priority. Updated 2026-06-17 after the GPU session (consolidation + MLP
champion + Phase D seq2seq/TFT).

---

## ✅ Done this session
- **1. Retrain ladder with lag features** — linreg/gbm retrained at 9%; effect marginal
  (gbm BSS +0.282→+0.284), confirming tabular saturation. rf discarded.
- **2. MLP on GPU** — `TorchMultiTaskModel` (PyTorch, `torch_models.py`); beats gbm
  (BSS +0.311 / HSS 0.472) and is the registered champion.
- **3. Phase D (seq2seq + TFT)** — `seq_dataset.py` + `seq_models.py` + `tft_models.py`
  (TFT with quantile heads). Both beat official/climatology; neither beats the tabular
  champion on BSS, but the TFT gives calibrated quantile distributions + vis MAE 287 m.
  Tests added (49 total).

---

## 1. PROB/TEMPO generation from TFT quantiles  *(in progress)*
**labels:** `ml`, `phase-d`, `product`

Turn the TFT's per-horizon q10/q50/q90 + calibrated P(adverse) into PROB30/TEMPO
groups — the actual probabilistic-TAF capability the quantiles unlock. Score the
generated groups against observed exceedance.

## 2. Batch the promotion gate  *(in progress)*
**labels:** `infra`, `ml`

`wx promote` (verifier path) is CPU/SQL-bound: ~1.5 s/TAF × 37k TAFs × 2 forecasters =
hours, GPU idle. Batch `build_inference_features` over all 2025 TAF-hours in one query
and score vectorized so the paired-bootstrap gate runs in minutes. Then re-run the MLP
gate for a verifier-path paired CI (currently promoted via tabular_eval, like gbm).

## 3. Phase D+ — more capable models (see docs/PHASE_D_ROADMAP.md)
**labels:** `ml`, `gpu`, `phase-d`

The TFT is NOT data-saturated (@25%→@40%: BSS +0.189→+0.229), so the sequence branch
has real headroom. Concrete sub-tickets, cheapest-first:
- **3a. LSTM seq2seq + attention** *(in progress)* — bidirectional LSTM encoder +
  cross-attention decoder; `lstm` rung, A/B vs the GRU seq2seq.
- **3b. Scale + HPO** — TFT data-scaling curve (40→60→max), Optuna over hidden/heads/
  depth/dropout/lr/`cat_loss_weight`/`past_len`/`horizon`.
- **3c. CRPS + focal loss** — CRPS as a first-class probabilistic objective; focal loss
  for the rare adverse class.
- **3d. DeepAR / N-HiTS / PatchTST** — stronger probabilistic/multi-horizon backbones.
- **3e. Full TFT Variable Selection Networks** — current TFT is lite (joint GRN proj).
- **3f. Ensemble** — stack tabular MLP/gbm (short-lead) + TFT (long-lead + distributions).

## 3g. Wire seq/TFT models as Forecasters for the gate
**labels:** `ml`, `infra`, `phase-d`

Seq models are scored only via the fast tabular-style eval in `train_seq.py`. Add a
`SeqForecaster` with a batched sequence-inference path so Phase-D models run the real
(now-batched) promotion gate. Prerequisite for promoting any Phase-D winner.

## 4. Validate Perfect-Prognosis optimism
**labels:** `ml`, `validation`, `data`

ERA5 is joined at the valid hour, so skill is optimistic vs real forecasts. Ingest an
IFS/ECMWF reforecast archive (issued at T0, valid at t) and re-evaluate the champion.

## 5. Generalize the memory guard beyond the dev box
**labels:** `infra`, `nice-to-have`

`PEAK_BYTES_PER_ROW`/`mem_fraction` are hand-tuned. Auto-calibrate bytes/row from a tiny
probe build so the guard is correct on any machine.

## 6. First-class `wx` commands for the seq models
**labels:** `infra`, `nice-to-have`

`seq2seq`/`tft` only run via `scripts/train_seq.py`. Add `wx train-seq` so they're
first-class alongside `wx train`/`wx promote`, and wire a seq `Forecaster` for the gate.
