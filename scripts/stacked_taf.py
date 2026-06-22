"""Autonomous Q7 — the deliverable: PROB/TEMPO TAFs from the stacked ensemble.

Fit the logistic stacker over {MLP, gbm, TFT, LSTM} calibrated P(adverse) on 2024 val
(all horizons), apply on 2025 test, and drive prob_groups generation with the stacked
probability + the TFT's quantile distribution. Validates the regenerated operational TAF
(BSS via the verifier's adverse_probability) vs the official TAF, and prints sample TAFs.
Saves the fitted stacker to data/models/stacker.joblib.

    uv run python scripts/stacked_taf.py [sample_pct] [n_test]
"""

import sys

import numpy as np
import pandas as pd

from wx.ai.dataset import build_inference_features_batch
from wx.ai.models import MultiTaskModel
from wx.ai.prob_groups import hours_from_quantiles
from wx.ai.seq_dataset import ADVERSE_CODES, build_sequences, split_sequences
from wx.ai.seq_models import SeqForecastModel
from wx.ai.tft_models import TFTModel
from wx.ai.torch_models import TorchMultiTaskModel
from wx.db.connection import connect
from wx.verification.scores import adverse_probability, brier_skill_score
from wx.ai.train import MODELS_DIR

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NTEST = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
BASES = ("mlp", "gbm", "tft", "lstm")


def _sub(batch, n, seed):
    import copy
    if len(batch.t0) <= n:
        return batch
    idx = np.random.default_rng(seed).choice(len(batch.t0), n, replace=False)
    s = copy.copy(batch)
    for f in ["icao", "t0", "x_past", "x_future", "x_static", "y_reg", "y_cat", "y_mask"]:
        setattr(s, f, getattr(batch, f)[idx])
    return s


def base_grids(con, models, batch):
    """{name: (N,H) calibrated P(adverse)} for all base models on the seq grid."""
    N, H = batch.y_mask.shape
    leads = batch.x_future[:, :, 10].astype(int)
    t0 = pd.to_datetime(batch.t0, utc=True)
    triples, keys = [], []
    for i in range(N):
        for h in range(H):
            icao = batch.stations[batch.icao[i]]
            vt = t0[i] + pd.Timedelta(hours=int(leads[i, h]))
            triples.append((icao, t0[i], vt)); keys.append((icao, int(t0[i].value), int(vt.value)))
    feats = build_inference_features_batch(con, triples)
    kidx = {(r.icao, int(r.t0.value), int(r.valid_time.value)): j
            for j, r in enumerate(feats.itertuples())}
    pos = np.array([kidx.get(k, -1) for k in keys])
    g = {}
    g["mlp"] = np.where(pos >= 0, np.asarray(models["mlp"].predict_adverse_proba(feats))[pos.clip(0)], np.nan).reshape(N, H)
    g["gbm"] = np.where(pos >= 0, np.asarray(models["gbm"].predict_adverse_proba(feats))[pos.clip(0)], np.nan).reshape(N, H)
    g["tft"] = models["tft"].predict_adverse_proba(batch)
    g["lstm"] = models["lstm"].predict_adverse_proba(batch)
    for k in g:
        g[k] = np.where(np.isnan(g[k]), g["tft"], g[k])
    return g


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    con = connect(read_only=True)
    print(f"building sequences ({SAMPLE}%) ...", flush=True)
    b = build_sequences(con, sample_pct=SAMPLE)
    _, va, te = split_sequences(b)
    va = _sub(va, 12000, 1); te = _sub(te, NTEST, 2)
    models = {"mlp": TorchMultiTaskModel.load("data/models/mlp.joblib"),
              "gbm": MultiTaskModel.load("data/models/gbm.joblib"),
              "tft": TFTModel.load("data/models/tft.joblib"),
              "lstm": SeqForecastModel.load("data/models/lstm.joblib")}

    # fit stacker on ALL val horizons (so it generalises across the full TAF window)
    gv = base_grids(con, models, va)
    ov = va.y_mask == 1
    Xv = np.column_stack([gv[k][ov] for k in BASES]); yv = np.isin(va.y_cat, ADVERSE_CODES)[ov].astype(int)
    stacker = LogisticRegression(max_iter=1000).fit(Xv, yv)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(stacker.predict_proba(Xv)[:, 1], yv)
    print("stacker coefs (mlp,gbm,tft,lstm):", np.round(stacker.coef_[0], 2),
          "intercept", round(float(stacker.intercept_[0]), 2), flush=True)

    gt = base_grids(con, models, te)
    N, H = te.y_mask.shape
    flat = np.column_stack([gt[k].reshape(-1) for k in BASES])
    stacked = iso.predict(stacker.predict_proba(flat)[:, 1]).reshape(N, H)

    # generate PROB/TEMPO TAFs: stacked P(adverse) + TFT quantile distribution
    quants = models["tft"].predict_quantiles(te)
    leads = te.x_future[:, :, 10].astype(int)
    t0 = pd.to_datetime(te.t0, utc=True)
    obs = te.y_mask == 1
    events = np.isin(te.y_cat, ADVERSE_CODES)
    reg_p, ev = [], []
    timelines = []
    for i in range(N):
        vh = [t0[i] + pd.Timedelta(hours=int(L)) for L in leads[i]]
        hrs = hours_from_quantiles(quants[i], stacked[i], vh)
        timelines.append((i, hrs))
        for h, eh in enumerate(hrs):
            if obs[i, h]:
                reg_p.append(adverse_probability(eh)); ev.append(int(events[i, h]))

    print(f"\nGenerated PROB/TEMPO TAFs on {N} test issues ({len(ev)} verified hours):", flush=True)
    print(f"  stacked-ensemble continuous BSS : {brier_skill_score(list(stacked[obs]), list(events[obs].astype(int))):+.3f}", flush=True)
    print(f"  regenerated TAF (PROB/TEMPO) BSS: {brier_skill_score(reg_p, ev):+.3f}", flush=True)
    print(f"  official TAF BSS                : -0.108", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"stacker": stacker, "iso": iso, "bases": BASES}, MODELS_DIR / "stacker.joblib")
    print("saved stacker -> data/models/stacker.joblib", flush=True)

    # sample TAFs: a few issues that contained an adverse hour
    print("\n=== sample generated TAFs (issues with adverse hours) ===", flush=True)
    shown = 0
    for i, hrs in timelines:
        if not events[i][obs[i]].any():
            continue
        icao = te.stations[te.icao[i]]
        print(f"\n{icao} issued {pd.Timestamp(t0[i]):%Y-%m-%d %H}Z:", flush=True)
        for h, eh in enumerate(hrs):
            if leads[i][h] not in (1, 3, 6, 9, 12, 18, 24):
                continue
            tag = ""
            if eh.prob:
                tag = f"  PROB{eh.prob['probability']} {eh.prob['flight_category']}"
            o = te.y_cat[i, h]
            ostr = {0: "LIFR", 1: "IFR", 2: "MVFR", 3: "VFR"}.get(int(o), "—") if obs[i, h] else "—"
            print(f"  +{leads[i][h]:>2}h vis={eh.prevailing['vis_m']:>5.0f} "
                  f"cat={eh.prevailing['flight_category']:<4}{tag:<18} | obs={ostr}", flush=True)
        shown += 1
        if shown >= 4:
            break
    con.close()


if __name__ == "__main__":
    main()
