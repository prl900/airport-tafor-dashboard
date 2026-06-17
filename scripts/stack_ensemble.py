"""Autonomous Q6 — stacked / per-lead ensemble over base models {MLP, gbm, TFT, LSTM}.

All base models' calibrated P(adverse) are aligned on the SAME (icao, T0, lead) points
(seq grid; tabular models scored via batched features for the matching triples). We fit
several stackers on val and report test BSS/HSS at the 9 standard leads:
  - MLP alone (reference champion)
  - best 2-model linear blend (the current ensemble)
  - global logistic regression over the 4 probs
  - per-lead logistic regression
Isotonic recalibration is applied to each stacker's output on val.

    uv run python scripts/stack_ensemble.py [sample_pct] [n_subsample]
"""

import sys

import numpy as np
import pandas as pd

from wx.ai.dataset import build_inference_features_batch
from wx.ai.models import MultiTaskModel
from wx.ai.seq_dataset import ADVERSE_CODES, build_sequences, split_sequences
from wx.ai.seq_models import SeqForecastModel
from wx.ai.tft_models import TFTModel
from wx.ai.torch_models import TorchMultiTaskModel
from wx.db.connection import connect
from wx.verification.scores import brier_skill_score, contingency_outcome, skill_scores

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
LEADS = [1, 2, 3, 6, 9, 12, 18, 24, 30]


def _sub(batch, n, seed):
    import copy
    if len(batch.t0) <= n:
        return batch
    idx = np.random.default_rng(seed).choice(len(batch.t0), n, replace=False)
    s = copy.copy(batch)
    for f in ["icao", "t0", "x_past", "x_future", "x_static", "y_reg", "y_cat", "y_mask"]:
        setattr(s, f, getattr(batch, f)[idx])
    return s


def tabular_grid(con, models, batch):
    """{name: (N,H) calibrated P(adverse)} for tabular models, aligned to the seq grid."""
    N, H = batch.y_mask.shape
    leads = batch.x_future[:, :, 10].astype(int)
    t0 = pd.to_datetime(batch.t0, utc=True)
    triples, keys = [], []
    for i in range(N):
        for h in range(H):
            icao = batch.stations[batch.icao[i]]
            vt = t0[i] + pd.Timedelta(hours=int(leads[i, h]))
            triples.append((icao, t0[i], vt))
            keys.append((icao, int(t0[i].value), int(vt.value)))
    feats = build_inference_features_batch(con, triples)
    keyidx = {(r.icao, int(r.t0.value), int(r.valid_time.value)): j
              for j, r in enumerate(feats.itertuples())}
    pos = np.array([keyidx.get(k, -1) for k in keys])
    out = {}
    for name, m in models.items():
        p = np.asarray(m.predict_adverse_proba(feats))
        grid = np.where(pos >= 0, p[pos.clip(0)], np.nan).reshape(N, H)
        out[name] = grid
    return out


def lead_mask(batch):
    obs = batch.y_mask == 1
    sel = np.zeros_like(obs)
    for L in LEADS:
        sel[:, L - 1] = True
    return sel & obs


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    con = connect(read_only=True)
    print(f"building sequences ({SAMPLE}%) ...", flush=True)
    b = build_sequences(con, sample_pct=SAMPLE)
    _, va, te = split_sequences(b)
    va, te = _sub(va, NSUB, 1), _sub(te, NSUB, 2)

    mlp = TorchMultiTaskModel.load("data/models/mlp.joblib")
    gbm = MultiTaskModel.load("data/models/gbm.joblib")
    tft = TFTModel.load("data/models/tft.joblib")
    lstm = SeqForecastModel.load("data/models/lstm.joblib")

    def features(batch):
        tab = tabular_grid(con, {"mlp": mlp, "gbm": gbm}, batch)
        p_tft = tft.predict_adverse_proba(batch)
        p_lstm = lstm.predict_adverse_proba(batch)
        cols = {"mlp": tab["mlp"], "gbm": tab["gbm"], "tft": p_tft, "lstm": p_lstm}
        for k in cols:                                   # NaN (no tabular anchor) -> TFT
            cols[k] = np.where(np.isnan(cols[k]), p_tft, cols[k])
        sel = lead_mask(batch)
        X = np.column_stack([cols[k][sel] for k in ("mlp", "gbm", "tft", "lstm")])
        ev = np.isin(batch.y_cat, ADVERSE_CODES)[sel].astype(int)
        leads = batch.x_future[:, :, 10].astype(int)[sel]
        return X, ev, leads

    Xv, yv, lv = features(va)
    Xt, yt, lt = features(te)
    print(f"val pts={len(yv)} test pts={len(yt)} base rate={yt.mean():.3f}\n", flush=True)

    def report(name, pv, pt):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv)
        pcal = iso.predict(pt)
        from wx.ai.models import _best_hss_threshold
        thr = _best_hss_threshold(iso.predict(pv), yv)
        out = [contingency_outcome(bool(p >= thr), bool(e)) for p, e in zip(pcal, yt)]
        print(f"  {name:24s} BSS={brier_skill_score(list(pcal), list(yt)):+.3f} "
              f"HSS={skill_scores(out)['HSS']:.3f}", flush=True)

    print(f"TEST (9 leads, n={len(yt)}):", flush=True)
    report("MLP alone", Xv[:, 0], Xt[:, 0])
    # current 2-model linear blend (mlp,tft) weight from earlier (~0.6/0.4)
    report("linear blend mlp+tft", 0.6 * Xv[:, 0] + 0.4 * Xv[:, 2],
           0.6 * Xt[:, 0] + 0.4 * Xt[:, 2])
    # global logistic stacker over 4 probs
    lr = LogisticRegression(max_iter=1000).fit(Xv, yv)
    report("logistic stack (4)", lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1])
    # per-lead logistic stacker
    pv_pl, pt_pl = np.zeros(len(yv)), np.zeros(len(yt))
    for L in LEADS:
        mv, mt = lv == L, lt == L
        if mv.sum() > 50 and len(np.unique(yv[mv])) > 1:
            lrl = LogisticRegression(max_iter=1000).fit(Xv[mv], yv[mv])
            pv_pl[mv] = lrl.predict_proba(Xv[mv])[:, 1]
            pt_pl[mt] = lrl.predict_proba(Xt[mt])[:, 1]
    report("per-lead logistic (4)", pv_pl, pt_pl)
    con.close()


if __name__ == "__main__":
    main()
