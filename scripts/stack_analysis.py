"""Autonomous analysis — characterize the production stack {MLP, gbm, TFT}: reliability
(calibration) of its probabilities, per-lead skill decay, and per-region skill. These
are the properties that matter for a probabilistic TAF product."""

import sys

import numpy as np

from wx.ai.seq_dataset import ADVERSE_CODES, build_sequences, split_sequences
from wx.db.connection import connect
from wx.verification.scores import brier_skill_score
from stack_ensemble import _sub, tabular_grid  # type: ignore

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
PROD = ("mlp", "gbm", "tft")
LEADS = [1, 2, 3, 6, 9, 12, 18, 24, 30]
REGIONS = {"LEMD": "pen", "LEBL": "pen", "LEMG": "pen", "LEZL": "pen", "LEAL": "pen",
           "LEVC": "pen", "LEBB": "pen", "LEST": "pen", "LEGE": "pen", "LEVT": "pen",
           "LEXJ": "pen", "LEAS": "pen", "LERS": "pen", "LEMI": "pen", "LEPA": "bal",
           "LEMH": "bal", "LEIB": "bal", "GCLP": "can", "GCXO": "can", "GCTS": "can",
           "GCFV": "can", "GCRR": "can", "GCLA": "can", "GEML": "naf"}


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from wx.ai.models import MultiTaskModel
    from wx.ai.seq_models import SeqForecastModel
    from wx.ai.tft_models import TFTModel
    from wx.ai.torch_models import TorchMultiTaskModel

    con = connect(read_only=True)
    print(f"building sequences ({SAMPLE}%) ...", flush=True)
    b = build_sequences(con, sample_pct=SAMPLE)
    _, va, te = split_sequences(b)
    va, te = _sub(va, 12000, 1), _sub(te, NSUB, 2)
    models = {"mlp": TorchMultiTaskModel.load("data/models/mlp.joblib"),
              "gbm": MultiTaskModel.load("data/models/gbm.joblib"),
              "tft": TFTModel.load("data/models/tft.joblib")}

    def stacked_grid(batch):
        tab = tabular_grid(con, {"mlp": models["mlp"], "gbm": models["gbm"]}, batch)
        g = {"mlp": tab["mlp"], "gbm": tab["gbm"], "tft": models["tft"].predict_adverse_proba(batch)}
        for k in g:
            g[k] = np.where(np.isnan(g[k]), g["tft"], g[k])
        return g

    gv, gt = stacked_grid(va), stacked_grid(te)

    def flat(g, batch, with_meta=False):
        sel = np.zeros_like(batch.y_mask, bool)
        for L in LEADS:
            sel[:, L - 1] = True
        sel &= batch.y_mask == 1
        X = np.column_stack([g[k][sel] for k in PROD])
        ev = np.isin(batch.y_cat, ADVERSE_CODES)[sel].astype(int)
        if not with_meta:
            return X, ev
        leads = batch.x_future[:, :, 10].astype(int)[sel]
        icao = np.array([batch.stations[i] for i in batch.icao])[:, None].repeat(batch.y_mask.shape[1], 1)[sel]
        return X, ev, leads, icao

    Xv, yv = flat(gv, va)
    Xt, yt, lt, it = flat(gt, te, with_meta=True)
    lr = LogisticRegression(max_iter=1000).fit(Xv, yv)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(lr.predict_proba(Xv)[:, 1], yv)
    pt = iso.predict(lr.predict_proba(Xt)[:, 1])
    print(f"test pts={len(yt)} base rate={yt.mean():.3f}  overall BSS={brier_skill_score(list(pt), list(yt)):+.3f}\n", flush=True)

    print("=== reliability (calibration) ===", flush=True)
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        m = (pt >= lo) & (pt < hi)
        if m.sum():
            print(f"  pred [{lo:.2f},{hi:.2f}): n={m.sum():6d} mean_pred={pt[m].mean():.3f} obs_freq={yt[m].mean():.3f}", flush=True)

    print("\n=== per-lead BSS ===", flush=True)
    for L in LEADS:
        m = lt == L
        if m.sum():
            print(f"  +{L:>2}h: BSS={brier_skill_score(list(pt[m]), list(yt[m].astype(int))):+.3f} (n={m.sum()}, rate={yt[m].mean():.3f})", flush=True)

    print("\n=== per-region BSS ===", flush=True)
    reg = np.array([REGIONS.get(x, "?") for x in it])
    for r in ("pen", "bal", "can", "naf"):
        m = reg == r
        if m.sum() and yt[m].sum() > 0:
            print(f"  {r}: BSS={brier_skill_score(list(pt[m]), list(yt[m].astype(int))):+.3f} (n={m.sum()}, rate={yt[m].mean():.3f})", flush=True)
    con.close()


if __name__ == "__main__":
    main()
