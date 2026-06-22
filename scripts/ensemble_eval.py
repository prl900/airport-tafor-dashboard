"""Phase D — ensemble experiment: does blending the tabular MLP (strong short-lead)
with the TFT (strong long-lead + distributions) beat the MLP champion alone?

Both models predict P(adverse) on the SAME (icao, T0, lead) points: the TFT over its
sequence test set, the MLP over the matching tabular features (built batched for the
same triples). We grid-search a blend weight on val (min Brier), tune an HSS threshold
on val, and report test BSS/HSS for ensemble vs each model alone — all at the 9 standard
leads for comparability with the ladder.

    uv run python scripts/ensemble_eval.py [sample_pct] [n_subsample]
"""

import sys

import numpy as np
import pandas as pd

from wx.ai.dataset import build_inference_features_batch
from wx.ai.models import MultiTaskModel, _best_hss_threshold
from wx.ai.seq_dataset import ADVERSE_CODES, build_sequences, split_sequences
from wx.ai.tft_models import TFTModel
from wx.ai.torch_models import TorchMultiTaskModel
from wx.db.connection import connect
from wx.verification.scores import brier_skill_score, contingency_outcome, skill_scores

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
LEADS = [1, 2, 3, 6, 9, 12, 18, 24, 30]


def _subsample(batch, n, seed=0):
    import copy
    if len(batch.t0) <= n:
        return batch
    idx = np.random.default_rng(seed).choice(len(batch.t0), n, replace=False)
    s = copy.copy(batch)
    for f in ["icao", "t0", "x_past", "x_future", "x_static", "y_reg", "y_cat", "y_mask"]:
        setattr(s, f, getattr(batch, f)[idx])
    return s


def mlp_grid(con, mlp, batch):
    """MLP calibrated P(adverse) aligned to the (N,H) sequence grid (NaN where the
    tabular anchor is missing)."""
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
    p = mlp.predict_adverse_proba(feats)
    lut = {(r.icao, int(r.t0.value), int(r.valid_time.value)): pv
           for r, pv in zip(feats.itertuples(), p)}
    grid = np.full(N * H, np.nan)
    for j, k in enumerate(keys):
        if k in lut:
            grid[j] = lut[k]
    return grid.reshape(N, H)


def metrics(p, events, thr):
    pe = p >= thr
    out = [contingency_outcome(bool(a), bool(c)) for a, c in zip(pe, events == 1)]
    return brier_skill_score(list(p), list(events)), skill_scores(out)["HSS"]


def main():
    con = connect(read_only=True)
    print(f"building sequences ({SAMPLE}%) ...", flush=True)
    b = build_sequences(con, sample_pct=SAMPLE)
    _, va, te = split_sequences(b)
    va, te = _subsample(va, NSUB, 1), _subsample(te, NSUB, 2)
    mlp = TorchMultiTaskModel.load("data/models/mlp.joblib")
    tft = TFTModel.load("data/models/tft.joblib")

    def pack(batch):
        p_tft = tft.predict_adverse_proba(batch)
        p_mlp = mlp_grid(con, mlp, batch)
        p_mlp = np.where(np.isnan(p_mlp), p_tft, p_mlp)   # fall back to TFT if no anchor
        obs = batch.y_mask == 1
        sel = np.zeros_like(obs)
        for L in LEADS:
            sel[:, L - 1] = True
        sel &= obs
        ev = np.isin(batch.y_cat, ADVERSE_CODES)[sel].astype(int)
        return p_mlp[sel], p_tft[sel], ev

    vm, vt, ve = pack(va)
    tm, tt, tev = pack(te)

    # blend weight on val (min Brier), then HSS threshold on val
    ws = np.linspace(0, 1, 21)
    briers = [np.mean((w * vm + (1 - w) * vt - ve) ** 2) for w in ws]
    w = float(ws[int(np.argmin(briers))])
    vblend = w * vm + (1 - w) * vt
    thr = _best_hss_threshold(vblend, ve)
    print(f"\nval-chosen blend weight w(MLP)={w:.2f}  (TFT={1 - w:.2f})  thr={thr:.3f}", flush=True)

    print(f"\nTEST (9 standard leads, n={len(tev)}, base rate {tev.mean():.3f}):", flush=True)
    for name, p in [("MLP alone", tm), ("TFT alone", tt), ("ENSEMBLE", w * tm + (1 - w) * tt)]:
        # per-model HSS threshold tuned on val for fairness
        vt_p = {"MLP alone": vm, "TFT alone": vt, "ENSEMBLE": vblend}[name]
        th = _best_hss_threshold(vt_p, ve)
        bss, hss = metrics(p, tev, th)
        print(f"  {name:9s}  BSS={bss:+.3f}  HSS={hss:.3f}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
