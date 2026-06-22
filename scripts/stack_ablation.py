"""Autonomous ablation — which base models matter in the stack? Build the 4 base-model
P(adverse) grids once, then fit a logistic stack over every useful subset (singles,
pairs, drop-one, full) on 2024 val and score on 2025 test (9 leads). Also a per-region
breakdown of the full stack."""

import sys
from itertools import combinations

import numpy as np

from wx.ai.seq_dataset import ADVERSE_CODES, build_sequences, split_sequences
from wx.db.connection import connect
from wx.verification.scores import brier_skill_score, contingency_outcome, skill_scores
# reuse the aligned-grid + base-model loading from the stacking experiment
from stack_ensemble import _sub, tabular_grid  # type: ignore

BASES = ("mlp", "gbm", "tft", "lstm")
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
LEADS = [1, 2, 3, 6, 9, 12, 18, 24, 30]


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
              "tft": TFTModel.load("data/models/tft.joblib"),
              "lstm": SeqForecastModel.load("data/models/lstm.joblib")}

    def grids(batch):
        tab = tabular_grid(con, {"mlp": models["mlp"], "gbm": models["gbm"]}, batch)
        g = {"mlp": tab["mlp"], "gbm": tab["gbm"],
             "tft": models["tft"].predict_adverse_proba(batch),
             "lstm": models["lstm"].predict_adverse_proba(batch)}
        for k in g:
            g[k] = np.where(np.isnan(g[k]), g["tft"], g[k])
        sel = np.zeros_like(batch.y_mask, bool)
        for L in LEADS:
            sel[:, L - 1] = True
        sel &= batch.y_mask == 1
        ev = np.isin(batch.y_cat, ADVERSE_CODES)[sel].astype(int)
        return {k: g[k][sel] for k in BASES}, ev, sel, batch

    gv, yv, _, _ = grids(va)
    gt, yt, selt, teb = grids(te)
    print(f"val={len(yv)} test={len(yt)} base rate={yt.mean():.3f}\n", flush=True)

    def stack_bss(subset):
        Xv = np.column_stack([gv[k] for k in subset])
        Xt = np.column_stack([gt[k] for k in subset])
        lr = LogisticRegression(max_iter=1000).fit(Xv, yv)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            lr.predict_proba(Xv)[:, 1], yv)
        from wx.ai.models import _best_hss_threshold
        pv = iso.predict(lr.predict_proba(Xv)[:, 1])
        pt = iso.predict(lr.predict_proba(Xt)[:, 1])
        thr = _best_hss_threshold(pv, yv)
        out = [contingency_outcome(bool(p >= thr), bool(e)) for p, e in zip(pt, yt)]
        return brier_skill_score(list(pt), list(yt)), skill_scores(out)["HSS"]

    print("=== subset stacks (test 9 leads): BSS / HSS ===", flush=True)
    subsets = ([(k,) for k in BASES]                                   # singles
               + [tuple(c) for c in combinations(BASES, 2)]            # pairs
               + [tuple(s for s in BASES if s != drop) for drop in BASES]  # drop-one
               + [BASES])                                              # full
    seen = set()
    rows = []
    for sub in subsets:
        key = tuple(sorted(sub))
        if key in seen:
            continue
        seen.add(key)
        bss, hss = stack_bss(sub)
        rows.append(("+".join(sub), bss, hss))
    for name, bss, hss in sorted(rows, key=lambda r: r[1], reverse=True):
        print(f"  {name:20s} BSS={bss:+.3f} HSS={hss:.3f}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
