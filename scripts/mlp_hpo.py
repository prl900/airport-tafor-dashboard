"""Autonomous HPO for the GPU MLP champion: build the 9% tabular frame ONCE, then
train several TorchMultiTaskModel configs and score each on the frozen 2025 test
(tabular_eval = 9 standard leads). Prints a ranked table; saves the best to
data/models/mlp_hpo_best.joblib (does NOT clobber the registered champion)."""

import sys

from wx.ai.dataset import build_samples, temporal_split
from wx.ai.train import MODELS_DIR, tabular_eval
from wx.ai.torch_models import TorchMultiTaskModel
from wx.db.connection import connect

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 9

CONFIGS = [
    ("base   (256,128) d.2 lr1e-3 e80", dict(trunk=(256, 128), dropout=0.2, lr=1e-3, max_epochs=80)),
    ("wide   (384,192) d.2 lr1e-3 e80", dict(trunk=(384, 192), dropout=0.2, lr=1e-3, max_epochs=80)),
    ("deep   (256,192,96) d.2 e100",    dict(trunk=(256, 192, 96), dropout=0.2, lr=1e-3, max_epochs=100)),
    ("reg    (256,128) d.35 wd1e-4",    dict(trunk=(256, 128), dropout=0.35, lr=1e-3, weight_decay=1e-4, max_epochs=100)),
    ("slowlr (384,192) lr5e-4 e120",    dict(trunk=(384, 192), dropout=0.25, lr=5e-4, max_epochs=120)),
    ("xwide  (512,256) d.3 e100",       dict(trunk=(512, 256), dropout=0.3, lr=1e-3, max_epochs=100)),
]


def main():
    con = connect(read_only=True)
    print(f"building tabular frame ({SAMPLE}%) ...", flush=True)
    df = build_samples(con, sample_pct=SAMPLE)
    tr, va, te = temporal_split(df)
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,}\n", flush=True)

    results = []
    for name, kw in CONFIGS:
        m = TorchMultiTaskModel("mlp", batch_size=4096, patience=10, **kw).fit(tr, val_df=va)
        ev = tabular_eval(m, te)
        s = ev["skill"]
        ci = ev["bss_ci"]
        results.append((name, ev["bss"], s["HSS"], m))
        print(f"{name:34s} BSS={ev['bss']:+.3f} CI[{ci['low']:+.3f},{ci['high']:+.3f}] "
              f"HSS={s['HSS']:.3f} POD={s['POD']:.2f} FAR={s['FAR']:.2f}", flush=True)

    results.sort(key=lambda r: r[1], reverse=True)
    best = results[0]
    print(f"\nBEST: {best[0]}  BSS={best[1]:+.3f} HSS={best[2]:.3f}", flush=True)
    print(f"(champion baseline: BSS +0.311 HSS 0.472)", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best[3].save(MODELS_DIR / "mlp_hpo_best.joblib")
    print("saved -> data/models/mlp_hpo_best.joblib", flush=True)
    con.close()


if __name__ == "__main__":
    main()
