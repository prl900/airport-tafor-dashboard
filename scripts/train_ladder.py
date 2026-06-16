"""Train the full model ladder on the (sampled) full dataset and summarize vs the
references (climatology BSS=0, official TAF BSS). Models saved to data/models/."""

import sys

from wx.ai.train import official_bss, train_and_evaluate
from wx.db.connection import connect

SAMPLE_PCT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# rungs after the sample_pct arg, else the full ladder (fast rungs first; RF last)
RUNGS = sys.argv[2:] or ["linreg", "gbm", "mlp", "rf"]

con = connect(read_only=True)
off = official_bss(con)
print(f"references: climatology BSS=0.000  official TAF BSS={off:.3f}  (sample {SAMPLE_PCT}%)\n",
      flush=True)

results = {}
for rung in RUNGS:
    rec = train_and_evaluate(con, rung, sample_pct=SAMPLE_PCT)
    t = rec["test"]
    s = t["skill"]
    results[rung] = rec
    print(f"{rung:7s} n_train={rec['n_train']:>8,} | HSS={s['HSS']:.3f} POD={s['POD']:.2f} "
          f"FAR={s['FAR']:.2f} | Brier={t['brier']:.4f} BSS={t['bss']:+.3f} | "
          f"visMAE={t['mae']['vis']:.0f}m windMAE={t['mae']['wind']:.1f}kt", flush=True)

print("\n=== ladder summary (test 2025) ===", flush=True)
best = max(results, key=lambda r: results[r]["test"]["bss"])
print(f"best BSS: {best} ({results[best]['test']['bss']:+.3f}) vs official {off:+.3f}", flush=True)
con.close()
