"""Train the full model ladder on the (sampled) full dataset and summarize vs the
references (climatology BSS=0, official TAF BSS). Models saved to data/models/."""

import sys

from wx.ai.dataset import estimate_build_memory
from wx.ai.train import official_bss, train_and_evaluate
from wx.db.connection import connect

SAMPLE_PCT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# rungs after the sample_pct arg, else the full ladder (fast rungs first; RF last)
RUNGS = sys.argv[2:] or ["linreg", "gbm", "mlp", "rf"]

con = connect(read_only=True)

# Memory guard: a full (or too-large) build OOM-kills this box (~22.7M rows ->
# ~50-70 GB at 100%). Estimate the peak for the requested sample and clamp down to
# the largest fraction that fits available RAM, so the experiment stays in bounds.
est = estimate_build_memory(con, sample_pct=SAMPLE_PCT)
print(f"memory: ~{est['est_peak_gb']:.1f} GB est. peak for {SAMPLE_PCT}% "
      f"({est['rows']:,} rows) | budget {est['budget_gb']:.1f} GB of "
      f"{est['avail_gb']:.1f} GB available | max safe sample {est['max_safe_pct']}%",
      flush=True)
if est["over_budget"] and est["max_safe_pct"]:
    print(f"  -> {SAMPLE_PCT}% over budget; clamping to {est['max_safe_pct']}%", flush=True)
    SAMPLE_PCT = est["max_safe_pct"]

off = official_bss(con)
print(f"references: climatology BSS=0.000  official TAF BSS={off:.3f}  (sample {SAMPLE_PCT}%)\n",
      flush=True)

results = {}
for rung in RUNGS:
    rec = train_and_evaluate(con, rung, sample_pct=SAMPLE_PCT)
    t = rec["test"]
    s = t["skill"]
    ci = t.get("bss_ci") or {}
    valbss = (rec.get("val") or {}).get("bss")
    results[rung] = rec
    ci_str = (f"[{ci['low']:+.3f},{ci['high']:+.3f}]"
              if ci.get("low") is not None else "[n/a]")
    valbss_str = f"{valbss:+.3f}" if valbss is not None else "n/a"
    print(f"{rung:7s} n_train={rec['n_train']:>8,} thr={t.get('threshold', 0.5):.2f} | "
          f"HSS={s['HSS']:.3f} POD={s['POD']:.2f} FAR={s['FAR']:.2f} | "
          f"Brier={t['brier']:.4f} BSS={t['bss']:+.3f} CI{ci_str} "
          f"valBSS={valbss_str} | visMAE={t['mae']['vis']:.0f}m "
          f"windMAE={t['mae']['wind']:.1f}kt", flush=True)

print("\n=== ladder summary (test 2025) ===", flush=True)
best = max(results, key=lambda r: results[r]["test"]["bss"])
print(f"best BSS: {best} ({results[best]['test']['bss']:+.3f}) vs official {off:+.3f}", flush=True)
con.close()
