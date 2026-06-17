"""Phase D driver — train the seq2seq model and score it with the SAME Brier/BSS/HSS
benchmark as the tabular ladder, pooled over observed (issue, horizon) points on the
frozen 2025 test. Appends to data/research_log.jsonl and saves data/models/<rung>.joblib.

    uv run python scripts/train_seq.py [sample_pct] [rung]
"""

import sys

import numpy as np

from wx.ai.evaluate import log_experiment
from wx.ai.seq_dataset import ADVERSE_CODES, TARGET_REG, build_sequences, split_sequences
from wx.ai.seq_models import SeqForecastModel
from wx.ai.tft_models import TFTModel
from wx.ai.train import MODELS_DIR, bootstrap_bss, official_bss


def build_seq_model(rung: str):
    """Rungs: seq2seq (GRU), lstm (biLSTM + cross-attention), tft (attention+quantiles)."""
    if rung == "tft":
        return TFTModel(rung)
    if rung == "lstm":
        return SeqForecastModel(rung, cell="lstm", bidirectional=True, attention=True)
    return SeqForecastModel(rung)
from wx.db.connection import connect
from wx.verification.scores import (
    brier_score,
    brier_skill_score,
    contingency_outcome,
    skill_scores,
)

SAMPLE_PCT = int(sys.argv[1]) if len(sys.argv) > 1 else 25
RUNG = sys.argv[2] if len(sys.argv) > 2 else "seq2seq"


def seq_eval(model, batch) -> dict:
    """Pooled metrics over observed (sample, horizon) points — comparable to the ladder."""
    reg, _ = model._forward(batch)
    p_adv = model.predict_adverse_proba(batch)
    pred_evt = model.predict_adverse_event(batch)
    obs = batch.y_mask == 1
    events = (np.isin(batch.y_cat, ADVERSE_CODES) & obs)[obs].astype(int)
    p = p_adv[obs]
    pe = pred_evt[obs]
    outcomes = [contingency_outcome(bool(a), bool(b)) for a, b in zip(pe, events == 1)]

    def mae(j, col):
        d = np.abs(reg[..., j][obs] - batch.y_reg[..., j][obs])
        return float(np.nanmean(d)) if d.size else None

    return {
        "n": int(obs.sum()),
        "skill": skill_scores(outcomes),
        "brier": brier_score(list(p), list(events)),
        "bss": brier_skill_score(list(p), list(events)),
        "bss_ci": bootstrap_bss(np.asarray(p, float), np.asarray(events, int)),
        "threshold": float(model.adverse_threshold),
        "calibrated": model.calibrator is not None,
        "mae": {"vis": mae(0, "vis"), "ceiling": mae(1, "ceiling"), "wind": mae(2, "wspd")},
    }


def main():
    con = connect(read_only=True)
    print(f"building sequences (sample {SAMPLE_PCT}%)...", flush=True)
    batch = build_sequences(con, sample_pct=SAMPLE_PCT)
    tr, va, te = split_sequences(batch)
    print(f"sequences: train={len(tr.t0):,} val={len(va.t0):,} test={len(te.t0):,} "
          f"| stations={len(batch.stations)} horizons={batch.y_reg.shape[1]}", flush=True)

    model = build_seq_model(RUNG).fit(tr, val=va)
    val_m = seq_eval(model, va) if len(va.t0) else None
    test_m = seq_eval(model, te)
    off = official_bss(con)

    record = {"rung": RUNG, "icaos": None, "sample_pct": SAMPLE_PCT,
              "kind": "sequence", "n_train": int(len(tr.t0)),
              "n_val": int(len(va.t0)), "n_test": int(len(te.t0)),
              "val": val_m, "test": test_m,
              "reference": {"climatology_bss": 0.0, "official_bss": off}}
    log_experiment(record)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{RUNG}.joblib")

    s, ci = test_m["skill"], test_m["bss_ci"]
    cis = f"[{ci['low']:+.3f},{ci['high']:+.3f}]" if ci.get("low") is not None else "[n/a]"
    print(f"\n{RUNG} (test 2025, {test_m['n']:,} obs-horizons)", flush=True)
    print(f"  HSS={s['HSS']:.3f} POD={s['POD']:.2f} FAR={s['FAR']:.2f} | "
          f"Brier={test_m['brier']:.4f} BSS={test_m['bss']:+.3f} CI{cis} | "
          f"visMAE={test_m['mae']['vis']:.0f}m windMAE={test_m['mae']['wind']:.1f}kt", flush=True)
    print(f"  references: climatology BSS=0.000  official TAF BSS={off:+.3f}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
