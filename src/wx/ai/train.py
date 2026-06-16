"""Phase C/D — training driver: fit a ladder rung, evaluate on the frozen test split.

Evaluation is fast and vectorized (tabular), on the held-out test rows: the model's
P(IFR-or-worse) is scored with Brier/BSS (the fair, hedging-aware metric), the
predicted category gives HSS, and the element regressors give MAE. The reference
bars are climatology (BSS = 0 by construction) and the official TAF's BSS — so we
can see directly whether a model clears them. Appends to the research log.
"""

from __future__ import annotations

import numpy as np

from wx.ai.dataset import LEADS_DEFAULT, build_samples, temporal_split
from wx.ai.evaluate import log_experiment
from wx.ai.models import MultiTaskModel
from wx.config import DATA_DIR
from wx.verification.scores import (
    brier_score,
    brier_skill_score,
    contingency_outcome,
    is_adverse,
    skill_scores,
)

MODELS_DIR = DATA_DIR / "models"


def tabular_eval(model: MultiTaskModel, df) -> dict:
    """Vectorized test-set metrics: HSS (category), Brier/BSS (P-adverse), element MAE."""
    preds = model.predict(df)
    p_adv = model.predict_adverse_proba(df)
    true_event = df["y_cat"].isin(("IFR", "LIFR")).tolist()
    pred_event = preds["pred_cat"].isin(("IFR", "LIFR")).tolist()
    outcomes = [contingency_outcome(pe, te) for pe, te in zip(pred_event, true_event)]
    events = [1 if e else 0 for e in true_event]

    def mae(pred_col, true_col):
        d = np.abs(preds[pred_col].to_numpy(float) - df[true_col].to_numpy(float))
        return float(np.nanmean(d)) if len(d) else None

    return {
        "n": len(df),
        "skill": skill_scores(outcomes),
        "brier": brier_score(list(p_adv), events),
        "bss": brier_skill_score(list(p_adv), events),
        "mae": {"vis": mae("pred_vis_m", "y_vis_m"),
                "ceiling": mae("pred_ceiling_ft", "y_ceiling_ft"),
                "wind": mae("pred_wspd", "y_wspd")},
    }


def official_bss(con, icaos=None) -> float | None:
    """Pooled Brier Skill Score of the official TAF over the test period (the bar)."""
    where = "scoring_profile = 'categorical' AND valid_hour >= TIMESTAMPTZ '2025-01-01'"
    if icaos:
        where += f" AND icao IN ({','.join(repr(i) for i in icaos)})"
    rows = con.execute(
        f"SELECT fcst_prob, obs_category FROM verification_hourly WHERE {where}"
    ).fetchall()
    if not rows:
        return None
    probs = [r[0] for r in rows]
    events = [1 if is_adverse(r[1]) else 0 for r in rows]
    return brier_skill_score(probs, events)


def train_and_evaluate(con, rung, icaos=None, train_end="2024-01-01", val_end="2025-01-01",
                       leads=LEADS_DEFAULT, sample_pct=5, save=True) -> dict:
    """Fit `rung` on train (<train_end), evaluate on the frozen test split (>=val_end)."""
    df = build_samples(con, icaos=icaos, leads=leads, sample_pct=sample_pct)
    tr, va, te = temporal_split(df, train_end=train_end, val_end=val_end)
    if tr.empty or te.empty:
        raise ValueError(f"empty split: train={len(tr)} test={len(te)}")

    model = MultiTaskModel(rung).fit(tr)
    metrics = tabular_eval(model, te)

    record = {"rung": rung, "icaos": icaos, "sample_pct": sample_pct,
              "n_train": len(tr), "n_test": len(te), "test": metrics,
              "reference": {"climatology_bss": 0.0, "official_bss": official_bss(con, icaos)}}
    log_experiment(record)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / f"{rung}.joblib")
    return record
