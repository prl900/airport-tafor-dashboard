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

# Rungs trained as a GPU PyTorch multi-task net rather than sklearn estimators.
TORCH_RUNGS = {"mlp"}


def build_model(rung: str):
    """Model factory: sklearn ladder rungs vs the GPU PyTorch multi-task net.

    Both expose the same MultiTaskModel surface (fit/predict/predict_adverse_*/
    calibrator/threshold/save/load), so everything downstream is rung-agnostic."""
    if rung in TORCH_RUNGS:
        from wx.ai.torch_models import TorchMultiTaskModel

        return TorchMultiTaskModel(rung)
    return MultiTaskModel(rung)


def tabular_eval(model: MultiTaskModel, df) -> dict:
    """Vectorized split metrics: HSS (calibrated adverse decision), Brier/BSS
    (calibrated P-adverse), element MAE, plus a bootstrap CI on BSS."""
    preds = model.predict(df)
    p_adv = model.predict_adverse_proba(df)
    true_event = df["y_cat"].isin(("IFR", "LIFR")).tolist()
    # HSS now comes from the val-tuned adverse threshold, not the classifier argmax —
    # decoupled from the calibrated probability that BSS scores.
    pred_event = list(model.predict_adverse_event(df))
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
        "bss_ci": bootstrap_bss(np.asarray(p_adv, float), np.asarray(events, int)),
        "threshold": float(model.adverse_threshold),
        "calibrated": model.calibrator is not None,
        "mae": {"vis": mae("pred_vis_m", "y_vis_m"),
                "ceiling": mae("pred_ceiling_ft", "y_ceiling_ft"),
                "wind": mae("pred_wspd", "y_wspd")},
    }


def bootstrap_bss(probs, events, n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap 95% CI on the test-set BSS by resampling rows. `wins` is True when
    the CI excludes 0 on the positive side (significantly beats climatology)."""
    n = len(events)
    if n == 0:
        return {"low": None, "high": None, "wins": False}
    rng = np.random.default_rng(seed)
    base = events.mean()
    bs_ref = float(np.mean((base - events) ** 2)) or None
    if not bs_ref:
        return {"low": None, "high": None, "wins": False}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs = float(np.mean((probs[idx] - events[idx]) ** 2))
        vals.append(1.0 - bs / bs_ref)
    lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    return {"low": lo, "high": hi, "wins": lo > 0}


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

    # Val-gate: fit estimators on train, then fit the isotonic calibrator + adverse
    # threshold on val. Val metrics are the selection signal; test stays frozen.
    model = build_model(rung).fit(tr, val_df=va)
    val_metrics = tabular_eval(model, va) if not va.empty else None
    metrics = tabular_eval(model, te)

    record = {"rung": rung, "icaos": icaos, "sample_pct": sample_pct,
              "n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "val": val_metrics, "test": metrics,
              "reference": {"climatology_bss": 0.0, "official_bss": official_bss(con, icaos)}}
    log_experiment(record)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / f"{rung}.joblib")
    return record
