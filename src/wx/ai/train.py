"""Phase C — training driver: fit a ladder rung, evaluate vs the champion/skyline.

Trains on the causal frame, evaluates the resulting ModelForecaster over official-TAF
windows in the validation split (and optionally the frozen test split), compares to
the official-TAF skyline with a paired bootstrap, and appends to the research log.
This is the unit the auto-research controller (Phase E) will call repeatedly.
"""

from __future__ import annotations

from datetime import timezone

import pandas as pd

from wx.ai.dataset import LEADS_DEFAULT, build_samples, temporal_split
from wx.ai.evaluate import bootstrap_hss_diff, evaluate, log_experiment, result_summary
from wx.ai.generate import ClimatologyForecaster, OfficialForecaster, PersistenceForecaster
from wx.ai.models import ModelForecaster, MultiTaskModel
from wx.config import DATA_DIR

MODELS_DIR = DATA_DIR / "models"


def train_and_evaluate(con, rung, icaos=None, train_end="2024-01-01",
                       val_end="2025-01-01", leads=LEADS_DEFAULT, save=True):
    """Fit `rung`, evaluate on the validation window vs official + baselines."""
    df = build_samples(con, icaos=icaos, leads=leads)
    tr, va, te = temporal_split(df, train_end=train_end, val_end=val_end)
    if tr.empty or va.empty:
        raise ValueError(f"empty split: train={len(tr)} val={len(va)} (need more data)")

    model = MultiTaskModel(rung).fit(tr)
    fc = ModelForecaster(model, name=f"model:{rung}")

    vs = pd.Timestamp(train_end, tz=timezone.utc)
    ve = pd.Timestamp(val_end, tz=timezone.utc)
    r_model = evaluate(con, fc, vs, ve, icaos)
    r_off = evaluate(con, OfficialForecaster(), vs, ve, icaos)
    r_pers = evaluate(con, PersistenceForecaster(), vs, ve, icaos)
    r_clim = evaluate(con, ClimatologyForecaster(), vs, ve, icaos)

    vs_official = bootstrap_hss_diff(r_off.outcomes, r_model.outcomes)
    vs_persist = bootstrap_hss_diff(r_pers.outcomes, r_model.outcomes)

    record = {
        "rung": rung, "icaos": icaos, "n_train": len(tr), "n_val": len(va),
        "val_window": [str(vs), str(ve)],
        "model": result_summary(r_model),
        "official": result_summary(r_off),
        "persistence": result_summary(r_pers),
        "climatology": result_summary(r_clim),
        "vs_official": vs_official,     # beats the skyline?
        "vs_persistence": vs_persist,   # beats the floor?
    }
    log_experiment(record)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / f"{rung}.joblib")

    return model, record
