"""Assess how useful each NWP predictor is, so operational inclusion is an empirical
decision rather than a guess.

Two complementary views, both reported in the model's own metrics (adverse-category HSS
and visibility/ceiling MAE) rather than abstract estimator scores:

- ``permutation_importance``: shuffle a feature group in the eval frame and measure the
  skill it destroys. Cheap (no retrain). A group that is all-NaN (variable not yet
  ingested) scores ~0 — the honest "no data yet" signal.
- ``ablation``: drop a feature group, retrain, and measure the skill lost vs the full
  model. The gold standard for "is this variable worth carrying operationally", but pays a
  full retrain per group.

Operational-inclusion rule of thumb: keep a variable only if it gives a real skill gain
here AND is available in the serve-time source (ECMWF Open Data). A variable that helps but
is missing at serve time would create a train/serve mismatch — see docs/ML_PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wx.ai.dataset import feature_columns

# Curated feature groups (label -> exact f_* columns). Each NWP variable is its own group
# so importance/ablation answer "is THIS variable pulling its weight". Tendencies live with
# their base variable. Columns absent from a given frame are silently ignored.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "wind": ("f_et_wspd", "f_et_wdir_sin", "f_et_wdir_cos", "f_et_wdir_known",
             "f_et_gust", "f_tend_wspd"),
    "t2m": ("f_et_t2m", "f_tend_t2m"),
    "spread": ("f_et_spread", "f_tend_spread"),
    "tcc": ("f_et_tcc", "f_tend_tcc"),
    "cloud_layers": ("f_et_lcc", "f_et_mcc", "f_et_hcc", "f_tend_lcc"),
    "cbh": ("f_et_cbh",),
    "tp": ("f_et_tp",),
    "msl": ("f_et_msl", "f_tend_msl"),
    # Candidate predictors under assessment:
    "cape": ("f_et_cape",),
    "blh": ("f_et_blh",),
    "tcwv": ("f_et_tcwv",),
    "skt": ("f_et_skt", "f_tend_skt"),
}


def _headline(metrics: dict) -> dict:
    """Pull the comparable scalars out of a tabular_eval() dict."""
    skill = metrics.get("skill") or {}
    mae = metrics.get("mae") or {}
    return {
        "hss": skill.get("HSS"),
        "bss": metrics.get("bss"),
        "mae_vis": mae.get("vis"),
        "mae_ceiling": mae.get("ceiling"),
    }


def _present(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c in df.columns]


def permutation_importance(model, eval_df: pd.DataFrame, groups: dict | None = None,
                           n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Group permutation importance on ``eval_df`` (use the val or frozen-test split).

    For each group, the group's columns are jointly shuffled ``n_repeats`` times and the
    model re-evaluated; importance is the mean skill lost. Positive ``d_hss`` / positive
    ``d_mae_*`` mean the variable was helping (shuffling hurt). Sorted by HSS impact.
    """
    from wx.ai.train import tabular_eval

    groups = groups or FEATURE_GROUPS
    rng = np.random.default_rng(seed)
    base = _headline(tabular_eval(model, eval_df))
    rows = []
    for label, cols in groups.items():
        cols = _present(eval_df, cols)
        if not cols:
            continue
        all_nan = bool(eval_df[cols].isna().all().all())
        deltas = {"d_hss": [], "d_mae_vis": [], "d_mae_ceiling": []}
        for _ in range(n_repeats):
            perm = eval_df.copy()
            idx = rng.permutation(len(perm))
            perm[cols] = perm[cols].to_numpy()[idx]
            m = _headline(tabular_eval(model, perm))
            if base["hss"] is not None and m["hss"] is not None:
                deltas["d_hss"].append(base["hss"] - m["hss"])
            for k in ("mae_vis", "mae_ceiling"):
                if base[k] is not None and m[k] is not None:
                    deltas[f"d_{k}"].append(m[k] - base[k])  # MAE rises when var removed
        rows.append({
            "group": label,
            "n_features": len(cols),
            "all_nan": all_nan,
            "d_hss": float(np.mean(deltas["d_hss"])) if deltas["d_hss"] else None,
            "d_mae_vis": float(np.mean(deltas["d_mae_vis"])) if deltas["d_mae_vis"] else None,
            "d_mae_ceiling": (float(np.mean(deltas["d_mae_ceiling"]))
                              if deltas["d_mae_ceiling"] else None),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("d_hss", ascending=False, na_position="last").reset_index(drop=True)


def ablation(con, rung: str = "gbm", groups: dict | None = None, *, icaos=None,
             leads=None, sample_pct: int = 5, nwp_source: str = "era5",
             train_end="2024-01-01", val_end="2025-01-01") -> pd.DataFrame:
    """Leave-one-group-out ablation: retrain dropping each group's columns and report the
    test-set skill lost vs the full-feature model. Positive ``d_hss`` = the group helped.
    Expensive (one retrain per group); use a modest ``sample_pct`` for screening."""
    from wx.ai.dataset import LEADS_DEFAULT, build_samples, temporal_split
    from wx.ai.models import MultiTaskModel
    from wx.ai.train import tabular_eval

    groups = groups or FEATURE_GROUPS
    leads = leads or LEADS_DEFAULT
    df = build_samples(con, icaos=icaos, leads=leads, sample_pct=sample_pct,
                       nwp_source=nwp_source)
    tr, va, te = temporal_split(df, train_end=train_end, val_end=val_end)
    if tr.empty or te.empty:
        raise ValueError(f"empty split: train={len(tr)} test={len(te)}")

    full = _headline(tabular_eval(MultiTaskModel(rung).fit(tr, val_df=va), te))
    rows = [{"group": "(full model)", "n_features": len(feature_columns(df)),
             "hss": full["hss"], "d_hss": 0.0, "mae_vis": full["mae_vis"],
             "mae_ceiling": full["mae_ceiling"]}]
    for label, cols in groups.items():
        cols = _present(df, cols)
        if not cols:
            continue
        keep = [c for c in df.columns if c not in cols]
        m = MultiTaskModel(rung).fit(tr[keep], val_df=va[keep])
        red = _headline(tabular_eval(m, te[keep]))
        rows.append({
            "group": f"-{label}", "n_features": len(cols),
            "hss": red["hss"],
            "d_hss": (full["hss"] - red["hss"]) if (full["hss"] is not None
                                                    and red["hss"] is not None) else None,
            "mae_vis": red["mae_vis"], "mae_ceiling": red["mae_ceiling"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("d_hss", ascending=False, na_position="last").reset_index(drop=True)
