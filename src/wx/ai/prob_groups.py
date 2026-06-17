"""Phase D — turn TFT quantile forecasts into PROB/TEMPO groups.

A TAF can only express uncertainty in discrete buckets: a firm *prevailing* state
(p=1 for the adverse event), a **PROB30** (0.3), or a **PROB40 / TEMPO** (0.4) — that
is exactly the mapping the verifier's ``adverse_probability`` inverts. The TFT gives a
*continuous* calibrated P(adverse) per hour plus a predictive distribution (q10/q50/q90)
for each element, so generation is:

  * **prevailing** = the median (q50) forecast → its flight category;
  * if the hour's P(adverse) clears the model's decision threshold → commit the
    prevailing state to adverse (the HSS operating point, p→1);
  * otherwise, if P(adverse) is non-trivial, attach a **PROB/TEMPO** group whose
    conditions are the **q10 "bad case"** (the low-visibility tail) and whose
    probability bucket is the one closest to the calibrated P(adverse).

Round-tripping these groups back through ``adverse_probability`` reproduces (up to the
TAF's 0/0.3/0.4/1 quantization) the model's calibrated probability — so the generated
TAF keeps the skill, now as an operational PROB/TEMPO product.
"""

from __future__ import annotations

import math

import numpy as np

from wx.ai.seq_dataset import TARGET_REG
from wx.ai.tft_models import MEDIAN_IDX
from wx.parsing.normalize import flight_category
from wx.verification.timeline import ExpectedHour

ADVERSE = {"IFR", "LIFR"}
_VIS, _CEIL, _WSPD, _WSIN, _WCOS = (TARGET_REG.index(k) for k in
                                    ("vis", "ceiling", "wspd", "wdir_sin", "wdir_cos"))
# The only adverse-probabilities a TAF can express (verifier's adverse_probability):
# none, PROB30, PROB40/TEMPO, or a firm prevailing commitment.
_BUCKETS = [
    (0.0, (False, None, 0.0)),
    (0.30, (False, "PROB30", 0.30)),
    (0.40, (False, "PROB40", 0.40)),
    (1.0, (True, None, 1.0)),
]
EMIT_FLOOR = 0.15            # midpoint(0, 0.30): below this -> no group
COMMIT_HIGH = 0.70          # midpoint(0.40, 1.0): above this -> commit prevailing adverse


def quantize(p: float):
    """Snap a calibrated P(adverse) to the NEAREST TAF construct.

    Returns (commit_prevailing_adverse, group_type, group_probability). Using the
    nearest bucket (not the HSS decision threshold) is what preserves calibration:
    committing prevailing-adverse (p=1) only kicks in near p>=0.7, so a hedge-worthy
    0.2-0.4 hour becomes a PROB group, not a false firm IFR."""
    return min(_BUCKETS, key=lambda lv: abs(p - lv[0]))[1]


def _state(vec: np.ndarray) -> dict:
    """Element vector (TARGET_REG order) -> a prevailing/overlay conditions dict."""
    vis = max(0.0, float(vec[_VIS]))
    ceiling = float(vec[_CEIL])
    wdir = (math.degrees(math.atan2(float(vec[_WSIN]), float(vec[_WCOS]))) % 360
            if abs(vec[_WSIN]) + abs(vec[_WCOS]) > 1e-6 else None)
    return {
        "vis_m": vis,
        "ceiling_ft": ceiling,
        "wind_spd_kt": max(0.0, float(vec[_WSPD])),
        "wind_dir_deg": wdir,
        "flight_category": flight_category(ceiling, vis),
    }


# Category-representative conditions for adverse groups. The TFT's vis/ceiling quantile
# regression collapses to "clear" even at q10 (the rare low-vis tail isn't captured), so
# the skill lives in the calibrated P(adverse), not the element distribution. We therefore
# fill an adverse group's conditions from these representative values (clamped against the
# model's own forecast) so the generated TAF is self-consistent and realistic.
_ADVERSE_COND = {"IFR": (2500.0, 600.0), "LIFR": (700.0, 200.0)}  # (vis_m, ceiling_ft)


def _force_adverse(state: dict) -> dict:
    """Make a state read as adverse (category + consistent low vis/ceiling)."""
    cat = state["flight_category"] if state["flight_category"] in ADVERSE else "IFR"
    vis_cap, ceil_cap = _ADVERSE_COND[cat]
    cur_ceil = state["ceiling_ft"]
    return {
        **state,
        "flight_category": cat,
        "vis_m": min(state["vis_m"], vis_cap),
        "ceiling_ft": min(cur_ceil, ceil_cap) if cur_ceil is not None else ceil_cap,
    }


def hours_from_quantiles(quantiles, p_adverse, valid_hours) -> list[ExpectedHour]:
    """Build a PROB/TEMPO-annotated ExpectedHour timeline for one issue.

    `quantiles` (H, len(TARGET_REG), Q), `p_adverse` (H,), `valid_hours` length H."""
    out = []
    for h, vh in enumerate(valid_hours):
        commit, gtype, gprob = quantize(float(p_adverse[h]))
        prob = None
        if commit:
            # firm adverse commitment: use the q10 bad-case CONDITIONS (low vis/ceiling)
            # so the prevailing state is internally consistent (not "vis 10km, cat LIFR").
            prevailing = _force_adverse(_state(quantiles[h, :, 0]))
        else:
            prevailing = _state(quantiles[h, :, MEDIAN_IDX])     # clear median
            if gtype is not None:
                bad = _force_adverse(_state(quantiles[h, :, 0]))  # q10 low-vis tail
                prob = {**bad, "probability": int(round(gprob * 100)), "group_type": gtype}
        out.append(ExpectedHour(vh, prevailing=prevailing, prob=prob))
    return out


def generate_for_batch(model, batch):
    """All issues in a SeqBatch -> list of (per-issue) ExpectedHour timelines.

    `valid_hours` are reconstructed as T0 + lead (lead lives in x_future col 10)."""
    import pandas as pd

    q = model.predict_quantiles(batch)               # (N, H, T, Q)
    p = model.predict_adverse_proba(batch)           # (N, H)
    leads = batch.x_future[:, :, 10]                 # (N, H) integer lead hours
    t0 = pd.to_datetime(batch.t0, utc=True)
    timelines = []
    for i in range(len(batch.t0)):
        vh = [t0[i] + pd.Timedelta(hours=int(L)) for L in leads[i]]
        timelines.append(hours_from_quantiles(q[i], p[i], vh))
    return timelines
