"""Phase D — PROB/TEMPO generation from TFT quantiles. The key property: the generated
groups round-trip back through the verifier's adverse_probability to the quantized
calibrated probability, so the TAF product keeps the model's skill."""

from datetime import datetime, timedelta, timezone

import numpy as np

from wx.ai.prob_groups import (
    EMIT_FLOOR,
    hours_from_quantiles,
    quantize,
)
from wx.ai.seq_dataset import TARGET_REG
from wx.verification.scores import adverse_probability

UTC = timezone.utc
Q = 3  # q10, q50, q90


def _quantiles_one_hour(median, q10):
    """(1, len(TARGET_REG), Q) with given median + q10 (q90 = median)."""
    arr = np.zeros((1, len(TARGET_REG), Q), dtype=float)
    arr[0, :, 1] = median
    arr[0, :, 0] = q10
    arr[0, :, 2] = median
    return arr


# element order: vis, ceiling, wspd, wdir_sin, wdir_cos
CLEAR = [9999, 30000, 5, 0, 1]      # VFR
FOG = [300, 100, 5, 0, 1]           # LIFR


def test_quantize_buckets():
    assert quantize(0.9) == (True, None, 1.0)            # commit prevailing adverse
    assert quantize(0.05) == (False, None, 0.0)          # nothing
    assert quantize(0.30)[1] == "PROB30"
    assert quantize(0.40)[1] == "PROB40"
    assert quantize(EMIT_FLOOR - 0.01) == (False, None, 0.0)


def test_commit_prevailing_when_over_threshold():
    q = _quantiles_one_hour(FOG, FOG)
    eh = hours_from_quantiles(q, np.array([0.8]), [datetime(2025, 1, 1, tzinfo=UTC)])[0]
    assert eh.prevailing["flight_category"] in ("IFR", "LIFR")
    assert adverse_probability(eh) == 1.0
    assert eh.prob is None


def test_prob_group_from_bad_tail():
    """Median clear but q10 foggy + moderate P(adverse) -> a PROB group at 0.4."""
    q = _quantiles_one_hour(CLEAR, FOG)
    eh = hours_from_quantiles(q, np.array([0.4]), [datetime(2025, 1, 1, tzinfo=UTC)])[0]
    assert eh.prevailing["flight_category"] == "VFR"      # prevailing stays clear
    assert eh.prob is not None and eh.prob["group_type"] == "PROB40"
    assert eh.prob["flight_category"] in ("IFR", "LIFR")  # bad-case tail is adverse
    assert adverse_probability(eh) == 0.4                 # round-trips to the bucket


def test_no_group_when_probability_trivial():
    q = _quantiles_one_hour(CLEAR, FOG)
    eh = hours_from_quantiles(q, np.array([0.02]), [datetime(2025, 1, 1, tzinfo=UTC)])[0]
    assert eh.prob is None
    assert adverse_probability(eh) == 0.0


def test_timeline_length_matches_horizons():
    H = 6
    q = np.tile(_quantiles_one_hour(CLEAR, FOG), (H, 1, 1))
    p = np.linspace(0.0, 0.9, H)
    base = datetime(2025, 1, 1, tzinfo=UTC)
    hrs = hours_from_quantiles(q, p, [base + timedelta(hours=i) for i in range(H)])
    assert len(hrs) == H
    # probability is non-decreasing in p across the buckets {0, .3, .4, 1}
    probs = [adverse_probability(e) for e in hrs]
    assert probs == sorted(probs)
