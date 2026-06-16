"""Phase D — probabilistic scoring: TAF hedging credited via Brier, not punished."""

from datetime import datetime, timezone

from wx.verification.scores import (
    adverse_probability,
    brier_score,
    brier_skill_score,
)
from wx.verification.timeline import ExpectedHour

UTC = timezone.utc


def _eh(prevailing_cat=None, tempo_cat=None, prob_cat=None, prob=None):
    return ExpectedHour(
        valid_hour=datetime(2023, 1, 1, tzinfo=UTC),
        prevailing={"flight_category": prevailing_cat},
        tempo={"flight_category": tempo_cat} if tempo_cat else None,
        prob=({"flight_category": prob_cat, "probability": prob} if prob_cat else None),
    )


def test_adverse_probability_from_groups():
    assert adverse_probability(_eh(prevailing_cat="LIFR")) == 1.0     # firm adverse
    assert adverse_probability(_eh(prevailing_cat="VFR")) == 0.0      # firm benign
    assert adverse_probability(_eh("VFR", prob_cat="IFR", prob=30)) == 0.3   # PROB30
    assert adverse_probability(_eh("VFR", prob_cat="IFR", prob=40)) == 0.4   # PROB40
    assert adverse_probability(_eh("VFR", tempo_cat="IFR")) == 0.4           # bare TEMPO


def test_brier_rewards_hedging_over_false_alarm():
    # A PROB30 fog that does NOT verify: categorically a full false alarm, but the
    # Brier penalty is only (0.3-0)^2 = 0.09 — hedging is credited.
    assert brier_score([0.3], [0]) == 0.09
    # A firm IFR forecast that misses costs the full (1-0)^2 = 1.0.
    assert brier_score([1.0], [0]) == 1.0
    # Perfect probabilistic calibration on a rare event beats over-forecasting.
    hedged = brier_score([0.3, 0.3, 0.3, 0.3], [0, 0, 0, 1])     # ~calibrated-ish
    overforecast = brier_score([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 1])
    assert hedged < overforecast


def test_brier_skill_score_sign():
    events = [0, 0, 0, 1, 0, 0, 1, 0]            # 25% base rate
    skilled = [0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1]
    assert brier_skill_score(skilled, events) > 0    # beats base-rate climatology
    anti = [0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]
    assert brier_skill_score(anti, events) < 0       # worse than climatology
