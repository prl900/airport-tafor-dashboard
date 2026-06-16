"""Align METAR observations to the hourly TAF expected-state grid."""

from __future__ import annotations

from datetime import timedelta

from wx.verification.timeline import ExpectedHour

DEFAULT_TOLERANCE = timedelta(minutes=40)


def nearest_obs(hour, obs_sorted: list[dict], tolerance: timedelta = DEFAULT_TOLERANCE):
    """Return the observation closest to ``hour`` within tolerance, else None.

    ``obs_sorted`` is a list of dicts with an ``observed_at`` datetime, sorted
    ascending. METARs are typically issued half-hourly, so a 40-minute window
    reliably picks the matching report for each top-of-hour.
    """
    best = None
    best_gap = tolerance
    for o in obs_sorted:
        gap = abs(o["observed_at"] - hour)
        if gap <= best_gap:
            best, best_gap = o, gap
        elif o["observed_at"] > hour and gap > tolerance:
            break  # sorted: nothing closer ahead
    return best


def align(expected: list[ExpectedHour], obs: list[dict]) -> list[tuple[ExpectedHour, dict | None]]:
    obs_sorted = sorted(obs, key=lambda o: o["observed_at"])
    return [(eh, nearest_obs(eh.valid_hour, obs_sorted)) for eh in expected]
