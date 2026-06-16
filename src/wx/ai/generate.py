"""Candidate-TAF forecasters.

A Forecaster turns (station, issue time, validity window) into an hourly
expected-state timeline — the SAME ``ExpectedHour`` structure produced by
expanding an official TAF. That is the whole point: a candidate forecast is
scored by the identical Phase-2 verifier, so "did the AI beat the official TAF?"
is an apples-to-apples question.

Two baselines ship here:
  * PersistenceForecaster — carry the latest observation forward (a famously
    hard-to-beat short-range baseline).
  * ClimatologyForecaster — the station's hour-of-day median/modal conditions.

A trained ML model implements the same ``generate`` method and slots in unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import duckdb

from wx.ai.features import hourly_climatology, latest_obs_before
from wx.parsing.normalize import flight_category
from wx.verification.timeline import ELEMENTS, ExpectedHour


def _hours(valid_from: datetime, valid_to: datetime):
    h = valid_from.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = valid_to.astimezone(timezone.utc)
    while h < end:
        yield h
        h += timedelta(hours=1)


def _prevailing(state: dict | None) -> dict:
    """Normalise an obs/clim dict into a prevailing-conditions dict."""
    base = {e: (state or {}).get(e) for e in ELEMENTS}
    base["flight_category"] = (state or {}).get("flight_category") or flight_category(
        base.get("ceiling_ft"), base.get("vis_m")
    )
    return base


class Forecaster(ABC):
    name: str

    @abstractmethod
    def generate(
        self, con: duckdb.DuckDBPyConnection, icao: str,
        issued_at: datetime, valid_from: datetime, valid_to: datetime,
    ) -> list[ExpectedHour]:
        raise NotImplementedError


class PersistenceForecaster(Forecaster):
    name = "persistence"

    def generate(self, con, icao, issued_at, valid_from, valid_to) -> list[ExpectedHour]:
        anchor = latest_obs_before(con, icao, issued_at)
        prevailing = _prevailing(anchor)
        return [ExpectedHour(h, dict(prevailing)) for h in _hours(valid_from, valid_to)]


class ClimatologyForecaster(Forecaster):
    name = "climatology"

    def generate(self, con, icao, issued_at, valid_from, valid_to) -> list[ExpectedHour]:
        clim = hourly_climatology(con, icao)
        out = []
        for h in _hours(valid_from, valid_to):
            out.append(ExpectedHour(h, _prevailing(clim.get(h.hour))))
        return out


FORECASTERS: dict[str, Forecaster] = {
    f.name: f for f in (PersistenceForecaster(), ClimatologyForecaster())
}
