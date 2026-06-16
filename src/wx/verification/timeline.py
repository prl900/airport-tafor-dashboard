"""Expand a parsed TAF's change groups into an hourly expected-state timeline.

For each valid hour we resolve:
  * ``prevailing`` — the single most likely state, built from BASE then applying
    FM (instantaneous, full replace) and BECMG (gradual, complete by its end,
    partial merge) transitions in chronological order.
  * ``tempo`` / ``prob`` — temporary or probabilistic alternatives valid that hour
    (overlaid, not part of the prevailing state).

The output is the apples-to-apples reference that both the official TAF and a
future AI-generated TAF are scored against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from wx.parsing.normalize import flight_category

ELEMENTS = ("wind_dir_deg", "wind_spd_kt", "wind_gust_kt", "vis_m", "ceiling_ft")


@dataclass
class ExpectedHour:
    valid_hour: datetime
    prevailing: dict
    tempo: dict | None = None
    prob: dict | None = None  # includes a 'probability' key when present


def _conditions(group: dict) -> dict:
    c = {k: group.get(k) for k in ELEMENTS}
    c["flight_category"] = group.get("flight_category")
    return c


def _merge(base: dict, new: dict, replace: bool) -> dict:
    """FM = full replace; BECMG = override only the elements the group specifies."""
    out = dict(new) if replace else {**base, **{k: v for k, v in new.items() if v is not None}}
    out["flight_category"] = flight_category(out.get("ceiling_ft"), out.get("vis_m"))
    return out


def expand(groups: list[dict], valid_from: datetime, valid_to: datetime) -> list[ExpectedHour]:
    base = next((g for g in groups if g["group_type"] == "BASE"), None)
    prevailing0 = _conditions(base) if base else {e: None for e in ELEMENTS}

    # Chronological prevailing transitions. FM effective at its start; BECMG by end.
    transitions = []
    for g in groups:
        gt = g["group_type"]
        if gt == "FM":
            transitions.append((g["valid_from"], _conditions(g), True))
        elif gt == "BECMG":
            transitions.append((g["valid_to"], _conditions(g), False))
    transitions.sort(key=lambda t: t[0])

    overlays = [g for g in groups if g["group_type"].startswith(("TEMPO", "PROB"))]

    hours: list[ExpectedHour] = []
    h = valid_from.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = valid_to.astimezone(timezone.utc)
    while h < end:
        prevailing = dict(prevailing0)
        for eff_time, cond, replace in transitions:
            if eff_time <= h:
                prevailing = _merge(prevailing, cond, replace)
        prevailing["flight_category"] = flight_category(
            prevailing.get("ceiling_ft"), prevailing.get("vis_m")
        )

        tempo = prob = None
        for g in overlays:
            if g["valid_from"] <= h < g["valid_to"]:
                cond = _conditions(g)
                if g["group_type"].startswith("PROB"):
                    prob = {**cond, "probability": g.get("probability")}
                else:
                    tempo = cond
        hours.append(ExpectedHour(h, prevailing, tempo, prob))
        h += timedelta(hours=1)
    return hours
