"""Parse a raw METAR string into a normalised observation record."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from metar_taf_parser.parser.parser import MetarParser

from wx.parsing.normalize import NormalizedConditions, normalize_conditions
from wx.parsing.timeutil import resolve_time

_parser = MetarParser()


@dataclass
class ParsedMetar:
    icao: str
    observed_at: datetime
    conditions: NormalizedConditions
    temp_c: float | None
    dewpoint_c: float | None
    qnh_hpa: float | None


def parse_metar(raw_text: str, reference: datetime) -> ParsedMetar:
    """Parse ``raw_text``; ``reference`` anchors the day/hour to an absolute time."""
    m = _parser.parse(raw_text.strip())
    cond = normalize_conditions(m)

    if m.day is not None and m.time is not None:
        observed_at = resolve_time(reference, m.day, m.time.hour, m.time.minute)
    else:
        observed_at = reference.astimezone(timezone.utc)

    return ParsedMetar(
        icao=m.station,
        observed_at=observed_at,
        conditions=cond,
        temp_c=float(m.temperature) if m.temperature is not None else None,
        dewpoint_c=float(m.dew_point) if m.dew_point is not None else None,
        qnh_hpa=float(m.altimeter) if m.altimeter is not None else None,
    )
