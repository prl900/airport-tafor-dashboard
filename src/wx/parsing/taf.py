"""Parse a raw TAF string into a header + decomposed change groups.

The base forecast becomes a 'BASE' group spanning the whole validity; each
trend (FM/BECMG/TEMPO, optionally with a PROB) becomes its own group with an
absolute validity window. FM groups are instantaneous changes — we set their
end to the TAF's end-of-validity here; Phase 2's timeline expansion refines
that to "until the next FM".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from metar_taf_parser.parser.parser import TAFParser

from wx.parsing.normalize import NormalizedConditions, normalize_conditions
from wx.parsing.timeutil import resolve_time

_parser = TAFParser()


@dataclass
class TafGroup:
    group_type: str  # BASE|FM|BECMG|TEMPO|PROB30|PROB40|PROB30_TEMPO|PROB40_TEMPO
    probability: int | None
    valid_from: datetime | None
    valid_to: datetime | None
    conditions: NormalizedConditions


@dataclass
class ParsedTaf:
    icao: str
    issued_at: datetime
    valid_from: datetime | None
    valid_to: datetime | None
    groups: list[TafGroup] = field(default_factory=list)


def _group_type(trend) -> str:
    name = trend.type.name  # 'FM' | 'BECMG' | 'TEMPO' | 'PROB'
    prob = getattr(trend, "probability", None)
    if prob and name == "TEMPO":
        return f"PROB{prob}_TEMPO"
    if prob:
        return f"PROB{prob}"
    return name


def parse_taf(raw_text: str, reference: datetime) -> ParsedTaf:
    """Parse ``raw_text``; ``reference`` anchors day/hour tokens to absolute time."""
    t = _parser.parse(raw_text.strip())

    v = t.validity
    valid_from = resolve_time(reference, v.start_day, v.start_hour)
    valid_to = resolve_time(valid_from, v.end_day, v.end_hour)

    groups: list[TafGroup] = [
        TafGroup("BASE", None, valid_from, valid_to, normalize_conditions(t))
    ]

    for tr in t.trends:
        tv = tr.validity
        g_from = resolve_time(valid_from, tv.start_day, tv.start_hour, getattr(tv, "start_minutes", 0) or 0)
        if getattr(tv, "end_day", None) is not None:
            g_to = resolve_time(g_from, tv.end_day, tv.end_hour)
        else:
            g_to = valid_to  # FM: persists to end-of-TAF (refined in Phase 2)
        groups.append(
            TafGroup(
                group_type=_group_type(tr),
                probability=getattr(tr, "probability", None),
                valid_from=g_from,
                valid_to=g_to,
                conditions=normalize_conditions(tr),
            )
        )

    return ParsedTaf(
        icao=t.station,
        issued_at=reference,
        valid_from=valid_from,
        valid_to=valid_to,
        groups=groups,
    )
