"""Normalisation helpers: turn metar-taf-parser objects into canonical SI
components plus a derived ceiling and flight category.

Spain uses ICAO/metric conventions (visibility in metres). The flight-category
band table is kept here as a module constant so it can be swapped for an
FAA/statute-mile profile without touching the parsers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Flight-category thresholds. Ceiling in feet AGL, visibility in metres.
# Defaults approximate the FAA VFR/MVFR/IFR/LIFR bands (1/3/5 SM -> ~1600/4800/8000 m).
CEILING_BANDS_FT = {"LIFR": 500, "IFR": 1000, "MVFR": 3000}  # below value => that category
VIS_BANDS_M = {"LIFR": 1600, "IFR": 4800, "MVFR": 8000}
_CEILING_COVERS = {"BKN", "OVC", "OVX"}  # layers that constitute a ceiling


def vis_to_meters(distance: str | None) -> float | None:
    """Parse the parser's visibility string (e.g. '6000', '>10000', '9999')."""
    if not distance:
        return None
    s = str(distance).strip().lstrip(">").lstrip("<").strip()
    if s.upper().endswith("SM"):  # statute miles (rare for Spain)
        try:
            return float(s[:-2].strip()) * 1609.34
        except ValueError:
            return None
    try:
        v = float(s)
    except ValueError:
        return None
    # 9999 is the ICAO code for "10 km or more"; the lib renders it as '>10000'.
    return 10000.0 if v >= 9999 else v


def wind_components(wind) -> tuple[int | None, float | None, float | None]:
    """Return (direction_deg, speed_kt, gust_kt). Handles VRB and m/s units."""
    if wind is None:
        return None, None, None
    deg = wind.degrees
    if isinstance(deg, str):  # 'VRB'
        deg = None
    spd = _to_knots(wind.speed, wind.unit)
    gust = _to_knots(wind.gust, wind.unit)
    return deg, spd, gust


def _to_knots(value, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit and unit.upper() in {"MPS", "M/S"}:
        return float(value) * 1.94384
    if unit and unit.upper() in {"KMH", "KPH"}:
        return float(value) / 1.852
    return float(value)


def ceiling_ft(clouds, vertical_visibility=None) -> int | None:
    """Lowest BKN/OVC layer (feet), or vertical visibility when sky-obscured."""
    heights = [
        c.height
        for c in (clouds or [])
        if c.height is not None and _cover_name(c) in _CEILING_COVERS
    ]
    candidates = list(heights)
    if vertical_visibility is not None:
        candidates.append(int(vertical_visibility))
    return min(candidates) if candidates else None


def _cover_name(cloud) -> str | None:
    q = getattr(cloud, "quantity", None)
    return q.name if q is not None else None


def flight_category(ceil_ft: int | None, vis_m: float | None) -> str | None:
    """Worst (most restrictive) of the ceiling and visibility categories.

    With neither ceiling nor visibility known we cannot classify -> None.
    A missing ceiling is treated as unlimited (does not restrict)."""
    if ceil_ft is None and vis_m is None:
        return None
    order = ["LIFR", "IFR", "MVFR", "VFR"]
    by_ceiling = _band(ceil_ft, CEILING_BANDS_FT) if ceil_ft is not None else "VFR"
    by_vis = _band(vis_m, VIS_BANDS_M) if vis_m is not None else "VFR"
    return min((by_ceiling, by_vis), key=order.index)


def _band(value: float, bands: dict[str, float]) -> str:
    for cat in ("LIFR", "IFR", "MVFR"):
        if value < bands[cat]:
            return cat
    return "VFR"


def clouds_to_json(clouds) -> list[dict]:
    out = []
    for c in clouds or []:
        out.append(
            {
                "cover": _cover_name(c),
                "base_ft": c.height,
                "cb": getattr(getattr(c, "type", None), "name", None) in {"CB", "CUMULONIMBUS"},
            }
        )
    return out


def weather_to_strings(weather_conditions) -> list[str]:
    """Render parsed weather groups back to tokens like '+RA', 'BR'."""
    out = []
    for w in weather_conditions or []:
        intensity = getattr(w, "intensity", None)
        prefix = {"LIGHT": "-", "HEAVY": "+", "IN_VICINITY": "VC"}.get(
            getattr(intensity, "name", ""), ""
        )
        desc = getattr(getattr(w, "descriptive", None), "value", "") or ""
        phen = "".join(getattr(p, "value", str(p)) for p in (w.phenomenons or []))
        out.append(f"{prefix}{desc}{phen}")
    return out


@dataclass
class NormalizedConditions:
    """Canonical weather state shared by METAR observations and TAF groups."""

    wind_dir_deg: int | None = None
    wind_spd_kt: float | None = None
    wind_gust_kt: float | None = None
    vis_m: float | None = None
    ceiling_ft: int | None = None
    flight_category: str | None = None
    clouds: list[dict] = field(default_factory=list)
    weather: list[str] = field(default_factory=list)


def normalize_conditions(obj) -> NormalizedConditions:
    """Build NormalizedConditions from any parser object exposing wind/visibility/
    clouds/weather_conditions (works for both Metar and TAF trend objects)."""
    wind_dir, wind_spd, wind_gust = wind_components(getattr(obj, "wind", None))
    vis = vis_to_meters(getattr(getattr(obj, "visibility", None), "distance", None))
    if getattr(obj, "cavok", None):
        vis = max(vis or 0.0, 10000.0)
    ceil = ceiling_ft(getattr(obj, "clouds", None), getattr(obj, "vertical_visibility", None))
    return NormalizedConditions(
        wind_dir_deg=wind_dir,
        wind_spd_kt=wind_spd,
        wind_gust_kt=wind_gust,
        vis_m=vis,
        ceiling_ft=ceil,
        flight_category=flight_category(ceil, vis),
        clouds=clouds_to_json(getattr(obj, "clouds", None)),
        weather=weather_to_strings(getattr(obj, "weather_conditions", None)),
    )
