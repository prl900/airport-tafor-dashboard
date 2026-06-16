"""TAF-vs-METAR scoring — pure functions over (expected hour, observation).

Headline metric: a contingency table for the operationally significant
"IFR-or-worse" event (ceiling < 1000 ft and/or visibility < ~3 SM), giving
POD / FAR / CSI / Heidke. We also record per-element errors and a pragmatic
weighted skill score per hour. Keeping these pure means an AI-generated TAF is
scored by exactly the same judge.
"""

from __future__ import annotations

from wx.verification.timeline import ExpectedHour

CATEGORY_RANK = {"LIFR": 0, "IFR": 1, "MVFR": 2, "VFR": 3}
ADVERSE = {"IFR", "LIFR"}  # the verified adverse event


def is_adverse(cat: str | None) -> bool:
    return cat in ADVERSE


def contingency_outcome(fcst_event: bool, obs_event: bool) -> str:
    if fcst_event and obs_event:
        return "hit"
    if obs_event and not fcst_event:
        return "miss"
    if fcst_event and not obs_event:
        return "false_alarm"
    return "correct_neg"


def angular_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _forecast_categories(eh: ExpectedHour) -> list[str]:
    cats = [eh.prevailing.get("flight_category")]
    if eh.tempo:
        cats.append(eh.tempo.get("flight_category"))
    if eh.prob:
        cats.append(eh.prob.get("flight_category"))
    return [c for c in cats if c]


def weighted_score(eh: ExpectedHour, obs_cat: str | None) -> float:
    """Pragmatic per-hour skill: prevailing exact match scores highest, an
    off-by-one-category prevailing forecast scores partial, and a TEMPO/PROB that
    captures the observed category earns partial credit when prevailing missed."""
    if obs_cat is None or obs_cat not in CATEGORY_RANK:
        return 0.0
    prev = eh.prevailing.get("flight_category")
    if prev == obs_cat:
        return 3.0
    score = 0.0
    if prev in CATEGORY_RANK and abs(CATEGORY_RANK[prev] - CATEGORY_RANK[obs_cat]) == 1:
        score = 1.0
    if eh.tempo and eh.tempo.get("flight_category") == obs_cat:
        score = max(score, 2.0)
    if eh.prob and eh.prob.get("flight_category") == obs_cat:
        score = max(score, 1.0 if (eh.prob.get("probability") or 0) < 40 else 1.5)
    return score


def score_hour(eh: ExpectedHour, obs: dict | None) -> dict | None:
    """Build a verification_hourly row for one hour, or None if unverifiable."""
    if obs is None:
        return None
    obs_cat = obs.get("flight_category")
    prev = eh.prevailing

    fcst_event = any(is_adverse(c) for c in _forecast_categories(eh))
    obs_event = is_adverse(obs_cat)

    def err(a, b):
        return (a - b) if (a is not None and b is not None) else None

    return {
        "scoring_profile": "categorical",
        "fcst_category": prev.get("flight_category"),
        "obs_category": obs_cat,
        "category_outcome": contingency_outcome(fcst_event, obs_event),
        "wind_err_kt": err(prev.get("wind_spd_kt"), obs.get("wind_spd_kt")),
        "dir_err_deg": angular_diff(prev.get("wind_dir_deg"), obs.get("wind_dir_deg")),
        "temp_err_c": None,  # base TAF carries no spot temperature
        "vis_err_m": err(prev.get("vis_m"), obs.get("vis_m")),
        "ceiling_err_ft": err(prev.get("ceiling_ft"), obs.get("ceiling_ft")),
        "weighted_score": weighted_score(eh, obs_cat),
    }


# --- aggregate skill scores over many contingency outcomes -------------------


def skill_scores(outcomes: list[str]) -> dict:
    """POD, FAR, CSI, bias and Heidke Skill Score from a list of outcomes."""
    h = outcomes.count("hit")
    m = outcomes.count("miss")
    f = outcomes.count("false_alarm")
    c = outcomes.count("correct_neg")
    n = h + m + f + c
    pod = h / (h + m) if (h + m) else None
    far = f / (h + f) if (h + f) else None
    csi = h / (h + m + f) if (h + m + f) else None
    bias = (h + f) / (h + m) if (h + m) else None
    # Heidke skill score
    hss = None
    if n:
        exp = ((h + m) * (h + f) + (c + m) * (c + f)) / n
        if (n - exp):
            hss = (h + c - exp) / (n - exp)
    return {"n": n, "hits": h, "misses": m, "false_alarms": f, "correct_neg": c,
            "POD": pod, "FAR": far, "CSI": csi, "bias": bias, "HSS": hss}
