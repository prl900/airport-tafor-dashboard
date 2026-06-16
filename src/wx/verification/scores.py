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
TEMPO_PROB = 0.4           # implied probability of a bare TEMPO group (no explicit PROB)


def is_adverse(cat: str | None) -> bool:
    return cat in ADVERSE


def adverse_probability(eh) -> float:
    """Forecast probability of the IFR-or-worse event for one hour.

    The prevailing forecast is a firm commitment (p=1 if adverse, else 0); a
    TEMPO/PROB group that mentions an adverse category adds probability mass
    (PROB30->0.3, PROB40->0.4, bare TEMPO->0.4). This is what lets the Brier score
    credit a TAF's hedging instead of treating every PROB30 FG as a false alarm.
    Baselines with only a prevailing state collapse to a deterministic 0/1.
    """
    if is_adverse(eh.prevailing.get("flight_category")):
        return 1.0
    p = 0.0
    if getattr(eh, "tempo", None) and is_adverse(eh.tempo.get("flight_category")):
        p = max(p, TEMPO_PROB)
    if getattr(eh, "prob", None) and is_adverse(eh.prob.get("flight_category")):
        pr = eh.prob.get("probability")
        p = max(p, (pr / 100.0) if pr else TEMPO_PROB)
    return p


def brier_score(probs: list[float], events: list[int]) -> float | None:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    pairs = [(p, e) for p, e in zip(probs, events) if p is not None and e is not None]
    if not pairs:
        return None
    return sum((p - e) ** 2 for p, e in pairs) / len(pairs)


def brier_skill_score(probs: list[float], events: list[int],
                      reference_probs: list[float] | None = None) -> float | None:
    """BSS vs a reference forecast (default: the event's own base rate).

    BSS = 1 - BS / BS_ref. Positive => more skilful than the reference; 0 => equal.
    """
    bs = brier_score(probs, events)
    valid_events = [e for e in events if e is not None]
    if bs is None or not valid_events:
        return None
    if reference_probs is None:
        base = sum(valid_events) / len(valid_events)        # climatological base rate
        bs_ref = sum((base - e) ** 2 for e in valid_events) / len(valid_events)
    else:
        bs_ref = brier_score(reference_probs, events)
    if not bs_ref:
        return None
    return 1.0 - bs / bs_ref


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
        "fcst_prob": adverse_probability(eh),   # P(IFR-or-worse) for the Brier score
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
