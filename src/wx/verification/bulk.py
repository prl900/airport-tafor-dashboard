"""Scale-out verification: load each station's data ONCE and align in memory.

The per-TAF query pattern (a METAR range-load + group-load per TAF) does not scale
to ~240k TAFs — DuckDB point/range lookups are ~5-24 ms each and indexes barely
help, so it would run for hours. Here we bulk-load a station's observations, TAF
groups and TAFs in a few queries, then expand/align/score entirely in Python
(bisect for nearest-obs), and bulk-insert the results. Official TAF, persistence
and climatology all share this path.
"""

from __future__ import annotations

import bisect
from datetime import timedelta

from wx.db import repositories as repo
from wx.parsing.normalize import flight_category
from wx.verification.scores import score_hour
from wx.verification.timeline import ELEMENTS, ExpectedHour, expand

_OBS_COLS = ("observed_at", "wind_dir_deg", "wind_spd_kt", "vis_m", "ceiling_ft", "flight_category")
_GRP_COLS = ("taf_forecast_id", "group_type", "probability", "valid_from", "valid_to",
             "wind_dir_deg", "wind_spd_kt", "wind_gust_kt", "vis_m", "ceiling_ft", "flight_category")
_TOL_S = 40 * 60  # nearest-obs tolerance (seconds)


def _station_obs(con, icao):
    rows = con.execute(
        f"SELECT {', '.join(_OBS_COLS)} FROM metar_obs WHERE icao = ? ORDER BY observed_at",
        [icao],
    ).fetchall()
    obs = [dict(zip(_OBS_COLS, r)) for r in rows]
    times = [o["observed_at"] for o in obs]
    return obs, times


def _groups_by_taf(con, icao):
    rows = con.execute(
        f"""SELECT {', '.join(_GRP_COLS)} FROM taf_group g
            WHERE g.taf_forecast_id IN (SELECT id FROM taf_forecast WHERE icao = ?)""",
        [icao],
    ).fetchall()
    out: dict[int, list[dict]] = {}
    for r in rows:
        d = dict(zip(_GRP_COLS, r))
        out.setdefault(d["taf_forecast_id"], []).append(d)
    return out


def _pending_tafs(con, icao, profile):
    """TAFs for a station not yet scored under `profile`."""
    return con.execute(
        """SELECT id, issued_at, valid_from, valid_to FROM taf_forecast f
           WHERE f.icao = ? AND f.valid_from IS NOT NULL AND f.valid_to IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM verification_hourly v
                             WHERE v.taf_forecast_id = f.id AND v.scoring_profile = ?)
           ORDER BY f.issued_at""",
        [icao, profile],
    ).fetchall()


def _nearest(times, obs, hour):
    i = bisect.bisect_left(times, hour)
    best, best_gap = None, _TOL_S
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(obs):
            gap = abs((obs[j]["observed_at"] - hour).total_seconds())
            if gap <= best_gap:
                best, best_gap = obs[j], gap
    return best


def _hourly(valid_from, valid_to):
    h = valid_from.replace(minute=0, second=0, microsecond=0)
    out = []
    while h < valid_to:
        out.append(h)
        h += timedelta(hours=1)
    return out


def _hour_climatology(obs):
    """Per hour-of-day prevailing dict (median vis/ceiling, modal category)."""
    from statistics import median, mode

    buckets: dict[int, dict[str, list]] = {}
    for o in obs:
        b = buckets.setdefault(o["observed_at"].hour, {"vis": [], "ceil": [], "cat": []})
        if o["vis_m"] is not None:
            b["vis"].append(o["vis_m"])
        if o["ceiling_ft"] is not None:
            b["ceil"].append(o["ceiling_ft"])
        if o["flight_category"]:
            b["cat"].append(o["flight_category"])
    clim = {}
    for hod, b in buckets.items():
        vis = median(b["vis"]) if b["vis"] else None
        ceil = median(b["ceil"]) if b["ceil"] else None
        cat = mode(b["cat"]) if b["cat"] else flight_category(ceil, vis)
        clim[hod] = {"vis_m": vis, "ceiling_ft": ceil, "wind_spd_kt": None,
                     "wind_dir_deg": None, "flight_category": cat}
    return clim


def _prevailing_from_obs(o: dict) -> dict:
    return {"vis_m": o["vis_m"], "ceiling_ft": o["ceiling_ft"], "wind_spd_kt": o["wind_spd_kt"],
            "wind_dir_deg": o["wind_dir_deg"], "flight_category": o["flight_category"]}


def _expected_for(profile, taf, groups_by_taf, obs, times, clim) -> list[ExpectedHour]:
    fid, issued_at, vf, vt = taf
    if profile == "categorical":           # the official TAF
        return expand(groups_by_taf.get(fid, []), vf, vt)
    if profile == "persistence":
        anchor = _nearest(times, obs, issued_at) or (
            obs[bisect.bisect_left(times, issued_at) - 1] if obs else None)
        if anchor is None:
            return []
        prevailing = _prevailing_from_obs(anchor)
        return [ExpectedHour(h, dict(prevailing)) for h in _hourly(vf, vt)]
    if profile == "climatology":
        return [ExpectedHour(h, dict(clim.get(h.hour, {e: None for e in ELEMENTS})))
                for h in _hourly(vf, vt)]
    raise ValueError(profile)


def score_profile(con, icao: str, profile: str) -> int:
    """Score every (pending) TAF for one station under `profile`, in memory."""
    tafs = _pending_tafs(con, icao, profile)
    if not tafs:
        return 0
    obs, times = _station_obs(con, icao)
    groups = _groups_by_taf(con, icao) if profile == "categorical" else {}
    clim = _hour_climatology(obs) if profile == "climatology" else {}

    rows = []
    for taf in tafs:
        fid, issued_at, vf, vt = taf
        for eh in _expected_for(profile, taf, groups, obs, times, clim):
            o = _nearest(times, obs, eh.valid_hour)
            s = score_hour(eh, o)
            if s is None:
                continue
            lead_h = int((eh.valid_hour - issued_at).total_seconds() // 3600)
            rows.append((fid, icao, eh.valid_hour, lead_h, profile, s["fcst_category"],
                         s["obs_category"], s["category_outcome"], s["wind_err_kt"],
                         s["dir_err_deg"], s["temp_err_c"], s["vis_err_m"],
                         s["ceiling_err_ft"], s["weighted_score"]))
    return repo.store_verification(con, rows)


def run_profiles(con, profiles, icaos=None) -> dict:
    """Score the given profiles across stations. Returns {profile: rows_written}."""
    stations = icaos or [r[0] for r in con.execute(
        "SELECT DISTINCT icao FROM taf_forecast ORDER BY icao").fetchall()]
    totals = {p: 0 for p in profiles}
    for icao in stations:
        for p in profiles:
            totals[p] += score_profile(con, icao, p)
    return totals
