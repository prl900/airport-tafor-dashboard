"""Phase B — benchmark harness.

Evaluate any Forecaster over a time split by running it through the SAME verifier
(scores.py) used for the official TAF, then compare challengers to the champion
with a paired bootstrap on HSS. A challenger is promoted only if it beats the
champion on the frozen test set with the HSS-difference CI excluding zero.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import duckdb

from wx.ai.generate import Forecaster
from wx.config import DATA_DIR
from wx.verification.align import align
from wx.verification.runner import _load_obs
from wx.verification.scores import score_hour, skill_scores

RESEARCH_LOG = DATA_DIR / "research_log.jsonl"


@dataclass
class EvalResult:
    forecaster: str
    n_tafs: int
    skill: dict                  # POD/FAR/CSI/HSS/... over the IFR event
    mean_weighted: float | None
    mae: dict                    # vis/ceiling/wind element MAE
    lead_hss: dict               # HSS by 6h lead bucket
    outcomes: list = field(default_factory=list, repr=False)  # for significance tests


def evaluate(con: duckdb.DuckDBPyConnection, forecaster: Forecaster,
             start: datetime, end: datetime, icaos=None) -> EvalResult:
    """Score a forecaster over every official TAF issued in [start, end).

    Pure: computes metrics without writing to verification_hourly, so the frozen
    test set can be scored repeatedly without side effects.
    """
    where, params = "issued_at >= ? AND issued_at < ?", [start, end]
    if icaos:
        where += f" AND icao IN ({','.join(['?'] * len(icaos))})"
        params += list(icaos)
    tafs = con.execute(
        f"""SELECT icao, issued_at, valid_from, valid_to FROM taf_forecast
            WHERE valid_from IS NOT NULL AND valid_to IS NOT NULL AND {where}
            ORDER BY issued_at""",
        params,
    ).fetchall()

    outcomes, weighted, leads = [], [], {}
    abs_err = {"vis": [], "ceiling": [], "wind": []}
    n = 0
    for icao, issued_at, vf, vt in tafs:
        expected = forecaster.generate(con, icao, issued_at, vf, vt)
        if not expected:
            continue
        n += 1
        obs = _load_obs(con, icao, expected[0].valid_hour, vt)
        for eh, o in align(expected, obs):
            s = score_hour(eh, o)
            if s is None:
                continue
            outcomes.append(s["category_outcome"])
            if s["weighted_score"] is not None:
                weighted.append(s["weighted_score"])
            lead = int((eh.valid_hour - issued_at).total_seconds() // 3600)
            leads.setdefault((lead // 6) * 6, []).append(s["category_outcome"])
            for k, col in (("vis", "vis_err_m"), ("ceiling", "ceiling_err_ft"), ("wind", "wind_err_kt")):
                if s[col] is not None:
                    abs_err[k].append(abs(s[col]))

    return EvalResult(
        forecaster=forecaster.name,
        n_tafs=n,
        skill=skill_scores(outcomes),
        mean_weighted=(sum(weighted) / len(weighted)) if weighted else None,
        mae={k: (sum(v) / len(v) if v else None) for k, v in abs_err.items()},
        lead_hss={b: skill_scores(o)["HSS"] for b, o in sorted(leads.items())},
        outcomes=outcomes,
    )


def bootstrap_hss_diff(champion: list[str], challenger: list[str],
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap CI on HSS(challenger) - HSS(champion).

    Pairs are resampled together (same TAF-hours), so the comparison controls for
    which hours are hard. Returns the point diff and 95% CI; 'wins' is True when the
    CI excludes zero on the positive side.
    """
    m = min(len(champion), len(challenger))
    champ_hss = skill_scores(champion)["HSS"]
    chal_hss = skill_scores(challenger)["HSS"]
    if m == 0 or champ_hss is None or chal_hss is None:
        return {"diff": None, "ci_low": None, "ci_high": None, "wins": False}
    rng = random.Random(seed)
    base = chal_hss - champ_hss
    diffs = []
    idx = range(m)
    for _ in range(n_boot):
        sample = [rng.randrange(m) for _ in idx]
        a = skill_scores([champion[i] for i in sample])["HSS"]
        b = skill_scores([challenger[i] for i in sample])["HSS"]
        if a is not None and b is not None:
            diffs.append(b - a)
    if not diffs:
        return {"diff": base, "ci_low": None, "ci_high": None, "wins": False}
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    return {"diff": base, "ci_low": lo, "ci_high": hi, "wins": lo > 0}


def log_experiment(record: dict) -> None:
    """Append an experiment record to the research log (Karpathy auto-research trail)."""
    RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with RESEARCH_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def result_summary(r: EvalResult) -> dict:
    """Compact, log-friendly view of an EvalResult (drops the big outcomes list)."""
    d = asdict(r)
    d.pop("outcomes", None)
    return d
