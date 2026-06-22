"""Phase B/C — the champion/challenger promotion gate.

A challenger replaces the champion only if it beats it on the **frozen test set**
(2025) with a paired bootstrap CI on the HSS difference that excludes zero — the
contract from docs/ML_PLAN.md. The incumbent champion starts as the official TAF
(the skyline); once an ML model wins, it is recorded in ``champion.json`` and
becomes the bar the next challenger must clear.

Evaluation is paired: both forecasters are scored over the *same* TAF-hours, and
only hours where both produced a verifiable score are kept, so the bootstrap
controls for which hours are intrinsically hard.
"""

from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from wx.ai.dataset import build_inference_features_batch
from wx.ai.evaluate import bootstrap_hss_diff, log_experiment
from wx.ai.generate import Forecaster, OfficialForecaster
from wx.ai.models import ModelForecaster, MultiTaskModel, _hourly
from wx.config import DATA_DIR
from wx.verification.align import align
from wx.verification.scores import brier_skill_score, score_hour, skill_scores

MODELS_DIR = DATA_DIR / "models"
CHAMPION_PATH = MODELS_DIR / "champion.json"
TEST_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _batch_load_obs(con, tafs, start, end) -> dict:
    """Load every test-window METAR once and group by station (sorted), so each TAF's
    obs window is a bisect slice instead of one SQL query per TAF (~37k queries)."""
    icaos = sorted({t[0] for t in tafs})
    if not icaos:
        return {}
    cols = ("observed_at", "wind_dir_deg", "wind_spd_kt", "vis_m", "ceiling_ft", "flight_category")
    rows = con.execute(
        f"SELECT icao, {', '.join(cols)} FROM metar_obs "
        f"WHERE icao IN ({','.join(['?'] * len(icaos))}) AND observed_at >= ? AND observed_at < ? "
        "ORDER BY icao, observed_at",
        [*icaos, start, end],
    ).fetchall()
    by_icao: dict = {}
    for r in rows:
        by_icao.setdefault(r[0], ([], []))  # (times, obs dicts)
        by_icao[r[0]][0].append(r[1])
        by_icao[r[0]][1].append(dict(zip(cols, r[1:])))
    return by_icao


def _obs_window(obs_index, icao, vf, vt) -> list:
    """Slice the pre-loaded obs for one TAF window [vf, vt) via bisect."""
    pack = obs_index.get(icao)
    if not pack:
        return []
    times, obs = pack
    return obs[bisect.bisect_left(times, vf):bisect.bisect_left(times, vt)]


def _model_hour_lookups(con, forecasters, tafs) -> dict:
    """Per ModelForecaster, a {(icao, t0_ns, valid_ns): ExpectedHour} map built from a
    SINGLE batched feature frame over every TAF-hour. This replaces ~1.5 s of per-TAF
    ASOF SQL per forecaster with one batched build the models share — the gate's
    bottleneck. Non-model forecasters (official/persistence) fall back to generate()."""
    model_fcs = {id(f): f for f in forecasters if isinstance(f, ModelForecaster)}
    if not model_fcs:
        return {}
    triples = [(icao, issued_at, h)
               for icao, issued_at, vf, vt in tafs for h in _hourly(vf, vt)]
    feats = build_inference_features_batch(con, triples)
    lookups = {}
    if feats.empty:
        return {fid: {} for fid in model_fcs}
    icaos = feats["icao"].tolist()
    t0_ns = [int(t.value) for t in feats["t0"]]
    vt_ns = [int(t.value) for t in feats["valid_time"]]
    for fid, fc in model_fcs.items():
        hours = fc.hours_from_feats(feats)
        lookups[fid] = {(icaos[i], t0_ns[i], vt_ns[i]): eh for i, eh in enumerate(hours)}
    return lookups


def _scored_by_hour(con, forecaster, icao, issued_at, vf, vt, obs, lookups) -> dict:
    """{valid_hour: score_hour dict} for one forecaster over one TAF window."""
    fid = id(forecaster)
    if fid in lookups:                       # batched model path: assemble from the lookup
        lk, t0n = lookups[fid], int(pd.Timestamp(issued_at).value)
        expected = [lk[(icao, t0n, int(pd.Timestamp(h).value))]
                    for h in _hourly(vf, vt) if (icao, t0n, int(pd.Timestamp(h).value)) in lk]
    else:
        expected = forecaster.generate(con, icao, issued_at, vf, vt)
    if not expected:
        return {}
    scored = {}
    for eh, o in align(expected, obs):
        s = score_hour(eh, o)
        if s is not None:
            scored[eh.valid_hour] = s
    return scored


def evaluate_paired(con: duckdb.DuckDBPyConnection, champion: Forecaster,
                    challenger: Forecaster, start=TEST_START, end=TEST_END,
                    icaos=None) -> dict:
    """Score both forecasters over the same TAFs; keep hours both could verify.

    Returns paired outcome lists (for HSS) and prob/event lists (for BSS)."""
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

    # Build the shared causal features for every TAF-hour once (batched), so the gate
    # is no longer ~1.5 s of ASOF SQL per TAF per forecaster.
    lookups = _model_hour_lookups(con, [champion, challenger], tafs)
    # ...and load all obs once (one query) instead of ~37k per-TAF window queries.
    obs_index = _batch_load_obs(con, tafs, start, end)

    champ_out, chal_out = [], []
    champ_probs, chal_probs, events = [], [], []
    for icao, issued_at, vf, vt in tafs:
        obs = _obs_window(obs_index, icao, vf, vt)
        if not obs:
            continue
        a = _scored_by_hour(con, champion, icao, issued_at, vf, vt, obs, lookups)
        b = _scored_by_hour(con, challenger, icao, issued_at, vf, vt, obs, lookups)
        for h in sorted(a.keys() & b.keys()):     # only hours both verified
            champ_out.append(a[h]["category_outcome"])
            chal_out.append(b[h]["category_outcome"])
            ev = 1 if a[h]["obs_category"] in ("IFR", "LIFR") else 0
            events.append(ev)
            champ_probs.append(a[h]["fcst_prob"])
            chal_probs.append(b[h]["fcst_prob"])

    return {
        "n": len(events),
        "champ_outcomes": champ_out, "chal_outcomes": chal_out,
        "champ_probs": champ_probs, "chal_probs": chal_probs, "events": events,
    }


def gate(con: duckdb.DuckDBPyConnection, challenger: Forecaster, icaos=None,
         n_boot: int = 1000) -> dict:
    """Run the promotion gate for `challenger` vs the current champion on frozen test.

    Decision = the paired HSS-difference CI excludes zero on the positive side
    (the plan's primary metric). BSS is reported alongside as the probabilistic view.
    """
    champion = load_champion_forecaster(con)
    paired = evaluate_paired(con, champion, challenger, icaos=icaos)
    boot = bootstrap_hss_diff(paired["champ_outcomes"], paired["chal_outcomes"], n_boot=n_boot)
    champ_skill = skill_scores(paired["champ_outcomes"])
    chal_skill = skill_scores(paired["chal_outcomes"])
    return {
        "kind": "promotion_gate",
        "challenger": challenger.name,
        "champion": getattr(champion, "name", "official"),
        "n_paired": paired["n"],
        "champion_hss": champ_skill["HSS"],
        "challenger_hss": chal_skill["HSS"],
        "champion_bss": brier_skill_score(paired["champ_probs"], paired["events"]),
        "challenger_bss": brier_skill_score(paired["chal_probs"], paired["events"]),
        "hss_diff": boot,                 # diff / ci_low / ci_high / wins
        "promote": bool(boot["wins"]),
    }


# --- champion registry (local JSON) ----------------------------------------

def load_champion() -> dict | None:
    """The recorded champion, or None when the official TAF is still the incumbent."""
    if CHAMPION_PATH.exists():
        return json.loads(CHAMPION_PATH.read_text())
    return None


def load_champion_forecaster(con) -> Forecaster:
    """Forecaster for the current champion; the official TAF until a model wins."""
    champ = load_champion()
    if champ and champ.get("model_path") and Path(champ["model_path"]).exists():
        model = MultiTaskModel.load(champ["model_path"])
        return ModelForecaster(model, name=f"champion:{champ.get('rung', model.rung)}")
    return OfficialForecaster()


def save_champion(rung: str, model_path: str | Path, decision: dict) -> dict:
    """Record a new champion in champion.json (after it has won the gate)."""
    record = {
        "rung": rung,
        "model_path": str(model_path),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "beat": decision.get("champion"),
        "test_hss": decision.get("challenger_hss"),
        "test_bss": decision.get("challenger_bss"),
        "n_paired": decision.get("n_paired"),
        "hss_diff": decision.get("hss_diff"),
    }
    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAMPION_PATH.write_text(json.dumps(record, indent=2, default=str))
    return record


def promote_if_better(con, rung: str, model_path: str | Path | None = None,
                      icaos=None, register: bool = True) -> dict:
    """Gate a saved model rung against the champion; register it if it wins.

    Logs the decision to the research log either way (the auto-research trail)."""
    model_path = Path(model_path) if model_path else MODELS_DIR / f"{rung}.joblib"
    model = MultiTaskModel.load(model_path)
    challenger = ModelForecaster(model, name=f"model:{rung}")
    decision = gate(con, challenger, icaos=icaos)
    log_experiment(decision)
    if decision["promote"] and register:
        decision["registered"] = save_champion(rung, model_path, decision)
    return decision
