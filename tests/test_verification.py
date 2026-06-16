from datetime import datetime, timedelta, timezone

from wx.verification.align import nearest_obs
from wx.verification.scores import (
    angular_diff,
    contingency_outcome,
    score_hour,
    skill_scores,
    weighted_score,
)
from wx.verification.timeline import ExpectedHour, expand

UTC = timezone.utc


def _group(gt, vf, vt, vis=None, ceil=None, cat=None, wspd=None, wdir=None, prob=None):
    return {
        "group_type": gt, "probability": prob, "valid_from": vf, "valid_to": vt,
        "wind_dir_deg": wdir, "wind_spd_kt": wspd, "wind_gust_kt": None,
        "vis_m": vis, "ceiling_ft": ceil, "flight_category": cat,
    }


def test_expand_applies_fm_transition():
    vf = datetime(2023, 1, 1, 0, tzinfo=UTC)
    vt = datetime(2023, 1, 1, 6, tzinfo=UTC)
    fm = datetime(2023, 1, 1, 3, tzinfo=UTC)
    groups = [
        _group("BASE", vf, vt, vis=9999, ceil=None, cat="VFR", wspd=10, wdir=220),
        _group("FM", fm, vt, vis=2000, ceil=300, cat="LIFR", wspd=5, wdir=180),
    ]
    hours = expand(groups, vf, vt)
    assert len(hours) == 6
    assert hours[0].prevailing["flight_category"] == "VFR"     # before FM
    assert hours[2].prevailing["flight_category"] == "VFR"     # 02:00, still base
    assert hours[3].prevailing["flight_category"] == "LIFR"    # 03:00, FM in effect
    assert hours[3].prevailing["wind_spd_kt"] == 5


def test_expand_tempo_overlay():
    vf = datetime(2023, 1, 1, 0, tzinfo=UTC)
    vt = datetime(2023, 1, 1, 6, tzinfo=UTC)
    groups = [
        _group("BASE", vf, vt, vis=9999, cat="VFR"),
        _group("TEMPO", datetime(2023, 1, 1, 2, tzinfo=UTC),
               datetime(2023, 1, 1, 4, tzinfo=UTC), vis=1000, ceil=200, cat="LIFR"),
    ]
    hours = expand(groups, vf, vt)
    assert hours[1].tempo is None                  # 01:00 outside tempo
    assert hours[2].tempo["flight_category"] == "LIFR"   # 02:00 inside tempo
    assert hours[2].prevailing["flight_category"] == "VFR"  # prevailing unchanged


def test_contingency_and_skill():
    assert contingency_outcome(True, True) == "hit"
    assert contingency_outcome(False, True) == "miss"
    assert contingency_outcome(True, False) == "false_alarm"
    assert contingency_outcome(False, False) == "correct_neg"
    ss = skill_scores(["hit", "hit", "miss", "false_alarm", "correct_neg"])
    assert ss["POD"] == 2 / 3
    assert ss["FAR"] == 1 / 3
    assert ss["HSS"] is not None


def test_angular_diff():
    assert angular_diff(350, 10) == 20
    assert angular_diff(10, 350) == 20
    assert angular_diff(None, 10) is None


def test_score_hour_tempo_gives_partial_credit():
    eh = ExpectedHour(
        valid_hour=datetime(2023, 1, 1, 2, tzinfo=UTC),
        prevailing={"flight_category": "VFR", "vis_m": 9999, "ceiling_ft": None,
                    "wind_spd_kt": 10, "wind_dir_deg": 200},
        tempo={"flight_category": "LIFR"},
    )
    obs = {"flight_category": "LIFR", "vis_m": 800, "ceiling_ft": 200,
           "wind_spd_kt": 6, "wind_dir_deg": 180}
    row = score_hour(eh, obs)
    assert row["category_outcome"] == "hit"          # tempo warned of the adverse event
    assert row["fcst_category"] == "VFR"             # prevailing was VFR
    assert row["obs_category"] == "LIFR"
    assert row["weighted_score"] == 2.0              # tempo captured obs while prevailing missed
    assert row["vis_err_m"] == 9999 - 800


def test_score_hour_none_obs():
    eh = ExpectedHour(datetime(2023, 1, 1, tzinfo=UTC), {"flight_category": "VFR"})
    assert score_hour(eh, None) is None


def test_nearest_obs_window():
    hour = datetime(2023, 1, 1, 12, tzinfo=UTC)
    obs = [
        {"observed_at": hour - timedelta(minutes=50)},
        {"observed_at": hour + timedelta(minutes=20)},
        {"observed_at": hour + timedelta(minutes=90)},
    ]
    got = nearest_obs(hour, obs)
    assert got["observed_at"] == hour + timedelta(minutes=20)
