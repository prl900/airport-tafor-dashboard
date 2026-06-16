from datetime import datetime, timezone

import pytest

from wx.parsing.metar import parse_metar
from wx.parsing.normalize import flight_category, vis_to_meters, wind_components
from wx.parsing.taf import parse_taf
from wx.parsing.timeutil import resolve_time


def test_vis_to_meters():
    assert vis_to_meters("6000") == 6000.0
    assert vis_to_meters("9999") == 10000.0
    assert vis_to_meters(">10000") == 10000.0
    assert vis_to_meters(None) is None
    assert vis_to_meters("2SM") == pytest.approx(3218.68, rel=1e-3)


def test_flight_category_worst_of_two():
    assert flight_category(5000, 9999) == "VFR"
    assert flight_category(2000, 9999) == "MVFR"   # ceiling restricts
    assert flight_category(5000, 3000) == "IFR"     # visibility restricts
    assert flight_category(300, 9999) == "LIFR"
    assert flight_category(None, None) is None
    assert flight_category(None, 9999) == "VFR"     # no ceiling => unlimited


def test_resolve_time_month_rollover():
    ref = datetime(2023, 1, 31, 23, 0, tzinfo=timezone.utc)
    # day 1 hour 5 should resolve to Feb 1, not Jan 1
    assert resolve_time(ref, 1, 5) == datetime(2023, 2, 1, 5, 0, tzinfo=timezone.utc)
    # hour 24 -> next day 00:00
    assert resolve_time(ref, 31, 24) == datetime(2023, 2, 1, 0, 0, tzinfo=timezone.utc)


def test_parse_metar_components():
    ref = datetime(2023, 6, 16, 12, 0, tzinfo=timezone.utc)
    p = parse_metar("LEMD 161200Z 22015G25KT 6000 -RA BKN012 OVC025 14/11 Q1012 NOSIG", ref)
    assert p.icao == "LEMD"
    assert p.observed_at == datetime(2023, 6, 16, 12, 0, tzinfo=timezone.utc)
    c = p.conditions
    assert c.wind_dir_deg == 220
    assert c.wind_spd_kt == 15
    assert c.wind_gust_kt == 25
    assert c.vis_m == 6000
    assert c.ceiling_ft == 1200          # lowest BKN/OVC
    assert c.flight_category == "MVFR"   # ceiling 1200 ft
    assert "-RA" in c.weather
    assert p.temp_c == 14 and p.dewpoint_c == 11 and p.qnh_hpa == 1012


def test_parse_metar_cavok_vrb():
    ref = datetime(2023, 6, 16, 6, 0, tzinfo=timezone.utc)
    p = parse_metar("LEBL 160600Z VRB02KT CAVOK 18/12 Q1018", ref)
    assert p.conditions.wind_dir_deg is None         # VRB
    assert p.conditions.vis_m >= 10000               # CAVOK
    assert p.conditions.flight_category == "VFR"


def test_parse_taf_groups():
    ref = datetime(2023, 6, 16, 11, 0, tzinfo=timezone.utc)
    taf = (
        "TAF LEMD 161100Z 1612/1718 22012KT 9999 SCT025 "
        "BECMG 1615/1617 18008KT "
        "PROB30 TEMPO 1700/1706 3000 BR BKN004 "
        "FM171000 27015G25KT CAVOK"
    )
    p = parse_taf(taf, ref)
    assert p.icao == "LEMD"
    assert p.valid_from == datetime(2023, 6, 16, 12, 0, tzinfo=timezone.utc)
    assert p.valid_to == datetime(2023, 6, 17, 18, 0, tzinfo=timezone.utc)

    types = [g.group_type for g in p.groups]
    assert types[0] == "BASE"
    assert "BECMG" in types
    assert "PROB30_TEMPO" in types
    assert "FM" in types

    tempo = next(g for g in p.groups if g.group_type == "PROB30_TEMPO")
    assert tempo.probability == 30
    assert tempo.conditions.vis_m == 3000
    assert tempo.conditions.ceiling_ft == 400          # BKN004
    assert tempo.conditions.flight_category == "LIFR"

    fm = next(g for g in p.groups if g.group_type == "FM")
    assert fm.valid_from == datetime(2023, 6, 17, 10, 0, tzinfo=timezone.utc)
    assert fm.valid_to == p.valid_to                    # persists to end-of-TAF
