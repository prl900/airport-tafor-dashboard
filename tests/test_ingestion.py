from datetime import datetime, timezone

from wx.ingestion.metar_iem import IemMetarIngester
from wx.ingestion.ogimet import icao_prefix, parse_ogimet_csv

UTC = timezone.utc

IEM_CSV = """station,valid,metar
LEMD,2023-01-01 00:00,LEMD 010000Z 34003KT CAVOK 04/02 Q1027 NOSIG
LEMD,2023-01-01 00:30,LEMD 010030Z 35003KT 320V020 CAVOK 04/02 Q1026 NOSIG
LEMD,2023-12-31 23:00,LEMD 312300Z 00000KT 9999 SCT045 04/03 Q1026 NOSIG
"""

OGIMET_TAF_CSV = """ESTACION,ANO,MES,DIA,HORA,MINUTO,PARTE
LEMD,2023,01,01,05,00,TAF LEMD 010500Z 0106/0212 VRB05KT 9999 BKN040 TEMPO 0118/0124 20008KT=
LEBL,2023,01,01,05,00,TAF LEBL 010500Z 0106/0212 23010KT CAVOK=
LEMD,2023,01,01,11,00,TAF LEMD 011100Z 0112/0218 VRB05KT 9999 SCT050=
"""


def test_icao_prefix():
    assert icao_prefix("LEMD") == "LE"
    assert icao_prefix("GCLP") == "GC"
    assert icao_prefix("GEML") == "GE"


def test_iem_parse_csv_window_and_fields():
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2023, 6, 1, tzinfo=UTC)
    recs = IemMetarIngester._parse_csv(IEM_CSV, "LEMD", start, end)
    # The Dec 31 row is outside the window and must be excluded.
    assert len(recs) == 2
    assert recs[0]["observed_at"] == datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    assert recs[0]["source"] == "iem"
    assert recs[0]["raw_text"].startswith("LEMD 010000Z")


def test_ogimet_csv_filters_to_station_and_strips_equals():
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2023, 2, 1, tzinfo=UTC)
    rows = list(parse_ogimet_csv(OGIMET_TAF_CSV, "LEMD", start, end))
    assert len(rows) == 2  # LEBL row excluded
    ts, raw = rows[0]
    assert ts == datetime(2023, 1, 1, 5, 0, tzinfo=UTC)
    assert raw.startswith("TAF LEMD 010500Z")
    assert not raw.endswith("=")
