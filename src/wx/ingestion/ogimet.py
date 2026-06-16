"""Ogimet bulk-download ingesters (the official ``getmetar`` / ``gettafor`` tools).

These are the endpoints the Ogimet author (G. Ballester) provides for *mass*
download — not the interactive ``display_metars2.php`` page, which throttles
hard. Docs:
  https://www.ogimet.com/getmetar_help.phtml
  https://www.ogimet.com/gettafor_help.phtml

Key properties:
  * One request per **minute** per IP for these HD-backed tools (older than ~3
    months lives on slower storage). Up to ~200k messages per request.
  * The author recommends requesting **per year** (messages are stored in
    yearly tables).
  * The ``icao`` parameter is a **prefix** match: ``icao=LE`` returns every
    Spanish peninsular/Balearic station in one shot.

So we fetch one ``(prefix, year)`` granule at a time and cache it. Looping over
many stations under the same prefix therefore costs only one live request per
year — the rest are cache hits. Output is CSV:
  ``ESTACION,ANO,MES,DIA,HORA,MINUTO,<METAR|PARTE>``
with the raw message in the last column (TAFs end with ``=``).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from wx.config import settings
from wx.ingestion.base import Ingester

GETMETAR_URL = "http://www.ogimet.com/cgi-bin/getmetar"
GETTAFOR_URL = "http://www.ogimet.com/cgi-bin/gettafor"


def icao_prefix(icao: str) -> str:
    """Two-letter ICAO region prefix used to batch Ogimet requests (LE, GC, GE)."""
    return icao[:2].upper()


def parse_ogimet_csv(text: str, icao: str, start: datetime, end: datetime):
    """Yield (timestamp_utc, raw_text) for ``icao`` from an Ogimet bulk CSV.

    Columns: ESTACION,ANO,MES,DIA,HORA,MINUTO,<METAR|PARTE>. The header row and
    rows for other stations are skipped; TAF trailing ``=`` is stripped.
    """
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 7 or row[0] != icao:
            continue
        try:
            ts = datetime(
                int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue  # header row or malformed line
        if not (start <= ts < end):
            continue
        raw = row[6].strip().rstrip("=").strip()
        if raw:
            yield ts, raw


class _OgimetBulk(Ingester):
    """Shared machinery for getmetar/gettafor: fetch a full (prefix, year)
    granule, cache it, and yield rows for the requested station."""

    source = "ogimet"
    url: str

    def __init__(self) -> None:
        super().__init__(min_interval_s=settings.ogimet_min_interval_s, cache_subdir="ogimet")

    def _fetch_prefix_year(self, prefix: str, year: int) -> str:
        begin = f"{year}01010000"
        end = f"{year + 1}01010000"
        kind = "metar" if self.url == GETMETAR_URL else "taf"
        return self.fetch(
            self.url,
            params={"icao": prefix, "begin": begin, "end": end, "lang": "eng", "header": "yes"},
            cache_key=f"ogimet-{kind}-{prefix}-{year}",
        )

    def _rows_for_station(self, icao: str, start: datetime, end: datetime):
        """Yield (observed/issued datetime, raw_text) for ``icao`` within window."""
        prefix = icao_prefix(icao)
        for year in range(start.year, end.year + 1):
            text = self._fetch_prefix_year(prefix, year)
            yield from parse_ogimet_csv(text, icao, start, end)


class OgimetTafIngester(_OgimetBulk):
    url = GETTAFOR_URL

    def fetch_raw(self, icao: str, start: datetime, end: datetime) -> list[dict]:
        return [
            {
                "icao": icao,
                "issued_at": ts,
                "valid_from": None,  # filled by the parse stage
                "valid_to": None,
                "raw_text": raw,
                "source": "ogimet",
            }
            for ts, raw in self._rows_for_station(icao, start, end)
        ]


class OgimetMetarIngester(_OgimetBulk):
    url = GETMETAR_URL

    def fetch_raw(self, icao: str, start: datetime, end: datetime) -> list[dict]:
        return [
            {"icao": icao, "observed_at": ts, "raw_text": raw, "source": "ogimet"}
            for ts, raw in self._rows_for_station(icao, start, end)
        ]
