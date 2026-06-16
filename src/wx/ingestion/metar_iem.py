"""METAR ingester for the Iowa Environmental Mesonet (IEM) ASOS archive.

Endpoint returns CSV ``station,valid,metar`` where ``valid`` is UTC
``YYYY-MM-DD HH:MM``. We fetch one station-year per request (well within IEM's
~1 request/second / station-year limits) and store the raw METAR text.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from wx.config import settings
from wx.ingestion.base import Ingester


class IemMetarIngester(Ingester):
    source = "iem"

    def __init__(self) -> None:
        super().__init__(min_interval_s=settings.iem_min_interval_s)

    def fetch_raw(self, icao: str, start: datetime, end: datetime) -> list[dict]:
        records: list[dict] = []
        for year in range(start.year, end.year + 1):
            # Always request the FULL calendar year so the per-year cache key holds
            # the whole year regardless of the requested sub-range; filter afterwards.
            # (A sub-year request cached under the year key would otherwise poison it.)
            y0 = datetime(year, 1, 1, tzinfo=timezone.utc)
            y1 = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            text = self.fetch(
                settings.iem_base_url,
                params={
                    "station": icao,
                    "data": "metar",
                    "year1": y0.year, "month1": 1, "day1": 1,
                    "year2": y1.year, "month2": 1, "day2": 1,
                    "tz": "Etc/UTC",
                    "format": "onlycomma",
                    "latlon": "no", "missing": "M", "trace": "T",
                },
                cache_key=f"iem-metar-{icao}-{year}",
            )
            records.extend(self._parse_csv(text, icao, max(start, y0), min(end, y1)))
        return records

    @staticmethod
    def _parse_csv(text: str, icao: str, start: datetime, end: datetime) -> list[dict]:
        out: list[dict] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            raw = (row.get("metar") or "").strip()
            valid = (row.get("valid") or "").strip()
            if not raw or not valid:
                continue
            try:
                observed_at = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            if not (start <= observed_at < end):
                continue
            out.append(
                {"icao": icao, "observed_at": observed_at, "raw_text": raw, "source": "iem"}
            )
        return out
