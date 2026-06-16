"""Resolve METAR/TAF day-of-month + hour tokens to absolute UTC timestamps.

METAR/TAF encode only the day-of-month and hour (and aviation uses hour 24 to
mean 00:00 of the next day). We anchor those to a reference datetime (the issue
or observation time) and disambiguate month boundaries by choosing the candidate
closest to the reference.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def resolve_time(reference: datetime, day: int, hour: int, minute: int = 0) -> datetime:
    """Return an absolute UTC datetime for a (day, hour) close to ``reference``.

    Handles hour == 24 (-> 00:00 next day) and month rollover in both directions.
    """
    ref = reference.astimezone(timezone.utc)
    extra_days = 0
    if hour >= 24:
        extra_days, hour = divmod(hour, 24)

    # Build candidates anchored to the reference month and the neighbouring months,
    # then pick the one nearest the reference (handles end-of-month wrap either way).
    candidates: list[datetime] = []
    for month_offset in (-1, 0, 1):
        year = ref.year
        month = ref.month + month_offset
        if month < 1:
            month, year = 12, year - 1
        elif month > 12:
            month, year = 1, year + 1
        try:
            base = datetime(year, month, day, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            continue  # e.g. day 31 in a 30-day month
        candidates.append(base + timedelta(days=extra_days, hours=hour, minutes=minute))

    return min(candidates, key=lambda c: abs((c - ref).total_seconds()))
