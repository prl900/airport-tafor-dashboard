"""Base ingester: polite, cached HTTP fetching with retry/backoff.

The pipeline is deliberately split into two idempotent stages:

  fetch_raw()  -> store_raw()    (this module's concern)
  parse()      -> store_parsed() (the parsing/ layer's concern)

Every HTTP response is cached on disk under ``data/raw_cache`` keyed by a stable
hash of the request, so re-parsing history never triggers a re-download and a
fragile source (Ogimet) is hit at most once per unique request.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from wx.config import RAW_CACHE_DIR, settings


class RateLimiter:
    """Simple monotonic-clock spacing between calls (per process)."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last = time.monotonic()


class Ingester(ABC):
    """Common HTTP + cache + throttle machinery for a data source."""

    #: short tag stored in the ``source`` column ('iem', 'ogimet', 'aemet', 'era5')
    source: str

    def __init__(self, min_interval_s: float, cache_subdir: str | None = None) -> None:
        self.limiter = RateLimiter(min_interval_s)
        self.cache_dir = RAW_CACHE_DIR / (cache_subdir or self.source)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=settings.http_timeout_s,
            headers={"User-Agent": "airport-tafor-dashboard/0.1 (research; contact pablo@haizea.com.au)"},
            follow_redirects=True,
        )

    # -- caching ------------------------------------------------------------
    def _cache_path(self, cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.txt"

    def fetch(self, url: str, params: dict | None = None, cache_key: str | None = None) -> str:
        """Return response text, served from cache when available.

        ``cache_key`` should uniquely and stably identify the request (e.g.
        ``"LEMD-2023"``). Falls back to the URL+params if not provided.
        """
        key = cache_key or f"{url}?{sorted((params or {}).items())}"
        cache_path = self._cache_path(key)
        if cache_path.exists():
            return cache_path.read_text()

        text = self._fetch_remote(url, params)
        cache_path.write_text(text)
        return text

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=60))
    def _fetch_remote(self, url: str, params: dict | None) -> str:
        self.limiter.wait()
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        return resp.text

    # -- contract for subclasses -------------------------------------------
    @abstractmethod
    def fetch_raw(self, icao: str, start, end) -> list:
        """Fetch raw messages for a station over [start, end). Returns parsed-out
        ``RawRecord``-like dicts ready for ``store_raw``."""
        raise NotImplementedError

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Ingester":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
