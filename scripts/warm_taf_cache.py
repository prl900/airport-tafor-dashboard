"""Warm the Ogimet TAF cache (network -> data/raw_cache/ogimet) WITHOUT touching the
DB, so it can run in parallel with the METAR backfill. Single process => respects
Ogimet's 1 request/minute/IP limit. A response that looks empty/rate-limited is
NOT left in the cache (so it can be re-fetched), avoiding silent data loss."""

import sys

from wx.config import AIRPORTS
from wx.ingestion.ogimet import OgimetTafIngester, icao_prefix

YEARS = range(2020, 2026)  # 2020-2025 (the range we keep); 2026 fetched by the store step
PREFIXES = sorted({icao_prefix(a.icao) for a in AIRPORTS})  # GC, GE, LE


def looks_valid(text: str) -> bool:
    # gettafor CSV has a header + data rows; a blocked/empty reply is tiny.
    return text.count("\n") > 1 and "TAF" in text


def main() -> int:
    ing = OgimetTafIngester()
    todo = [(p, y) for p in PREFIXES for y in YEARS]
    print(f"warming {len(todo)} Ogimet granules: prefixes={PREFIXES} years={list(YEARS)}",
          flush=True)
    for i, (prefix, year) in enumerate(todo, 1):
        cache_path = ing._cache_path(f"ogimet-taf-{prefix}-{year}")
        if cache_path.exists() and looks_valid(cache_path.read_text()):
            print(f"[{i}/{len(todo)}] {prefix}-{year}: cached", flush=True)
            continue
        text = ing._fetch_prefix_year(prefix, year)   # paced by the 62s limiter
        if not looks_valid(text):
            cache_path.unlink(missing_ok=True)         # don't keep an empty/blocked reply
            print(f"[{i}/{len(todo)}] {prefix}-{year}: EMPTY/blocked — not cached", flush=True)
        else:
            rows = text.count("\n")
            print(f"[{i}/{len(todo)}] {prefix}-{year}: ok ({rows} lines)", flush=True)
    ing.close()
    print("TAF cache warming complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
