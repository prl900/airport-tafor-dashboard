"""Application settings and the seed list of Spanish airports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../airport-tafor-dashboard
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_CACHE_DIR = DATA_DIR / "raw_cache"
ERA5_DIR = DATA_DIR / "era5"


class Settings(BaseSettings):
    """Runtime settings, overridable via environment (prefix ``WX_``) or a .env file."""

    model_config = SettingsConfigDict(env_prefix="WX_", env_file=".env", extra="ignore")

    db_path: Path = ROOT_DIR / "wx.duckdb"

    # Ingestion politeness
    iem_base_url: str = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    iem_min_interval_s: float = 1.0  # IEM asks for <= 1 request/second
    ogimet_base_url: str = "https://www.ogimet.com/display_metars2.php"
    ogimet_min_interval_s: float = 5.0  # Ogimet is fragile — be gentle
    http_timeout_s: float = 60.0

    # AEMET OpenData (authoritative spot-check) — set WX_AEMET_API_KEY to enable
    aemet_api_key: str | None = None

    # ERA5 / Iberia bounding box [North, West, South, East]
    era5_area: tuple[float, float, float, float] = (44.0, -10.0, 35.0, 4.0)


settings = Settings()


@dataclass(frozen=True)
class Airport:
    icao: str
    name: str
    lat: float
    lon: float
    elevation_m: int
    region: str  # 'peninsula' | 'balearic' | 'canary' | 'north_africa'


# Seed set of ~24 Spanish airports (peninsular + Balearic + Canary + enclaves).
# Kept here so it is trivially extensible; loaded into the `stations` table by initdb.
AIRPORTS: list[Airport] = [
    Airport("LEMD", "Madrid–Barajas Adolfo Suárez", 40.4936, -3.5668, 610, "peninsula"),
    Airport("LEBL", "Barcelona–El Prat", 41.2971, 2.0785, 4, "peninsula"),
    Airport("LEMG", "Málaga–Costa del Sol", 36.6749, -4.4991, 16, "peninsula"),
    Airport("LEZL", "Sevilla", 37.4180, -5.8931, 34, "peninsula"),
    Airport("LEAL", "Alicante–Elche", 38.2822, -0.5582, 43, "peninsula"),
    Airport("LEVC", "Valencia", 39.4893, -0.4816, 73, "peninsula"),
    Airport("LEBB", "Bilbao", 43.3011, -2.9106, 42, "peninsula"),
    Airport("LEST", "Santiago de Compostela", 42.8963, -8.4151, 370, "peninsula"),
    Airport("LEGE", "Girona–Costa Brava", 41.9010, 2.7605, 143, "peninsula"),
    Airport("LEVT", "Vitoria", 42.8828, -2.7244, 513, "peninsula"),
    Airport("LEXJ", "Santander–Seve Ballesteros", 43.4271, -3.8200, 5, "peninsula"),
    Airport("LEAS", "Asturias", 43.5636, -6.0346, 127, "peninsula"),
    Airport("LERS", "Reus", 41.1474, 1.1672, 71, "peninsula"),
    Airport("LEMI", "Región de Murcia (Corvera)", 37.8030, -1.1250, 161, "peninsula"),
    Airport("LEPA", "Palma de Mallorca", 39.5517, 2.7388, 8, "balearic"),
    Airport("LEMH", "Menorca", 39.8626, 4.2186, 91, "balearic"),
    Airport("LEIB", "Ibiza", 38.8729, 1.3731, 7, "balearic"),
    Airport("GCLP", "Gran Canaria", 27.9319, -15.3866, 24, "canary"),
    Airport("GCXO", "Tenerife Norte–Ciudad de La Laguna", 28.4827, -16.3415, 633, "canary"),
    Airport("GCTS", "Tenerife Sur", 28.0445, -16.5725, 64, "canary"),
    Airport("GCFV", "Fuerteventura", 28.4527, -13.8638, 26, "canary"),
    Airport("GCRR", "Lanzarote–César Manrique", 28.9455, -13.6052, 14, "canary"),
    Airport("GCLA", "La Palma", 28.6265, -17.7556, 33, "canary"),
    Airport("GEML", "Melilla", 35.2798, -2.9563, 47, "north_africa"),
]

AIRPORTS_BY_ICAO: dict[str, Airport] = {a.icao: a for a in AIRPORTS}
