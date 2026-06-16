"""Phase A — causal feature/target frame for ML TAFOR generation.

A sample is one (airport, issue time T0, lead h, valid time t = T0+h). Causality is
enforced *structurally*: the current-METAR anchor is an ASOF join on
``observed_at <= t0`` (DuckDB picks the most recent obs at or before T0), so no
observation after T0 can enter the features. The target is the METAR at t — a label
only, never a feature.

Perfect-Prognosis NWP: ERA5 is joined at the valid hour t (and at T0, plus the
T0->t tendency). ERA5 is reanalysis, so this is the PP training signal that, at
deployment, would be replaced by an IFS forecast issued at T0 valid at t.

Returns a pandas DataFrame with engineered features (prefix f_) and targets (y_).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEADS_DEFAULT = (1, 2, 3, 6, 9, 12, 18, 24, 30)
ANCHOR_MAX_AGE_H = 6        # drop samples whose newest obs <= T0 is staler than this
NO_CEILING_FT = 30000       # sentinel "unlimited" ceiling for the regression target

_ERA5_FIELDS = ("wind10m_spd", "wind10m_dir", "gust", "t2m_c", "d2m_c",
                "tcc", "cbh_m", "tp_mm", "mslp_hpa")


def _era5_cols(alias: str, prefix: str) -> str:
    return ", ".join(f"{alias}.{c} AS {prefix}_{c}" for c in _ERA5_FIELDS)


def build_samples(con, icaos=None, leads=LEADS_DEFAULT, start=None, end=None) -> pd.DataFrame:
    """Build the causal sample frame from the DB (metar_obs targets + nwp_point)."""
    leads_vals = ", ".join(f"({int(h)})" for h in leads)
    icao_filter = ""
    params: list = [start, start, end, end]
    if icaos:
        icao_filter = f"AND icao IN ({','.join(['?'] * len(icaos))})"
        params += list(icaos)

    sql = f"""
    WITH leads(h) AS (VALUES {leads_vals}),
    targets AS (
        SELECT icao, observed_at AS valid_time, vis_m AS y_vis_m,
               ceiling_ft AS y_ceiling_ft, wind_spd_kt AS y_wspd,
               wind_dir_deg AS y_wdir, flight_category AS y_cat
        FROM metar_obs
        WHERE (CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at >= ?)
          AND (CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at < ?)
          {icao_filter}
    ),
    grid AS (
        SELECT t.*, l.h AS lead_h,
               t.valid_time - to_hours(l.h) AS t0
        FROM targets t, leads l
    )
    SELECT g.y_vis_m, g.y_ceiling_ft, g.y_wspd, g.y_wdir, g.y_cat,
           {_RAW_SELECT}
    {_JOINS}
    """
    df = con.execute(sql, params).df()
    return engineer(df, with_targets=True)


# Shared raw-feature SELECT + join clause, used by both training (build_samples) and
# inference (build_inference_features) so the feature construction is identical.
_RAW_SELECT = f"""
           g.icao, g.t0, g.valid_time, g.lead_h,
           a.observed_at AS o0_time, a.vis_m AS o0_vis, a.ceiling_ft AS o0_ceiling,
           a.wind_spd_kt AS o0_wspd, a.wind_dir_deg AS o0_wdir, a.temp_c AS o0_temp,
           a.dewpoint_c AS o0_dew, a.flight_category AS o0_cat,
           {_era5_cols('et', 'et')},
           {_era5_cols('e0', 'e0')},
           s.elevation_m, s.lat, s.lon, s.region
"""
_JOINS = """
    FROM grid g
    ASOF LEFT JOIN metar_obs a
        ON a.icao = g.icao AND a.observed_at <= g.t0
    LEFT JOIN nwp_point et
        ON et.icao = g.icao AND et.source = 'era5'
       AND et.valid_time = date_trunc('hour', g.valid_time)
    LEFT JOIN nwp_point e0
        ON e0.icao = g.icao AND e0.source = 'era5'
       AND e0.valid_time = date_trunc('hour', g.t0)
    JOIN stations s ON s.icao = g.icao
"""


def build_inference_features(con, icao: str, issued_at, valid_hours) -> pd.DataFrame:
    """Build the SAME causal features for explicit (icao, issued_at, valid hours),
    with no target join — used by ModelForecaster at inference time."""
    if not valid_hours:
        return pd.DataFrame()
    vh = ", ".join([f"(TIMESTAMPTZ '{pd.Timestamp(t).tz_convert('UTC')}')" for t in valid_hours])
    sql = f"""
    WITH grid AS (
        SELECT '{icao}' AS icao,
               TIMESTAMPTZ '{pd.Timestamp(issued_at).tz_convert('UTC')}' AS t0,
               v.t AS valid_time,
               CAST(date_diff('hour',
                    TIMESTAMPTZ '{pd.Timestamp(issued_at).tz_convert('UTC')}', v.t) AS INTEGER) AS lead_h
        FROM (VALUES {vh}) v(t)
    )
    SELECT {_RAW_SELECT}
    {_JOINS}
    """
    df = con.execute(sql).df()
    return engineer(df, with_targets=False)


def engineer(df: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    """Add derived features (f_*) and (optionally) targets (y_*); drop anchorless rows."""
    if df.empty:
        return df

    # Require a fresh current-METAR anchor (the Markov state).
    df = df.dropna(subset=["o0_time"]).copy()
    age_h = (df["t0"] - df["o0_time"]).dt.total_seconds() / 3600.0
    df = df[age_h <= ANCHOR_MAX_AGE_H].copy()

    f = {}
    # --- current-state (T0) features ---
    f["f_o0_vis"] = df["o0_vis"]
    f["f_o0_ceiling"] = df["o0_ceiling"].fillna(NO_CEILING_FT)
    f["f_o0_has_ceiling"] = df["o0_ceiling"].notna().astype(int)
    f["f_o0_wspd"] = df["o0_wspd"]
    f["f_o0_spread"] = df["o0_temp"] - df["o0_dew"]      # T-Td: fog proxy
    f["f_o0_temp"] = df["o0_temp"]
    f.update(_circular("f_o0_wdir", df["o0_wdir"]))

    # --- ERA5 @t (PP forecast proxy) ---
    f["f_et_wspd"] = df["et_wind10m_spd"]
    f["f_et_gust"] = df["et_gust"]
    f["f_et_t2m"] = df["et_t2m_c"]
    f["f_et_spread"] = df["et_t2m_c"] - df["et_d2m_c"]
    f["f_et_tcc"] = df["et_tcc"]
    f["f_et_cbh"] = df["et_cbh_m"]
    f["f_et_tp"] = df["et_tp_mm"]
    f["f_et_msl"] = df["et_mslp_hpa"]
    f.update(_circular("f_et_wdir", df["et_wind10m_dir"]))

    # --- ERA5 T0->t tendency (the evolution signal) ---
    f["f_tend_t2m"] = df["et_t2m_c"] - df["e0_t2m_c"]
    f["f_tend_spread"] = (df["et_t2m_c"] - df["et_d2m_c"]) - (df["e0_t2m_c"] - df["e0_d2m_c"])
    f["f_tend_tcc"] = df["et_tcc"] - df["e0_tcc"]
    f["f_tend_msl"] = df["et_mslp_hpa"] - df["e0_mslp_hpa"]
    f["f_tend_wspd"] = df["et_wind10m_spd"] - df["e0_wind10m_spd"]

    # --- lead & temporal (cyclical) ---
    f["f_lead_h"] = df["lead_h"]
    hod = df["valid_time"].dt.hour
    doy = df["valid_time"].dt.dayofyear
    f["f_hod_sin"], f["f_hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    f["f_doy_sin"], f["f_doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)

    # --- static ---
    f["f_elevation"] = df["elevation_m"]
    f["f_lat"], f["f_lon"] = df["lat"], df["lon"]

    feat = pd.DataFrame(f, index=df.index)
    keys = df[["icao", "region", "t0", "valid_time", "lead_h", "o0_time"]]
    parts = [keys, feat]

    if with_targets:
        y = {
            "y_vis_m": df["y_vis_m"],
            "y_ceiling_ft": df["y_ceiling_ft"].fillna(NO_CEILING_FT),
            "y_has_ceiling": df["y_ceiling_ft"].notna().astype(int),
            "y_wspd": df["y_wspd"],
            "y_cat": df["y_cat"],
        }
        y.update(_circular("y_wdir", df["y_wdir"]))   # plain dict: adds the new keys
        parts.append(pd.DataFrame(y, index=df.index))

    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return out


def _circular(name: str, deg: pd.Series) -> dict:
    """sin/cos encoding of a wind direction (NaN -> 0, paired with a 'known' flag)."""
    rad = np.deg2rad(deg.astype(float))
    return {
        f"{name}_sin": np.sin(rad).fillna(0.0),
        f"{name}_cos": np.cos(rad).fillna(0.0),
        f"{name}_known": deg.notna().astype(int),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("y_")]


# --- temporal splits (blocked, no random shuffling) ------------------------

def temporal_split(df: pd.DataFrame, train_end="2024-01-01", val_end="2025-01-01"):
    """train < train_end <= val < val_end <= test, keyed on issue time t0."""
    t0 = pd.to_datetime(df["t0"], utc=True)
    train_end = pd.Timestamp(train_end, tz="UTC")
    val_end = pd.Timestamp(val_end, tz="UTC")
    return (
        df[t0 < train_end],
        df[(t0 >= train_end) & (t0 < val_end)],
        df[t0 >= val_end],
    )
