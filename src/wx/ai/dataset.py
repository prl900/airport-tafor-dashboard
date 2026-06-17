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

import math

import numpy as np
import pandas as pd

LEADS_DEFAULT = (1, 2, 3, 6, 9, 12, 18, 24, 30)
ANCHOR_MAX_AGE_H = 6        # drop samples whose newest obs <= T0 is staler than this
NO_CEILING_FT = 30000       # sentinel "unlimited" ceiling for the regression target

# --- memory guard ----------------------------------------------------------
# Peak resident memory of a build+train scales ~linearly with the number of grid
# rows (target obs x leads): the DuckDB result and the engineered frame briefly
# coexist (concat), then a dense transformed matrix + per-target estimators sit
# on top. Measured on the 5% sample (1.16M rows): build peak ~2.8 GB, full train
# peak ~3.5 GB across every sklearn rung (linreg/gbm/rf) -> ~3.5 KB/grid-row.
# A full-data build is ~22.7M rows (~50-70 GB) and OOM-kills a 15 GB box, so we
# budget a fraction of *available* RAM and fail fast with the max safe sample_pct
# rather than letting the kernel kill the process. Constant carries model+copy
# overhead margin; bump it if a heavier estimator or more features are added.
# Raised 3600->4200 when the METAR-lag features widened the frame (33->54 feats,
# +36 raw/engineered columns); conservative (errs toward a smaller safe sample).
PEAK_BYTES_PER_ROW = 4200
DEFAULT_MEM_FRACTION = 0.6


def _available_ram_bytes() -> int | None:
    """Linux MemAvailable in bytes; falls back to MemTotal, then None (non-Linux)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024  # values are in kB
        return info.get("MemAvailable") or info.get("MemTotal")
    except (OSError, ValueError, IndexError):
        return None


def _count_grid_rows(con, icaos, leads, start, end, sample_pct) -> int:
    """Cheap COUNT of grid rows (target obs in window x leads) — no materialization."""
    where = ["(CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at >= ?)",
             "(CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at < ?)"]
    params: list = [start, start, end, end]
    if icaos:
        where.append(f"icao IN ({','.join(['?'] * len(icaos))})")
        params += list(icaos)
    if sample_pct is not None and sample_pct < 100:
        where.append(f"(hash(observed_at) % 100) < {int(sample_pct)}")
    n_obs = con.execute(
        f"SELECT count(*) FROM metar_obs WHERE {' AND '.join(where)}", params
    ).fetchone()[0]
    return n_obs * len(leads)


def estimate_build_memory(con, icaos=None, leads=LEADS_DEFAULT, start=None, end=None,
                          sample_pct=None, mem_fraction=DEFAULT_MEM_FRACTION) -> dict:
    """Estimate peak RAM for a build at `sample_pct` and the largest safe sample_pct.

    Returns {rows, est_peak_gb, avail_gb, budget_gb, max_safe_pct, over_budget}.
    `max_safe_pct` is None when available RAM can't be read (non-Linux)."""
    avail = _available_ram_bytes()
    rows = _count_grid_rows(con, icaos, leads, start, end, sample_pct)
    est_peak = rows * PEAK_BYTES_PER_ROW
    budget = int(avail * mem_fraction) if avail else None
    max_safe_pct = None
    if budget:
        rows_full = _count_grid_rows(con, icaos, leads, start, end, 100)
        if rows_full:
            max_safe_pct = max(1, math.floor(budget / PEAK_BYTES_PER_ROW / rows_full * 100))
    return {
        "rows": rows,
        "est_peak_gb": est_peak / 1e9,
        "avail_gb": (avail or 0) / 1e9,
        "budget_gb": (budget or 0) / 1e9,
        "max_safe_pct": max_safe_pct,
        "over_budget": bool(budget and est_peak > budget),
    }


def check_memory_budget(con, icaos=None, leads=LEADS_DEFAULT, start=None, end=None,
                        sample_pct=None, mem_fraction=DEFAULT_MEM_FRACTION) -> dict:
    """Raise MemoryError (naming the max safe sample_pct) if the requested build would
    likely exceed `mem_fraction` of available RAM. Returns the estimate dict."""
    est = estimate_build_memory(con, icaos, leads, start, end, sample_pct, mem_fraction)
    if est["over_budget"]:
        raise MemoryError(
            f"build_samples(sample_pct={sample_pct}) would peak ~{est['est_peak_gb']:.1f} GB "
            f"({est['rows']:,} rows), over the {est['budget_gb']:.1f} GB budget "
            f"({mem_fraction:.0%} of {est['avail_gb']:.1f} GB available). Use "
            f"sample_pct<={est['max_safe_pct']} on this machine, free RAM, or raise "
            f"mem_fraction (pass mem_guard=False to override)."
        )
    return est

_ERA5_FIELDS = ("wind10m_spd", "wind10m_dir", "gust", "t2m_c", "d2m_c",
                "tcc", "cbh_m", "tp_mm", "mslp_hpa")

# Lagged-METAR history: the recent trajectory of the Markov state. All anchored at
# observed_at <= T0 - lag (strictly causal). Tendencies (current - lag) capture
# whether conditions are deteriorating -- the strongest signal for fog/adverse onset.
_LAG_HOURS = (1, 3, 6)
_OBS_LAG_MAP = {"vis_m": "vis", "ceiling_ft": "ceiling", "wind_spd_kt": "wspd",
                "temp_c": "temp", "dewpoint_c": "dew"}


def _era5_cols(alias: str, prefix: str) -> str:
    return ", ".join(f"{alias}.{c} AS {prefix}_{c}" for c in _ERA5_FIELDS)


def _lag_cols(lag: int) -> str:
    a = f"a{lag}"
    return ", ".join(f"{a}.{src} AS o{lag}_{dst}" for src, dst in _OBS_LAG_MAP.items())


def _lag_joins() -> str:
    return "\n".join(
        f"    ASOF LEFT JOIN metar_obs a{lag}\n"
        f"        ON a{lag}.icao = g.icao AND a{lag}.observed_at <= g.t0 - to_hours({lag})"
        for lag in _LAG_HOURS
    )


def build_samples(con, icaos=None, leads=LEADS_DEFAULT, start=None, end=None,
                  sample_pct: int | None = None, mem_guard: bool = True,
                  mem_fraction: float = DEFAULT_MEM_FRACTION) -> pd.DataFrame:
    """Build the causal sample frame from the DB (metar_obs targets + nwp_point).

    ``sample_pct`` (1-100) keeps a reproducible hash-based subset of target obs —
    the dense (obs x lead) grid is ~22M rows at full size, so training samples a
    tractable fraction. The hash on observed_at is deterministic across runs.

    ``mem_guard`` (default on) refuses builds whose estimated peak RAM exceeds
    ``mem_fraction`` of available memory, raising MemoryError with the largest safe
    ``sample_pct`` — this is the guard against OOM-killing the machine on a full
    (or too-large) build. Pass ``mem_guard=False`` to override."""
    if mem_guard:
        check_memory_budget(con, icaos=icaos, leads=leads, start=start, end=end,
                            sample_pct=sample_pct, mem_fraction=mem_fraction)
    leads_vals = ", ".join(f"({int(h)})" for h in leads)
    icao_filter = ""
    params: list = [start, start, end, end]
    if icaos:
        icao_filter = f"AND icao IN ({','.join(['?'] * len(icaos))})"
        params += list(icaos)
    sample_filter = ""
    if sample_pct is not None and sample_pct < 100:
        sample_filter = f"AND (hash(observed_at) % 100) < {int(sample_pct)}"

    sql = f"""
    WITH leads(h) AS (VALUES {leads_vals}),
    targets AS (
        SELECT icao, observed_at AS valid_time, vis_m AS y_vis_m,
               ceiling_ft AS y_ceiling_ft, wind_spd_kt AS y_wspd,
               wind_dir_deg AS y_wdir, flight_category AS y_cat
        FROM metar_obs
        WHERE (CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at >= ?)
          AND (CAST(? AS TIMESTAMPTZ) IS NULL OR observed_at < ?)
          {icao_filter} {sample_filter}
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
           {_lag_cols(1)},
           {_lag_cols(3)},
           {_lag_cols(6)},
           {_era5_cols('et', 'et')},
           {_era5_cols('e0', 'e0')},
           s.elevation_m, s.lat, s.lon, s.region
"""
_JOINS = f"""
    FROM grid g
    ASOF LEFT JOIN metar_obs a
        ON a.icao = g.icao AND a.observed_at <= g.t0
{_lag_joins()}
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

    # --- recent-history lags & tendencies (deterioration signal; all tau <= T0) ---
    o0_ceil = df["o0_ceiling"].fillna(NO_CEILING_FT)
    o0_spread = df["o0_temp"] - df["o0_dew"]
    for lag in _LAG_HOURS:
        lag_ceil = df[f"o{lag}_ceiling"].fillna(NO_CEILING_FT)
        lag_spread = df[f"o{lag}_temp"] - df[f"o{lag}_dew"]
        f[f"f_o{lag}_vis"] = df[f"o{lag}_vis"]
        f[f"f_o{lag}_ceiling"] = lag_ceil
        f[f"f_o{lag}_spread"] = lag_spread
        # current - lag: negative vis/ceiling trend = closing in; shrinking T-Td = fog risk
        f[f"f_dvis_{lag}"] = df["o0_vis"] - df[f"o{lag}_vis"]
        f[f"f_dceil_{lag}"] = o0_ceil - lag_ceil
        f[f"f_dspread_{lag}"] = o0_spread - lag_spread
        f[f"f_dwspd_{lag}"] = df["o0_wspd"] - df[f"o{lag}_wspd"]

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
