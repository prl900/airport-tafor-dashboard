"""Phase D — sequence dataset adapter for multi-horizon probabilistic models.

The tabular frame (``dataset.build_samples``) flattens each (T0, lead) into one row.
A sequence model instead wants, per (station, issue time T0):

  * **past**  — the observed METAR trajectory on an hourly grid ending at T0
                (the encoder context; strictly ``observed_at <= T0``);
  * **future**— known-future covariates at each horizon t = T0+1..T0+H: the ERA5
                forecast proxy + cyclical time + lead (the decoder inputs);
  * **static**— station attributes (lat/lon/elevation + a station id for embedding);
  * **target**— the METAR-derived state at each horizon (vis/ceiling/wind/category),
                with a mask where no verifying obs exists.

Causality mirrors the tabular path: the past grid is built by an ASOF join
(``observed_at <= hour``) so nothing after T0 leaks; ERA5 is the Perfect-Prognosis
stand-in for an IFS forecast issued at T0. Splits are by T0 (train<2024 / val 2024 /
frozen test 2025), identical to ``temporal_split``.

Returns plain numpy arrays (+ a T0 index) so the torch model stays decoupled from
pandas/DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wx.ai.dataset import ANCHOR_MAX_AGE_H, NO_CEILING_FT

PAST_LEN = 12          # hours of observed history fed to the encoder
HORIZON = 30           # forecast horizons (T0+1 .. T0+30h), covers TAF validity
CAT_CODE = {"LIFR": 0, "IFR": 1, "MVFR": 2, "VFR": 3}
ADVERSE_CODES = (0, 1)  # LIFR, IFR
_ERA5_FIELDS = ("wind10m_spd", "wind10m_dir", "gust", "t2m_c", "d2m_c",
                "tcc", "cbh_m", "tp_mm", "mslp_hpa")

# Past (encoder) feature names, in array order.
PAST_FEATURES = ["vis", "ceiling", "has_ceiling", "wspd", "wdir_sin", "wdir_cos",
                 "spread", "temp", "cat_code", "fresh"]
# Future (decoder) covariate names, in array order.
FUTURE_FEATURES = ["et_wspd", "et_wdir_sin", "et_wdir_cos", "et_gust", "et_t2m",
                   "et_spread", "et_tcc", "et_cbh", "et_tp", "et_msl",
                   "lead_h", "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
STATIC_FEATURES = ["lat", "lon", "elevation"]
# Target names (regression), in array order; category is carried separately.
TARGET_REG = ["vis", "ceiling", "wspd", "wdir_sin", "wdir_cos"]


@dataclass
class SeqBatch:
    """Materialized sequence tensors (numpy). N samples, K past steps, H horizons."""
    icao: np.ndarray          # (N,) station id index (for embedding)
    t0: np.ndarray            # (N,) issue time (datetime64) — for temporal_split
    x_past: np.ndarray        # (N, K, len(PAST_FEATURES)) float32
    x_future: np.ndarray      # (N, H, len(FUTURE_FEATURES)) float32
    x_static: np.ndarray      # (N, len(STATIC_FEATURES)) float32
    y_reg: np.ndarray         # (N, H, len(TARGET_REG)) float32
    y_cat: np.ndarray         # (N, H) int64 category code, -1 where missing
    y_mask: np.ndarray        # (N, H) float32 — 1 where a verifying obs exists
    stations: list            # icao strings indexed by the id in `icao`


def _circular(deg: pd.Series):
    rad = np.deg2rad(deg.astype(float))
    return np.sin(rad).fillna(0.0).to_numpy(), np.cos(rad).fillna(0.0).to_numpy()


def _hourly_metar_state(con, icao: str):
    """Build the hourly METAR grid for one station, returning (encoder, target).

    Raw obs are pulled once and resampled in pandas (fast, and gives an honest target
    mask) two ways on the same hourly index:
      * **encoder** — forward-filled last-known state (continuous context) with a
        ``fresh`` flag = a real obs landed within ANCHOR_MAX_AGE of this hour;
      * **target**  — the obs *nearest within 30 min* of the hour, NaN (masked) when
        none exists, so we never train/score on a carried-forward stale observation.
    """
    df = con.execute(
        "SELECT observed_at, vis_m, ceiling_ft, wind_spd_kt, wind_dir_deg, "
        "temp_c, dewpoint_c, flight_category FROM metar_obs WHERE icao = ? "
        "ORDER BY observed_at", [icao]
    ).df()
    if df.empty:
        return None, None
    df = df.set_index("observed_at")
    df = df[~df.index.duplicated(keep="last")]
    idx = pd.date_range(df.index.min().floor("h"), df.index.max().floor("h"),
                        freq="h", tz="UTC")

    near = df.reindex(idx, method="nearest", tolerance=pd.Timedelta("30min"))
    ff = df.reindex(idx, method="ffill")
    # fresh = a real obs landed within ANCHOR_MAX_AGE of this hour (carry-forward age)
    last_obs_time = pd.Series(df.index, index=df.index).reindex(idx, method="ffill")
    age_h = (idx - pd.DatetimeIndex(last_obs_time)).total_seconds() / 3600.0
    fresh = pd.Series(age_h <= ANCHOR_MAX_AGE_H, index=idx).fillna(False).astype(float)

    enc = pd.DataFrame(index=idx)
    enc["vis"] = ff["vis_m"]
    enc["ceiling"] = ff["ceiling_ft"].fillna(NO_CEILING_FT)
    enc["has_ceiling"] = ff["ceiling_ft"].notna().astype(float)
    enc["wspd"] = ff["wind_spd_kt"]
    enc["wdir_sin"], enc["wdir_cos"] = _circular(ff["wind_dir_deg"])
    enc["spread"] = ff["temp_c"] - ff["dewpoint_c"]
    enc["temp"] = ff["temp_c"]
    enc["cat_code"] = ff["flight_category"].map(CAT_CODE)
    enc["fresh"] = fresh

    tgt = pd.DataFrame(index=idx)
    tgt["vis"] = near["vis_m"]
    tgt["ceiling"] = near["ceiling_ft"].fillna(NO_CEILING_FT)
    tgt["wspd"] = near["wind_spd_kt"]
    tgt["wdir_sin"], tgt["wdir_cos"] = _circular(near["wind_dir_deg"])
    tgt["cat_code"] = near["flight_category"].map(CAT_CODE)
    tgt["has_obs"] = near["flight_category"].notna().astype(float)
    return enc, tgt


def _hourly_era5(con, icao: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """ERA5 hourly covariates reindexed onto the METAR hourly grid."""
    fields = ", ".join(_ERA5_FIELDS)
    df = con.execute(
        f"SELECT valid_time, {fields} FROM nwp_point WHERE icao = ? AND source = 'era5' "
        "ORDER BY valid_time", [icao]
    ).df()
    if df.empty:
        return pd.DataFrame(index=index, columns=list(FUTURE_FEATURES[:10]), dtype=float)
    df = df.set_index("valid_time").reindex(index)
    out = pd.DataFrame(index=index)
    out["et_wspd"] = df["wind10m_spd"]
    out["et_wdir_sin"], out["et_wdir_cos"] = _circular(df["wind10m_dir"])
    out["et_gust"] = df["gust"]
    out["et_t2m"] = df["t2m_c"]
    out["et_spread"] = df["t2m_c"] - df["d2m_c"]
    out["et_tcc"] = df["tcc"]
    out["et_cbh"] = df["cbh_m"]
    out["et_tp"] = df["tp_mm"]
    out["et_msl"] = df["mslp_hpa"]
    return out


def _windows(arr: np.ndarray, win: int) -> np.ndarray:
    """Sliding windows of length `win` over axis 0: (n-win+1, win, *feat)."""
    from numpy.lib.stride_tricks import sliding_window_view

    v = sliding_window_view(arr, win, axis=0)            # (n-win+1, *feat, win)
    return np.moveaxis(v, -1, 1)                          # (n-win+1, win, *feat)


def _station_sequences(con, icao: str, sid: int, station_static: dict,
                       past_len: int, horizon: int, sample_pct: int | None):
    """Build all (past, future, static, target) windows for one station (vectorized)."""
    enc, tgt = _hourly_metar_state(con, icao)
    if enc is None or len(enc) < past_len + horizon + 1:
        return None
    era5 = _hourly_era5(con, icao, enc.index)
    hours = enc.index
    n = len(enc)
    K, H = past_len, horizon

    # Valid T0 indices: a fresh anchor now, and a full past+future window in range.
    fresh = enc["fresh"].to_numpy()
    lo, hi = K - 1, n - H - 1
    t0_pos = np.arange(lo, hi + 1)
    t0_pos = t0_pos[fresh[t0_pos] == 1.0]
    if sample_pct is not None and sample_pct < 100:
        # epoch-hour hash, resolution-independent (DuckDB timestamps are microseconds):
        # numpy casts to datetime64[h] -> integer hours since epoch, deterministic.
        naive = hours[t0_pos].tz_convert("UTC").tz_localize(None)
        epoch_h = naive.values.astype("datetime64[h]").astype(np.int64)
        t0_pos = t0_pos[(np.abs(epoch_h) % 100) < sample_pct]
    if len(t0_pos) == 0:
        return None

    # full-grid feature matrices, then slice with sliding windows
    past_mat = np.nan_to_num(enc[PAST_FEATURES].to_numpy(np.float32))
    hod = hours.hour.to_numpy(); doy = hours.dayofyear.to_numpy()
    fut_full = np.zeros((n, len(FUTURE_FEATURES)), dtype=np.float32)
    fut_full[:, :10] = np.nan_to_num(era5[FUTURE_FEATURES[:10]].to_numpy(np.float32))
    # col 10 (lead) filled per-window below; 11..14 are absolute-hour cyclical time
    fut_full[:, 11] = np.sin(2 * np.pi * hod / 24); fut_full[:, 12] = np.cos(2 * np.pi * hod / 24)
    fut_full[:, 13] = np.sin(2 * np.pi * doy / 365); fut_full[:, 14] = np.cos(2 * np.pi * doy / 365)
    reg_full = np.nan_to_num(tgt[TARGET_REG].to_numpy(np.float32))
    cat_full = tgt["cat_code"].to_numpy()
    obs_full = tgt["has_obs"].to_numpy(np.float32)

    pw = _windows(past_mat, K)            # (n-K+1, K, Fp); window w covers [w:w+K]
    fw = _windows(fut_full, H)            # (n-H+1, H, Ff); window w covers [w:w+H]
    rw = _windows(reg_full, H)
    cw = _windows(cat_full, H)
    mw = _windows(obs_full, H)

    # past window ending at T0=i starts at i-K+1; future window starts at i+1
    x_past = pw[t0_pos - K + 1]
    fut = fw[t0_pos + 1].copy()
    fut[:, :, 10] = np.arange(1, H + 1, dtype=np.float32)     # lead per horizon
    y_reg = rw[t0_pos + 1]
    cat_win = cw[t0_pos + 1]
    y_mask = mw[t0_pos + 1].astype(np.float32)
    y_cat = np.where(y_mask == 1.0, np.nan_to_num(cat_win, nan=-1), -1).astype(np.int64)

    N = len(t0_pos)
    static = np.array([[station_static["lat"], station_static["lon"],
                        station_static["elevation"]]], dtype=np.float32).repeat(N, axis=0)
    return {
        "icao": np.full(N, sid, dtype=np.int64),
        "t0": hours[t0_pos].to_numpy(),
        "x_past": x_past, "x_future": fut, "x_static": static,
        "y_reg": y_reg, "y_cat": y_cat, "y_mask": y_mask,
    }


def build_sequences(con, icaos=None, past_len: int = PAST_LEN, horizon: int = HORIZON,
                    sample_pct: int | None = None) -> SeqBatch:
    """Build the sequence dataset across stations. ``sample_pct`` subsamples T0s by a
    deterministic hash on the issue hour (reproducible)."""
    if icaos is None:
        icaos = [r[0] for r in con.execute(
            "SELECT DISTINCT icao FROM metar_obs ORDER BY icao").fetchall()]
    static_rows = {r[0]: {"lat": r[1], "lon": r[2], "elevation": r[3]}
                   for r in con.execute(
                       "SELECT icao, lat, lon, elevation_m FROM stations").fetchall()}

    parts, stations = [], []
    for icao in icaos:
        if icao not in static_rows:
            continue
        sid = len(stations)
        res = _station_sequences(con, icao, sid, static_rows[icao],
                                 past_len, horizon, sample_pct)
        if res is not None:
            parts.append(res)
            stations.append(icao)

    if not parts:
        raise ValueError("no sequences built (check station data / sample_pct)")

    def cat(key):
        return np.concatenate([p[key] for p in parts], axis=0)

    return SeqBatch(
        icao=cat("icao"), t0=cat("t0"),
        x_past=cat("x_past"), x_future=cat("x_future"), x_static=cat("x_static"),
        y_reg=cat("y_reg"), y_cat=cat("y_cat"), y_mask=cat("y_mask"),
        stations=stations,
    )


def split_sequences(batch: SeqBatch, train_end="2024-01-01", val_end="2025-01-01"):
    """Split a SeqBatch by T0 into (train, val, test) — same boundaries as tabular."""
    t0 = pd.to_datetime(batch.t0, utc=True)
    train_end = pd.Timestamp(train_end, tz="UTC")
    val_end = pd.Timestamp(val_end, tz="UTC")
    tr = np.asarray(t0 < train_end)
    va = np.asarray((t0 >= train_end) & (t0 < val_end))
    te = np.asarray(t0 >= val_end)

    def sub(mask):
        return SeqBatch(
            icao=batch.icao[mask], t0=batch.t0[mask],
            x_past=batch.x_past[mask], x_future=batch.x_future[mask],
            x_static=batch.x_static[mask], y_reg=batch.y_reg[mask],
            y_cat=batch.y_cat[mask], y_mask=batch.y_mask[mask], stations=batch.stations,
        )
    return sub(tr), sub(va), sub(te)
