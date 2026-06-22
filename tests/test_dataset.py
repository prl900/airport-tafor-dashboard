"""Phase A — causal dataset builder: the leakage guarantee is the critical test."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from wx.ai.dataset import (
    build_inference_features,
    build_inference_features_batch,
    build_samples,
    feature_columns,
    target_columns,
    temporal_split,
)
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture
def con(tmp_path):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    # Hourly METARs across two days: VFR except a 06Z LIFR fog.
    base = datetime(2023, 1, 1, tzinfo=UTC)
    rid = 0
    for hours in range(48):
        ts = base + timedelta(hours=hours)
        fog = ts.hour == 6
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?, 'LEMD', ?, ?, ?, ?, ?, ?, ?, ?)""",
            [rid, rid, ts, 400 if fog else 9999, 200 if fog else None,
             5, 220, 8.0, 7.5 if fog else 2.0, "LIFR" if fog else "VFR"],
        )
        rid += 1
    yield c
    c.close()


def test_leakage_audit_anchor_never_in_future(con):
    """Every feature anchor must be at or before T0, and T0 strictly before valid."""
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3, 6))
    assert len(df) > 0
    t0 = pd.to_datetime(df["t0"], utc=True)
    anchor = pd.to_datetime(df["o0_time"], utc=True)
    valid = pd.to_datetime(df["valid_time"], utc=True)
    assert (anchor <= t0).all(), "feature uses an observation after T0 — leakage!"
    assert (t0 < valid).all(), "T0 must be strictly before the valid time"


def test_targets_include_wind_direction(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3))
    tcols = target_columns(df)
    for col in ("y_vis_m", "y_ceiling_ft", "y_wspd", "y_cat",
                "y_wdir_sin", "y_wdir_cos", "y_wdir_known"):
        assert col in tcols, f"missing target {col}"


def test_lead_and_features_present(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 6, 12))
    fcols = feature_columns(df)
    assert "f_lead_h" in fcols and "f_o0_spread" in fcols and "f_hod_sin" in fcols
    assert set(df["lead_h"].unique()) <= {1, 6, 12}
    # fog spread feature: when the anchor is the 06Z fog ob, T-Td spread is small
    assert df["f_o0_spread"].min() < 1.0


def test_batched_inference_features_match_per_issue(con):
    """The gate's batched feature build must be identical to per-issue builds."""
    issued = datetime(2023, 1, 1, 12, tzinfo=UTC)
    hours = [issued + timedelta(hours=h) for h in (1, 3, 6)]
    single = build_inference_features(con, "LEMD", issued, hours)
    batch = build_inference_features_batch(con, [("LEMD", issued, h) for h in hours])
    assert len(single) == len(batch) > 0
    fcols = feature_columns(single)
    a = single.sort_values("valid_time")[fcols].reset_index(drop=True)
    b = batch.sort_values("valid_time")[fcols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_temporal_split_is_chronological(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1,))
    tr, va, te = temporal_split(df, train_end="2023-01-02", val_end="2023-01-03")
    assert len(tr) and len(va)
    if len(tr) and len(va):
        assert pd.to_datetime(tr["t0"], utc=True).max() < pd.to_datetime(va["t0"], utc=True).min()
