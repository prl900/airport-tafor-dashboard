"""Phase D — sequence dataset adapter: window alignment + causality are the
critical guarantees, plus the hash-sampling ratio and honest target masking."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from wx.ai.seq_dataset import (
    FUTURE_FEATURES,
    PAST_FEATURES,
    TARGET_REG,
    build_sequences,
    split_sequences,
)
from wx.db.connection import connect

UTC = timezone.utc
START = datetime(2023, 1, 1, tzinfo=UTC)


def _insert_hourly(c, icao, n_hours, skip=()):
    """Hourly METARs where vis encodes the absolute hour index (for alignment checks).
    Hours in `skip` are omitted, creating gaps to exercise the target mask."""
    for i in range(n_hours):
        if i in skip:
            continue
        ts = START + timedelta(hours=i)
        fog = ts.hour in (5, 6, 7)
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [hash((icao, i)) & 0x7FFFFFFF, i, icao, ts, float(i),
             200 if fog else None, 6, 200, 8.0, 7.6 if fog else 1.0,
             "IFR" if fog else "VFR"],
        )


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path_factory.mktemp("seq") / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    c.execute("INSERT INTO stations VALUES ('LEBL','Barcelona',41.3,2.1,4,'peninsula')")
    _insert_hourly(c, "LEMD", 24 * 40)         # 40 days
    _insert_hourly(c, "LEBL", 24 * 40, skip={100, 101, 102})  # a 3-hour gap
    yield c
    c.close()


def test_shapes_and_lead_encoding(con):
    b = build_sequences(con, icaos=["LEMD"], sample_pct=100)
    assert b.x_past.shape[1:] == (12, len(PAST_FEATURES))
    assert b.x_future.shape[1:] == (30, len(FUTURE_FEATURES))
    assert b.y_reg.shape[1:] == (30, len(TARGET_REG))
    assert b.y_cat.shape[1] == 30 and b.y_mask.shape[1] == 30
    # lead column (index 10) must be 1..30 for every sample
    assert (b.x_future[:, :, 10] == np.arange(1, 31)).all()
    # y_cat is -1 exactly where the horizon has no verifying obs
    assert (b.y_cat[b.y_mask == 0] == -1).all()
    assert set(np.unique(b.y_mask)) <= {0.0, 1.0}


def test_window_alignment_is_causal(con):
    """vis encodes the absolute hour, so the last past step == T0 and horizon j
    targets T0+j+1 — proving the past is causal (<=T0) and futures are aligned."""
    b = build_sequences(con, icaos=["LEMD"], sample_pct=100)
    t0_idx = ((pd.to_datetime(b.t0, utc=True) - pd.Timestamp(START))
              / pd.Timedelta(hours=1)).to_numpy().round().astype(int)
    vis_past_last = b.x_past[:, -1, 0]          # PAST_FEATURES[0] == "vis"
    assert np.allclose(vis_past_last, t0_idx), "last past step is not T0"
    for j in (0, 5, 29):                        # horizons +1, +6, +30
        assert np.allclose(b.y_reg[:, j, 0], t0_idx + j + 1), f"horizon {j} misaligned"


def test_sampling_ratio_is_reproducible(con):
    full = build_sequences(con, icaos=["LEMD"], sample_pct=100)
    half = build_sequences(con, icaos=["LEMD"], sample_pct=50)
    ratio = len(half.t0) / len(full.t0)
    assert 0.4 < ratio < 0.6, f"sample_pct=50 kept {ratio:.0%}, expected ~50%"
    # deterministic: same call -> same T0 set
    again = build_sequences(con, icaos=["LEMD"], sample_pct=50)
    assert (half.t0 == again.t0).all()


def test_target_mask_drops_gaps(con):
    """The 3-hour LEBL gap must show up as masked horizons (no carried-forward obs)."""
    b = build_sequences(con, icaos=["LEBL"], sample_pct=100)
    assert (b.y_mask == 0).any(), "gap did not produce any masked targets"


def test_temporal_split_partitions_by_t0(con):
    b = build_sequences(con, icaos=["LEMD"], sample_pct=100)
    tr, va, te = split_sequences(b, train_end="2023-01-21", val_end="2023-01-31")
    assert len(tr.t0) and len(va.t0) and len(te.t0)
    assert len(tr.t0) + len(va.t0) + len(te.t0) == len(b.t0)
    assert pd.to_datetime(tr.t0, utc=True).max() < pd.to_datetime(va.t0, utc=True).min()
    assert pd.to_datetime(va.t0, utc=True).max() < pd.to_datetime(te.t0, utc=True).min()
