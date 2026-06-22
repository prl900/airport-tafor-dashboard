"""Phase D — TFT with quantiles: same SeqForecastModel contract plus the quantile
distribution (monotone, median == point prediction). Skipped if torch absent. CPU."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("torch")

from wx.ai.seq_dataset import TARGET_REG, build_sequences, split_sequences
from wx.ai.tft_models import MEDIAN_IDX, QUANTILES, TFTModel
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path_factory.mktemp("tft") / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    c.execute("INSERT INTO stations VALUES ('LEBL','Barcelona',41.3,2.1,4,'peninsula')")
    base = datetime(2023, 1, 1, tzinfo=UTC)
    for icao in ("LEMD", "LEBL"):
        for i in range(24 * 40):
            ts = base + timedelta(hours=i)
            fog = ts.hour in (5, 6, 7)
            c.execute(
                """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
                   ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                [hash((icao, i)) & 0x7FFFFFFF, i, icao, ts, 500 if fog else 9999,
                 200 if fog else None, 6, 200, 8.0, 7.6 if fog else 1.0,
                 "IFR" if fog else "VFR"],
            )
    b = build_sequences(c, sample_pct=100)
    tr, va, te = split_sequences(b, train_end="2023-01-25", val_end="2023-02-05")
    c.close()
    return tr, va, te


@pytest.fixture(scope="module")
def model(splits):
    tr, va, _ = splits
    return TFTModel("tft", hidden=32, emb_dim=4, n_heads=2,
                    max_epochs=2, batch_size=128).fit(tr, val=va)


def test_adverse_proba_and_calibration(splits, model):
    *_, te = splits
    p = model.predict_adverse_proba(te)
    assert p.shape == (len(te.t0), te.y_reg.shape[1])
    assert (0.0 <= p).all() and (p <= 1.0).all()
    assert model.calibrator is not None and 0.0 < model.adverse_threshold < 1.0


def test_quantile_shape_and_median_is_point(splits, model):
    *_, te = splits
    q = model.predict_quantiles(te)
    H, T, Q = te.y_reg.shape[1], len(TARGET_REG), len(QUANTILES)
    assert q.shape == (len(te.t0), H, T, Q)
    # the point prediction the eval driver uses is exactly the median quantile
    point, _ = model._forward(te)
    assert np.allclose(point, q[..., MEDIAN_IDX])


def test_quantiles_roughly_ordered(splits, model):
    """Pinball loss should push q10 <= q90 on average (not strictly per-point early)."""
    *_, te = splits
    q = model.predict_quantiles(te)
    assert q[..., 0].mean() <= q[..., -1].mean()


def test_save_load_roundtrip_is_exact(splits, model, tmp_path):
    *_, te = splits
    p = model.predict_adverse_proba(te)
    path = tmp_path / "tft.joblib"
    model.save(path)
    m2 = TFTModel.load(path)
    assert m2._net is None
    assert np.max(np.abs(p - m2.predict_adverse_proba(te))) == 0.0
