"""Phase D — seq2seq model: fit/predict shape contract over (sample, horizon),
calibration, and device-safe (un)pickling. Skipped if torch absent. Runs on CPU."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("torch")

from wx.ai.seq_dataset import build_sequences, split_sequences
from wx.ai.seq_models import SeqForecastModel
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path_factory.mktemp("seqm") / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    c.execute("INSERT INTO stations VALUES ('LEBL','Barcelona',41.3,2.1,4,'peninsula')")
    base = datetime(2023, 1, 1, tzinfo=UTC)
    for icao in ("LEMD", "LEBL"):
        for i in range(24 * 40):  # 40 days hourly
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
    return SeqForecastModel("seq2seq", hidden=32, emb_dim=4,
                            max_epochs=2, batch_size=256).fit(tr, val=va)


def test_adverse_proba_shape_and_calibration(splits, model):
    *_, te = splits
    p = model.predict_adverse_proba(te)
    assert p.shape == (len(te.t0), te.y_reg.shape[1])     # (N, H)
    assert (0.0 <= p).all() and (p <= 1.0).all()
    assert model.calibrator is not None and 0.0 < model.adverse_threshold < 1.0
    assert model.predict_adverse_event(te).shape == p.shape


def test_forward_regression_shape(splits, model):
    *_, te = splits
    reg, cat_p = model._forward(te)
    assert reg.shape == te.y_reg.shape                    # (N, H, 5)
    assert cat_p.shape[:2] == (len(te.t0), te.y_reg.shape[1])
    assert np.isfinite(reg).all()


def test_save_load_roundtrip_is_exact(splits, model, tmp_path):
    *_, te = splits
    p = model.predict_adverse_proba(te)
    path = tmp_path / "seq2seq.joblib"
    model.save(path)
    m2 = SeqForecastModel.load(path)
    assert m2._net is None
    assert np.max(np.abs(p - m2.predict_adverse_proba(te))) == 0.0


@pytest.mark.parametrize("cell,kw", [
    ("lstm", {"bidirectional": True, "attention": True, "n_heads": 2}),
    ("tcn", {}),
    ("transformer", {"n_heads": 2}),
])
def test_backbone_variants(splits, tmp_path, cell, kw):
    """Each recent-architecture backbone fits, predicts a valid (N,H) probability,
    and round-trips through save/load."""
    tr, va, te = splits
    m = SeqForecastModel(cell, cell=cell, hidden=32, emb_dim=4,
                         max_epochs=2, batch_size=256, **kw).fit(tr, val=va)
    p = m.predict_adverse_proba(te)
    assert p.shape == (len(te.t0), te.y_reg.shape[1])
    assert (0.0 <= p).all() and (p <= 1.0).all()
    path = tmp_path / f"{cell}.joblib"
    m.save(path)
    assert np.max(np.abs(p - SeqForecastModel.load(path).predict_adverse_proba(te))) == 0.0
