"""Phase C+ — GPU PyTorch multi-task MLP rung: fit/predict contract, calibration,
device-safe (un)pickling, and Forecaster wiring. Skipped if torch/sklearn absent.
Runs on CPU."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from wx.ai.dataset import build_samples, temporal_split
from wx.ai.models import ModelForecaster
from wx.ai.torch_models import TorchMultiTaskModel
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path_factory.mktemp("torch") / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    base = datetime(2023, 1, 1, tzinfo=UTC)
    for i in range(24 * 30):  # 30 days hourly
        ts = base + timedelta(hours=i)
        fog = ts.hour in (5, 6, 7)
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?,'LEMD',?,?,?,?,?,?,?,?)""",
            [i, i, ts, 500 if fog else 9999, 200 if fog else None, 6, 200,
             8.0, 7.6 if fog else 1.0, "IFR" if fog else "VFR"],
        )
    df = build_samples(c, icaos=["LEMD"], leads=(1, 3, 6))
    tr, va, te = temporal_split(df, train_end="2023-01-21", val_end="2023-01-26")
    yield c, tr, va, te
    c.close()


@pytest.fixture(scope="module")
def model(data):
    _, tr, va, _ = data
    return TorchMultiTaskModel("mlp", max_epochs=3, batch_size=512).fit(tr, val_df=va)


def test_predict_contract_and_calibration(data, model):
    *_, te = data
    preds = model.predict(te)
    for col in ("pred_vis_m", "pred_ceiling_ft", "pred_wspd",
                "pred_wdir_sin", "pred_wdir_cos", "pred_cat", "pred_has_ceiling"):
        assert col in preds.columns
    assert preds["pred_cat"].isin(["VFR", "MVFR", "IFR", "LIFR"]).all()
    p = model.predict_adverse_proba(te)
    assert p.shape == (len(te),) and (0.0 <= p).all() and (p <= 1.0).all()
    assert model.calibrator is not None and 0.0 < model.adverse_threshold < 1.0
    assert model.predict_adverse_event(te).shape == (len(te),)


def test_save_load_roundtrip_is_exact(data, model, tmp_path):
    *_, te = data
    p = model.predict_adverse_proba(te)
    path = tmp_path / "mlp.joblib"
    model.save(path)
    m2 = TorchMultiTaskModel.load(path)
    assert m2._net is None  # rebuilt lazily, not pickled live (device-safe)
    assert np.max(np.abs(p - m2.predict_adverse_proba(te))) == 0.0


def test_prevailing_adverse_matches_calibrated_decision(data, model):
    """Regression guard: the forecaster's prevailing-category adverse flag must equal
    the CALIBRATED decision, never the class-weighted argmax (which over-calls adverse
    and silently wrecked the verifier-path HSS)."""
    *_, te = data
    hours = ModelForecaster(model).hours_from_feats(te)
    cat_adv = np.array([eh.prevailing["flight_category"] in ("IFR", "LIFR") for eh in hours])
    assert (cat_adv == np.asarray(model.predict_adverse_event(te))).all()


def test_wraps_as_forecaster(data, model):
    """Slots into ModelForecaster and emits a verifier-ready ExpectedHour timeline."""
    con = data[0]
    fc = ModelForecaster(model, name="model:mlp")
    hours = fc.generate(con, "LEMD", datetime(2023, 1, 28, 0, tzinfo=UTC),
                        datetime(2023, 1, 28, 1, tzinfo=UTC),
                        datetime(2023, 1, 28, 6, tzinfo=UTC))
    assert hours and hours[0].prevailing["flight_category"] in {"VFR", "MVFR", "IFR", "LIFR"}
