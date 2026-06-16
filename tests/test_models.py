"""Phase C — model + ModelForecaster wiring (trains on synthetic data, predicts,
emits a verifier-ready ExpectedHour timeline). Skipped if scikit-learn is absent."""

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sklearn")

from wx.ai.dataset import build_samples
from wx.ai.models import ModelForecaster, MultiTaskModel
from wx.db.connection import connect

UTC = timezone.utc


@pytest.fixture
def con(tmp_path):
    from wx.db.connection import SCHEMA_PATH

    c = connect(tmp_path / "t.duckdb")
    c.execute(SCHEMA_PATH.read_text())
    c.execute("INSERT INTO stations VALUES ('LEMD','Madrid',40.5,-3.5,610,'peninsula')")
    base = datetime(2023, 1, 1, tzinfo=UTC)
    for i in range(24 * 20):  # 20 days hourly
        ts = base + timedelta(hours=i)
        fog = ts.hour in (5, 6, 7)
        c.execute(
            """INSERT INTO metar_obs (id, raw_metar_id, icao, observed_at, vis_m,
               ceiling_ft, wind_spd_kt, wind_dir_deg, temp_c, dewpoint_c, flight_category)
               VALUES (?,?,'LEMD',?,?,?,?,?,?,?,?)""",
            [i, i, ts, 500 if fog else 9999, 200 if fog else None, 6, 200,
             8.0, 7.6 if fog else 1.0, "IFR" if fog else "VFR"],
        )
    yield c
    c.close()


def test_model_trains_and_forecasts(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3, 6))
    model = MultiTaskModel("linreg").fit(df)
    fc = ModelForecaster(model)

    issued = datetime(2023, 1, 18, 0, tzinfo=UTC)
    vf = datetime(2023, 1, 18, 1, tzinfo=UTC)
    vt = datetime(2023, 1, 18, 9, tzinfo=UTC)
    hours = fc.generate(con, "LEMD", issued, vf, vt)

    assert len(hours) > 0
    eh = hours[0]
    assert set(eh.prevailing) >= {"vis_m", "ceiling_ft", "wind_spd_kt",
                                  "wind_dir_deg", "flight_category"}
    assert eh.prevailing["flight_category"] in {"VFR", "MVFR", "IFR", "LIFR"}
    assert eh.prevailing["vis_m"] >= 0


def test_predict_columns(con):
    df = build_samples(con, icaos=["LEMD"], leads=(1, 3))
    model = MultiTaskModel("gbm").fit(df)
    preds = model.predict(df.head(5))
    for col in ("pred_vis_m", "pred_ceiling_ft", "pred_has_ceiling",
                "pred_wspd", "pred_cat"):
        assert col in preds.columns
