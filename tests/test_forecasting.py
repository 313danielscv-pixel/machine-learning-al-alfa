import numpy as np
import pandas as pd
import pytest

from ml_al_alfa.forecasting import (
    _calendar_features,
    _make_features,
    forecast_to_date,
    train_forecast_and_save,
)


def make_daily_frame(length: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "births": 1000 + np.arange(length) * 0.5 + dates.dayofweek * 25,
        }
    )


def test_forecast_features_use_only_past_values():
    frame = make_daily_frame(40)
    features, target = _make_features(frame, "births", [])
    first_date = features.index[0]
    expected_target = frame.loc[frame["date"] == first_date, "births"].iloc[0]
    previous_day = frame.loc[frame["date"] == first_date - pd.Timedelta(days=1), "births"].iloc[0]
    assert target.loc[first_date] == expected_target
    assert features.loc[first_date, "lag_1"] == previous_day


def test_forecasting_uses_chronological_test_and_can_predict_future(tmp_path):
    bundle = train_forecast_and_save(
        make_daily_frame(),
        dataset_name="births",
        output_path=tmp_path / "births_model.joblib",
    )
    predictions = forecast_to_date(
        bundle,
        variant_label="Calendario y datos pasados",
        days_ahead=3,
    )
    last_history_date = pd.Timestamp(bundle["forecast"]["latest_date"])
    assert bundle["task"] == "forecasting"
    assert bundle["baseline_name"] == "Baseline 7 días"
    assert bundle["split_method"].startswith("cronológico")
    assert len(predictions) == 3
    assert predictions[0][0] == last_history_date + pd.Timedelta(days=1)
    assert all(np.isfinite(value) for _, value in predictions)


def test_forecast_rejects_missing_daily_dates():
    frame = make_daily_frame(50).drop(index=20)
    with pytest.raises(ValueError, match="faltan 1 fechas"):
        _make_features(frame, "births", [])


def test_birth_features_mark_federal_holidays_and_friday_the_13th():
    holiday = _calendar_features(pd.Timestamp("2024-07-04"), "births")
    friday_13 = _calendar_features(pd.Timestamp("2024-09-13"), "births")
    assert holiday["is_us_federal_holiday"] == 1
    assert friday_13["is_friday_13"] == 1
