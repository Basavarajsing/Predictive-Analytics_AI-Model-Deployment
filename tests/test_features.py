import pandas as pd
import pytest

from src.predictive_analytics.features import FEATURE_COLUMNS, make_future_features, make_supervised_frame


def test_supervised_features_use_past_observations():
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    sales = pd.Series([float(i) for i in range(40)], index=index)
    frame = make_supervised_frame(sales)
    row = frame.loc[index[28]]
    assert row["lag_1"] == 27.0
    assert row["lag_7"] == 21.0
    assert row["lag_28"] == 0.0
    assert row["target"] == 28.0
    assert set(FEATURE_COLUMNS).issubset(frame.columns)


def test_future_feature_date_must_be_next_day():
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    history = pd.Series(range(30), index=index, dtype=float)
    with pytest.raises(ValueError, match="consecutive"):
        make_future_features(history, pd.Timestamp("2024-02-01"))


def test_make_supervised_frame_requires_complete_calendar():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
    sales = pd.Series([1.0, 2.0, 4.0], index=index)
    with pytest.raises(ValueError, match="consecutive"):
        make_supervised_frame(sales)
