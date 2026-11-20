from typing import Any

import pandas as pd

from .features import FEATURE_COLUMNS, make_future_features


def forecast_values(
    artifact: dict[str, Any], history: pd.Series, horizon: int
) -> list[dict[str, Any]]:
    if not 1 <= horizon <= 30:
        raise ValueError("Forecast horizon must be between 1 and 30 days.")
    values = pd.Series(history, dtype=float).sort_index()
    if len(values) < 28 or not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("The forecast artifact requires at least 28 dated daily observations.")

    predictions: list[dict[str, Any]] = []
    for _ in range(horizon):
        forecast_date = values.index[-1] + pd.Timedelta(days=1)
        if artifact["model_name"] == "Seasonal Naive (7-day)":
            prediction = float(values.iloc[-7])
        else:
            features = make_future_features(values, forecast_date).loc[:, FEATURE_COLUMNS]
            prediction = float(artifact["estimator"].predict(features)[0])
        values.loc[forecast_date] = prediction
        predictions.append({"date": forecast_date.date().isoformat(), "predicted_sales": prediction})
    return predictions
