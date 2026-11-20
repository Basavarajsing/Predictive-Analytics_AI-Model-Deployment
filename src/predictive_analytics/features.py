import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
]


def make_supervised_frame(daily_sales: pd.Series) -> pd.DataFrame:
    """Build one-step-ahead features using only sales observed before each date."""
    series = _validate_daily_series(daily_sales)
    features = pd.DataFrame(index=series.index)
    for lag in (1, 7, 14, 28):
        features[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    features["rolling_mean_7"] = shifted.rolling(7, min_periods=7).mean()
    features["rolling_mean_28"] = shifted.rolling(28, min_periods=28).mean()
    features["day_of_week"] = series.index.dayofweek
    features["day_of_month"] = series.index.day
    features["month"] = series.index.month
    features["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
    day_of_year = series.index.dayofyear
    angle = 2 * np.pi * day_of_year / 365.25
    features["day_of_year_sin"] = np.sin(angle)
    features["day_of_year_cos"] = np.cos(angle)
    features["target"] = series
    return features.dropna(subset=FEATURE_COLUMNS + ["target"])


def make_future_features(history: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
    """Build the feature row for a date after the observed daily history."""
    series = _validate_daily_series(history)
    date = pd.Timestamp(date).normalize()
    expected_date = series.index[-1] + pd.Timedelta(days=1)
    if date != expected_date:
        raise ValueError(f"Forecast dates must be consecutive; expected {expected_date.date()}.")
    row = {
        "lag_1": float(series.iloc[-1]),
        "lag_7": float(series.iloc[-7]),
        "lag_14": float(series.iloc[-14]),
        "lag_28": float(series.iloc[-28]),
        "rolling_mean_7": float(series.iloc[-7:].mean()),
        "rolling_mean_28": float(series.iloc[-28:].mean()),
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "is_weekend": int(date.dayofweek >= 5),
        "day_of_year_sin": float(np.sin(2 * np.pi * date.dayofyear / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * date.dayofyear / 365.25)),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _validate_daily_series(daily_sales: pd.Series) -> pd.Series:
    series = pd.Series(daily_sales, dtype=float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Daily sales must have a DatetimeIndex.")
    if series.index.has_duplicates or series.isna().any():
        raise ValueError("Daily sales must have unique dates and no missing values.")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if not series.index.equals(expected):
        raise ValueError("Daily sales must contain a consecutive calendar-day index.")
    if len(series) < 28:
        raise ValueError("At least 28 daily observations are required to build lag features.")
    return series
