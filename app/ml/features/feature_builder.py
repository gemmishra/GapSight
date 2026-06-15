"""Feature construction for gap prediction models."""

import numpy as np
import pandas as pd

from app.ml.features.data_loader import REQUIRED_OHLCV_COLUMNS

BASIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "previous_close",
    "previous_open",
    "previous_high",
    "previous_low",
    "previous_volume",
    "previous_day_return_percent",
    "previous_day_range_percent",
    "previous_candle_body_percent",
    "close_position_in_range",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5",
    "volatility_10",
)


def _require_ohlcv_columns(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Feature input is missing required columns: "
            + ", ".join(missing_columns)
        )


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features available before the current day's open."""
    _require_ohlcv_columns(df)

    source = df.sort_values("date").reset_index(drop=True).copy()
    features = pd.DataFrame({"date": source["date"]})

    features["previous_close"] = source["close"].shift(1)
    features["previous_open"] = source["open"].shift(1)
    features["previous_high"] = source["high"].shift(1)
    features["previous_low"] = source["low"].shift(1)
    features["previous_volume"] = source["volume"].shift(1)

    day_return_percent = ((source["close"] - source["open"]) / source["open"]) * 100
    day_range_percent = ((source["high"] - source["low"]) / source["close"]) * 100
    candle_body_percent = (
        (source["close"] - source["open"]).abs() / source["open"]
    ) * 100
    range_points = (source["high"] - source["low"]).replace(0, np.nan)
    close_position = (source["close"] - source["low"]) / range_points

    features["previous_day_return_percent"] = day_return_percent.shift(1)
    features["previous_day_range_percent"] = day_range_percent.shift(1)
    features["previous_candle_body_percent"] = candle_body_percent.shift(1)
    features["close_position_in_range"] = close_position.shift(1)

    shifted_close = source["close"].shift(1)
    features["ma_5"] = shifted_close.rolling(window=5).mean()
    features["ma_10"] = shifted_close.rolling(window=10).mean()
    features["ma_20"] = shifted_close.rolling(window=20).mean()

    daily_return_percent = source["close"].pct_change() * 100
    shifted_return = daily_return_percent.shift(1)
    features["volatility_5"] = shifted_return.rolling(window=5).std()
    features["volatility_10"] = shifted_return.rolling(window=10).std()

    return features.dropna(subset=BASIC_FEATURE_COLUMNS).reset_index(drop=True)
