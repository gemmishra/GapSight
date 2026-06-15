"""Label construction for gap prediction models."""

import numpy as np
import pandas as pd


def _require_label_columns(df: pd.DataFrame) -> None:
    required_columns = {"date", "open", "close"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Label input is missing required columns: " + ", ".join(missing_columns)
        )


def build_gap_labels(
    df: pd.DataFrame,
    flat_threshold_percent: float = 0.40,
) -> pd.DataFrame:
    """Build gap size and direction labels from OHLCV rows."""
    _require_label_columns(df)

    labels = df.sort_values("date").reset_index(drop=True).copy()
    if "previous_close" not in labels.columns:
        labels["previous_close"] = labels["close"].shift(1)

    labels["gap_points"] = labels["open"] - labels["previous_close"]
    labels["gap_percent"] = (labels["gap_points"] / labels["previous_close"]) * 100
    labels["gap_direction"] = np.select(
        [
            labels["gap_percent"] >= flat_threshold_percent,
            labels["gap_percent"] <= -flat_threshold_percent,
        ],
        ["GAP_UP", "GAP_DOWN"],
        default="FLAT",
    )

    return labels.dropna(subset=["previous_close"]).reset_index(drop=True)
