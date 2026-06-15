"""Shared utilities for GapSight model training."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.symbol_resolver import resolve_symbol

TARGET_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "symbol",
        "gap_direction",
        "gap_percent",
        "gap_points",
    }
)


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    return value


def processed_dataset_path(symbol: str) -> Path:
    """Return the expected processed dataset path for a supported symbol."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    return Path(settings.PROCESSED_DATA_DIR) / f"{canonical_symbol}_training.csv"


def load_training_dataset(symbol: str) -> pd.DataFrame:
    """Load the processed training dataset for a supported symbol."""
    dataset_path = processed_dataset_path(symbol)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed training dataset not found for {symbol}: {dataset_path}. "
            "Build it first with: python scripts/build_dataset.py --symbol BANKNIFTY"
        )

    df = pd.read_csv(dataset_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model feature columns while excluding targets and metadata."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in TARGET_COLUMNS]


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split oldest rows into train and newest rows into test."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(df) < 2:
        raise ValueError("At least two rows are required for train/test split")

    sorted_df = df.sort_values("date").reset_index(drop=True) if "date" in df else df
    split_index = int(len(sorted_df) * (1 - test_size))
    split_index = max(1, min(split_index, len(sorted_df) - 1))

    train_df = sorted_df.iloc[:split_index].reset_index(drop=True)
    test_df = sorted_df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def save_json(data: dict, path: str) -> None:
    """Save JSON data, converting numpy scalar types when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_json_safe(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def ensure_model_dir_exists() -> Path:
    """Create and return the model artifact directory."""
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
