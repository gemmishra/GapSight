"""Training dataset assembly for GapSight."""

from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.ml.features.data_loader import load_raw_ohlcv
from app.ml.features.feature_builder import BASIC_FEATURE_COLUMNS, build_basic_features
from app.ml.features.label_builder import build_gap_labels
from app.services.symbol_resolver import resolve_symbol

LABEL_COLUMNS: tuple[str, ...] = ("gap_direction", "gap_percent", "gap_points")


def build_training_dataset(symbol: str) -> pd.DataFrame:
    """Build the aligned feature and label dataset for a supported symbol."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    raw_df = load_raw_ohlcv(canonical_symbol)
    features_df = build_basic_features(raw_df)
    labels_df = build_gap_labels(raw_df)

    dataset = features_df.merge(
        labels_df[["date", *LABEL_COLUMNS]],
        on="date",
        how="inner",
    )
    dataset.insert(1, "symbol", canonical_symbol)

    ordered_columns = [
        "date",
        "symbol",
        *BASIC_FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    ]
    return dataset.loc[:, ordered_columns].reset_index(drop=True)


def save_training_dataset(
    symbol: str,
    dataset: pd.DataFrame | None = None,
) -> str:
    """Save the processed training dataset and return the saved file path."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    output_df = dataset if dataset is not None else build_training_dataset(canonical_symbol)
    output_dir = Path(settings.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{canonical_symbol}_training.csv"
    output_df.to_csv(output_path, index=False)

    return str(output_path)
