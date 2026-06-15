"""Dataset diagnostics for model training runs."""

from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.training.training_utils import (
    ensure_model_dir_exists,
    get_feature_columns,
    load_training_dataset,
    processed_dataset_path,
    save_json,
)
from app.services.symbol_resolver import resolve_symbol


def build_dataset_diagnostics(
    symbol: str,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a compact diagnostics summary for a processed dataset."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    dataset = df.copy() if df is not None else load_training_dataset(canonical_symbol)
    feature_columns = get_feature_columns(dataset)
    class_distribution = (
        dataset["gap_direction"].value_counts(dropna=False).to_dict()
        if "gap_direction" in dataset.columns
        else {}
    )

    diagnostics: dict[str, Any] = {
        "symbol": canonical_symbol,
        "training_dataset_path": str(processed_dataset_path(canonical_symbol)),
        "row_count": len(dataset),
        "column_count": len(dataset.columns),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_distribution": class_distribution,
        "missing_values": dataset.isna().sum().to_dict(),
    }

    if "date" in dataset.columns and not dataset.empty:
        diagnostics["date_min"] = dataset["date"].min().date().isoformat()
        diagnostics["date_max"] = dataset["date"].max().date().isoformat()

    if "gap_percent" in dataset.columns:
        diagnostics["gap_percent_summary"] = (
            dataset["gap_percent"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .to_dict()
        )

    return diagnostics


def save_dataset_diagnostics(symbol: str) -> dict[str, Any]:
    """Save dataset diagnostics and return both data and path."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    diagnostics = build_dataset_diagnostics(canonical_symbol)
    diagnostics_path = (
        ensure_model_dir_exists() / f"{canonical_symbol}_dataset_diagnostics.json"
    )
    save_json(diagnostics, str(diagnostics_path))

    return {
        "diagnostics": diagnostics,
        "path": str(Path(diagnostics_path)),
    }
