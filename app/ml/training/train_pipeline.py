"""End-to-end model training pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ml.training.dataset_diagnostics import save_dataset_diagnostics
from app.ml.training.train_classifier import train_gap_direction_classifier
from app.ml.training.train_regressor import train_gap_size_regressor
from app.ml.training.training_utils import ensure_model_dir_exists, save_json
from app.ml.training.walk_forward_validation import run_walk_forward_validation
from app.services.symbol_resolver import resolve_symbol


def train_all_models(symbol: str) -> dict[str, Any]:
    """Train all currently supported models for a symbol."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    diagnostics_result = save_dataset_diagnostics(canonical_symbol)
    classifier_result = train_gap_direction_classifier(canonical_symbol)
    regressor_result = train_gap_size_regressor(canonical_symbol)
    walk_forward_result = run_walk_forward_validation(canonical_symbol)

    metadata = {
        "symbol": canonical_symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset_path": classifier_result["dataset_path"],
        "classifier_model_path": classifier_result["model_path"],
        "regressor_model_path": regressor_result["model_path"],
        "classifier_metrics_path": classifier_result["metrics_path"],
        "regressor_metrics_path": regressor_result["metrics_path"],
        "classifier_comparison_path": classifier_result["comparison_path"],
        "regressor_comparison_path": regressor_result["comparison_path"],
        "diagnostics_path": diagnostics_result["path"],
        "walk_forward_validation_path": walk_forward_result["path"],
        "selected_classifier_type": classifier_result["model_type"],
        "selected_regressor_type": regressor_result["model_type"],
        "feature_columns": classifier_result["feature_columns"],
        "classifier_metrics": classifier_result["metrics"],
        "regressor_metrics": regressor_result["metrics"],
        "beats_baseline": regressor_result["metrics"]["beats_baseline"],
        "classifier_target": "gap_direction",
        "regressor_target": "gap_percent",
        "train_rows": classifier_result["train_rows"],
        "test_rows": classifier_result["test_rows"],
    }

    metadata_path = (
        ensure_model_dir_exists() / f"{canonical_symbol}_model_metadata.json"
    )
    save_json(metadata, str(metadata_path))

    return {
        "symbol": canonical_symbol,
        "diagnostics": diagnostics_result,
        "classifier": classifier_result,
        "regressor": regressor_result,
        "walk_forward": walk_forward_result,
        "metadata": metadata,
        "metadata_path": str(Path(metadata_path)),
    }
