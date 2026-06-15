"""Prediction orchestration for trained GapSight models."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.ml.features.data_loader import load_raw_ohlcv
from app.ml.features.feature_builder import build_basic_features
from app.services.alert_formatter import format_prediction_alert
from app.services.explanation_service import generate_basic_explanation
from app.services.prediction_presentation import enrich_prediction_presentation
from app.services.symbol_resolver import resolve_symbol


class PredictionError(Exception):
    """Base exception for prediction failures."""


class UnsupportedSymbolError(PredictionError):
    """Raised when a symbol cannot be resolved to a supported symbol."""


class MissingRawDataError(PredictionError):
    """Raised when raw OHLCV data is missing."""


class MissingModelArtifactError(PredictionError):
    """Raised when a required model artifact is missing."""


class MissingMetadataError(PredictionError):
    """Raised when model metadata is missing or invalid."""


class FeatureMismatchError(PredictionError):
    """Raised when runtime features do not match trained model metadata."""


def _artifact_path(symbol: str, filename_suffix: str) -> Path:
    return Path(settings.MODEL_DIR) / f"{symbol}_{filename_suffix}"


def _load_json(path: Path, missing_exception: type[PredictionError]) -> dict[str, Any]:
    if not path.exists():
        raise missing_exception(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_joblib(path: Path) -> Any:
    if not path.exists():
        raise MissingModelArtifactError(f"Required model file not found: {path}")

    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise MissingModelArtifactError(
            "joblib is required for inference. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return joblib.load(path)


def _load_prediction_artifacts(symbol: str) -> dict[str, Any]:
    classifier_model_path = _artifact_path(symbol, "gap_direction_classifier.joblib")
    regressor_model_path = _artifact_path(symbol, "gap_size_regressor.joblib")
    metadata_path = _artifact_path(symbol, "model_metadata.json")
    classifier_metrics_path = _artifact_path(symbol, "classifier_metrics.json")
    regressor_metrics_path = _artifact_path(symbol, "regressor_metrics.json")
    walk_forward_path = _artifact_path(symbol, "walk_forward_validation.json")

    return {
        "classifier_bundle": _load_joblib(classifier_model_path),
        "regressor_bundle": _load_joblib(regressor_model_path),
        "metadata": _load_json(metadata_path, MissingMetadataError),
        "classifier_metrics": _load_json(
            classifier_metrics_path,
            MissingModelArtifactError,
        ),
        "regressor_metrics": _load_json(
            regressor_metrics_path,
            MissingModelArtifactError,
        ),
        "walk_forward_validation": (
            _load_json(walk_forward_path, MissingModelArtifactError)
            if walk_forward_path.exists()
            else None
        ),
    }


def classifier_quality_status(classifier_metrics: dict[str, Any]) -> str:
    """Classify the current classifier baseline quality."""
    accuracy = float(classifier_metrics.get("accuracy", 0.0))
    weighted_f1 = float(
        classifier_metrics.get(
            "weighted_f1",
            classifier_metrics.get("f1_score", 0.0),
        )
    )
    if accuracy >= 0.60 and weighted_f1 >= 0.55:
        return "usable_baseline"
    return "weak_baseline"


def regressor_quality_status(
    regressor_metrics: dict[str, Any],
    walk_forward_validation: dict[str, Any] | None,
) -> str:
    """Classify the current regressor baseline quality."""
    if not bool(regressor_metrics.get("beats_baseline", False)):
        return "weak_baseline"

    summary = (walk_forward_validation or {}).get("summary", {})
    fold_count = int(summary.get("fold_count", 0) or 0)
    wins = int(summary.get("regressor_beats_baseline_fold_count", 0) or 0)
    if fold_count > 0 and wins == fold_count:
        return "usable_baseline"
    return "usable_but_unstable_baseline"


def build_model_quality(
    classifier_metrics: dict[str, Any],
    regressor_metrics: dict[str, Any],
    walk_forward_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build model quality flags from saved training metrics."""
    summary = (walk_forward_validation or {}).get("summary", {})
    return {
        "classifier_status": classifier_quality_status(classifier_metrics),
        "regressor_status": regressor_quality_status(
            regressor_metrics,
            walk_forward_validation,
        ),
        "classifier_accuracy": classifier_metrics.get("accuracy"),
        "classifier_weighted_f1": classifier_metrics.get(
            "weighted_f1",
            classifier_metrics.get("f1_score"),
        ),
        "regressor_mae": regressor_metrics.get("mae"),
        "regressor_beats_baseline": regressor_metrics.get("beats_baseline"),
        "walk_forward_regressor_baseline_wins": summary.get(
            "regressor_beats_baseline_fold_count"
        ),
        "walk_forward_fold_count": summary.get("fold_count"),
    }


def _validate_feature_columns(
    feature_row: pd.Series,
    metadata: dict[str, Any],
    classifier_bundle: dict[str, Any],
    regressor_bundle: dict[str, Any],
) -> list[str]:
    metadata_feature_columns = metadata.get("feature_columns")
    if not isinstance(metadata_feature_columns, list) or not metadata_feature_columns:
        raise MissingMetadataError("Metadata is missing non-empty feature_columns")

    missing_runtime_columns = [
        column for column in metadata_feature_columns if column not in feature_row.index
    ]
    if missing_runtime_columns:
        raise FeatureMismatchError(
            "Runtime feature row is missing trained feature columns: "
            + ", ".join(missing_runtime_columns)
        )

    for artifact_name, bundle in (
        ("classifier", classifier_bundle),
        ("regressor", regressor_bundle),
    ):
        artifact_feature_columns = bundle.get("feature_columns")
        if artifact_feature_columns and artifact_feature_columns != metadata_feature_columns:
            raise FeatureMismatchError(
                f"{artifact_name} feature columns do not match metadata feature_columns"
            )

    return metadata_feature_columns


def _inverse_direction_prediction(classifier_bundle: dict[str, Any], prediction: Any) -> str:
    label_encoder = classifier_bundle.get("label_encoder")
    if label_encoder is None:
        return str(prediction)
    return str(label_encoder.inverse_transform([int(prediction)])[0])


def _class_probabilities(
    classifier_bundle: dict[str, Any],
    feature_frame: pd.DataFrame,
    direction: str,
) -> tuple[dict[str, float], float | None]:
    model = classifier_bundle["model"]
    if not hasattr(model, "predict_proba"):
        return {}, None

    probabilities = model.predict_proba(feature_frame)[0]
    label_encoder = classifier_bundle.get("label_encoder")
    if label_encoder is not None:
        labels = [str(label) for label in label_encoder.classes_]
    else:
        labels = [str(index) for index in range(len(probabilities))]

    probability_map = {
        label: float(probability)
        for label, probability in zip(labels, probabilities, strict=False)
    }
    return probability_map, probability_map.get(direction, max(probability_map.values()))


def predict_latest(symbol: str) -> dict[str, Any]:
    """Predict latest BANKNIFTY gap direction and size using trained models."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise UnsupportedSymbolError(f"Unsupported symbol: {symbol}")

    try:
        raw_df = load_raw_ohlcv(canonical_symbol)
    except FileNotFoundError as exc:
        raise MissingRawDataError(str(exc)) from exc

    features_df = build_basic_features(raw_df)
    if features_df.empty:
        raise FeatureMismatchError(
            f"Not enough raw data to build latest features for {canonical_symbol}"
        )

    artifacts = _load_prediction_artifacts(canonical_symbol)
    classifier_bundle = artifacts["classifier_bundle"]
    regressor_bundle = artifacts["regressor_bundle"]
    metadata = artifacts["metadata"]
    classifier_metrics = artifacts["classifier_metrics"]
    regressor_metrics = artifacts["regressor_metrics"]
    walk_forward_validation = artifacts["walk_forward_validation"]

    latest_feature_row = features_df.iloc[-1]
    feature_columns = _validate_feature_columns(
        feature_row=latest_feature_row,
        metadata=metadata,
        classifier_bundle=classifier_bundle,
        regressor_bundle=regressor_bundle,
    )
    feature_frame = pd.DataFrame([latest_feature_row.loc[feature_columns].to_dict()])
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any(axis=None):
        raise FeatureMismatchError("Runtime feature row contains non-numeric values")

    classifier_model = classifier_bundle["model"]
    regressor_model = regressor_bundle["model"]
    direction_prediction = classifier_model.predict(feature_frame)[0]
    direction = _inverse_direction_prediction(classifier_bundle, direction_prediction)
    class_probabilities, direction_confidence = _class_probabilities(
        classifier_bundle=classifier_bundle,
        feature_frame=feature_frame,
        direction=direction,
    )

    predicted_gap_percent = float(regressor_model.predict(feature_frame)[0])
    previous_close = float(latest_feature_row["previous_close"])
    predicted_gap_points = previous_close * predicted_gap_percent / 100
    expected_open = previous_close + predicted_gap_points
    regressor_mae = float(regressor_metrics.get("mae", 0.0))
    expected_open_min = previous_close * (
        1 + (predicted_gap_percent - regressor_mae) / 100
    )
    expected_open_max = previous_close * (
        1 + (predicted_gap_percent + regressor_mae) / 100
    )

    prediction = {
        "symbol": canonical_symbol,
        "prediction_date": date.today(),
        "latest_data_date": pd.to_datetime(raw_df["date"].max()).date(),
        "previous_close": previous_close,
        "direction": direction,
        "direction_confidence": direction_confidence,
        "class_probabilities": class_probabilities,
        "predicted_gap_percent": predicted_gap_percent,
        "predicted_gap_points": predicted_gap_points,
        "expected_open": expected_open,
        "expected_open_min": expected_open_min,
        "expected_open_max": expected_open_max,
        "model_quality": build_model_quality(
            classifier_metrics=classifier_metrics,
            regressor_metrics=regressor_metrics,
            walk_forward_validation=walk_forward_validation,
        ),
        "reasons": [],
        "disclaimer": (
            "Research baseline only. This is not financial advice and does not "
            "guarantee market behavior."
        ),
    }
    prediction = enrich_prediction_presentation(prediction)
    prediction["formatted_alert"] = format_prediction_alert(prediction)
    prediction["reasons"] = generate_basic_explanation(prediction)
    return prediction
