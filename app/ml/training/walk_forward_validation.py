"""Lightweight walk-forward validation for GapSight models."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.training.train_classifier import _evaluate_classifier
from app.ml.training.train_regressor import _evaluate_regressor
from app.ml.training.training_utils import (
    ensure_model_dir_exists,
    get_feature_columns,
    load_training_dataset,
    save_json,
)
from app.services.symbol_resolver import resolve_symbol


def build_walk_forward_folds(
    row_count: int,
    min_train_ratio: float = 0.60,
    test_window_ratio: float = 0.05,
    step_ratio: float = 0.05,
) -> list[dict[str, int]]:
    """Create expanding-window fold boundaries by row position."""
    if row_count < 10:
        raise ValueError("At least 10 rows are required for walk-forward validation")

    min_train_size = max(1, int(row_count * min_train_ratio))
    test_window = max(1, int(row_count * test_window_ratio))
    step_size = max(1, int(row_count * step_ratio))

    folds: list[dict[str, int]] = []
    train_end = min_train_size
    while train_end < row_count:
        test_end = min(train_end + test_window, row_count)
        if test_end <= train_end:
            break
        folds.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            }
        )
        train_end += step_size

    return folds


def _import_walk_forward_dependencies() -> dict[str, Any]:
    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Walk-forward validation dependencies are missing. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    return {
        "DummyRegressor": DummyRegressor,
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "LabelEncoder": LabelEncoder,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    }


def _average_metric(folds: list[dict[str, Any]], section: str, metric: str) -> float:
    values = [fold[section][metric] for fold in folds]
    return float(np.mean(values)) if values else float("nan")


def run_walk_forward_validation(
    symbol: str,
    min_train_ratio: float = 0.60,
    test_window_ratio: float = 0.05,
    step_ratio: float = 0.05,
) -> dict[str, Any]:
    """Run lightweight expanding-window validation and save the report."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    df = load_training_dataset(canonical_symbol)
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for walk-forward validation")

    dependencies = _import_walk_forward_dependencies()
    folds = build_walk_forward_folds(
        row_count=len(df),
        min_train_ratio=min_train_ratio,
        test_window_ratio=test_window_ratio,
        step_ratio=step_ratio,
    )

    fold_results: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_df = df.iloc[fold["train_start"] : fold["train_end"]]
        test_df = df.iloc[fold["test_start"] : fold["test_end"]]
        x_train = train_df.loc[:, feature_columns]
        x_test = test_df.loc[:, feature_columns]

        label_encoder = dependencies["LabelEncoder"]()
        label_encoder.fit(df["gap_direction"])
        y_train_classifier = label_encoder.transform(train_df["gap_direction"])
        y_test_classifier = label_encoder.transform(test_df["gap_direction"])
        class_labels = label_encoder.classes_.tolist()

        classifier = dependencies["Pipeline"](
            [
                ("scaler", dependencies["StandardScaler"]()),
                (
                    "classifier",
                    dependencies["LogisticRegression"](
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        classifier.fit(x_train, y_train_classifier)
        classifier_pred = classifier.predict(x_test)
        classifier_metrics = _evaluate_classifier(
            model_name="LogisticRegression",
            dependencies=dependencies,
            y_test=y_test_classifier,
            y_pred=classifier_pred,
            class_labels=class_labels,
        )

        regressor = dependencies["Pipeline"](
            [
                ("scaler", dependencies["StandardScaler"]()),
                ("regressor", dependencies["Ridge"]()),
            ]
        )
        baseline_regressor = dependencies["DummyRegressor"](strategy="mean")
        y_train_regressor = train_df["gap_percent"]
        y_test_regressor = test_df["gap_percent"]
        regressor.fit(x_train, y_train_regressor)
        baseline_regressor.fit(x_train, y_train_regressor)
        regressor_pred = regressor.predict(x_test)
        baseline_pred = baseline_regressor.predict(x_test)
        regressor_metrics = _evaluate_regressor(
            model_name="Ridge",
            dependencies=dependencies,
            y_test=y_test_regressor,
            y_pred=regressor_pred,
            actual_directions=test_df["gap_direction"],
        )
        baseline_mae = dependencies["mean_absolute_error"](
            y_test_regressor,
            baseline_pred,
        )
        regressor_metrics["baseline_mae"] = baseline_mae
        regressor_metrics["beats_baseline"] = regressor_metrics["mae"] < baseline_mae

        fold_results.append(
            {
                "fold": fold_index,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start_date": train_df["date"].min().date().isoformat(),
                "train_end_date": train_df["date"].max().date().isoformat(),
                "test_start_date": test_df["date"].min().date().isoformat(),
                "test_end_date": test_df["date"].max().date().isoformat(),
                "classifier": classifier_metrics,
                "regressor": regressor_metrics,
            }
        )

    summary = {
        "fold_count": len(fold_results),
        "classifier_model_type": "LogisticRegression",
        "regressor_model_type": "Ridge",
        "avg_classifier_accuracy": _average_metric(
            fold_results,
            "classifier",
            "accuracy",
        ),
        "avg_classifier_macro_f1": _average_metric(
            fold_results,
            "classifier",
            "macro_f1",
        ),
        "avg_classifier_weighted_f1": _average_metric(
            fold_results,
            "classifier",
            "weighted_f1",
        ),
        "avg_regressor_mae": _average_metric(fold_results, "regressor", "mae"),
        "avg_regressor_rmse": _average_metric(fold_results, "regressor", "rmse"),
        "avg_regressor_r2": _average_metric(fold_results, "regressor", "r2"),
        "avg_regressor_baseline_mae": _average_metric(
            fold_results,
            "regressor",
            "baseline_mae",
        ),
        "regressor_beats_baseline_fold_count": sum(
            1 for fold in fold_results if fold["regressor"]["beats_baseline"]
        ),
    }
    validation = {
        "symbol": canonical_symbol,
        "method": "expanding_window",
        "min_train_ratio": min_train_ratio,
        "test_window_ratio": test_window_ratio,
        "step_ratio": step_ratio,
        "feature_columns": feature_columns,
        "summary": summary,
        "folds": fold_results,
    }

    validation_path = (
        ensure_model_dir_exists() / f"{canonical_symbol}_walk_forward_validation.json"
    )
    save_json(validation, str(validation_path))

    return {
        "validation": validation,
        "path": str(Path(validation_path)),
    }
