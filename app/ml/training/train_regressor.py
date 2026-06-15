"""Gap-size regressor training entry point."""

from typing import Any

import numpy as np
import pandas as pd

from app.ml.training.training_utils import (
    chronological_train_test_split,
    ensure_model_dir_exists,
    get_feature_columns,
    load_training_dataset,
    processed_dataset_path,
    save_json,
)
from app.services.symbol_resolver import resolve_symbol


def _import_regressor_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model training dependencies are missing. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    return {
        "joblib": joblib,
        "DummyRegressor": DummyRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    }


def _regressor_candidates(dependencies: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        {
            "model_name": "Ridge",
            "model": dependencies["Pipeline"](
                [
                    ("scaler", dependencies["StandardScaler"]()),
                    ("regressor", dependencies["Ridge"]()),
                ]
            ),
            "is_baseline": False,
        },
        {
            "model_name": "RandomForestRegressor",
            "model": dependencies["RandomForestRegressor"](
                n_estimators=300,
                max_depth=6,
                random_state=42,
            ),
            "is_baseline": False,
        },
    ]

    try:
        from xgboost import XGBRegressor

        candidates.append(
            {
                "model_name": "XGBRegressor",
                "model": XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                ),
                "is_baseline": False,
            }
        )
    except Exception:
        candidates.append(
            {
                "model_name": "XGBRegressor",
                "model": None,
                "is_baseline": False,
                "error": "XGBoost is not available",
            }
        )

    candidates.append(
        {
            "model_name": "BaselineMeanRegressor",
            "model": dependencies["DummyRegressor"](strategy="mean"),
            "is_baseline": True,
        }
    )
    return candidates


def _gap_direction_from_percent(value: float, threshold_percent: float = 0.40) -> str:
    if value >= threshold_percent:
        return "GAP_UP"
    if value <= -threshold_percent:
        return "GAP_DOWN"
    return "FLAT"


def _evaluate_regressor(
    model_name: str,
    dependencies: dict[str, Any],
    y_test: pd.Series,
    y_pred: Any,
    actual_directions: pd.Series,
) -> dict[str, Any]:
    mse = dependencies["mean_squared_error"](y_test, y_pred)
    predicted_directions = [
        _gap_direction_from_percent(float(value)) for value in y_pred
    ]
    directional_accuracy = (
        pd.Series(predicted_directions, index=actual_directions.index)
        .eq(actual_directions)
        .mean()
    )
    return {
        "model_name": model_name,
        "mae": dependencies["mean_absolute_error"](y_test, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": dependencies["r2_score"](y_test, y_pred),
        "directional_accuracy_from_regression": directional_accuracy,
    }


def select_best_regressor_result(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select best non-baseline regressor by MAE, RMSE, then R2."""
    successful_real_results = [
        result
        for result in results
        if result.get("status") == "success" and not result.get("is_baseline", False)
    ]
    if not successful_real_results:
        raise ValueError("No real regressor models trained successfully")

    return min(
        successful_real_results,
        key=lambda result: (
            result["metrics"]["mae"],
            result["metrics"]["rmse"],
            -result["metrics"]["r2"],
        ),
    )


def get_baseline_mae(results: list[dict[str, Any]]) -> float:
    """Return MAE from the successful baseline mean regressor result."""
    for result in results:
        if result.get("status") == "success" and result.get("is_baseline", False):
            return float(result["metrics"]["mae"])
    raise ValueError("Baseline regressor result is missing")


def beats_baseline(best_mae: float, baseline_mae: float) -> bool:
    """Return whether the selected real regressor improves on baseline MAE."""
    return best_mae < baseline_mae


def train_gap_size_regressor(
    symbol: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train and persist the gap percent regressor."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    df = load_training_dataset(canonical_symbol)
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for regressor training")

    train_df, test_df = chronological_train_test_split(df, test_size=test_size)
    dependencies = _import_regressor_dependencies()
    x_train = train_df.loc[:, feature_columns]
    x_test = test_df.loc[:, feature_columns]
    y_train = train_df["gap_percent"]
    y_test = test_df["gap_percent"]

    comparison_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    for candidate in _regressor_candidates(dependencies):
        model_name = candidate["model_name"]
        model = candidate.get("model")
        is_baseline = candidate["is_baseline"]
        if model is None:
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "is_baseline": is_baseline,
                    "error": candidate.get("error", "Model unavailable"),
                }
            )
            continue

        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            model_metrics = _evaluate_regressor(
                model_name=model_name,
                dependencies=dependencies,
                y_test=y_test,
                y_pred=y_pred,
                actual_directions=test_df["gap_direction"],
            )
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "success",
                    "is_baseline": is_baseline,
                    "metrics": model_metrics,
                }
            )
            trained_models[model_name] = model
        except Exception as exc:
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "is_baseline": is_baseline,
                    "error": str(exc),
                }
            )

    best_result = select_best_regressor_result(comparison_results)
    best_model_name = best_result["model_name"]
    baseline_mae = get_baseline_mae(comparison_results)
    best_beats_baseline = beats_baseline(
        best_mae=float(best_result["metrics"]["mae"]),
        baseline_mae=baseline_mae,
    )
    metrics = {
        **best_result["metrics"],
        "model_type": best_model_name,
        "baseline_mae": baseline_mae,
        "beats_baseline": best_beats_baseline,
        "selection_metric": "mae",
    }
    comparison = {
        "symbol": canonical_symbol,
        "target": "gap_percent",
        "selection_rule": [
            "lowest mae",
            "lowest rmse",
            "highest r2",
        ],
        "models": comparison_results,
        "best_model_name": best_model_name,
        "baseline_mae": baseline_mae,
        "beats_baseline": best_beats_baseline,
    }

    model_dir = ensure_model_dir_exists()
    model_path = model_dir / f"{canonical_symbol}_gap_size_regressor.joblib"
    metrics_path = model_dir / f"{canonical_symbol}_regressor_metrics.json"
    comparison_path = model_dir / f"{canonical_symbol}_regressor_model_comparison.json"
    dependencies["joblib"].dump(
        {
            "model": trained_models[best_model_name],
            "feature_columns": feature_columns,
            "target": "gap_percent",
            "model_type": best_model_name,
        },
        model_path,
    )
    save_json(metrics, str(metrics_path))
    save_json(comparison, str(comparison_path))

    return {
        "symbol": canonical_symbol,
        "dataset_path": str(processed_dataset_path(canonical_symbol)),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
        "metrics": metrics,
        "comparison": comparison,
        "feature_columns": feature_columns,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target": "gap_percent",
        "model_type": best_model_name,
    }


def train_regressor(symbol: str, test_size: float = 0.2) -> dict[str, Any]:
    """Backward-friendly alias for regressor training."""
    return train_gap_size_regressor(symbol=symbol, test_size=test_size)
