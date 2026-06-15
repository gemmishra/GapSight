"""Direction classifier training entry point."""

from typing import Any

from app.ml.training.training_utils import (
    chronological_train_test_split,
    ensure_model_dir_exists,
    get_feature_columns,
    load_training_dataset,
    processed_dataset_path,
    save_json,
)
from app.services.symbol_resolver import resolve_symbol


def _import_classifier_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_sample_weight
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model training dependencies are missing. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    return {
        "joblib": joblib,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "LabelEncoder": LabelEncoder,
        "compute_sample_weight": compute_sample_weight,
    }


def _classifier_candidates(dependencies: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        {
            "model_name": "LogisticRegression",
            "model": dependencies["Pipeline"](
                [
                    ("scaler", dependencies["StandardScaler"]()),
                    (
                        "classifier",
                        dependencies["LogisticRegression"](
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "uses_sample_weight": False,
        },
        {
            "model_name": "RandomForestClassifier",
            "model": dependencies["RandomForestClassifier"](
                n_estimators=300,
                max_depth=6,
                random_state=42,
                class_weight="balanced",
            ),
            "uses_sample_weight": False,
        },
    ]

    try:
        from xgboost import XGBClassifier

        candidates.append(
            {
                "model_name": "XGBClassifier",
                "model": XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                ),
                "uses_sample_weight": True,
            }
        )
    except Exception:
        candidates.append(
            {
                "model_name": "XGBClassifier",
                "model": None,
                "uses_sample_weight": True,
                "error": "XGBoost is not available",
            }
        )

    return candidates


def _evaluate_classifier(
    model_name: str,
    dependencies: dict[str, Any],
    y_test: Any,
    y_pred: Any,
    class_labels: list[str],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "accuracy": dependencies["accuracy_score"](y_test, y_pred),
        "macro_precision": dependencies["precision_score"](
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": dependencies["recall_score"](
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": dependencies["f1_score"](
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": dependencies["f1_score"](
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": dependencies["confusion_matrix"](
            y_test,
            y_pred,
            labels=list(range(len(class_labels))),
        ).tolist(),
        "classification_report": dependencies["classification_report"](
            y_test,
            y_pred,
            labels=list(range(len(class_labels))),
            target_names=class_labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def select_best_classifier_result(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select classifier by macro F1, then weighted F1, then accuracy."""
    successful_results = [result for result in results if result.get("status") == "success"]
    if not successful_results:
        raise ValueError("No classifier models trained successfully")

    return max(
        successful_results,
        key=lambda result: (
            result["metrics"]["macro_f1"],
            result["metrics"]["weighted_f1"],
            result["metrics"]["accuracy"],
        ),
    )


def train_gap_direction_classifier(
    symbol: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train and persist the gap direction classifier."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    df = load_training_dataset(canonical_symbol)
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for classifier training")

    train_df, test_df = chronological_train_test_split(df, test_size=test_size)
    dependencies = _import_classifier_dependencies()
    label_encoder = dependencies["LabelEncoder"]()
    y_train = label_encoder.fit_transform(train_df["gap_direction"])
    y_test = label_encoder.transform(test_df["gap_direction"])
    if len(label_encoder.classes_) < 2:
        raise ValueError("Classifier training requires at least two gap_direction classes")

    class_labels = label_encoder.classes_.tolist()
    x_train = train_df.loc[:, feature_columns]
    x_test = test_df.loc[:, feature_columns]
    sample_weight = dependencies["compute_sample_weight"](
        class_weight="balanced",
        y=y_train,
    )

    comparison_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    for candidate in _classifier_candidates(dependencies):
        model_name = candidate["model_name"]
        model = candidate.get("model")
        if model is None:
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": candidate.get("error", "Model unavailable"),
                }
            )
            continue

        try:
            if candidate["uses_sample_weight"]:
                model.fit(x_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            metrics = _evaluate_classifier(
                model_name=model_name,
                dependencies=dependencies,
                y_test=y_test,
                y_pred=y_pred,
                class_labels=class_labels,
            )
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "success",
                    "metrics": metrics,
                }
            )
            trained_models[model_name] = model
        except Exception as exc:
            comparison_results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    best_result = select_best_classifier_result(comparison_results)
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]
    metrics = {
        **best_result["metrics"],
        "model_type": best_model_name,
        "precision": best_result["metrics"]["macro_precision"],
        "recall": best_result["metrics"]["macro_recall"],
        "f1_score": best_result["metrics"]["weighted_f1"],
        "class_labels": class_labels,
        "selection_metric": "macro_f1",
    }
    comparison = {
        "symbol": canonical_symbol,
        "target": "gap_direction",
        "selection_rule": [
            "highest macro_f1",
            "highest weighted_f1",
            "highest accuracy",
        ],
        "class_labels": class_labels,
        "models": comparison_results,
        "best_model_name": best_model_name,
    }

    model_dir = ensure_model_dir_exists()
    model_path = model_dir / f"{canonical_symbol}_gap_direction_classifier.joblib"
    metrics_path = model_dir / f"{canonical_symbol}_classifier_metrics.json"
    comparison_path = model_dir / f"{canonical_symbol}_classifier_model_comparison.json"
    dependencies["joblib"].dump(
        {
            "model": best_model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "target": "gap_direction",
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
        "target": "gap_direction",
        "model_type": best_model_name,
    }


def train_classifier(symbol: str, test_size: float = 0.2) -> dict[str, Any]:
    """Backward-friendly alias for classifier training."""
    return train_gap_direction_classifier(symbol=symbol, test_size=test_size)
