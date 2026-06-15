from app.ml.training.train_classifier import select_best_classifier_result
from app.ml.training.train_regressor import (
    beats_baseline,
    get_baseline_mae,
    select_best_regressor_result,
)


def test_classifier_selector_chooses_higher_macro_f1() -> None:
    results = [
        {
            "model_name": "ModelA",
            "status": "success",
            "metrics": {
                "macro_f1": 0.40,
                "weighted_f1": 0.90,
                "accuracy": 0.95,
            },
        },
        {
            "model_name": "ModelB",
            "status": "success",
            "metrics": {
                "macro_f1": 0.50,
                "weighted_f1": 0.60,
                "accuracy": 0.70,
            },
        },
    ]

    assert select_best_classifier_result(results)["model_name"] == "ModelB"


def test_regressor_comparison_includes_baseline() -> None:
    results = [
        {
            "model_name": "Ridge",
            "status": "success",
            "is_baseline": False,
            "metrics": {"mae": 0.5, "rmse": 0.7, "r2": 0.1},
        },
        {
            "model_name": "BaselineMeanRegressor",
            "status": "success",
            "is_baseline": True,
            "metrics": {"mae": 0.6, "rmse": 0.8, "r2": 0.0},
        },
    ]

    assert get_baseline_mae(results) == 0.6
    assert select_best_regressor_result(results)["model_name"] == "Ridge"


def test_beats_baseline_flag() -> None:
    assert beats_baseline(best_mae=0.49, baseline_mae=0.50) is True
    assert beats_baseline(best_mae=0.50, baseline_mae=0.50) is False
