from datetime import date

import pandas as pd
import pytest

from app.ml.inference import predictor
from app.ml.inference.predictor import (
    FeatureMismatchError,
    MissingModelArtifactError,
    UnsupportedSymbolError,
    build_model_quality,
    predict_latest,
)


class DummyLabelEncoder:
    classes_ = ["FLAT", "GAP_DOWN", "GAP_UP"]

    def inverse_transform(self, values):
        return [self.classes_[int(value)] for value in values]


class DummyClassifier:
    def predict(self, _features):
        return [2]

    def predict_proba(self, _features):
        return [[0.1, 0.2, 0.7]]


class DummyRegressor:
    def predict(self, _features):
        return [1.2]


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-14", "2026-06-15"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [100.0, 102.0],
            "volume": [1000, 1100],
        }
    )


def _feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-15"),
                "previous_close": 100.0,
                "ma_5": 99.0,
            }
        ]
    )


def _artifacts(feature_columns=None):
    columns = feature_columns or ["previous_close", "ma_5"]
    return {
        "classifier_bundle": {
            "model": DummyClassifier(),
            "label_encoder": DummyLabelEncoder(),
            "feature_columns": columns,
        },
        "regressor_bundle": {
            "model": DummyRegressor(),
            "feature_columns": columns,
        },
        "metadata": {"feature_columns": columns},
        "classifier_metrics": {"accuracy": 0.65, "weighted_f1": 0.60},
        "regressor_metrics": {"mae": 0.4, "beats_baseline": True},
        "walk_forward_validation": {
            "summary": {
                "fold_count": 8,
                "regressor_beats_baseline_fold_count": 4,
            }
        },
    }


def test_predict_latest_rejects_unsupported_symbol() -> None:
    with pytest.raises(UnsupportedSymbolError, match="Unsupported symbol"):
        predict_latest("NIFTY")


def test_predict_latest_missing_model_file_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(predictor, "load_raw_ohlcv", lambda _symbol: _raw_df())
    monkeypatch.setattr(predictor, "build_basic_features", lambda _df: _feature_df())
    monkeypatch.setattr("app.ml.inference.predictor.settings.MODEL_DIR", str(tmp_path))

    with pytest.raises(MissingModelArtifactError, match="Required model file not found"):
        predict_latest("BANKNIFTY")


def test_model_quality_flags_unstable_regressor() -> None:
    quality = build_model_quality(
        classifier_metrics={"accuracy": 0.65, "weighted_f1": 0.60},
        regressor_metrics={"mae": 0.4, "beats_baseline": True},
        walk_forward_validation={
            "summary": {
                "fold_count": 8,
                "regressor_beats_baseline_fold_count": 4,
            }
        },
    )

    assert quality["classifier_status"] == "usable_baseline"
    assert quality["regressor_status"] == "usable_but_unstable_baseline"


def test_predict_latest_response_contains_required_keys(monkeypatch) -> None:
    monkeypatch.setattr(predictor, "load_raw_ohlcv", lambda _symbol: _raw_df())
    monkeypatch.setattr(predictor, "build_basic_features", lambda _df: _feature_df())
    monkeypatch.setattr(
        predictor,
        "_load_prediction_artifacts",
        lambda _symbol: _artifacts(),
    )

    response = predict_latest("BANK NIFTY")

    required_keys = {
        "symbol",
        "prediction_date",
        "latest_data_date",
        "previous_close",
        "direction",
        "direction_confidence",
        "confidence_label",
        "direction_label",
        "class_probabilities",
        "predicted_gap_percent",
        "predicted_gap_points",
        "gap_interpretation",
        "expected_open",
        "expected_open_min",
        "expected_open_max",
        "model_quality",
        "reliability_warning",
        "formatted_alert",
        "reasons",
        "disclaimer",
    }
    assert required_keys.issubset(response)
    assert response["symbol"] == "BANKNIFTY"
    assert response["prediction_date"] == date.today()
    assert response["latest_data_date"].isoformat() == "2026-06-15"
    assert response["direction"] == "GAP_UP"
    assert response["direction_confidence"] == 0.7
    assert response["confidence_label"] == "HIGH"
    assert response["direction_label"] == "Likely Gap Up"
    assert response["gap_interpretation"] == "Strong expected gap"
    assert response["predicted_gap_points"] == pytest.approx(1.2)
    assert response["expected_open"] == pytest.approx(101.2)
    assert response["model_quality"]["regressor_status"] == (
        "usable_but_unstable_baseline"
    )
    assert "BANKNIFTY Pre-Market Prediction" in response["formatted_alert"]


def test_predict_latest_feature_mismatch_gives_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(predictor, "load_raw_ohlcv", lambda _symbol: _raw_df())
    monkeypatch.setattr(predictor, "build_basic_features", lambda _df: _feature_df())
    monkeypatch.setattr(
        predictor,
        "_load_prediction_artifacts",
        lambda _symbol: _artifacts(feature_columns=["previous_close", "missing_feature"]),
    )

    with pytest.raises(FeatureMismatchError, match="missing trained feature columns"):
        predict_latest("BANKNIFTY")
