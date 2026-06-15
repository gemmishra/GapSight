import asyncio

import pytest
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

from app.api import routes
from app.main import app


def _openclaw_prediction() -> dict:
    return {
        "symbol": "BANKNIFTY",
        "direction_label": "Likely Flat / Neutral Opening",
        "confidence_label": "LOW",
        "predicted_gap_percent": 0.1,
        "predicted_gap_points": 55.0,
        "expected_open": 55229.32,
        "expected_open_min": 55008.67,
        "expected_open_max": 55449.98,
        "model_quality": {
            "classifier_status": "usable_baseline",
            "regressor_status": "usable_but_unstable_baseline",
        },
        "reliability_warning": ["Direction confidence is low."],
        "formatted_alert": "BANKNIFTY Pre-Market Prediction",
    }


def test_predict_alert_route_returns_plain_text(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "predict_latest",
        lambda _symbol: {"formatted_alert": "BANKNIFTY Pre-Market Prediction"},
    )

    response = asyncio.run(routes.predict_symbol_alert("BANKNIFTY"))

    assert isinstance(response, PlainTextResponse)
    assert response.media_type == "text/plain"
    assert response.body.decode() == "BANKNIFTY Pre-Market Prediction"


def test_send_alert_api_route_exists() -> None:
    schema = app.openapi()

    assert "/api/v1/alerts/{symbol}/send" in schema["paths"]
    assert "post" in schema["paths"]["/api/v1/alerts/{symbol}/send"]


def test_openclaw_json_endpoint_returns_compact_keys(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", False)
    monkeypatch.setattr(routes, "predict_latest", lambda _symbol: _openclaw_prediction())

    response = asyncio.run(routes.openclaw_predict_symbol("BANKNIFTY"))

    assert set(response) == {
        "symbol",
        "direction_label",
        "confidence_label",
        "predicted_gap_percent",
        "predicted_gap_points",
        "expected_open",
        "expected_open_min",
        "expected_open_max",
        "model_quality",
        "reliability_warning",
        "formatted_alert",
    }
    assert response["formatted_alert"] == "BANKNIFTY Pre-Market Prediction"


def test_openclaw_text_endpoint_returns_plain_text(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", False)
    monkeypatch.setattr(routes, "predict_latest", lambda _symbol: _openclaw_prediction())

    response = asyncio.run(routes.openclaw_predict_symbol_alert("BANKNIFTY"))

    assert isinstance(response, PlainTextResponse)
    assert response.media_type == "text/plain"
    assert response.body.decode() == "BANKNIFTY Pre-Market Prediction"


def test_openclaw_auth_disabled_allows_request(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", False)
    monkeypatch.setattr(routes, "predict_latest", lambda _symbol: _openclaw_prediction())

    response = asyncio.run(routes.openclaw_predict_symbol("BANKNIFTY"))

    assert response["symbol"] == "BANKNIFTY"


def test_openclaw_auth_enabled_rejects_missing_token(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", True)
    monkeypatch.setattr(routes.settings, "OPENCLAW_API_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.openclaw_predict_symbol("BANKNIFTY"))

    assert exc_info.value.status_code == 401
    assert "Missing or invalid" in exc_info.value.detail


def test_openclaw_auth_enabled_accepts_valid_bearer_token(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", True)
    monkeypatch.setattr(routes.settings, "OPENCLAW_API_TOKEN", "secret")
    monkeypatch.setattr(routes, "predict_latest", lambda _symbol: _openclaw_prediction())

    response = asyncio.run(
        routes.openclaw_predict_symbol(
            "BANKNIFTY",
            authorization="Bearer secret",
        )
    )

    assert response["symbol"] == "BANKNIFTY"


def test_normal_prediction_endpoint_does_not_require_openclaw_token(monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "ENABLE_OPENCLAW_AUTH", True)
    monkeypatch.setattr(routes.settings, "OPENCLAW_API_TOKEN", "secret")
    prediction = _openclaw_prediction()
    prediction.update(
        {
            "prediction_date": "2026-06-16",
            "latest_data_date": "2026-06-12",
            "previous_close": 55176.75,
            "direction": "FLAT",
            "direction_confidence": 0.34,
            "class_probabilities": {"FLAT": 0.34},
            "gap_interpretation": "Very small expected gap",
            "reasons": [],
            "disclaimer": "Research baseline only.",
        }
    )
    monkeypatch.setattr(routes, "predict_latest", lambda _symbol: prediction)

    response = asyncio.run(routes.predict_symbol("BANKNIFTY"))

    assert response.symbol == "BANKNIFTY"
