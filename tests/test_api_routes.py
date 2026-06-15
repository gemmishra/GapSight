import asyncio

from fastapi.responses import PlainTextResponse

from app.api import routes
from app.main import app


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
