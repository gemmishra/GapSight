import pytest

from app.services import notification_service
from app.services.notification_service import (
    MissingNotificationConfigError,
    UnsupportedNotificationChannelError,
    send_discord_webhook,
    send_prediction_alert,
)


def test_missing_discord_webhook_gives_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.notification_service.settings.DISCORD_WEBHOOK_URL",
        "",
    )

    with pytest.raises(MissingNotificationConfigError, match="DISCORD_WEBHOOK_URL"):
        send_discord_webhook("hello")


def test_unsupported_channel_gives_clear_error() -> None:
    with pytest.raises(UnsupportedNotificationChannelError, match="Unsupported"):
        send_prediction_alert("BANKNIFTY", channel="telegram")


def test_send_prediction_alert_uses_formatted_alert(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        notification_service,
        "predict_latest",
        lambda _symbol: {
            "symbol": "BANKNIFTY",
            "formatted_alert": "real formatted alert",
        },
    )

    def fake_send_discord_webhook(message: str) -> dict:
        captured["message"] = message
        return {
            "success": True,
            "status_code": 204,
            "message": "Discord alert sent successfully.",
        }

    monkeypatch.setattr(
        notification_service,
        "send_discord_webhook",
        fake_send_discord_webhook,
    )

    result = send_prediction_alert("BANKNIFTY", channel="discord")

    assert captured["message"] == "real formatted alert"
    assert result["success"] is True
    assert result["symbol"] == "BANKNIFTY"
    assert result["channel"] == "discord"


def test_send_discord_webhook_posts_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.notification_service.settings.DISCORD_WEBHOOK_URL",
        "https://discord.example/webhook",
    )

    captured = {}

    class FakeResponse:
        status_code = 204

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    result = send_discord_webhook("alert text")

    assert captured == {
        "url": "https://discord.example/webhook",
        "json": {"content": "alert text"},
        "timeout": 15,
    }
    assert result["success"] is True
    assert result["status_code"] == 204
