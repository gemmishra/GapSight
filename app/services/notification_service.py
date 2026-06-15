"""Prediction notification service."""

from typing import Any

from app.core.config import settings
from app.ml.inference.predictor import predict_latest


class NotificationError(Exception):
    """Base exception for notification failures."""


class MissingNotificationConfigError(NotificationError):
    """Raised when a required notification setting is missing."""


class UnsupportedNotificationChannelError(NotificationError):
    """Raised when a notification channel is not supported yet."""


def send_discord_webhook(message: str) -> dict[str, Any]:
    """Send a plain-text message to the configured Discord webhook."""
    webhook_url = settings.DISCORD_WEBHOOK_URL.strip()
    if not webhook_url:
        raise MissingNotificationConfigError(
            "DISCORD_WEBHOOK_URL is not configured. Add it to .env before "
            "sending Discord alerts."
        )

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise MissingNotificationConfigError(
            "requests is required to send Discord alerts. Install dependencies "
            "with: pip install -r requirements.txt"
        ) from exc

    response = requests.post(
        webhook_url,
        json={"content": message},
        timeout=15,
    )
    success = 200 <= response.status_code < 300
    return {
        "success": success,
        "status_code": response.status_code,
        "message": (
            "Discord alert sent successfully."
            if success
            else f"Discord webhook returned status {response.status_code}."
        ),
    }


def send_prediction_alert(symbol: str, channel: str = "discord") -> dict[str, Any]:
    """Generate the latest prediction alert and send it to a notification channel."""
    normalized_channel = channel.strip().lower()
    if normalized_channel != "discord":
        raise UnsupportedNotificationChannelError(
            f"Unsupported notification channel: {channel}. Currently supported: discord"
        )

    prediction = predict_latest(symbol)
    result = send_discord_webhook(prediction["formatted_alert"])
    return {
        **result,
        "symbol": prediction["symbol"],
        "channel": normalized_channel,
    }
