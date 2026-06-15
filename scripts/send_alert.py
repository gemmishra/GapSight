"""Send a GapSight prediction alert to a notification channel."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.inference.predictor import PredictionError
from app.services.notification_service import NotificationError, send_prediction_alert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a GapSight prediction alert.")
    parser.add_argument(
        "--symbol",
        default=settings.DEFAULT_SYMBOL,
        help=f"Symbol to alert. Defaults to {settings.DEFAULT_SYMBOL}.",
    )
    parser.add_argument(
        "--channel",
        default="discord",
        help="Notification channel. Currently supported: discord.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = send_prediction_alert(symbol=args.symbol, channel=args.channel)
    except (NotificationError, PredictionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result["message"])
    print(f"Channel: {result['channel']}")
    print(f"Symbol: {result['symbol']}")
    print(f"Status code: {result['status_code']}")
    print(f"Success: {str(result['success']).lower()}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
