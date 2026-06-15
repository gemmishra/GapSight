"""Run the complete GapSight daily alert workflow."""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.features.dataset_builder import save_training_dataset
from app.ml.inference.predictor import PredictionError, predict_latest
from app.ml.training.train_pipeline import train_all_models
from app.services.market_data_service import download_and_save_raw_ohlcv
from app.services.notification_service import NotificationError, send_prediction_alert
from app.services.symbol_resolver import resolve_symbol

EXPECTED_ERRORS = (
    FileNotFoundError,
    NotificationError,
    PredictionError,
    RuntimeError,
    ValueError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GapSight daily alert workflow.")
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the formatted alert without sending a notification.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models before generating and sending the alert.",
    )
    return parser.parse_args()


def _print_header(symbol: str, channel: str, retrain: bool, dry_run: bool) -> None:
    print("Starting GapSight daily alert workflow...")
    print(f"Symbol: {symbol}")
    print(f"Channel: {channel}")
    print(f"Retrain: {str(retrain).lower()}")
    print(f"Dry run: {str(dry_run).lower()}")
    print()


def _print_data_date_warning(prediction: dict[str, Any]) -> None:
    latest_data_date = prediction["latest_data_date"]
    if isinstance(latest_data_date, str):
        latest_data_date = date.fromisoformat(latest_data_date)

    if latest_data_date != date.today():
        print(f"Latest available data date: {latest_data_date.isoformat()}")
        print(
            "Note: Market data may not include today yet. This is normal before "
            "market close or on holidays/weekends."
        )
        print()


def run_daily_alert_workflow(
    symbol: str,
    channel: str = "discord",
    dry_run: bool = False,
    retrain: bool = False,
) -> dict[str, Any]:
    """Run the daily alert workflow and return execution details."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    _print_header(
        symbol=canonical_symbol,
        channel=channel,
        retrain=retrain,
        dry_run=dry_run,
    )

    print("Step 1/4: Downloading latest data...")
    raw_path = download_and_save_raw_ohlcv(canonical_symbol)
    print(f"Saved raw data: {raw_path}")

    print("Step 2/4: Building dataset...")
    dataset_path = save_training_dataset(canonical_symbol)
    print(f"Saved training dataset: {dataset_path}")

    training_result = None
    if retrain:
        print("Retrain enabled. Training models before prediction...")
        training_result = train_all_models(canonical_symbol)
        print(f"Saved model metadata: {training_result['metadata_path']}")

    print("Step 3/4: Generating prediction...")
    prediction = predict_latest(canonical_symbol)
    _print_data_date_warning(prediction)

    if dry_run:
        print("Step 4/4: Dry run enabled. Notification not sent.")
        print()
        print(prediction["formatted_alert"])
        print()
        print("Success.")
        return {
            "success": True,
            "symbol": canonical_symbol,
            "channel": channel,
            "dry_run": True,
            "retrain": retrain,
            "raw_path": raw_path,
            "dataset_path": dataset_path,
            "prediction": prediction,
            "training_result": training_result,
            "notification_result": None,
        }

    print("Step 4/4: Sending alert...")
    notification_result = send_prediction_alert(canonical_symbol, channel=channel)
    print(notification_result["message"])
    print()
    print("Success.")
    return {
        "success": notification_result["success"],
        "symbol": canonical_symbol,
        "channel": channel,
        "dry_run": False,
        "retrain": retrain,
        "raw_path": raw_path,
        "dataset_path": dataset_path,
        "prediction": prediction,
        "training_result": training_result,
        "notification_result": notification_result,
    }


def main() -> int:
    args = parse_args()

    try:
        result = run_daily_alert_workflow(
            symbol=args.symbol,
            channel=args.channel,
            dry_run=args.dry_run,
            retrain=args.retrain,
        )
    except EXPECTED_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
