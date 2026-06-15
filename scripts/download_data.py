"""Download raw GapSight OHLCV data from yfinance."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.market_data_service import download_and_save_raw_ohlcv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GapSight OHLCV data.")
    parser.add_argument(
        "--symbol",
        default=settings.DEFAULT_SYMBOL,
        help=f"Internal symbol or alias. Defaults to {settings.DEFAULT_SYMBOL}.",
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date in YYYY-MM-DD format. Defaults to 2015-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        saved_path = download_and_save_raw_ohlcv(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        saved_df = pd.read_csv(saved_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved raw data: {saved_path}")
    print(f"Rows downloaded: {len(saved_df)}")
    if not saved_df.empty:
        print(f"First date: {saved_df['date'].iloc[0]}")
        print(f"Last date: {saved_df['date'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
