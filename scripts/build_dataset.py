"""Build a processed GapSight training dataset from raw OHLCV data."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.features.dataset_builder import build_training_dataset, save_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GapSight training dataset.")
    parser.add_argument(
        "--symbol",
        default=settings.DEFAULT_SYMBOL,
        help=f"Symbol to build. Defaults to {settings.DEFAULT_SYMBOL}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        dataset = build_training_dataset(args.symbol)
        saved_path = save_training_dataset(args.symbol, dataset=dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved dataset: {saved_path}")
    print(f"Rows: {len(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
