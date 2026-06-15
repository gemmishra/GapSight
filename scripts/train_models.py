"""Train GapSight ML models from a processed training dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.training.train_pipeline import train_all_models


def _format_class_distribution(distribution: dict) -> str:
    return ", ".join(f"{label}={count}" for label, count in distribution.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GapSight models.")
    parser.add_argument(
        "--symbol",
        default=settings.DEFAULT_SYMBOL,
        help=f"Symbol to train. Defaults to {settings.DEFAULT_SYMBOL}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = train_all_models(args.symbol)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    classifier = result["classifier"]
    regressor = result["regressor"]
    diagnostics = result["diagnostics"]["diagnostics"]
    classifier_metrics = classifier["metrics"]
    regressor_metrics = regressor["metrics"]

    print(f"Dataset rows: {diagnostics['row_count']}")
    print(
        "Class distribution: "
        f"{_format_class_distribution(diagnostics['class_distribution'])}"
    )
    print(f"Selected classifier: {classifier['model_type']}")
    print(f"Classifier model: {classifier['model_path']}")
    print(f"Classifier macro F1: {classifier_metrics['macro_f1']:.4f}")
    print(f"Classifier weighted F1: {classifier_metrics['weighted_f1']:.4f}")
    print(f"Classifier accuracy: {classifier_metrics['accuracy']:.4f}")
    print(f"Selected regressor: {regressor['model_type']}")
    print(f"Regressor model: {regressor['model_path']}")
    print(f"Regressor MAE: {regressor_metrics['mae']:.4f}")
    print(f"Regressor RMSE: {regressor_metrics['rmse']:.4f}")
    print(f"Regressor R2: {regressor_metrics['r2']:.4f}")
    print(f"Baseline MAE: {regressor_metrics['baseline_mae']:.4f}")
    print(f"Beats baseline: {str(regressor_metrics['beats_baseline']).lower()}")
    if not regressor_metrics["beats_baseline"]:
        print("Warning: selected regressor does not beat the baseline mean regressor.")
    print(f"Metadata: {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
