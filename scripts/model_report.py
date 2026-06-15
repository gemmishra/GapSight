"""Print a GapSight model training report from saved artifacts."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.training.training_utils import ensure_model_dir_exists
from app.services.symbol_resolver import resolve_symbol


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summarize_models(comparison: dict[str, Any], metric_names: list[str]) -> None:
    for model in comparison["models"]:
        if model["status"] != "success":
            print(f"- {model['model_name']}: failed ({model.get('error', 'unknown')})")
            continue
        metrics = model["metrics"]
        metric_text = ", ".join(
            f"{metric}={_fmt(metrics[metric])}"
            for metric in metric_names
            if metric in metrics
        )
        print(f"- {model['model_name']}: {metric_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a GapSight model report.")
    parser.add_argument(
        "--symbol",
        default=settings.DEFAULT_SYMBOL,
        help=f"Symbol to report. Defaults to {settings.DEFAULT_SYMBOL}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_symbol = resolve_symbol(args.symbol)
    if canonical_symbol is None:
        print(f"Error: Unsupported symbol: {args.symbol}", file=sys.stderr)
        return 1

    model_dir = ensure_model_dir_exists()
    try:
        diagnostics = _load_json(model_dir / f"{canonical_symbol}_dataset_diagnostics.json")
        classifier_comparison = _load_json(
            model_dir / f"{canonical_symbol}_classifier_model_comparison.json"
        )
        regressor_comparison = _load_json(
            model_dir / f"{canonical_symbol}_regressor_model_comparison.json"
        )
        walk_forward = _load_json(
            model_dir / f"{canonical_symbol}_walk_forward_validation.json"
        )
        metadata = _load_json(model_dir / f"{canonical_symbol}_model_metadata.json")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            f"Run first: python scripts/train_models.py --symbol {canonical_symbol}",
            file=sys.stderr,
        )
        return 1

    print(f"GapSight model report: {canonical_symbol}")
    print("\nDataset diagnostics")
    print(f"- Rows: {diagnostics['row_count']}")
    print(f"- Date range: {diagnostics.get('date_min')} to {diagnostics.get('date_max')}")
    print(f"- Feature count: {diagnostics['feature_count']}")
    print(f"- Class distribution: {diagnostics['class_distribution']}")

    print("\nClassifier comparison")
    print(f"- Selected: {metadata['selected_classifier_type']}")
    _summarize_models(
        classifier_comparison,
        ["macro_f1", "weighted_f1", "accuracy"],
    )

    print("\nRegressor comparison")
    print(f"- Selected: {metadata['selected_regressor_type']}")
    print(f"- Baseline MAE: {_fmt(regressor_comparison['baseline_mae'])}")
    print(f"- Beats baseline: {str(regressor_comparison['beats_baseline']).lower()}")
    _summarize_models(
        regressor_comparison,
        ["mae", "rmse", "r2", "directional_accuracy_from_regression"],
    )

    wf_summary = walk_forward["summary"]
    print("\nWalk-forward validation")
    print(f"- Folds: {wf_summary['fold_count']}")
    print(f"- Avg classifier macro F1: {_fmt(wf_summary['avg_classifier_macro_f1'])}")
    print(f"- Avg classifier weighted F1: {_fmt(wf_summary['avg_classifier_weighted_f1'])}")
    print(f"- Avg classifier accuracy: {_fmt(wf_summary['avg_classifier_accuracy'])}")
    print(f"- Avg regressor MAE: {_fmt(wf_summary['avg_regressor_mae'])}")
    print(f"- Avg regressor RMSE: {_fmt(wf_summary['avg_regressor_rmse'])}")
    print(f"- Avg regressor R2: {_fmt(wf_summary['avg_regressor_r2'])}")
    print(
        "- Regressor beats baseline folds: "
        f"{wf_summary['regressor_beats_baseline_fold_count']}/"
        f"{wf_summary['fold_count']}"
    )

    print("\nSelected models")
    print(f"- Classifier: {metadata['classifier_model_path']}")
    print(f"- Regressor: {metadata['regressor_model_path']}")

    warnings: list[str] = []
    if not metadata["beats_baseline"]:
        warnings.append("Selected regressor does not beat the baseline mean regressor.")
    if wf_summary["regressor_beats_baseline_fold_count"] < wf_summary["fold_count"]:
        warnings.append("Regressor does not beat baseline in every walk-forward fold.")

    print("\nWarnings")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
