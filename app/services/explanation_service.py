"""Human-readable prediction explanation service."""

from typing import Any


def generate_basic_explanation(prediction: dict[str, Any]) -> list[str]:
    """Generate honest, model-based explanations without inventing causes."""
    symbol = prediction.get("symbol", "the symbol")
    reasons = [
        f"Prediction is generated using latest available {symbol} OHLCV features.",
        "Direction confidence is based on classifier probability.",
        "Gap-size range is estimated using historical regressor MAE.",
    ]

    regressor_status = prediction.get("model_quality", {}).get("regressor_status")
    if regressor_status == "usable_but_unstable_baseline":
        reasons.append(
            "Gap-size model currently beats baseline only partially in "
            "walk-forward validation, so treat exact gap estimate cautiously."
        )
    elif regressor_status == "weak_baseline":
        reasons.append(
            "Gap-size model does not currently beat the baseline reliably, so "
            "treat the gap estimate as weak."
        )

    return reasons
