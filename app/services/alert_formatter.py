"""Format prediction output into a reusable text alert."""

from typing import Any


def _signed_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _format_price(value: float) -> str:
    return f"{value:,.2f}"


def format_prediction_alert(prediction: dict[str, Any]) -> str:
    """Return a clean text alert for future notification integrations."""
    warnings = prediction.get("reliability_warning", [])
    if warnings:
        notes = "\n".join(f"* {warning}" for warning in warnings)
    else:
        notes = "* No major reliability warnings."

    return (
        f"{prediction['symbol']} Pre-Market Prediction\n\n"
        f"Direction: {prediction['direction_label']}\n"
        f"Confidence: {prediction['confidence_label']}\n"
        f"Estimated gap: {_signed_percent(prediction['predicted_gap_percent'])}\n"
        f"Expected open: {_format_price(prediction['expected_open'])}\n"
        "Expected range: "
        f"{_format_price(prediction['expected_open_min'])} - "
        f"{_format_price(prediction['expected_open_max'])}\n\n"
        "Model quality:\n\n"
        f"* Direction model: {prediction['model_quality']['classifier_status']}\n"
        f"* Gap-size model: {prediction['model_quality']['regressor_status']}\n\n"
        "Notes:\n\n"
        f"{notes}\n\n"
        "Disclaimer: Probability-based ML estimate, not financial advice."
    )
