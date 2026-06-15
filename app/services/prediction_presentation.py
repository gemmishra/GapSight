"""User-facing presentation helpers for prediction output."""

from typing import Any


DIRECTION_LABELS: dict[str, str] = {
    "GAP_UP": "Likely Gap Up",
    "GAP_DOWN": "Likely Gap Down",
    "FLAT": "Likely Flat / Neutral Opening",
}


def confidence_label(confidence: float | None) -> str:
    """Convert numeric classifier confidence to a user-facing label."""
    if confidence is None:
        return "LOW"
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    return "LOW"


def direction_label(direction: str) -> str:
    """Convert a model direction class to user-friendly wording."""
    return DIRECTION_LABELS.get(direction, direction)


def gap_interpretation(predicted_gap_percent: float) -> str:
    """Convert predicted gap percentage magnitude into plain-language severity."""
    absolute_gap = abs(predicted_gap_percent)
    if absolute_gap < 0.20:
        return "Very small expected gap"
    if absolute_gap < 0.40:
        return "Mild expected gap"
    if absolute_gap < 0.90:
        return "Moderate expected gap"
    return "Strong expected gap"


def reliability_warnings(prediction: dict[str, Any]) -> list[str]:
    """Generate reliability warnings from confidence and model quality flags."""
    warnings: list[str] = []
    if prediction.get("confidence_label") == "LOW":
        warnings.append(
            "Direction confidence is low, so treat this as a weak signal."
        )

    regressor_status = prediction.get("model_quality", {}).get("regressor_status")
    if regressor_status == "usable_but_unstable_baseline":
        warnings.append(
            "Gap-size estimate is unstable across walk-forward validation and "
            "should be treated as an approximate range."
        )

    return warnings


def enrich_prediction_presentation(prediction: dict[str, Any]) -> dict[str, Any]:
    """Add user-facing labels and warnings to a prediction dictionary."""
    prediction["confidence_label"] = confidence_label(
        prediction.get("direction_confidence")
    )
    prediction["direction_label"] = direction_label(prediction["direction"])
    prediction["gap_interpretation"] = gap_interpretation(
        float(prediction["predicted_gap_percent"])
    )
    prediction["reliability_warning"] = reliability_warnings(prediction)
    return prediction
