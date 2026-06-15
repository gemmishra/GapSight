from app.services.alert_formatter import format_prediction_alert
from app.services.prediction_presentation import (
    confidence_label,
    direction_label,
    gap_interpretation,
    reliability_warnings,
)


def test_confidence_label_mapping() -> None:
    assert confidence_label(0.70) == "HIGH"
    assert confidence_label(0.50) == "MEDIUM"
    assert confidence_label(0.49) == "LOW"
    assert confidence_label(None) == "LOW"


def test_direction_label_mapping() -> None:
    assert direction_label("GAP_UP") == "Likely Gap Up"
    assert direction_label("GAP_DOWN") == "Likely Gap Down"
    assert direction_label("FLAT") == "Likely Flat / Neutral Opening"


def test_gap_interpretation_mapping() -> None:
    assert gap_interpretation(0.19) == "Very small expected gap"
    assert gap_interpretation(-0.20) == "Mild expected gap"
    assert gap_interpretation(0.40) == "Moderate expected gap"
    assert gap_interpretation(-0.90) == "Strong expected gap"


def test_reliability_warning_generation() -> None:
    warnings = reliability_warnings(
        {
            "confidence_label": "LOW",
            "model_quality": {
                "regressor_status": "usable_but_unstable_baseline",
            },
        }
    )

    assert warnings == [
        "Direction confidence is low, so treat this as a weak signal.",
        (
            "Gap-size estimate is unstable across walk-forward validation and "
            "should be treated as an approximate range."
        ),
    ]


def test_formatted_alert_contains_key_fields() -> None:
    alert = format_prediction_alert(
        {
            "symbol": "BANKNIFTY",
            "direction_label": "Likely Flat / Neutral Opening",
            "confidence_label": "LOW",
            "predicted_gap_percent": 0.10,
            "expected_open": 55229.32,
            "expected_open_min": 55008.0,
            "expected_open_max": 55450.0,
            "model_quality": {
                "classifier_status": "usable_baseline",
                "regressor_status": "usable_but_unstable_baseline",
            },
            "reliability_warning": [
                "Direction confidence is low, so treat this as a weak signal."
            ],
        }
    )

    assert "BANKNIFTY Pre-Market Prediction" in alert
    assert "Direction: Likely Flat / Neutral Opening" in alert
    assert "Confidence: LOW" in alert
    assert "Expected open: 55,229.32" in alert
    assert "Disclaimer: Probability-based ML estimate, not financial advice." in alert
