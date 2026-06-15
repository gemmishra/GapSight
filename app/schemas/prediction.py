"""Schemas for model prediction responses."""

from datetime import date
from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response contract for model-based inference."""

    symbol: str
    prediction_date: date
    latest_data_date: date
    previous_close: float
    direction: str
    direction_confidence: float | None
    confidence_label: str
    direction_label: str
    class_probabilities: dict[str, float]
    predicted_gap_percent: float
    predicted_gap_points: float
    gap_interpretation: str
    expected_open: float
    expected_open_min: float
    expected_open_max: float
    model_quality: dict[str, Any]
    reliability_warning: list[str]
    formatted_alert: str
    reasons: list[str]
    disclaimer: str
