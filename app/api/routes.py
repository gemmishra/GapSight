"""Versioned API routes for GapSight."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.ml.inference.predictor import (
    FeatureMismatchError,
    MissingMetadataError,
    MissingModelArtifactError,
    MissingRawDataError,
    PredictionError,
    UnsupportedSymbolError,
    predict_latest,
)
from app.schemas.prediction import PredictionResponse
from app.schemas.symbols import SupportedSymbol
from app.services.notification_service import (
    MissingNotificationConfigError,
    NotificationError,
    UnsupportedNotificationChannelError,
    send_prediction_alert,
)
from app.utils.constants import SUPPORTED_SYMBOLS

router = APIRouter()

T = TypeVar("T")


def _run_prediction_or_raise(action: Callable[[], T]) -> T:
    try:
        return action()
    except UnsupportedSymbolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MissingRawDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (MissingModelArtifactError, MissingMetadataError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FeatureMismatchError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_notification_or_raise(action: Callable[[], T]) -> T:
    try:
        return _run_prediction_or_raise(action)
    except UnsupportedNotificationChannelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MissingNotificationConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotificationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _verify_openclaw_auth(authorization: str | None) -> None:
    if not settings.ENABLE_OPENCLAW_AUTH:
        return

    expected_token = settings.OPENCLAW_API_TOKEN.strip()
    if not expected_token:
        raise HTTPException(
            status_code=401,
            detail="OpenClaw auth is enabled but OPENCLAW_API_TOKEN is not configured.",
        )

    expected_header = f"Bearer {expected_token}"
    if authorization != expected_header:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid OpenClaw bearer token.",
        )


def _compact_openclaw_prediction(prediction: dict) -> dict:
    return {
        "symbol": prediction["symbol"],
        "direction_label": prediction["direction_label"],
        "confidence_label": prediction["confidence_label"],
        "predicted_gap_percent": prediction["predicted_gap_percent"],
        "predicted_gap_points": prediction["predicted_gap_points"],
        "expected_open": prediction["expected_open"],
        "expected_open_min": prediction["expected_open_min"],
        "expected_open_max": prediction["expected_open_max"],
        "model_quality": prediction["model_quality"],
        "reliability_warning": prediction["reliability_warning"],
        "formatted_alert": prediction["formatted_alert"],
    }


@router.get("/supported-symbols", response_model=list[SupportedSymbol])
async def get_supported_symbols() -> list[SupportedSymbol]:
    """Return the symbols currently supported by GapSight."""
    return [SupportedSymbol(**symbol) for symbol in SUPPORTED_SYMBOLS]


@router.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_symbol(symbol: str) -> PredictionResponse:
    """Return latest model-based prediction for a supported symbol."""
    return PredictionResponse(**_run_prediction_or_raise(lambda: predict_latest(symbol)))


@router.get("/predict/{symbol}/alert", response_class=PlainTextResponse)
async def predict_symbol_alert(symbol: str) -> PlainTextResponse:
    """Return latest formatted prediction alert as plain text."""
    prediction = _run_prediction_or_raise(lambda: predict_latest(symbol))
    return PlainTextResponse(prediction["formatted_alert"])


@router.post("/alerts/{symbol}/send")
async def send_symbol_alert(symbol: str, channel: str = "discord") -> dict:
    """Send latest formatted prediction alert to a notification channel."""
    return _run_notification_or_raise(
        lambda: send_prediction_alert(symbol=symbol, channel=channel)
    )


@router.get("/openclaw/predict/{symbol}")
async def openclaw_predict_symbol(
    symbol: str,
    authorization: str | None = Header(default=None),
) -> dict:
    """Return compact model prediction for OpenClaw tool calls."""
    _verify_openclaw_auth(authorization)
    prediction = _run_prediction_or_raise(lambda: predict_latest(symbol))
    return _compact_openclaw_prediction(prediction)


@router.get("/openclaw/predict/{symbol}/alert", response_class=PlainTextResponse)
async def openclaw_predict_symbol_alert(
    symbol: str,
    authorization: str | None = Header(default=None),
) -> PlainTextResponse:
    """Return compact OpenClaw alert text for conversational responses."""
    _verify_openclaw_auth(authorization)
    prediction = _run_prediction_or_raise(lambda: predict_latest(symbol))
    return PlainTextResponse(prediction["formatted_alert"])
