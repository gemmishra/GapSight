"""Versioned API routes for GapSight."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

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
